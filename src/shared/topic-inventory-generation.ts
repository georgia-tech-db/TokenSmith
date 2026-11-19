import { localModelCatalog } from './model-catalog'
import type { EngineChatRequest, EngineChatResponse } from './engine'
import type { ChatSource } from './app-state'
import type { StudyDocument } from './study-scope'
import type { CourseMaterial, LocalModel, ModelRuntimeSettings, TokenSmithSettings } from './app-state'
import type { StudyDocumentRef } from './study-scope'
import { practiceSourceKey } from './practice'
import { buildTopicBatches, consolidateTopics, parseInventoryTopics, topicInventoryVersion,
  type InventoryTopic, type TopicInventory, type TopicBatch } from './topic-inventory'

export interface TopicInventoryStore {
  studyDocuments(): Promise<StudyDocument[]>
  topicSources(documents: StudyDocumentRef[]): Promise<ChatSource[]>
  loadTopicInventory(id: string): Promise<TopicInventory | null>
  saveTopicInventory(inventory: TopicInventory, revision: string): Promise<void>
  sendChatMessage(request: EngineChatRequest): Promise<EngineChatResponse>
}


export function topicExtractionStatus(batch: TopicBatch, current: number, total: number, saved: number, model: LocalModel): string {
  const ranges = batch.sources.flatMap(source => source.pageStart ? [source.pageStart, source.pageEnd ?? source.pageStart] : [])
  const lines = batch.sources.flatMap(source => source.lineFrom ? [source.lineFrom, source.lineTo ?? source.lineFrom] : [])
  const locator = ranges.length ? `Pages ${Math.min(...ranges)}–${Math.max(...ranges)}`
    : lines.length ? `Lines ${Math.min(...lines)}–${Math.max(...lines)}` : ''
  const name = localModelCatalog.find(item => item.id === model.ollamaModelName)?.displayName || model.name || model.ollamaModelName
  return ['Extracting topics', batch.sources[0]?.documentTitle, batch.area === batch.sources[0]?.documentTitle ? '' : batch.area, locator,
    total > 1 ? `Text batch ${current} of ${total}` : '', `${saved} topics saved`, name].filter(Boolean).join(' · ')
}

export async function generateTopicInventory(bridge: TopicInventoryStore, documents: StudyDocumentRef[], materials: CourseMaterial[],
  model: LocalModel, settings: TokenSmithSettings, modelSettings: ModelRuntimeSettings, requestId: string,
  signal: AbortSignal, progress: (completed: number, total: number, topics: InventoryTopic[], stage?: string) => void): Promise<InventoryTopic[]> {
  if (!['ollama', 'remote'].includes(model.engine) || model.role === 'embedder') throw new Error('Choose a generation model for chapter topics.')
  const collectionIds = [...new Set(documents.map(d => d.materialId.split('/')[0]))]
  const allDocuments = (await bridge.studyDocuments()).filter(d => collectionIds.includes(d.materialId.split('/')[0]))
  signal.throwIfAborted()
  const allSources = await bridge.topicSources(allDocuments)
  signal.throwIfAborted()
  const plans = collectionIds.map(id => ({ id, batches: buildTopicBatches(allSources.filter(s => s.materialId?.startsWith(`${id}/`))) }))
  const total = plans.reduce((n, p) => n + p.batches.length, 0)
  let completed = 0
  const result: InventoryTopic[] = []
  const selected = new Set(documents.map(d => `${d.materialId}:${d.documentId}`))
  const visible = (topics: InventoryTopic[]) => topics.map(t => ({ ...t, sources: t.sources.filter(s => selected.has(`${s.materialId}:${s.documentId}`)) })).filter(t => t.sources.length)
  for (const plan of plans) {
    const revision = materials.find(m => m.id.startsWith(`${plan.id}/`))?.collectionRevision
    if (!revision) throw new Error('Collection revision is unavailable. Reopen the library.')
    const saved = await bridge.loadTopicInventory(plan.id)
    signal.throwIfAborted()
    const inventory: TopicInventory = { version: topicInventoryVersion, collectionId: plan.id, indexRevision: revision, modelId: model.id,
      batches: {}, topics: [], complete: false, completed: 0, total: plan.batches.length }
    // Seed all valid checkpoints before new work so cancellation never loses later chapters.
    for (const batch of plan.batches) {
      const checkpoint = saved?.version === topicInventoryVersion && saved.modelId === model.id ? saved.batches[batch.id] : undefined
      const currentSources = new Map(batch.sources.map(s => [practiceSourceKey(s), s]))
      if (checkpoint?.fingerprint === batch.fingerprint && checkpoint.topics.every(t => t.sources.every(s => currentSources.has(practiceSourceKey(s)))))
        inventory.batches[batch.id] = { ...checkpoint, topics: checkpoint.topics.map(t => ({ ...t, sources: t.sources.map(s => currentSources.get(practiceSourceKey(s))!) })) }
    }
    for (const batch of plan.batches) {
      signal.throwIfAborted()
      if (!inventory.batches[batch.id]) {
        progress(completed, total, visible([...result, ...consolidateTopics(Object.values(inventory.batches).flatMap(b => b.topics))]), topicExtractionStatus(batch, completed + 1, total, consolidateTopics(Object.values(inventory.batches).flatMap(b => b.topics)).length, model))
        const evidence = batch.sources.map((s, i) => ({ ...s, materialId: 'chapter', documentId: 0, sourceUnitId: `term-${i + 1}` }))
        let validationError = ''
        for (let attempt = 0; attempt < 3; attempt++) {
          signal.throwIfAborted()
          if (attempt) progress(completed, total, visible([...result, ...inventory.topics]), `Retrying topic extraction (${attempt + 1}/3) · ${topicExtractionStatus(batch, completed + 1, total, inventory.topics.length, model)}`)
          const reply = await bridge.sendChatMessage({ requestId, practiceTask: 'chapter_topics', practiceAllowedIds: evidence.map(practiceSourceKey),
            model, settings, materials: [], messages: [], prompt: JSON.stringify({ chapter: batch.area, ...(validationError ? { correction: `The previous response failed validation: ${validationError} Return a corrected complete topics object. Use only the supplied source keys, titles of at most 100 characters without questions, 0–6 aliases, 2–5 nonempty objectives, and 1–6 source keys per topic.` } : {}) }), retrievedSources: evidence,
            conversationContextMode: 'standalone', reasoning: false,
            applicationSettings: { ...settings.application, suggestionMode: 'off', explanationDepthEnabled: false },
            modelSettings: { ...modelSettings, systemMessage: '', contextLengthMode: 'manual', contextLength: 32768,
              reasoningMode: 'off', thinking: false, maxLength: 4096 } })
          signal.throwIfAborted()
          const included = new Set(reply.sources.map(practiceSourceKey))
          if (evidence.some(s => !included.has(practiceSourceKey(s)))) throw new Error('The full chapter part did not fit the model context. No partial result was saved.')
          try {
            const topics = parseInventoryTopics(reply.text, { ...batch, sources: evidence }, plan.id).map(t => ({ ...t,
              sources: t.sources.map(s => batch.sources[evidence.findIndex(e => practiceSourceKey(e) === practiceSourceKey(s))]) }))
            inventory.batches[batch.id] = { fingerprint: batch.fingerprint, topics }
            break
          } catch (reason) {
            validationError = reason instanceof Error ? reason.message : 'Invalid topic response.'
            if (attempt === 2) throw new Error(`Topic extraction failed for text batch ${completed + 1} of ${total} after 3 attempts. ${inventory.topics.length} topics are saved. Retry extraction to resume. ${validationError}`)
          }
        }
      }
      inventory.topics = consolidateTopics(Object.values(inventory.batches).flatMap(b => b.topics))
      inventory.completed = Object.keys(inventory.batches).length; completed++
      signal.throwIfAborted()
      await bridge.saveTopicInventory(inventory, revision)
      progress(completed, total, visible([...result, ...inventory.topics]), `Saved topics · ${batch.sources[0]?.documentTitle || batch.area} · ${total > 1 ? `${completed} of ${total} text batches complete · ` : ''}${inventory.topics.length} topics saved`)
    }
    if (!plan.batches.length) throw new Error('No substantive chapter text was found in this collection.')
    inventory.complete = true
    signal.throwIfAborted()
    await bridge.saveTopicInventory(inventory, revision)
    result.push(...inventory.topics)
  }
  return visible(result)
}
