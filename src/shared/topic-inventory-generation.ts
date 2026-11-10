import type { EngineChatRequest, EngineChatResponse } from './engine'
import type { ChatSource } from './app-state'
import type { StudyDocument } from './study-scope'
import type { CourseMaterial, LocalModel, ModelRuntimeSettings, TokenSmithSettings } from './app-state'
import type { StudyDocumentRef } from './study-scope'
import { practiceSourceKey } from './practice'
import { buildTopicBatches, consolidateTopics, parseInventoryTopics, topicInventoryVersion,
  type InventoryTopic, type TopicInventory } from './topic-inventory'

export interface TopicInventoryStore {
  studyDocuments(): Promise<StudyDocument[]>
  topicSources(documents: StudyDocumentRef[]): Promise<ChatSource[]>
  loadTopicInventory(id: string): Promise<TopicInventory | null>
  saveTopicInventory(inventory: TopicInventory, revision: string): Promise<void>
  sendChatMessage(request: EngineChatRequest): Promise<EngineChatResponse>
}

export async function generateTopicInventory(bridge: TopicInventoryStore, documents: StudyDocumentRef[], materials: CourseMaterial[],
  model: LocalModel, settings: TokenSmithSettings, modelSettings: ModelRuntimeSettings, requestId: string,
  signal: AbortSignal, progress: (completed: number, total: number, topics: InventoryTopic[], stage?: string) => void): Promise<InventoryTopic[]> {
  if (model.engine !== 'ollama' || model.ollamaModelName !== 'gemma4:26b') throw new Error('Install local gemma4:26b in Models to generate chapter topics.')
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
        progress(completed, total, visible([...result, ...consolidateTopics(Object.values(inventory.batches).flatMap(b => b.topics))]), `Reading ${batch.area}`)
        const evidence = batch.sources.map((s, i) => ({ ...s, materialId: 'chapter', documentId: 0, sourceUnitId: `term-${i + 1}` }))
        const reply = await bridge.sendChatMessage({ requestId, practiceTask: 'chapter_topics', practiceAllowedIds: evidence.map(practiceSourceKey),
          model, settings, materials: [], messages: [], prompt: JSON.stringify({ chapter: batch.area }), retrievedSources: evidence,
          conversationContextMode: 'standalone', reasoning: false,
          applicationSettings: { ...settings.application, suggestionMode: 'off', explanationDepthEnabled: false },
          modelSettings: { ...modelSettings, systemMessage: '', contextLengthMode: 'manual', contextLength: 32768,
            reasoningMode: 'off', thinking: false, maxLength: 4096 } })
        signal.throwIfAborted()
        const included = new Set(reply.sources.map(practiceSourceKey))
        if (evidence.some(s => !included.has(practiceSourceKey(s)))) throw new Error('The full chapter part did not fit the model context. No partial result was saved.')
        const topics = parseInventoryTopics(reply.text, { ...batch, sources: evidence }, plan.id).map(t => ({ ...t,
          sources: t.sources.map(s => batch.sources[evidence.findIndex(e => practiceSourceKey(e) === practiceSourceKey(s))]) }))
        inventory.batches[batch.id] = { fingerprint: batch.fingerprint, topics }
      }
      inventory.topics = consolidateTopics(Object.values(inventory.batches).flatMap(b => b.topics))
      inventory.completed = Object.keys(inventory.batches).length; completed++
      signal.throwIfAborted()
      await bridge.saveTopicInventory(inventory, revision)
      progress(completed, total, visible([...result, ...inventory.topics]))
    }
    if (!plan.batches.length) throw new Error('No substantive chapter text was found in this collection.')
    inventory.complete = true
    signal.throwIfAborted()
    await bridge.saveTopicInventory(inventory, revision)
    result.push(...inventory.topics)
  }
  return visible(result)
}
