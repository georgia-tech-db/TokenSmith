import type { AppStateSnapshot, CourseMaterial, LocalModel, MaterialIndexProgress } from '../../shared/app-state'
import { defaultPreparation, preparationNeedsModel, preparationGenerator, type IndexMaterialOptions } from '../../shared/preparation'
import { generateTopicInventory, type TopicInventoryStore } from '../../shared/topic-inventory-generation'

export interface BookPreparationDependencies {
  state: () => Promise<AppStateSnapshot | null>
  materials?: () => Promise<CourseMaterial[]>
  index: (id: string, path: string, embedder: LocalModel | undefined, options: IndexMaterialOptions | undefined,
    progress: (value: MaterialIndexProgress) => void) => Promise<CourseMaterial>
  topics: (signal: AbortSignal) => TopicInventoryStore
  progress: (value: MaterialIndexProgress) => void
  cancelIndex: (id: string) => Promise<void>
}

/** Import is complete only after indexing AND chapter topics have been saved. */
export function createBookPreparation(deps: BookPreparationDependencies) {
  const jobs = new Map<string, AbortController>()
  const indexed = new Set<string>()
  let queue: Promise<unknown> = Promise.resolve()
  return {
    isCollectionIndexing(id: string) { return [...jobs.keys()].some(key => key.startsWith(`${id}/`) && !indexed.has(key)) },
    isCollectionBusy(id: string) { return [...jobs.keys()].some(key => key.startsWith(`${id}/`)) },
    async cancel(id: string) { jobs.get(id)?.abort(); await deps.cancelIndex(id) },
    async cancelCollection(id: string) { await Promise.all([...jobs.keys()].filter(key => key.startsWith(`${id}/`)).map(async key => { jobs.get(key)?.abort(); await deps.cancelIndex(key) })) },
    cancelAll() { for (const controller of jobs.values()) controller.abort() },
    run(id: string, path: string, embedder?: LocalModel, options?: IndexMaterialOptions, topicsOnly = false): Promise<CourseMaterial> {
      if (jobs.has(id)) return Promise.reject(new Error('This book is already being prepared.'))
      const controller = new AbortController(); jobs.set(id, controller)
      if (topicsOnly) indexed.add(id)
      const run = async () => {
        const signal = controller.signal
        signal.throwIfAborted()
        const state = await deps.state()
        const model = options?.preparationModel ?? (state && preparationGenerator(state.models, options?.preparation?.modelId, state.settings?.application.defaultModelId, options?.preparation?.ocr))
        if (!state || (preparationNeedsModel(options?.preparation ?? defaultPreparation()) && (!model || model.status !== 'ready' || model.role === 'embedder' || !['ollama', 'remote'].includes(model.engine)))) throw new Error('Choose an available preparation model in Libraries. Gemma 4 E4B is the default.')
        const material = topicsOnly ? (await deps.materials?.())?.find(item => item.id === id) : await deps.index(id, path, embedder, { ...options, preparationModel: model ?? undefined }, p => deps.progress({ ...p,
          phase: p.phase === 'complete' ? (options?.preparation?.extractTopics === false ? 'saving' : 'topics') : p.phase, percent: Math.min(75, p.percent * 0.75),
          message: p.phase === 'complete' ? (options?.preparation?.extractTopics === false ? 'Finishing preparation' : 'Preparing chapter topics') : p.message }))
        signal.throwIfAborted()
        if (!material || (!topicsOnly && material.status !== 'ready')) throw new Error(material?.error || 'Finish document indexing before preparing chapter topics.')
        if (!topicsOnly && options?.preparation?.extractTopics === false) {
          deps.progress({ materialId: id, phase: 'complete', percent: 100, message: 'Ready · Topic extraction off' })
          return { ...material, status: 'ready' as const, detail: 'Ready · Topic extraction off' }
        }
        if (!model) throw new Error('Choose a model for topic extraction.')
        indexed.add(id)
        const store = deps.topics(signal)
        const docs = (await store.studyDocuments()).filter(d => d.materialId === material.id)
        if (!docs.length) throw new Error('No indexed documents are available for chapter topics.')
        await generateTopicInventory(store, docs, [material], model, state.settings,
          { ...state.settings.modelDefaults, ...state.settings.modelSettingsById[model.id] }, `import-topics-${id}`, signal,
          (completed, total, _topics, stage) => deps.progress({ materialId: id, phase: 'topics', resumePhase: 'topics',
            percent: (topicsOnly ? 0 : 75) + (topicsOnly ? 99 : 24) * (total ? completed / total : 0),
            message: stage || `Saving chapter topics (${completed}/${total})` }))
        signal.throwIfAborted()
        deps.progress({ materialId: id, phase: 'complete', percent: 100, message: 'Ready for chat and practice' })
        return { ...material, status: 'ready' as const, detail: 'Ready for chat and practice' }
      }
      const result = queue.then(run).catch(error => {
        deps.progress({ materialId: id, phase: 'error', ...(indexed.has(id) ? { resumePhase: 'topics' as const } : {}), percent: 0, message: controller.signal.aborted ? 'Preparation paused. Resume in Libraries.' : (error as Error).message })
        throw error
      }).finally(() => { jobs.delete(id); indexed.delete(id) })
      queue = result.catch(() => undefined)
      return result
    }
  }
}
