import type { AppStateSnapshot, CourseMaterial, LocalModel, MaterialIndexProgress } from '../../shared/app-state'
import { materialCollectionId } from '../../shared/collections'
import { isPreparationGenerator, preparationNeedsModel, preparationGenerator, type IndexMaterialOptions } from '../../shared/preparation'
import type { LibraryTopicProgress } from '../../shared/topic-inventory'
import { generateTopicInventory, type TopicInventoryStore } from '../../shared/topic-inventory-generation'

export interface BookPreparationDependencies {
  state: () => Promise<AppStateSnapshot | null>
  materials: () => Promise<CourseMaterial[]>
  index: (id: string, path: string, embedder: LocalModel | undefined, options: IndexMaterialOptions | undefined,
    progress: (value: MaterialIndexProgress) => void) => Promise<CourseMaterial>
  topics: (signal: AbortSignal) => TopicInventoryStore
  progress: (value: MaterialIndexProgress) => void
  topicProgress: (value: LibraryTopicProgress) => void
  cancelIndex: (id: string) => Promise<void>
}

export function createBookPreparation(deps: BookPreparationDependencies) {
  // Material IDs are qualified with the collection ID. Topic jobs use the collection ID itself.
  const jobs = new Map<string, AbortController>()
  let queue: Promise<unknown> = Promise.resolve()

  function enqueue<T>(id: string, run: (signal: AbortSignal) => Promise<T>): Promise<T> {
    if (jobs.has(id)) return Promise.reject(new Error('Preparation is already running.'))
    const controller = new AbortController()
    jobs.set(id, controller)
    const result = queue.then(() => run(controller.signal)).finally(() => jobs.delete(id))
    queue = result.catch(() => undefined)
    return result
  }

  function extractTopics(collectionId: string, modelId?: string): Promise<void> {
    if (jobs.has(collectionId)) return Promise.reject(new Error('Practice topics are already being generated for this library.'))
    function report(status: LibraryTopicProgress['status'], message: string, completed = 0, total = 0) {
      deps.topicProgress({ collectionId, modelId, status, completed, total, message })
    }
    report('running', 'Queued for topic generation')
    return enqueue(collectionId, async signal => {
      try {
        signal.throwIfAborted()
        const state = await deps.state()
        const model = state && (preparationGenerator(state.models, modelId, state.settings.application.defaultModelId)
          ?? (modelId ? undefined : state.models.find(isPreparationGenerator)))
        if (!state || !model) throw new Error('Choose an available model in Models to generate Practice topics.')
        modelId = model.id
        const materials = (await deps.materials()).filter(item => materialCollectionId(item.id) === collectionId && item.status === 'ready')
        const store = deps.topics(signal)
        const docs = (await store.studyDocuments()).filter(doc => materials.some(item => item.id === doc.materialId))
        signal.throwIfAborted()
        if (!docs.length) throw new Error('Index documents in this library before generating Practice topics.')
        await generateTopicInventory(store, docs, materials, model, state.settings,
          { ...state.settings.modelDefaults, ...state.settings.modelSettingsById[model.id] }, `library-topics-${collectionId}`, signal,
          (completed, total, _topics, stage) => report('running', stage || 'Generating Practice topics', completed, total))
        report('complete', 'Practice topics ready')
      } catch (error) {
        if (signal.aborted) return report('paused', 'Practice topic generation paused')
        report('error', (error as Error).message)
        throw error
      }
    })
  }

  async function cancel(id: string) {
    jobs.get(id)?.abort()
    if (materialCollectionId(id)) await deps.cancelIndex(id)
  }

  return {
    isCollectionIndexing(id: string) { return [...jobs.keys()].some(key => materialCollectionId(key) === id) },
    cancel,
    cancelTopics(id: string) { if (!materialCollectionId(id)) jobs.get(id)?.abort() },
    extractTopics,
    async cancelCollection(id: string) {
      await Promise.all([...jobs.keys()].filter(key => key === id || materialCollectionId(key) === id).map(cancel))
    },
    cancelAll() { for (const controller of jobs.values()) controller.abort() },
    run(id: string, path: string, embedder?: LocalModel, options?: IndexMaterialOptions): Promise<CourseMaterial> {
      return enqueue(id, async signal => {
        try {
          signal.throwIfAborted()
          const preparation = options?.preparation
          let model: LocalModel | undefined
          if (preparation && preparationNeedsModel(preparation)) {
            const state = await deps.state()
            model = options?.preparationModel ?? (state
              ? preparationGenerator(state.models, preparation.modelId, state.settings.application.defaultModelId, preparation.ocr)
              : undefined)
            if (!model || !isPreparationGenerator(model)) throw new Error('Choose an available preparation model in Libraries.')
          }
          signal.throwIfAborted()
          const material = await deps.index(id, path, embedder, { ...options, preparationModel: model }, progress => {
            if (!signal.aborted && progress.phase !== 'complete') deps.progress(progress)
          })
          signal.throwIfAborted()
          if (material.status !== 'ready') throw new Error(material.error || 'Document indexing did not complete.')
          deps.progress({ materialId: id, phase: 'complete', percent: 100, message: 'Ready for chat' })

          // Explicit opt-in queues topic extraction after the committed index is usable.
          const collectionId = materialCollectionId(material.id)
          if (preparation?.extractTopics && collectionId && !jobs.has(collectionId)) {
            void extractTopics(collectionId, model?.id ?? preparation.modelId).catch(() => { /* Reported separately through topic progress. */ })
          }
          return material
        } catch (error) {
          deps.progress({ materialId: id, phase: 'error', percent: 0, message: signal.aborted ? 'Preparation paused. Resume in Libraries.' : (error as Error).message })
          throw error
        }
      })
    }
  }
}
