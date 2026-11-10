import type { AppStateSnapshot, CourseMaterial, LocalModel, MaterialIndexProgress } from '../../shared/app-state'
import type { IndexMaterialOptions } from '../../shared/preparation'
import { generateTopicInventory, type TopicInventoryStore } from '../../shared/topic-inventory-generation'

export interface BookPreparationDependencies {
  state: () => Promise<AppStateSnapshot | null>
  index: (id: string, path: string, embedder: LocalModel | undefined, options: IndexMaterialOptions | undefined,
    progress: (value: MaterialIndexProgress) => void) => Promise<CourseMaterial>
  topics: (signal: AbortSignal) => TopicInventoryStore
  progress: (value: MaterialIndexProgress) => void
  cancelIndex: (id: string) => Promise<void>
}

/** Import is complete only after indexing AND chapter topics have been saved. */
export function createBookPreparation(deps: BookPreparationDependencies) {
  const jobs = new Map<string, AbortController>()
  let queue: Promise<unknown> = Promise.resolve()
  return {
    async cancel(id: string) { jobs.get(id)?.abort(); await deps.cancelIndex(id) },
    async cancelCollection(id: string) { await Promise.all([...jobs.keys()].filter(key => key.startsWith(`${id}/`)).map(async key => { jobs.get(key)?.abort(); await deps.cancelIndex(key) })) },
    cancelAll() { for (const controller of jobs.values()) controller.abort() },
    run(id: string, path: string, embedder?: LocalModel, options?: IndexMaterialOptions): Promise<CourseMaterial> {
      if (jobs.has(id)) return Promise.reject(new Error('This book is already being prepared.'))
      const controller = new AbortController(); jobs.set(id, controller)
      const run = async () => {
        const signal = controller.signal
        signal.throwIfAborted()
        const state = await deps.state()
        const model = state?.models.find(m => m.engine === 'ollama' && m.ollamaModelName === 'gemma4:26b' && m.status === 'ready')
        if (!state || !model) throw new Error('Install local gemma4:26b in Models before preparing this book.')
        const material = await deps.index(id, path, embedder, options, p => deps.progress({ ...p,
          phase: p.phase === 'complete' ? 'topics' : p.phase, percent: Math.min(75, p.percent * 0.75),
          message: p.phase === 'complete' ? 'Preparing chapter topics' : p.message }))
        signal.throwIfAborted()
        if (material.status !== 'ready') throw new Error(material.error || 'Finish document indexing before preparing chapter topics.')
        const store = deps.topics(signal)
        const docs = (await store.studyDocuments()).filter(d => d.materialId === id)
        if (!docs.length) throw new Error('No indexed documents are available for chapter topics.')
        await generateTopicInventory(store, docs, [material], model, state.settings,
          { ...state.settings.modelDefaults, ...state.settings.modelSettingsById[model.id] }, `import-topics-${id}`, signal,
          (completed, total, _topics, stage) => deps.progress({ materialId: id, phase: 'topics',
            percent: 75 + 24 * (total ? completed / total : 0),
            message: stage || `Saving chapter topics (${completed}/${total})` }))
        signal.throwIfAborted()
        deps.progress({ materialId: id, phase: 'complete', percent: 100, message: 'Ready for chat and practice' })
        return { ...material, status: 'ready' as const, detail: 'Ready for chat and practice' }
      }
      const result = queue.then(run).catch(error => {
        deps.progress({ materialId: id, phase: 'error', percent: 0, message: controller.signal.aborted ? 'Preparation paused. Resume in Libraries.' : (error as Error).message })
        throw error
      }).finally(() => { jobs.delete(id) })
      queue = result.catch(() => undefined)
      return result
    }
  }
}
