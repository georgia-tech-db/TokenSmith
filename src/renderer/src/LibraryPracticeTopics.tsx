import { useEffect, useId, useState } from 'react'
import type { LocalModel } from '@shared/app-state'
import type { Collection } from '@shared/collections'
import { isPreparationGenerator, preparationGenerator } from '@shared/preparation'
import { hasCurrentTopicInventory, type LibraryTopicProgress, type TopicInventory } from '@shared/topic-inventory'
import { libraryError } from './library-documents'

export function LibraryPracticeTopics({ collection, models, defaultModelId, canGenerate, progress }: {
  collection: Collection
  models: LocalModel[]
  defaultModelId: string
  canGenerate: boolean
  progress?: LibraryTopicProgress
}) {
  const modelSelectId = useId()
  const [inventory, setInventory] = useState<TopicInventory | null>()
  const [modelId, setModelId] = useState('')
  const [error, setError] = useState('')
  const generators = models.filter(isPreparationGenerator)
  const model = generators.find(item => item.id === modelId)
    ?? generators.find(item => item.id === (progress?.modelId ?? inventory?.modelId))
    ?? preparationGenerator(generators, undefined, defaultModelId)
    ?? generators[0]
  const running = progress?.status === 'running'
  const current = hasCurrentTopicInventory(inventory, collection.indexRevision)
  const stale = inventory && inventory.indexRevision !== collection.indexRevision
  const resumable = progress?.status === 'paused' || (inventory && !inventory.complete && inventory.completed > 0)

  useEffect(() => {
    let active = true
    void window.tokensmith?.loadTopicInventory?.(collection.id)
      .then(value => { if (active) setInventory(value) })
      .catch(() => { if (active) setInventory(null) })
    return () => { active = false }
  }, [collection.id, collection.indexRevision, progress?.status])

  function run(action: () => Promise<void>) {
    setError('')
    void action().catch(reason => setError(libraryError(reason)))
  }

  const message = running ? progress.message
    : current ? 'Practice topics ready'
      : stale ? 'Documents changed. Update the Practice topics for this library.'
        : resumable ? 'Practice topic generation paused.'
          : inventory === undefined ? 'Checking saved Practice topics…' : 'Generate topics to browse by chapter in Practice.'

  return <section className="library-preparation-status" aria-label="Practice topics">
    <p role="status">{message}</p>
    {(error || progress?.status === 'error') && <p className="inline-error" role="alert">{error || progress?.message}</p>}
    {running ? <>
      <progress max={progress?.total || 1} value={progress?.total ? progress.completed : undefined} aria-label="Practice topic generation progress" />
      <div><button className="secondary-action" onClick={() => run(() => window.tokensmith!.cancelLibraryTopics(collection.id))}>Pause topic generation</button></div>
    </> : <>
      {model && <label htmlFor={modelSelectId}>Topic model</label>}
      <div className="collection-header-actions library-topic-actions">
        {model ? <select id={modelSelectId} value={model.id} onChange={event => setModelId(event.target.value)}>
          {generators.map(item => <option key={item.id} value={item.id}>{item.name}</option>)}
        </select> : <p>Connect a model in Models to generate Practice topics.</p>}
        <button className="secondary-action" disabled={!model || !canGenerate || inventory === undefined}
          onClick={() => model && run(() => window.tokensmith!.extractLibraryTopics(collection.id, model.id))}>
          {stale || current ? 'Update Practice topics' : resumable ? 'Resume Practice topics' : 'Generate Practice topics'}
        </button>
      </div>
    </>}
  </section>
}
