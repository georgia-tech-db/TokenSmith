import type { CourseMaterial } from './app-state'
import type { StudyDocumentRef } from './study-scope'
import { hasCurrentTopicInventory, type InventoryTopic, type TopicInventory } from './topic-inventory'

/** Read-only: opening Practice must never start a model or write an inventory. */
export async function loadSavedTopics(store: { loadTopicInventory: (id: string) => Promise<TopicInventory | null> },
  documents: StudyDocumentRef[], materials: CourseMaterial[]): Promise<InventoryTopic[]> {
  const selected = new Set(documents.map(d => `${d.materialId}:${d.documentId}`))
  const ids = [...new Set(documents.map(d => d.materialId.split('/')[0]))]
  const inventories = await Promise.all(ids.map(async id => {
    const saved = await store.loadTopicInventory(id)
    const revision = materials.find(m => m.collectionRevision && m.id.startsWith(`${id}/`))?.collectionRevision
    return hasCurrentTopicInventory(saved, revision) ? [saved] : []
  }))
  const available = inventories.flat()
  if (ids.length && !available.length) throw new Error('Generate Practice topics in Libraries to browse chapters, or enter your own topic below.')
  return available.flatMap(i => i.topics).map(t => ({ ...t, sources: t.sources.filter(s => selected.has(`${s.materialId}:${s.documentId}`)) })).filter(t => t.sources.length)
}
