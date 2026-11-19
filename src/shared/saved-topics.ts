import type { CourseMaterial } from './app-state'
import type { StudyDocumentRef } from './study-scope'
import { topicInventoryVersion, type InventoryTopic, type TopicInventory } from './topic-inventory'

/** Read-only: opening Practice must never start a model or write an inventory. */
export async function loadSavedTopics(store: { loadTopicInventory: (id: string) => Promise<TopicInventory | null> },
  documents: StudyDocumentRef[], materials: CourseMaterial[]): Promise<InventoryTopic[]> {
  const selected = new Set(documents.map(d => `${d.materialId}:${d.documentId}`))
  const ids = [...new Set(documents.map(d => d.materialId.split('/')[0]))]
  const inventories = await Promise.all(ids.map(async id => {
    const saved = await store.loadTopicInventory(id)
    const revision = materials.find(m => m.id.startsWith(`${id}/`))?.collectionRevision
    if (!saved || saved.version !== topicInventoryVersion || !saved.complete || !revision || saved.indexRevision !== revision)
      throw new Error('No chapter topics are available yet. Extract topics from the indexed documents to choose chapters and topics here.')
    return saved
  }))
  return inventories.flatMap(i => i.topics).map(t => ({ ...t, sources: t.sources.filter(s => selected.has(`${s.materialId}:${s.documentId}`)) })).filter(t => t.sources.length)
}
