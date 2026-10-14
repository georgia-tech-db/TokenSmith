import type { ChatSource, CourseMaterial } from '../../shared/app-state'
import type { StudyDocument, StudyDocumentRef } from '../../shared/study-scope'
import { collectionMaterialId, splitMaterialId } from '../../shared/collections'
import type { CollectionRegistry, MountedCollection } from './collection-registry'

export function localDocuments(item: MountedCollection, documents: StudyDocumentRef[]): StudyDocumentRef[] {
  return documents.map(document => {
    const { collectionId, localId } = splitMaterialId(document.materialId)
    if (collectionId !== item.id) throw new Error('The document belongs to a different collection.')
    return { materialId: localId, documentId: document.documentId }
  })
}
export function qualifyMaterial(registry: CollectionRegistry, item: MountedCollection, material: CourseMaterial): CourseMaterial {
  return { ...material, id: collectionMaterialId(item.id, material.id), collectionId: item.id, collectionTitle: item.name, collectionRevision: item.indexRevision,
    path: material.path ? registry.sourcePath(item, material.path, material.id) : undefined }
}
export function qualifyDocument(registry: CollectionRegistry, item: MountedCollection, document: StudyDocument): StudyDocument {
  return { ...document, materialId: collectionMaterialId(item.id, document.materialId), collectionId: item.id, collectionTitle: item.name,
    path: registry.sourcePath(item, document.path, document.materialId) }
}
export function qualifySource(registry: CollectionRegistry, item: MountedCollection, source: ChatSource): ChatSource {
  if (!source.materialId) throw new Error('The indexed passage is missing its folder reference.')
  return { ...source, collectionRevision: item.indexRevision, materialId: collectionMaterialId(item.id, source.materialId),
    collectionName: item.name,
    path: source.path ? registry.sourcePath(item, source.path, source.materialId) : undefined,
    thumbnailPath: source.thumbnailPath ? registry.sourcePath(item, source.thumbnailPath) : undefined }
}
// Scores from distinct indexes/embedding models are not directly comparable.
// Interleave their rankings, retaining each source's qualified identity.
export function mergeCollectionSources(groups: ChatSource[][], limit: number): ChatSource[] {
  const merged: ChatSource[] = []
  for (let rank = 0; merged.length < limit && groups.some(group => group.length > rank); rank++) {
    for (const group of groups) {
      if (group[rank]) merged.push(group[rank])
      if (merged.length === limit) break
    }
  }
  return merged
}
