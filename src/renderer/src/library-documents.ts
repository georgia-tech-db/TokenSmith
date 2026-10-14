import type { CourseMaterial, LocalModel } from '@shared/app-state'
import type { Collection } from '@shared/collections'
import { collectionMaterialId } from '@shared/collections'
import { defaultPreparation, type IndexMaterialOptions } from '@shared/preparation'
import type { StudyDocument } from '@shared/study-scope'

export const libraryError = (reason: unknown) => (reason instanceof Error ? reason.message : String(reason))
  .replace(/^Error invoking remote method '[^']+':\s*(?:Error:\s*)?/, '')

export interface DocumentImportTarget {
  collection?: Collection
  materials: CourseMaterial[]
  documents?: StudyDocument[]
  models: LocalModel[]
  selectedEmbeddingModelId: string
  onAddMaterials: (items: CourseMaterial[]) => void
  onStartMaterialIndexing: (id: string, path: string, model?: LocalModel, options?: IndexMaterialOptions) => void
}
export function librarySearchModel(target: Pick<DocumentImportTarget, 'models' | 'selectedEmbeddingModelId'>) {
  const models = target.models.filter(m => (m.role === 'embedder' || m.role === 'both') && m.status === 'ready')
  const model = models.find(m => m.id === target.selectedEmbeddingModelId) ?? models[0]
  if (!model) throw new Error('Add a search model in Models before adding documents.')
  return model
}
export function addLibraryDocuments(picked: CourseMaterial[], target: DocumentImportTarget) {
  const model = librarySearchModel(target)
  const paths = new Set([...target.materials.map(m => m.path), ...(target.documents ?? []).map(d => d.path)])
  const items = picked.filter(item => {
    if (!item.path || paths.has(item.path)) return false
    paths.add(item.path)
    return true
  }).map(item => ({ ...item, id: target.collection ? collectionMaterialId(target.collection.id, `document-${crypto.randomUUID()}`) : item.id,
    collectionId: target.collection?.id, collectionTitle: target.collection?.name,
    kind: 'document' as const, status: 'indexing' as const, isActive: false, preparation: defaultPreparation() }))
  if (!items.length) throw new Error('These documents are already in this library.')
  target.onAddMaterials(items)
  for (const item of items) target.onStartMaterialIndexing(item.id, item.path!, model, { title: item.title, preparation: item.preparation })
}
