import type { CourseMaterial, LocalModel } from '@shared/app-state'
import type { Collection } from '@shared/collections'
import { collectionMaterialId } from '@shared/collections'
import { preparationNeedsModel, preparationGenerator, type IndexMaterialOptions } from '@shared/preparation'
import type { StudyDocument } from '@shared/study-scope'

export const libraryError = (reason: unknown) => (reason instanceof Error ? reason.message : String(reason))
  .replace(/^Error invoking remote method '[^']+':\s*(?:Error:\s*)?/, '')

export interface DocumentImportTarget {
  collection?: Collection
  materials: CourseMaterial[]
  documents?: StudyDocument[]
  models: LocalModel[]
  selectedModelId?: string
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
export async function pickDocumentFolder(pick: () => Promise<import('./DocumentFolderPreparation').PreparedFolder>): Promise<CourseMaterial[]> {
  const picked = await pick()
  if (picked.canceled || !picked.path || !picked.preparation) return []
  return [{ id: crypto.randomUUID(), title: picked.title || 'Document folder', path: picked.path,
    kind: picked.kind ?? 'folder', status: 'paused', detail: 'Ready to prepare', addedAt: new Date().toISOString(),
    preparation: { ...picked.preparation } }]
}
export function addLibraryDocuments(picked: CourseMaterial[], target: DocumentImportTarget) {
  const paths = new Set([...target.materials.map(m => m.path), ...(target.documents ?? []).map(d => d.path)])
  const items = picked.filter(item => {
    if (!item.path || paths.has(item.path)) return false
    paths.add(item.path)
    return true
  }).map(item => ({ ...item, id: target.collection ? collectionMaterialId(target.collection.id, `document-${crypto.randomUUID()}`) : item.id,
    collectionId: target.collection?.id, collectionTitle: target.collection?.name,
    kind: item.kind, status: 'paused' as const, isActive: false, preparation: item.preparation, detail: 'Set up models to prepare this folder.' }))
  if (!items.length) throw new Error('These documents are already in this library.')
  target.onAddMaterials(items)
  const embedder = target.models.find(m => m.id === target.selectedEmbeddingModelId && m.status === 'ready')
    ?? target.models.find(m => (m.role === 'embedder' || m.role === 'both') && m.status === 'ready')
  if (embedder) for (const item of items) {
    const generator = preparationGenerator(target.models, item.preparation?.modelId, target.selectedModelId, item.preparation?.ocr)
    if (!generator && item.preparation && preparationNeedsModel(item.preparation)) continue
    if (item.preparation) target.onStartMaterialIndexing(item.id, item.path!, embedder, {
      title: item.title, preparation: generator ? { ...item.preparation, modelId: generator.id, modelName: generator.name } : item.preparation,
      preparationModel: generator
    })
  }
}
