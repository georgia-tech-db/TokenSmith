import type { StudyDocumentRef } from './study-scope'

export interface CollectionManifest {
  format: 'tokensmith-collection'
  version: 1
  id: string
  name: string
  indexRevision: string
  sourceBasePath?: string
}

export interface Collection extends CollectionManifest {
  path: string
  loaded: boolean
  error?: string
}

// Material IDs are qualified at the storage boundary. Document and chunk IDs
// remain local to their database, and always travel with a qualified material ID.
export function collectionMaterialId(collectionId: string, localId: string): string {
  return `${collectionId}/${localId}`
}

export function splitMaterialId(id: string): { collectionId: string; localId: string } {
  const slash = id.indexOf('/')
  if (slash < 1 || slash === id.length - 1) throw new Error('Open the library before using this document.')
  return { collectionId: id.slice(0, slash), localId: id.slice(slash + 1) }
}

export function materialCollectionId(id?: string): string | undefined {
  return id?.includes('/') ? id.slice(0, id.indexOf('/')) : undefined
}

export interface SavedCollectionTopics {
  version: 1
  indexRevision: string
  topics: Array<{ title: string; documents: StudyDocumentRef[] }>
}
