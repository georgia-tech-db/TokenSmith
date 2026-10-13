import type { AppStateSnapshot, ChatSource } from './app-state'
import type { EmbeddingOptions } from './embedding-settings'

export interface StudyDocumentRef { materialId: string; documentId: number }
export interface StudyDocument extends StudyDocumentRef {
  title: string
  path: string
  collectionName: string
  chunkCount: number
}
export interface LibrarySearchOptions extends EmbeddingOptions { documents?: StudyDocumentRef[] }
export const studyDocumentKey = (document: StudyDocumentRef) => `${document.materialId}:${document.documentId}`
export const studyScopeKey = (documents: StudyDocumentRef[]) => [...new Set(documents.map(studyDocumentKey))].sort().join('|')
export const scopeRefs = (documents: StudyDocumentRef[]): StudyDocumentRef[] =>
  [...new Map(documents.map(({ materialId, documentId }) => [studyDocumentKey({ materialId, documentId }), { materialId, documentId }])).values()]

export function sourcesInScope(sources: ChatSource[], scope: StudyDocumentRef[]): ChatSource[] {
  const keys = new Set(scope.map(studyDocumentKey))
  return sources.filter(source => source.materialId && source.documentId != null &&
    keys.has(studyDocumentKey({ materialId: source.materialId, documentId: Number(source.documentId) })))
}

export function currentStudyScope(state: AppStateSnapshot): StudyDocumentRef[] | undefined {
  if (state.practice?.mode === 'practice') {
    const session = state.practice.sessions.find(item => item.id === state.practice?.activeSessionId)
    return session ? scopeRefs(session.documents) : state.practice.draftScope
  }
  return state.conversations.find(item => item.id === state.activeConversationId)?.documentScope
}

// Older chats inherit the former collection selection once; an explicit empty scope stays empty.
export function initializeStudyScopes(state: AppStateSnapshot, documents: StudyDocument[]): AppStateSnapshot {
  const initial = state.studyScope ?? scopeRefs(documents.filter(document =>
    state.materials.some(material => material.id === document.materialId && material.isActive !== false)))
  if (state.studyScope && state.conversations.every(chat => chat.documentScope !== undefined) &&
      (!state.practice || state.practice.draftScope !== undefined)) return state
  return { ...state, studyScope: initial,
    conversations: state.conversations.map(chat => chat.documentScope === undefined ? { ...chat, documentScope: initial } : chat),
    practice: state.practice ? { ...state.practice, draftScope: state.practice.draftScope ?? initial } : state.practice }
}
