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

export function changeStudyScope(state: AppStateSnapshot, documents: StudyDocumentRef[]): AppStateSnapshot {
  const scope = scopeRefs(documents)
  const key = studyScopeKey(scope)
  const practice = state.practice
  const session = practice?.sessions.find(item => item.id === practice.activeSessionId)
  // Saved exercises retain their original evidence; a different selection needs a new session.
  const activeSessionId = session && studyScopeKey(session.documents) === key ? session.id : undefined
  return { ...state, studyScope: scope,
    conversations: state.conversations.map(chat => chat.id === state.activeConversationId ? { ...chat, documentScope: scope } : chat),
    practice: practice ? { ...practice, draftScope: scope, activeSessionId,
      draftTopic: studyScopeKey(state.studyScope ?? []) === key ? practice.draftTopic : undefined } : practice }
}

// Older chats inherit the former collection selection once; an explicit empty scope stays empty.
export function initializeStudyScopes(state: AppStateSnapshot, documents: StudyDocument[]): AppStateSnapshot {
  const initial = state.studyScope ?? scopeRefs(documents.filter(document =>
    state.materials.some(material => material.id === document.materialId && material.isActive !== false)))
  const session = state.practice?.sessions.find(item => item.id === state.practice?.activeSessionId)
  if (state.studyScope && state.conversations.every(chat => chat.documentScope !== undefined) &&
      (!state.practice || state.practice.draftScope !== undefined) &&
      (!session || studyScopeKey(session.documents) === studyScopeKey(initial))) return state
  return changeStudyScope({ ...state,
    conversations: state.conversations.map(chat => chat.documentScope === undefined ? { ...chat, documentScope: initial } : chat),
  }, initial)
}
