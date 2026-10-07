import assert from 'node:assert/strict'
import test from 'node:test'
import { requireTranspiledTs } from './ts-module-loader.mjs'

const { studyScopeKey, scopeRefs, sourcesInScope, currentStudyScope, initializeStudyScopes } = requireTranspiledTs('src/shared/study-scope.ts')
const documents = [
  { materialId: 'db', documentId: 1, title: 'Trees.md' },
  { materialId: 'db', documentId: 2, title: 'Buffer.txt' },
  { materialId: 'poems', documentId: 3, title: 'Trees.md' }
]
const base = { materials: [{ id: 'db', isActive: true }, { id: 'poems', isActive: false }],
  activeConversationId: 'a', conversations: [{ id: 'a' }, { id: 'b', documentScope: [] }],
  practice: { mode: 'chat', sessions: [] } }

test('old chats inherit the old collection selection once; explicit empty selection survives', () => {
  const next = initializeStudyScopes(base, documents)
  assert.deepEqual(next.conversations[0].documentScope, scopeRefs(documents.slice(0, 2)))
  assert.deepEqual(next.conversations[1].documentScope, [])
  assert.deepEqual(next.practice.draftScope, next.studyScope)
  assert.equal(initializeStudyScopes(next, documents), next)
})

test('each chat and started practice restores its own scope; drafts inherit the last selection', () => {
  const a = scopeRefs([documents[0]]), b = scopeRefs([documents[2]])
  const state = { ...base, studyScope: b, conversations: [{ id: 'a', documentScope: a }, { id: 'b', documentScope: b }],
    practice: { mode: 'chat', draftScope: b, sessions: [{ id: 'p', documents: [documents[1]] }], activeSessionId: 'p' } }
  assert.deepEqual(currentStudyScope(state), a)
  assert.deepEqual(currentStudyScope({ ...state, activeConversationId: 'b' }), b)
  assert.deepEqual(currentStudyScope({ ...state, practice: { ...state.practice, mode: 'practice' } }), scopeRefs([documents[1]]))
  assert.deepEqual(currentStudyScope({ ...state, practice: { ...state.practice, mode: 'practice', activeSessionId: undefined } }), b)
  const fresh = initializeStudyScopes({ ...state, conversations: [{ id: 'new' }] }, documents)
  assert.deepEqual(fresh.conversations[0].documentScope, b)
})

test('scope keys are order-independent and references contain no document content', () => {
  assert.equal(studyScopeKey(documents), studyScopeKey([...documents].reverse()))
  assert.deepEqual(scopeRefs([documents[0], documents[0]]), [{ materialId: 'db', documentId: 1 }])
  assert.notEqual(studyScopeKey([documents[0]]), studyScopeKey([{ ...documents[0], materialId: 'other' }]))
})

test('practice discussion evidence is restricted to the selected collection/document pair', () => {
  const sources = [...documents, { materialId: 'poems', documentId: 1 }, { title: 'No identity' }]
  assert.deepEqual(sourcesInScope(sources, [documents[0]]), [documents[0]])
  assert.deepEqual(sourcesInScope(sources, []), [])
  const stringId = { materialId: 'db', documentId: '1' }
  assert.deepEqual(sourcesInScope([stringId], [documents[0]]), [stringId])
})
