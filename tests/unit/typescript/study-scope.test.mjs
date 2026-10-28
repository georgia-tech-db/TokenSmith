import assert from 'node:assert/strict'
import test from 'node:test'
import { requireTranspiledTs } from './ts-module-loader.mjs'

const { studyScopeKey, scopeRefs, sourcesInScope, changeStudyScope, initializeStudyScopes } = requireTranspiledTs('src/shared/study-scope.ts')
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

test('one selection updates Chat and Practice without rewriting saved exercises or other chats', () => {
  const a = scopeRefs([documents[0]]), b = scopeRefs([documents[2]])
  const session = { id: 'p', documents: [documents[0]], questions: [{ draft: 'Keep this answer' }] }
  const state = { ...base, studyScope: a, conversations: [{ id: 'a', documentScope: a }, { id: 'b', documentScope: a }],
    practice: { mode: 'chat', draftScope: a, draftTopic: { title: 'Trees' }, sessions: [session], activeSessionId: 'p' } }
  for (const mode of ['chat', 'practice']) {
    const next = changeStudyScope({ ...state, practice: { ...state.practice, mode } }, b)
    assert.deepEqual(next.studyScope, b)
    assert.deepEqual(next.conversations[0].documentScope, b)
    assert.equal(next.conversations[1], state.conversations[1])
    assert.deepEqual(next.practice.draftScope, b)
    assert.equal(next.practice.activeSessionId, undefined)
    assert.equal(next.practice.draftTopic, undefined)
    assert.equal(next.practice.sessions[0], session)
    assert.equal(next.practice.mode, mode)
  }
  assert.equal(changeStudyScope(state, [...a, ...a]).practice.activeSessionId, 'p')
  assert.equal(changeStudyScope(state, [...a, ...a]).practice.draftTopic.title, 'Trees')
  const empty = changeStudyScope(state, [])
  assert.deepEqual(initializeStudyScopes(empty, documents).studyScope, [])
  assert.equal(empty.practice.activeSessionId, undefined)
  const fresh = initializeStudyScopes({ ...state, studyScope: b, conversations: [{ id: 'new' }] }, documents)
  assert.deepEqual(fresh.conversations[0].documentScope, b)
  assert.equal(fresh.practice.activeSessionId, undefined)
  assert.equal(fresh.practice.sessions[0], session)
})

test('restoring an older state detaches a practice session with a different document scope', () => {
  const state = { ...initializeStudyScopes(base, documents), practice: {
    mode: 'practice', draftScope: [], sessions: [{ id: 'p', documents: [documents[2]] }], activeSessionId: 'p'
  } }
  const next = initializeStudyScopes(state, documents)
  assert.deepEqual(next.studyScope, scopeRefs(documents.slice(0, 2)))
  assert.equal(next.practice.activeSessionId, undefined)
  assert.equal(next.practice.sessions, state.practice.sessions)
  assert.equal(initializeStudyScopes(next, documents), next)
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
