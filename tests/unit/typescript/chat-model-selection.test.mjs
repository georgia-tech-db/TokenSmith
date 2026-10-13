import assert from 'node:assert/strict'
import test from 'node:test'
import { requireTranspiledTs } from './ts-module-loader.mjs'

const { answerModel, isAnswerModel, migrateConversationModels, selectStudyModel, studyModelTarget } = requireTranspiledTs('src/shared/chat-model-selection.ts')
const local = { id: 'local', engine: 'ollama', role: 'generator', status: 'ready' }
const cloud = { id: 'cloud', engine: 'remote', role: 'generator', status: 'ready' }
const embedder = { id: 'embed', engine: 'ollama', role: 'embedder', status: 'ready' }
const state = () => ({
  models: [local, cloud, embedder], activeConversationId: 'a',
  conversations: [{ id: 'a', modelId: local.id, messages: [] }, { id: 'b', modelId: cloud.id, messages: [] }],
  practice: { mode: 'chat', sessions: [], draftModelId: local.id }, selectedEmbeddingModelId: embedder.id,
  settings: { application: { defaultModelId: local.id }, modelSettingsById: { local: { contextLength: 8192 } } }
})

test('only supported generator roles are eligible, independent of readiness', () => {
  for (const engine of ['ollama', 'remote']) {
    for (const role of [undefined, 'generator', 'both']) assert.equal(isAnswerModel({ engine, role, status: 'missing' }), true)
    for (const role of ['embedder', 'invalid']) assert.equal(isAnswerModel({ engine, role }), false)
  }
  assert.equal(isAnswerModel({ engine: 'python', role: 'generator' }), false)
})

test('choosing a model changes only the targeted chat, not defaults, practice, or embeddings', () => {
  const original = state()
  const changed = selectStudyModel(original, cloud.id, studyModelTarget(original))
  assert.equal(changed.conversations[0].modelId, cloud.id)
  assert.equal(original.conversations[0].modelId, local.id)
  assert.equal(changed.conversations[1], original.conversations[1])
  assert.equal(changed.practice, original.practice)
  assert.equal(changed.settings, original.settings)
  assert.equal(changed.selectedEmbeddingModelId, embedder.id)
})

test('missing and removed models never resolve to another local or cloud model', () => {
  const saved = state()
  saved.models = [cloud, embedder]
  assert.equal(answerModel(saved.models, saved.conversations[0].modelId), undefined)
  assert.equal(answerModel(saved.models, saved.settings.application.defaultModelId), undefined)
  for (const id of [undefined, '', 'missing', embedder.id]) assert.equal(answerModel(saved.models, id), undefined)
  assert.equal(selectStudyModel(saved, 'missing', studyModelTarget(saved)), saved)
  assert.equal(selectStudyModel(saved, embedder.id, studyModelTarget(saved)), saved)
})

test('unavailable selected models remain identifiable for repair', () => {
  const stopped = { ...local, status: 'needsRuntime' }
  const disconnected = { ...cloud, status: 'needsRuntime' }
  assert.equal(answerModel([cloud, stopped], local.id), stopped)
  assert.equal(answerModel([local, disconnected], cloud.id), disconnected)
})

test('old global selections migrate once without overriding explicit empty or missing choices', () => {
  const chats = [{ id: 'legacy', messages: [] }, { id: 'empty', modelId: '' }, { id: 'missing', modelId: 'deleted' }]
  const migrated = migrateConversationModels(chats, cloud.id)
  assert.deepEqual(migrated.map(chat => chat.modelId), [cloud.id, '', 'deleted'])
  assert.deepEqual(migrateConversationModels(JSON.parse(JSON.stringify(migrated)), local.id), migrated)
  assert.equal(chats[0].modelId, undefined)
})

test('a delayed connection or download targets its original chat, not the currently viewed chat', () => {
  const original = state()
  const target = studyModelTarget(original)
  const changed = selectStudyModel({ ...original, activeConversationId: 'b' }, cloud.id, target)
  assert.equal(changed.activeConversationId, 'b')
  assert.equal(changed.conversations[0].modelId, cloud.id)
  assert.equal(changed.conversations[1], original.conversations[1])
  const deleted = { ...original, conversations: original.conversations.slice(1) }
  assert.equal(selectStudyModel(deleted, cloud.id, target), deleted)
})

test('practice draft selection does not change chat or an existing practice session', () => {
  const original = state()
  original.practice.mode = 'practice'
  const target = studyModelTarget(original)
  const selected = selectStudyModel(original, cloud.id, target)
  assert.equal(selected.practice.draftModelId, cloud.id)
  assert.equal(selected.conversations, original.conversations)
  const started = { ...selected, practice: { ...selected.practice, activeSessionId: 'session', sessions: [
    { id: 'session', modelId: cloud.id, modelSettings: { contextLength: 16384 }, questions: [] }
  ] } }
  assert.equal(selectStudyModel(started, local.id, target), started)
})

test('only an explicit first choice establishes an empty default', () => {
  const original = state()
  original.settings.application.defaultModelId = ''
  assert.equal(answerModel(original.models, original.settings.application.defaultModelId), undefined)
  const selected = selectStudyModel(original, cloud.id, studyModelTarget(original))
  assert.equal(selected.settings.application.defaultModelId, cloud.id)
  const changed = selectStudyModel(selected, local.id, studyModelTarget(selected))
  assert.equal(changed.settings.application.defaultModelId, cloud.id)
})
