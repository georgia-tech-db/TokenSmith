import assert from 'node:assert/strict'
import { test } from 'node:test'
import { requireTranspiledTs } from './ts-module-loader.mjs'

const {
  normalizeAnswerModelDefaults,
  normalizeConversationModelSelection
} = requireTranspiledTs('src/shared/chat-model-selection.ts')

const local = { id: 'local-a', name: 'Local A', engine: 'ollama', role: 'generator', status: 'ready', addedAt: '' }
const online = { id: 'online-a', name: 'Online A', engine: 'remote', role: 'generator', status: 'ready', addedAt: '' }

test('separate local and online defaults can select online for new conversations', () => {
  assert.deepEqual(normalizeAnswerModelDefaults({
    defaultLocalModelId: local.id,
    defaultOnlineModelId: online.id,
    defaultChatMode: 'online'
  }, [local, online]), {
    defaultLocalModelId: local.id,
    defaultOnlineModelId: online.id,
    defaultChatMode: 'online'
  })
})

test('online cannot become the default mode before an online model is configured', () => {
  assert.deepEqual(normalizeAnswerModelDefaults({ defaultChatMode: 'online' }, [local]), {
    defaultLocalModelId: local.id,
    defaultOnlineModelId: '',
    defaultChatMode: 'local'
  })
})

test('a temporarily unavailable configured model keeps its default selection', () => {
  const stoppedLocal = { ...local, status: 'needsRuntime' }
  assert.deepEqual(normalizeAnswerModelDefaults({
    defaultLocalModelId: stoppedLocal.id,
    defaultChatMode: 'local'
  }, [stoppedLocal]), {
    defaultLocalModelId: stoppedLocal.id,
    defaultOnlineModelId: '',
    defaultChatMode: 'local'
  })
})

test('existing conversations retain independent mode-specific model choices', () => {
  const defaults = normalizeAnswerModelDefaults({
    defaultLocalModelId: local.id,
    defaultOnlineModelId: online.id,
    defaultChatMode: 'local'
  }, [local, online])
  const conversation = normalizeConversationModelSelection({
    id: 'chat', title: 'Chat', period: 'Today', messages: [],
    chatModelMode: 'online', localModelId: local.id, onlineModelId: online.id
  }, defaults, [local, online])
  assert.equal(conversation.chatModelMode, 'online')
  assert.equal(conversation.localModelId, local.id)
  assert.equal(conversation.onlineModelId, online.id)
})

test('legacy global selections migrate into the matching conversation slot', () => {
  const defaults = normalizeAnswerModelDefaults({ defaultModelId: local.id }, [local, online])
  const conversation = normalizeConversationModelSelection({
    id: 'legacy', title: 'Chat', period: 'Today', messages: []
  }, defaults, [local, online], online.id)
  assert.equal(conversation.chatModelMode, 'online')
  assert.equal(conversation.localModelId, local.id)
  assert.equal(conversation.onlineModelId, online.id)
})
