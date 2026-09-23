import assert from 'node:assert/strict'
import { test } from 'node:test'
import { requireTranspiledTs } from './ts-module-loader.mjs'

const { listOpenAiCompatibleModels, runRemoteStudyEngine } = requireTranspiledTs(
  'src/main/engine/remote-chat-service.ts'
)

const addedAt = new Date(0).toISOString()
const databaseSource = {
  title: 'Database Systems.pdf',
  locator: 'Page 4',
  excerpt: 'Transactions preserve atomicity and durability.'
}

function geminiChatModel(overrides = {}) {
  return {
    id: 'remote-gemini',
    name: 'Gemini 2.5 Flash',
    engine: 'remote',
    source: 'remote',
    status: 'ready',
    providerId: 'gemini',
    providerName: 'Gemini',
    baseUrl: 'https://generativelanguage.googleapis.com/v1beta/openai/',
    apiKey: 'gemini-key',
    remoteModelName: 'gemini-2.5-flash',
    addedAt,
    ...overrides
  }
}

function remoteStudyRequest(overrides = {}) {
  return {
    prompt: 'What is atomicity?',
    messages: [],
    materials: [],
    retrievedSources: [databaseSource],
    model: geminiChatModel(),
    settings: {},
    applicationSettings: {
      suggestionMode: 'off'
    },
    modelSettings: {
      maxLength: 512,
      temperature: 0.2,
      topP: 0.95
    },
    ...overrides
  }
}

async function withMockFetch(fetchImplementation, callback) {
  const originalFetch = globalThis.fetch
  globalThis.fetch = fetchImplementation

  try {
    return await callback()
  } finally {
    globalThis.fetch = originalFetch
  }
}

test('listOpenAiCompatibleModels normalizes Gemini chat model ids and sorts recent Gemini models first', async () => {
  const originalFetch = globalThis.fetch
  const requestedUrls = []

  globalThis.fetch = async (url, options) => {
    requestedUrls.push(String(url))
    assert.equal(options.headers.Accept, 'application/json')
    assert.equal(options.headers.Authorization, 'Bearer gemini-key')

    if (String(url).endsWith('/models')) {
      return {
        ok: true,
        json: async () => ({
          data: [
            { id: 'models/antigravity-preview-05-26' },
            { id: 'models/gemini-3.5-flash' },
            { id: 'models/gemini-2.5-pro' },
            { id: 'models/gemini-2.0-flash' },
            { id: 'models/gemini-embedding-001' },
            { id: 'models/text-embedding-004' }
          ]
        })
      }
    }

    throw new Error(`Unexpected request to ${String(url)}`)
  }

  try {
    const models = await listOpenAiCompatibleModels(
      ' gemini-key ',
      'https://generativelanguage.googleapis.com/v1beta/openai/'
    )

    assert.equal(requestedUrls[0], 'https://generativelanguage.googleapis.com/v1beta/openai/models')
    assert.deepEqual(requestedUrls, ['https://generativelanguage.googleapis.com/v1beta/openai/models'])
    assert.deepEqual(models, [
      'gemini-3.5-flash',
      'gemini-2.5-pro',
      'gemini-2.0-flash',
      'antigravity-preview-05-26'
    ])
  } finally {
    globalThis.fetch = originalFetch
  }
})

test('listOpenAiCompatibleModels lists Gemini embedding models when requested', async () => {
  const originalFetch = globalThis.fetch

  globalThis.fetch = async (url, options) => {
    assert.equal(String(url), 'https://generativelanguage.googleapis.com/v1beta/openai/models')
    assert.equal(options.headers.Authorization, 'Bearer gemini-key')

    return {
      ok: true,
      json: async () => ({
        data: [
          { id: 'models/gemini-3.5-flash' },
          { id: 'models/text-embedding-004' },
          { id: 'models/gemini-embedding-001' }
        ]
      })
    }
  }

  try {
    const models = await listOpenAiCompatibleModels(
      ' gemini-key ',
      'https://generativelanguage.googleapis.com/v1beta/openai/',
      'embedder'
    )

    assert.deepEqual(models, ['gemini-embedding-001', 'text-embedding-004'])
  } finally {
    globalThis.fetch = originalFetch
  }
})

test('runRemoteStudyEngine sends Gemini chat through OpenAI-compatible chat completions', async () => {
  let requestedUrl = ''
  let requestBody = null

  await withMockFetch(async (url, options) => {
    requestedUrl = String(url)
    requestBody = JSON.parse(String(options.body))
    assert.equal(options.headers['Content-Type'], 'application/json')
    assert.equal(options.headers.Authorization, 'Bearer gemini-key')

    return {
      ok: true,
      json: async () => ({
        choices: [
          {
            message: {
              content: 'Gemini answer.'
            }
          }
        ]
      })
    }
  }, async () => {
    const response = await runRemoteStudyEngine(remoteStudyRequest({
      model: geminiChatModel({
        remoteModelName: 'models/gemini-2.5-flash'
      }),
      modelSettings: {
        systemMessage: 'Answer from sources.',
        maxLength: 512,
        temperature: 0.2,
        topP: 0.95
      }
    }))

    assert.equal(requestedUrl, 'https://generativelanguage.googleapis.com/v1beta/openai/chat/completions')
    assert.equal(requestBody.model, 'gemini-2.5-flash')
    assert.match(requestBody.messages[0].content, /^Answer from sources\./)
    assert.match(requestBody.messages[0].content, /undergraduate understand the current question/)
    assert.match(requestBody.messages.at(-1).content, /Transactions preserve atomicity and durability/)
    assert.equal(requestBody.max_tokens, 512)
    assert.equal(response.engineId, 'tokensmith')
    assert.equal(response.text, 'Gemini answer.')
    assert.equal(response.followUpSuggestions, undefined)
  })
})

test('runRemoteStudyEngine never requests follow-up suggestions for cloud models even when enabled', async () => {
  const requestBodies = []

  await withMockFetch(async (_url, options) => {
    requestBodies.push(JSON.parse(String(options.body)))

    return {
      ok: true,
      json: async () => ({
        choices: [
          {
            message: {
              content: 'Atomicity makes a transaction all-or-nothing.'
            }
          }
        ]
      })
    }
  }, async () => {
    const response = await runRemoteStudyEngine(remoteStudyRequest({
      applicationSettings: {
        suggestionMode: 'on',
        followUpSuggestionCount: 4
      },
      modelSettings: {
        maxLength: 512,
        temperature: 0.2,
        topP: 0.95,
        suggestedFollowUpPrompt: 'Suggest follow-up questions.'
      }
    }))

    // Only the answer request is made; no second follow-up request.
    assert.equal(requestBodies.length, 1)
    assert.equal(response.text, 'Atomicity makes a transaction all-or-nothing.')
    assert.equal(response.followUpSuggestions, undefined)
    assert.equal(response.followUpError, undefined)
  })
})

test('runRemoteStudyEngine includes the remote endpoint and model when a provider returns an error', async () => {
  await withMockFetch(async () => ({
    ok: false,
    status: 404,
    text: async () => '{"error":{"message":"model not found"}}'
  }), async () => {
    await assert.rejects(
      runRemoteStudyEngine(remoteStudyRequest({
        prompt: 'What is RAID?',
        retrievedSources: [],
        model: geminiChatModel({
          name: 'Gemini 2.0 Flash',
          remoteModelName: 'models/gemini-2.0-flash'
        }),
        modelSettings: {}
      })),
      /HTTP 404 at POST https:\/\/generativelanguage\.googleapis\.com\/v1beta\/openai\/chat\/completions using model gemini-2\.0-flash/
    )
  })
})

test('listOpenAiCompatibleModels preserves generic OpenAI-compatible model ids', async () => {
  const originalFetch = globalThis.fetch

  globalThis.fetch = async () => ({
    ok: true,
    json: async () => ({
      data: [{ id: 'z-model' }, { id: 'models/provider-prefixed' }, { id: 'a-model' }]
    })
  })

  try {
    const models = await listOpenAiCompatibleModels('key', 'https://api.example.test/v1/')

    assert.deepEqual(models, ['a-model', 'models/provider-prefixed', 'z-model'])
  } finally {
    globalThis.fetch = originalFetch
  }
})
