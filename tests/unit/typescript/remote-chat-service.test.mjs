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

    if (String(url).endsWith('/models')) {
      assert.equal(options.headers['x-goog-api-key'], 'gemini-key')
      return {
        ok: true,
        json: async () => ({
          models: [
            { name: 'models/antigravity-preview-05-26', supportedGenerationMethods: ['generateContent'] },
            { name: 'models/gemini-3.5-flash', supportedGenerationMethods: ['generateContent'] },
            { name: 'models/gemini-2.5-pro', supportedGenerationMethods: ['generateContent'] },
            { name: 'models/gemini-2.0-flash', supportedGenerationMethods: ['generateContent'] },
            { name: 'models/gemini-embedding-001', supportedGenerationMethods: ['embedContent'] },
            { name: 'models/imagen-4.0', supportedGenerationMethods: ['predict'] }
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

    assert.equal(requestedUrls[0], 'https://generativelanguage.googleapis.com/v1beta/models')
    assert.deepEqual(requestedUrls, ['https://generativelanguage.googleapis.com/v1beta/models'])
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
    assert.equal(String(url), 'https://generativelanguage.googleapis.com/v1beta/models')
    assert.equal(options.headers['x-goog-api-key'], 'gemini-key')

    return {
      ok: true,
      json: async () => ({
        models: [
          { name: 'models/gemini-3.5-flash', supportedGenerationMethods: ['generateContent'] },
          { name: 'models/text-embedding-004', supportedGenerationMethods: ['embedContent'] },
          { name: 'models/gemini-embedding-001', supportedGenerationMethods: ['embedContent'] }
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
    assert.equal(requestBody.messages[0].content, 'Answer from sources.')
    assert.match(requestBody.messages.at(-1).content, /Transactions preserve atomicity and durability/)
    assert.equal(requestBody.max_tokens, 512)
    assert.equal(response.engineId, 'tokensmith')
    assert.equal(response.text, 'Gemini answer.')
    assert.deepEqual(response.followUpSuggestions, [])
  })
})

test('runRemoteStudyEngine asks the selected remote model for follow-up suggestions when enabled', async () => {
  const requestBodies = []

  await withMockFetch(async (_url, options) => {
    const body = JSON.parse(String(options.body))
    requestBodies.push(body)

    return {
      ok: true,
      json: async () => ({
        choices: [
          {
            message: {
              content:
                requestBodies.length === 1
                  ? 'Atomicity makes a transaction all-or-nothing.'
                  : '1. How does durability differ from atomicity?\n2. Why does rollback matter for atomicity?\n3. Which ACID property covers isolation?'
            }
          }
        ]
      })
    }
  }, async () => {
    const response = await runRemoteStudyEngine(remoteStudyRequest({
      applicationSettings: {
        suggestionMode: 'on',
        followUpSuggestionCount: 2
      },
      modelSettings: {
        maxLength: 512,
        temperature: 0.2,
        topP: 0.95,
        suggestedFollowUpPrompt: 'Suggest follow-up questions.'
      }
    }))

    assert.equal(requestBodies.length, 2)
    assert.equal(requestBodies[1].messages.at(-2).role, 'assistant')
    assert.equal(requestBodies[1].messages.at(-2).content, 'Atomicity makes a transaction all-or-nothing.')
    assert.equal(requestBodies[1].messages.at(-1).content, 'Generate 2 suggested follow-up questions.\nSuggest follow-up questions.')
    assert.deepEqual(response.followUpSuggestions, [
      'How does durability differ from atomicity?',
      'Why does rollback matter for atomicity?'
    ])
  })
})

test('runRemoteStudyEngine uses the study-oriented default follow-up prompt', async () => {
  const requestBodies = []

  await withMockFetch(async (_url, options) => {
    const body = JSON.parse(String(options.body))
    requestBodies.push(body)

    return {
      ok: true,
      json: async () => ({
        choices: [
          {
            message: {
              content:
                requestBodies.length === 1
                  ? 'The chapter covers physical storage media.'
                  : 'What are the main categories of storage media?\nHow do magnetic disks and flash storage compare?\nWhy does storage reliability matter?\nHow do storage choices affect system design?'
            }
          }
        ]
      })
    }
  }, async () => {
    const response = await runRemoteStudyEngine(remoteStudyRequest({
      prompt: 'What are the key ideas in this chapter?',
      retrievedSources: [
        {
          title: 'Chapter 12',
          locator: 'Page 1',
          excerpt: 'The chapter discusses physical storage media.'
        }
      ],
      applicationSettings: {
        suggestionMode: 'on',
        followUpSuggestionCount: 4
      },
      modelSettings: {
        maxLength: 512,
        temperature: 0.2,
        topP: 0.95
      }
    }))

    assert.equal(requestBodies.length, 2)
    const followUpPrompt = requestBodies[1].messages.at(-1).content
    assert.match(followUpPrompt, /Suggest 4 very short factual follow-up questions/i)
    assert.match(followUpPrompt, /previous conversation and excerpts/i)
    assert.doesNotMatch(followUpPrompt, /\{count\}/)
    assert.deepEqual(response.followUpSuggestions, [
      'What are the main categories of storage media?',
      'How do magnetic disks and flash storage compare?',
      'Why does storage reliability matter?',
      'How do storage choices affect system design?'
    ])
  })
})

test('runRemoteStudyEngine surfaces follow-up generation failures without replacing the answer', async () => {
  let requestCount = 0

  await withMockFetch(async () => {
    requestCount += 1
    if (requestCount === 1) {
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
    }

    return {
      ok: false,
      status: 500,
      text: async () => 'suggestion model failed'
    }
  }, async () => {
    const response = await runRemoteStudyEngine(remoteStudyRequest({
      applicationSettings: {
        suggestionMode: 'on',
        followUpSuggestionCount: 2
      }
    }))

    assert.equal(response.text, 'Atomicity makes a transaction all-or-nothing.')
    assert.deepEqual(response.followUpSuggestions, undefined)
    assert.match(response.followUpError, /suggestion model failed/)
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
