import assert from 'node:assert/strict'
import { test } from 'node:test'
import { requireTranspiledTs } from './ts-module-loader.mjs'

const { listOpenAiCompatibleModels, runRemoteStudyEngine } = requireTranspiledTs(
  'src/main/engine/remote-chat-service.ts'
)

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
  const originalFetch = globalThis.fetch
  let requestedUrl = ''
  let requestBody = null

  globalThis.fetch = async (url, options) => {
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
  }

  try {
    const response = await runRemoteStudyEngine({
      prompt: 'Who is Annita Demetriou?',
      messages: [],
      materials: [],
      retrievedSources: [
        {
          title: 'Annita Demetriou - Wikipedia.pdf',
          locator: 'Page 1',
          excerpt: 'Annita Demetriou is a Cypriot politician.'
        }
      ],
      model: {
        id: 'remote-gemini',
        name: 'Gemini 2.5 Flash',
        engine: 'remote',
        source: 'remote',
        status: 'ready',
        providerId: 'gemini',
        providerName: 'Gemini',
        baseUrl: 'https://generativelanguage.googleapis.com/v1beta/openai/',
        apiKey: 'gemini-key',
        remoteModelName: 'models/gemini-2.5-flash',
        addedAt: new Date(0).toISOString()
      },
      settings: {},
      applicationSettings: {
        suggestionMode: 'off'
      },
      modelSettings: {
        systemMessage: 'Answer from sources.',
        maxLength: 512,
        temperature: 0.2,
        topP: 0.95
      }
    })

    assert.equal(requestedUrl, 'https://generativelanguage.googleapis.com/v1beta/openai/chat/completions')
    assert.equal(requestBody.model, 'gemini-2.5-flash')
    assert.equal(requestBody.messages[0].content, 'Answer from sources.')
    assert.match(requestBody.messages.at(-1).content, /Annita Demetriou is a Cypriot politician/)
    assert.equal(requestBody.max_tokens, 512)
    assert.equal(response.engineId, 'tokensmith')
    assert.equal(response.text, 'Gemini answer.')
    assert.deepEqual(response.followUpSuggestions, [])
  } finally {
    globalThis.fetch = originalFetch
  }
})

test('runRemoteStudyEngine asks the selected remote model for follow-up suggestions when enabled', async () => {
  const originalFetch = globalThis.fetch
  const requestBodies = []

  globalThis.fetch = async (_url, options) => {
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
                  ? 'Annita Demetriou is a Cypriot politician.'
                  : '1. What office does Annita Demetriou hold?\n2. Which party does Annita Demetriou lead?\n3. When was Annita Demetriou born?'
            }
          }
        ]
      })
    }
  }

  try {
    const response = await runRemoteStudyEngine({
      prompt: 'Who is Annita Demetriou?',
      messages: [],
      materials: [],
      retrievedSources: [
        {
          title: 'Annita Demetriou - Wikipedia.pdf',
          locator: 'Page 1',
          excerpt: 'Annita Demetriou is a Cypriot politician.'
        }
      ],
      model: {
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
        addedAt: new Date(0).toISOString()
      },
      settings: {},
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
    })

    assert.equal(requestBodies.length, 2)
    assert.equal(requestBodies[1].messages.at(-2).role, 'assistant')
    assert.equal(requestBodies[1].messages.at(-2).content, 'Annita Demetriou is a Cypriot politician.')
    assert.equal(requestBodies[1].messages.at(-1).content, 'Generate 2 suggested follow-up questions.\nSuggest follow-up questions.')
    assert.deepEqual(response.followUpSuggestions, [
      'What office does Annita Demetriou hold?',
      'Which party does Annita Demetriou lead?'
    ])
  } finally {
    globalThis.fetch = originalFetch
  }
})

test('runRemoteStudyEngine uses the study-oriented default follow-up prompt', async () => {
  const originalFetch = globalThis.fetch
  const requestBodies = []

  globalThis.fetch = async (_url, options) => {
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
  }

  try {
    const response = await runRemoteStudyEngine({
      prompt: 'What are the key ideas in this chapter?',
      messages: [],
      materials: [],
      retrievedSources: [
        {
          title: 'Chapter 12',
          locator: 'Page 1',
          excerpt: 'The chapter discusses physical storage media.'
        }
      ],
      model: {
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
        addedAt: new Date(0).toISOString()
      },
      settings: {},
      applicationSettings: {
        suggestionMode: 'on',
        followUpSuggestionCount: 4
      },
      modelSettings: {
        maxLength: 512,
        temperature: 0.2,
        topP: 0.95
      }
    })

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
  } finally {
    globalThis.fetch = originalFetch
  }
})

test('runRemoteStudyEngine includes the remote endpoint and model when a provider returns an error', async () => {
  const originalFetch = globalThis.fetch

  globalThis.fetch = async () => ({
    ok: false,
    status: 404,
    text: async () => '{"error":{"message":"model not found"}}'
  })

  try {
    await assert.rejects(
      runRemoteStudyEngine({
        prompt: 'What is RAID?',
        messages: [],
        materials: [],
        retrievedSources: [],
        model: {
          id: 'remote-gemini',
          name: 'Gemini 2.0 Flash',
          engine: 'remote',
          source: 'remote',
          status: 'ready',
          providerId: 'gemini',
          providerName: 'Gemini',
          baseUrl: 'https://generativelanguage.googleapis.com/v1beta/openai/',
          apiKey: 'gemini-key',
          remoteModelName: 'models/gemini-2.0-flash',
          addedAt: new Date(0).toISOString()
        },
        settings: {},
        modelSettings: {}
      }),
      /HTTP 404 at POST https:\/\/generativelanguage\.googleapis\.com\/v1beta\/openai\/chat\/completions using model gemini-2\.0-flash/
    )
  } finally {
    globalThis.fetch = originalFetch
  }
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
