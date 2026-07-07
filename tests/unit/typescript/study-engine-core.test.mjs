import assert from 'node:assert/strict'
import { test } from 'node:test'
import { requireTranspiledTs } from './ts-module-loader.mjs'

const {
  generateStudyQuestionSuggestions,
  listStudyEngines,
  sendStudyChatMessage
} = requireTranspiledTs('src/main/engine/study-engine-core.ts')

const addedAt = new Date(0).toISOString()
const databaseSource = {
  title: 'Database Systems.pdf',
  locator: 'Page 4',
  excerpt: 'Transactions preserve atomicity and durability.'
}
const applicationSettings = {
  theme: 'light',
  fontSize: 'small',
  defaultModelId: 'llama-3-2-3b-instruct',
  suggestionMode: 'on',
  followUpSuggestionCount: 4,
  showSources: true,
  cpuThreads: 4
}
const modelSettings = {
  systemMessage: '',
  chatTemplate: '',
  suggestedFollowUpPrompt: '',
  contextLength: 2048,
  maxLength: 4096,
  promptBatchSize: 128,
  temperature: 0.7,
  topP: 0.4,
  topK: 40,
  minP: 0,
  repeatPenaltyTokens: 64,
  repeatPenalty: 1.18,
  gpuLayers: -1,
  device: 'applicationDefault'
}

function pythonChatModel(overrides = {}) {
  return {
    id: 'llama-3-2-3b-instruct',
    name: 'Llama 3.2 3B Instruct',
    engine: 'python',
    status: 'ready',
    addedAt,
    ...overrides
  }
}

function ollamaChatModel(overrides = {}) {
  return {
    id: 'ollama:llama3',
    name: 'Ollama llama3',
    engine: 'ollama',
    role: 'generator',
    status: 'ready',
    source: 'ollama',
    ollamaModelName: 'llama3',
    addedAt,
    ...overrides
  }
}

function remoteChatModel(overrides = {}) {
  return {
    id: 'remote-openai-gpt-test',
    name: 'gpt-test',
    engine: 'remote',
    source: 'remote',
    status: 'ready',
    providerId: 'openai',
    providerName: 'OpenAI',
    baseUrl: 'https://example.test/v1/',
    apiKey: 'test-key',
    remoteModelName: 'gpt-test',
    addedAt,
    ...overrides
  }
}

function chatRequest(overrides = {}) {
  return {
    prompt: 'What is atomicity?',
    messages: [],
    materials: [],
    retrievedSources: [databaseSource],
    model: pythonChatModel(),
    settings: {
      maxSources: 4,
      application: applicationSettings,
      modelDefaults: modelSettings,
      modelSettingsById: {}
    },
    applicationSettings,
    modelSettings,
    ...overrides
  }
}

function engineDependencies(overrides = {}) {
  return {
    getPythonEngineHealth: async () => ({ llamaCppAvailable: true }),
    generateOllamaStudyQuestionSuggestions: async () => {
      throw new Error('ollama suggestions should not be used')
    },
    runOllamaStudyEngine: async () => {
      throw new Error('ollama should not be used')
    },
    runPythonStudyEngine: async () => {
      throw new Error('python should not be used')
    },
    ...overrides
  }
}

test('listStudyEngines reports TokenSmith as ready when health succeeds', async () => {
  const engines = await listStudyEngines(engineDependencies())

  assert.deepEqual(engines, [
    {
      id: 'tokensmith',
      name: 'TokenSmith',
      status: 'ready',
      detail: 'Local indexing and vector retrieval are available. Ollama chat and embedding models are supported alongside cloud-based providers.'
    }
  ])
})

test('listStudyEngines explains when the local TokenSmith runtime is unavailable', async () => {
  const engines = await listStudyEngines(engineDependencies({
    getPythonEngineHealth: async () => {
      throw new Error('worker missing')
    }
  }))

  assert.equal(engines.length, 1)
  assert.equal(engines[0].id, 'tokensmith')
  assert.equal(engines[0].status, 'unavailable')
  assert.match(engines[0].detail, /local TokenSmith runtime is not available/i)
})

test('listStudyEngines reports retrieval-only mode without llama.cpp', async () => {
  const engines = await listStudyEngines(engineDependencies({
    getPythonEngineHealth: async () => ({ llamaCppAvailable: false })
  }))

  assert.equal(engines.length, 1)
  assert.equal(engines[0].status, 'ready')
  assert.match(engines[0].detail, /Local indexing and vector retrieval/i)
  assert.match(engines[0].detail, /Ollama chat and embedding models/i)
})

test('sendStudyChatMessage routes Ollama models to the Ollama study engine', async () => {
  let ollamaRequest = null
  let pythonWasCalled = false

  const response = await sendStudyChatMessage(
    chatRequest({ model: ollamaChatModel() }),
    engineDependencies({
      runOllamaStudyEngine: async (request) => {
        ollamaRequest = request
        return {
          engineId: 'tokensmith',
          modelName: request.model.name,
          text: 'Ollama answer.',
          sources: request.retrievedSources ?? []
        }
      },
      runPythonStudyEngine: async () => {
        pythonWasCalled = true
        throw new Error('python should not be used')
      }
    })
  )

  assert.equal(ollamaRequest.model.ollamaModelName, 'llama3')
  assert.equal(response.engineId, 'tokensmith')
  assert.equal(response.modelName, 'Ollama llama3')
  assert.equal(response.text, 'Ollama answer.')
  assert.deepEqual(response.sources, [databaseSource])
  assert.equal(pythonWasCalled, false)
})

test('sendStudyChatMessage rejects unsupported packaged local chat models', async () => {
  let pythonWasCalled = false

  await assert.rejects(
    sendStudyChatMessage(
      chatRequest(),
      engineDependencies({
        runPythonStudyEngine: async () => {
          pythonWasCalled = true
          throw new Error('python should not be used')
        }
      })
    ),
    /Python\/GGUF chat models are not packaged/i
  )

  assert.equal(pythonWasCalled, false)
})

test('sendStudyChatMessage propagates Ollama chat failures', async () => {
  await assert.rejects(
    sendStudyChatMessage(
      chatRequest({ model: ollamaChatModel() }),
      engineDependencies({
        runOllamaStudyEngine: async () => {
          throw new Error('Ollama is down')
        }
      })
    ),
    /Ollama is down/
  )
})

test('sendStudyChatMessage propagates remote chat failures', async () => {
  const originalFetch = globalThis.fetch
  globalThis.fetch = async () => ({
    ok: false,
    status: 500,
    text: async () => 'remote failed'
  })

  try {
    await assert.rejects(
      sendStudyChatMessage(chatRequest({ model: remoteChatModel() }), engineDependencies()),
      /remote failed/
    )
  } finally {
    globalThis.fetch = originalFetch
  }
})

test('sendStudyChatMessage sends remote models to an OpenAI-compatible chat endpoint', async () => {
  const originalFetch = globalThis.fetch
  let requestUrl = ''
  let requestBody = null

  globalThis.fetch = async (url, options) => {
    requestUrl = String(url)
    requestBody = JSON.parse(String(options.body))

    return {
      ok: true,
      json: async () => ({
        choices: [
          {
            message: {
              content: 'Remote answer.'
            }
          }
        ]
      })
    }
  }

  try {
    const response = await sendStudyChatMessage(
      chatRequest({
        model: remoteChatModel(),
        applicationSettings: {
          ...applicationSettings,
          suggestionMode: 'off'
        }
      }),
      engineDependencies()
    )

    assert.equal(requestUrl, 'https://example.test/v1/chat/completions')
    assert.equal(requestBody.model, 'gpt-test')
    assert.match(requestBody.messages.at(-1).content, /Transactions preserve atomicity and durability/)
    assert.equal(response.engineId, 'tokensmith')
    assert.equal(response.modelName, 'gpt-test')
    assert.equal(response.text, 'Remote answer.')
  } finally {
    globalThis.fetch = originalFetch
  }
})

test('generateStudyQuestionSuggestions rejects unsupported local chat models', async () => {
  await assert.rejects(
    generateStudyQuestionSuggestions(chatRequest(), engineDependencies()),
    /Question suggestions require an Ollama or remote chat model/
  )
})
