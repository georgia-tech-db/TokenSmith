import assert from 'node:assert/strict'
import { test } from 'node:test'
import { requireTranspiledTs } from './ts-module-loader.mjs'

const { listStudyEngines, sendStudyChatMessage } = requireTranspiledTs('src/main/engine/study-engine-core.ts')

const baseRequest = {
  prompt: 'Who is Annita Demetriou?',
  messages: [],
  materials: [],
  model: {
    id: 'llama-3-2-3b-instruct',
    name: 'Llama 3.2 3B Instruct',
    engine: 'python',
    status: 'ready',
    addedAt: new Date(0).toISOString()
  },
  settings: {
    maxSources: 4,
    application: {
      theme: 'light',
      fontSize: 'small',
      defaultModelId: 'llama-3-2-3b-instruct',
      suggestionMode: 'on',
      followUpSuggestionCount: 4,
      showSources: true,
      cpuThreads: 4
    },
    modelDefaults: {
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
    },
    modelSettingsById: {}
  }
}

test('listStudyEngines reports TokenSmith as ready when health succeeds', async () => {
  const engines = await listStudyEngines({
    getPythonEngineHealth: async () => ({ llamaCppAvailable: true }),
    runPythonStudyEngine: async () => {
      throw new Error('not used')
    }
  })

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
  const engines = await listStudyEngines({
    getPythonEngineHealth: async () => {
      throw new Error('worker missing')
    },
    runPythonStudyEngine: async () => {
      throw new Error('not used')
    }
  })

  assert.equal(engines.length, 1)
  assert.equal(engines[0].id, 'tokensmith')
  assert.equal(engines[0].status, 'unavailable')
  assert.match(engines[0].detail, /local TokenSmith runtime is not available/i)
})

test('listStudyEngines reports retrieval-only mode without llama.cpp', async () => {
  const engines = await listStudyEngines({
    getPythonEngineHealth: async () => ({ llamaCppAvailable: false }),
    runPythonStudyEngine: async () => {
      throw new Error('not used')
    }
  })

  assert.equal(engines.length, 1)
  assert.equal(engines[0].status, 'ready')
  assert.match(engines[0].detail, /Local indexing and vector retrieval/i)
  assert.match(engines[0].detail, /Ollama chat and embedding models/i)
})

test('sendStudyChatMessage sends Ollama models to the Ollama study engine', async () => {
  let ollamaWasCalled = false
  let pythonWasCalled = false
  const response = await sendStudyChatMessage(
    {
      ...baseRequest,
      retrievedSources: [
        {
          title: 'Annita Demetriou - Wikipedia.pdf',
          locator: 'Page 1',
          excerpt: 'Annita Demetriou is a Cypriot politician.'
        }
      ],
      model: {
        id: 'ollama:llama3',
        name: 'Ollama llama3',
        engine: 'ollama',
        role: 'generator',
        status: 'ready',
        source: 'ollama',
        ollamaModelName: 'llama3',
        addedAt: new Date(0).toISOString()
      }
    },
    {
      getPythonEngineHealth: async () => ({ llamaCppAvailable: false }),
      runOllamaStudyEngine: async (request) => {
        ollamaWasCalled = true
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
    }
  )

  assert.equal(response.engineId, 'tokensmith')
  assert.equal(response.modelName, 'Ollama llama3')
  assert.equal(response.text, 'Ollama answer.')
  assert.equal(response.sources.length, 1)
  assert.equal(ollamaWasCalled, true)
  assert.equal(pythonWasCalled, false)
})

test('sendStudyChatMessage explains that packaged local chat is disabled', async () => {
  let pythonWasCalled = false
  const response = await sendStudyChatMessage({
    ...baseRequest,
    retrievedSources: [
      {
        title: 'Annita Demetriou - Wikipedia.pdf',
        locator: 'Page 1',
        excerpt: 'Annita Demetriou is a Cypriot politician.'
      }
    ]
  }, {
    getPythonEngineHealth: async () => ({ llamaCppAvailable: true }),
    runPythonStudyEngine: async () => {
      pythonWasCalled = true
      throw new Error('python should not be used')
    }
  })

  assert.equal(response.engineId, 'tokensmith')
  assert.equal(response.modelName, 'Llama 3.2 3B Instruct')
  assert.match(response.text, /Python\/GGUF chat models are not packaged/i)
  assert.match(response.text, /Use Ollama for local chat/i)
  assert.equal(response.sources.length, 1)
  assert.equal(pythonWasCalled, false)
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
      {
        ...baseRequest,
        prompt: 'Who is Annita Demetriou?',
        applicationSettings: {
          ...baseRequest.settings.application,
          suggestionMode: 'off'
        },
        retrievedSources: [
          {
            title: 'Annita Demetriou - Wikipedia.pdf',
            locator: 'Page 1',
            excerpt: 'Annita Demetriou is a Cypriot politician.'
          }
        ],
        model: {
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
          addedAt: new Date(0).toISOString()
        }
      },
      {
        getPythonEngineHealth: async () => ({ llamaCppAvailable: true }),
        runPythonStudyEngine: async () => {
          throw new Error('python should not be used')
        }
      }
    )

    assert.equal(requestUrl, 'https://example.test/v1/chat/completions')
    assert.equal(requestBody.model, 'gpt-test')
    assert.match(requestBody.messages.at(-1).content, /Annita Demetriou is a Cypriot politician/)
    assert.equal(response.engineId, 'tokensmith')
    assert.equal(response.modelName, 'gpt-test')
    assert.equal(response.text, 'Remote answer.')
  } finally {
    globalThis.fetch = originalFetch
  }
})
