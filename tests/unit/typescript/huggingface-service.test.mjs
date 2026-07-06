import assert from 'node:assert/strict'
import { test } from 'node:test'
import { requireTranspiledTs } from './ts-module-loader.mjs'

const { searchHuggingFaceModels } = requireTranspiledTs('src/main/models/huggingface-service.ts')
const { remoteProviderCatalog } = requireTranspiledTs('src/shared/model-providers.ts')

test('searchHuggingFaceModels accepts nested GGUF siblings and stores a safe local filename', async () => {
  const originalFetch = globalThis.fetch
  const searchUrls = []
  const headUrls = []

  globalThis.fetch = async (url, options = {}) => {
    const href = String(url)

    if (href.startsWith('https://huggingface.co/api/models?')) {
      searchUrls.push(href)
      return {
        ok: true,
        json: async () => [
          {
            id: 'abc/a-model',
            author: 'abc',
            likes: 3,
            downloads: 208,
            lastModified: '2026-04-29T00:00:00.000Z',
            config: { model_type: 'llama' },
            siblings: [
              { rfilename: 'mmproj-model.gguf' },
              { rfilename: 'nested/model-f16.gguf' },
              { rfilename: 'nested/model-Q4_0.gguf' }
            ]
          }
        ]
      }
    }

    if (options.method === 'HEAD') {
      headUrls.push(href)
      return {
        headers: {
          get: (name) => (name.toLowerCase() === 'x-linked-size' ? '15728640' : null)
        }
      }
    }

    throw new Error(`Unexpected fetch: ${href}`)
  }

  try {
    const results = await searchHuggingFaceModels('abc', {
      sort: 'likes',
      direction: 'desc',
      limit: 20
    })

    assert.equal(results.length, 1)
    assert.equal(results[0].id, 'hf:abc/a-model:nested/model-Q4_0.gguf')
    assert.equal(results[0].sourceFilename, 'nested/model-Q4_0.gguf')
    assert.equal(results[0].filename.includes('/'), false)
    assert.match(results[0].filename, /^abc-a-model-model-Q4_0\.gguf$/)
    assert.equal(results[0].url, 'https://huggingface.co/abc/a-model/resolve/main/nested/model-Q4_0.gguf')
    assert.equal(results[0].sizeBytes, 15728640)
    assert.match(results[0].description.join('\n'), /File: nested\/model-Q4_0\.gguf/)
    assert.match(searchUrls[0], /filter=gguf/)
    assert.match(searchUrls[0], /search=abc/)
    assert.match(searchUrls[0], /full=true/)
    assert.match(searchUrls[0], /config=true/)
    assert.deepEqual(headUrls, ['https://huggingface.co/abc/a-model/resolve/main/nested/model-Q4_0.gguf'])
  } finally {
    globalThis.fetch = originalFetch
  }
})

test('remote provider catalog includes dynamic provider endpoints without static model filters', () => {
  const byId = new Map(remoteProviderCatalog.map((provider) => [provider.id, provider]))

  assert.equal(byId.get('gemini')?.baseUrl, 'https://generativelanguage.googleapis.com/v1beta/openai/')
  assert.equal(byId.get('gemini')?.apiKeyUrl, 'https://aistudio.google.com/apikey')
  assert.equal(byId.get('groq')?.baseUrl, 'https://api.groq.com/openai/v1/')
  assert.equal(byId.get('openai')?.baseUrl, 'https://api.openai.com/v1/')
  assert.equal(byId.get('mistral')?.baseUrl, 'https://api.mistral.ai/v1/')
  assert.ok(remoteProviderCatalog.every((provider) => !('modelWhitelist' in provider)))
})
