import assert from 'node:assert/strict'
import { test } from 'node:test'
import { requireTranspiledTs } from './ts-module-loader.mjs'

const { parseOllamaSearchResults, searchOllamaLibrary } = requireTranspiledTs('src/main/engine/ollama-library-search.ts')

const ollamaSearchHtml = `
  <ul>
    <li x-test-model>
      <a href="/library/llama3">
        <span x-test-search-response-title>llama3</span>
        <p>Meta Llama 3: The most capable openly available LLM to date</p>
        <span x-test-capability>tools</span>
        <span x-test-size>8b</span>
        <span x-test-size>70b</span>
        <span x-test-pull-count>24.6M</span>
        <span x-test-tag-count>68 Tags</span>
        <span x-test-updated>2 years ago</span>
      </a>
    </li>
    <li x-test-model>
      <a href="/library/nomic-embed-text">
        <span x-test-search-response-title>nomic-embed-text</span>
        <p>A high-performing open embedding model with a large token context window.</p>
        <span x-test-capability>embedding</span>
        <span x-test-pull-count>72.9M</span>
        <span x-test-tag-count>3 Tags</span>
        <span x-test-updated>2 years ago</span>
      </a>
    </li>
  </ul>
`

const ollamaLibraryHtml = `
  <ul>
    <li x-test-model>
      <a href="/library/nomic-embed-text">
        <div x-test-model-title title="nomic-embed-text">
          <h2><span>nomic-embed-text</span></h2>
          <p>A high-performing open embedding model with a large token context window.</p>
        </div>
        <span x-test-capability>embedding</span>
        <span x-test-pull-count>77.5M</span>
        <span x-test-tag-count>3</span>
        <span x-test-updated>2 years ago</span>
      </a>
    </li>
  </ul>
`

test('parseOllamaSearchResults extracts real Ollama library card metadata', () => {
  const results = parseOllamaSearchResults(ollamaSearchHtml)

  assert.equal(results.length, 2)
  assert.deepEqual(results[0], {
    name: 'llama3',
    description: 'Meta Llama 3: The most capable openly available LLM to date',
    url: 'https://ollama.com/library/llama3',
    capabilities: ['tools'],
    sizes: ['8b', '70b'],
    pulls: '24.6M',
    tagCount: 68,
    updated: '2 years ago'
  })
  assert.equal(results[1].name, 'nomic-embed-text')
  assert.deepEqual(results[1].capabilities, ['embedding'])
})

test('parseOllamaSearchResults supports Ollama library page title markup', () => {
  const results = parseOllamaSearchResults(ollamaLibraryHtml)

  assert.equal(results.length, 1)
  assert.equal(results[0].name, 'nomic-embed-text')
  assert.equal(results[0].description, 'A high-performing open embedding model with a large token context window.')
  assert.equal(results[0].url, 'https://ollama.com/library/nomic-embed-text')
  assert.equal(results[0].pulls, '77.5M')
  assert.equal(results[0].tagCount, 3)
})

test('searchOllamaLibrary uses Ollama library search and applies embedding category for embedders', async () => {
  const originalFetch = globalThis.fetch
  const requestedUrls = []

  globalThis.fetch = async (url, options = {}) => {
    requestedUrls.push(String(url))
    assert.equal(options.headers.Accept, 'text/html,application/xhtml+xml')
    return {
      ok: true,
      text: async () => ollamaSearchHtml
    }
  }

  try {
    const results = await searchOllamaLibrary('embed', 'embedder', 1)

    assert.equal(results.length, 1)
    assert.equal(results[0].name, 'nomic-embed-text')
    assert.match(requestedUrls[0], /^https:\/\/ollama\.com\/library\?/)
    assert.match(requestedUrls[0], /q=embed/)
    assert.match(requestedUrls[0], /c=embedding/)
  } finally {
    globalThis.fetch = originalFetch
  }
})

test('searchOllamaLibrary filters unrelated fuzzy Ollama results', async () => {
  const originalFetch = globalThis.fetch

  globalThis.fetch = async () => ({
    ok: true,
    text: async () => ollamaSearchHtml
  })

  try {
    const results = await searchOllamaLibrary('abcdefe', 'generator', 20)

    assert.deepEqual(results, [])
  } finally {
    globalThis.fetch = originalFetch
  }
})
