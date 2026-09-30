import assert from 'node:assert/strict'
import { spawnSync } from 'node:child_process'
import { mkdtemp, mkdir, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import { performance } from 'node:perf_hooks'
import { requireTranspiledTs } from '../unit/typescript/ts-module-loader.mjs'

const { createEmbeddingDeviceManager } = requireTranspiledTs('src/main/engine/embedding-device.ts')
const prepareDevice = createEmbeddingDeviceManager()
const base = process.env.TOKENSMITH_TEST_OLLAMA_URL || 'http://127.0.0.1:11434'
const model = { id: 'nomic-test', name: 'Nomic', engine: 'ollama', role: 'embedder',
  ollamaModelName: 'nomic-embed-text:latest', ollamaBaseUrl: base }
const python = process.env.TOKENSMITH_TEST_PYTHON || resolve('app_runtime/python', process.platform === 'win32' ? 'python.exe' : 'bin/python')
const root = await mkdtemp(join(tmpdir(), 'tokensmith-embedding-device-'))
const documents = join(root, 'documents')
await mkdir(documents)
await writeFile(join(documents, 'slotted-pages.md'), '# Stable record identifiers\n\n' +
  'A slotted page separates the slot array from record contents. A record identifier names the page and the slot. During compaction, records move but their slots remain stable. Updating the offset in the slot preserves external references to that record.\n'.repeat(4))
await writeFile(join(documents, 'trees.md'), '# B+ tree range scans\n\n' +
  'B+ tree leaves maintain keys in sorted order. Links between neighboring leaves form the sequence set. A range query locates its starting key and follows the leaf links until the end of the requested range.\n'.repeat(4))

const code = `
import json, sys, time
from python_engine import tokensmith_engine as engine
payload = json.loads(sys.argv[1])
vector_hits = 0
original_vector_search = engine.vector_search
def observe_vector_search(*args, **kwargs):
    global vector_hits
    hits = original_vector_search(*args, **kwargs)
    vector_hits += len(hits)
    return hits
engine.vector_search = observe_vector_search
start = time.perf_counter()
material = engine.index_material(payload)['material']
indexed = time.perf_counter()
result = engine.search_library({'userDataPath': payload['userDataPath'], 'materials': [material],
  'embeddingModels': [payload['model']], 'embeddingGpuEnabled': payload.get('embeddingGpuEnabled', True),
  'query': 'How do record identifiers remain valid when records move during compaction?', 'searchMode': 'hybrid', 'limit': 2})
assert vector_hits > 0, 'Hybrid retrieval must not silently fall back to keyword-only search'
print(json.dumps({'vectorHits': vector_hits, 'indexMs': round((indexed-start)*1000), 'searchMs': round((time.perf_counter()-indexed)*1000),
  'embeddingModel': material['embeddingModel'], 'sources': [{'path':s['path'], 'mode':s['retrievalMode']} for s in result['sources']]}))
`

try {
  const results = []
  for (const [label, enabled] of [['default', undefined], ['cpu', false], ['gpu', true]]) {
    const start = performance.now()
    await prepareDevice([model], enabled)
    const options = enabled === undefined ? {} : { embeddingGpuEnabled: enabled }
    const payload = { path: documents, userDataPath: join(root, label), model,
      preparation: { mode: 'basic', instructions: '', documentInstructions: {} }, ...options }
    const result = spawnSync(python, ['-c', code, JSON.stringify(payload)], { cwd: process.cwd(), encoding: 'utf8', timeout: 120_000 })
    assert.equal(result.status, 0, result.stderr || result.error?.message)
    const data = JSON.parse(result.stdout.trim())
    assert.equal(data.embeddingModel, 'ollama:nomic-embed-text:latest')
    assert.ok(data.sources.some(source => source.path.endsWith('slotted-pages.md')))
    assert.ok(data.sources.every(source => source.mode === 'hybrid'))
    const ps = await fetch(`${base}/api/ps`).then(response => response.json())
    const loaded = ps.models.find(item => item.name === model.ollamaModelName)
    assert.ok(loaded, 'Embedding model must be loaded after the query')
    if (enabled === false) assert.equal(loaded.size_vram, 0, 'CPU mode must use no GPU memory')
    if (label === 'gpu' && results[0].gpuBytes > 0) assert.ok(loaded.size_vram > 0, 'Returning to GPU must not reuse the CPU runner')
    const measurement = { mode: label, gpuBytes: loaded.size_vram, totalMs: Math.round(performance.now() - start), ...data }
    results.push(measurement)
    console.log(JSON.stringify(measurement))
  }
  console.log('Real Ollama embedding device integration passed (default -> CPU -> GPU).')
} finally {
  await rm(root, { recursive: true, force: true })
}
