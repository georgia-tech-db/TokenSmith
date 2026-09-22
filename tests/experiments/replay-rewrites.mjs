// Inspect actual answers from the rewrite-only experiment with unchanged hybrid retrieval.
import { spawn } from 'node:child_process'
import { once } from 'node:events'
import { readFileSync, writeFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { createInterface } from 'node:readline'
import { parseArgs } from 'node:util'
import { requireTranspiledTs } from '../unit/typescript/ts-module-loader.mjs'

const { values } = parseArgs({ options: { 'user-data': { type: 'string' },
  original: { type: 'string' }, rewrites: { type: 'string' }, out: { type: 'string' } } })
if (Object.keys(values).length < 4) throw new Error('--user-data, --original, --rewrites and --out are required')
const original = JSON.parse(readFileSync(values.original, 'utf8'))
const rewrites = JSON.parse(readFileSync(values.rewrites, 'utf8'))
const state = JSON.parse(readFileSync(resolve(values['user-data'], 'tokensmith-state.json'), 'utf8'))
const { studyChatMessages } = requireTranspiledTs('src/main/engine/study-chat-format.ts')
const worker = spawn(resolve('app_runtime/python/bin/python'),
  [resolve('tests/experiments/buzzdb-library-worker.py'), resolve(values['user-data'])],
  { env: { ...process.env, TOKENSMITH_LOG_FILE: '', PYTHONUNBUFFERED: '1' } })
const workerExit = once(worker, 'close')
const lines = createInterface({ input: worker.stdout })[Symbol.asyncIterator]()
worker.stderr.pipe(process.stderr)
const results = []

const cases = rewrites.resolutions.filter((r) =>
  r.caseId === 'rehash_example' || (r.model === 'qwen3:1.7b' && ['lru_example', 'slotted_typo'].includes(r.caseId)))
// These come directly from the model's output, not authored replacement questions.
for (const seedId of ['slots', 'vectors']) {
  const suggestions = rewrites.suggestions.find((r) => r.model === 'llama3:latest' && r.seedId === seedId)
  const question = suggestions?.questions?.[seedId === 'slots' ? 2 : 0]
  if (question) cases.push({ caseId: `${seedId}_generated_follow_up`, model: 'llama3:latest', question })
}

try {
  for (const item of cases) {
    const start = performance.now()
    worker.stdin.write(JSON.stringify({ query: item.question, limit: 4,
      materials: state.materials.filter((m) => m.isActive && m.status === 'ready'),
      embeddingModels: [{ engine: 'ollama', role: 'embedder', ollamaModelName: 'nomic-embed-text' }] }) + '\n')
    const line = await lines.next()
    if (line.done) throw new Error('Library worker exited')
    const response = JSON.parse(line.value)
    if (!response.ok) throw new Error(response.error)
    const sources = response.result.sources
    const searchSeconds = (performance.now() - start) / 1000
    const request = { prompt: item.question, messages: [], materials: [], retrievedSources: sources,
      model: { id: 'ollama:llama3', name: 'llama3:latest', engine: 'ollama', source: 'ollama',
        role: 'generator', ollamaModelName: 'llama3:latest', contextLength: 8192 },
      settings: {}, applicationSettings: { suggestionMode: 'off' },
      modelSettings: { contextLength: 8192, maxLength: 1024 }, conversationContextMode: 'standalone' }
    const generated = await fetch('http://127.0.0.1:11434/api/chat', { method: 'POST',
      headers: { 'Content-Type': 'application/json' }, signal: AbortSignal.timeout(180000),
      body: JSON.stringify({ model: original.generator, messages: studyChatMessages(request), stream: false,
        options: { num_ctx: 8192, num_predict: 768, temperature: 0, seed: 42 } }) })
    if (!generated.ok) throw new Error(await generated.text())
    const data = await generated.json()
    results.push({ caseId: item.caseId, resolver: item.model, query: item.question, sources,
      vectorHits: response.vectorHits, text: data.message.content, searchSeconds,
      totalSeconds: (performance.now() - start) / 1000 })
    writeFileSync(values.out, JSON.stringify(results, null, 2))
    console.log(`${item.caseId} ${item.model}: ${results.at(-1).totalSeconds.toFixed(2)}s`)
  }
} finally {
  worker.stdin.end()
  await workerExit
}
