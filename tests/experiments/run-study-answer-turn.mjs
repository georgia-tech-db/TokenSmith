import { spawn } from 'node:child_process'
import { once } from 'node:events'
import { existsSync, readFileSync, writeFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { createInterface } from 'node:readline'
import { parseArgs } from 'node:util'
import { requireTranspiledTs } from '../unit/typescript/ts-module-loader.mjs'

const { values } = parseArgs({ options: {
  'user-data': { type: 'string' }, session: { type: 'string' }, question: { type: 'string' },
  'rewrite-from': { type: 'string' }, model: { type: 'string', default: 'gemma4:e4b' }
} })
if (!values['user-data'] || !values.session || !values.question || !values['rewrite-from']) {
  throw new Error('--user-data, --session, --question and --rewrite-from are required')
}
const readJson = (path) => JSON.parse(readFileSync(path, 'utf8'))
const state = readJson(resolve(values['user-data'], 'tokensmith-state.json'))
const promptRequest = readJson(values['rewrite-from']).request
const { studyChatMessages, sourceContextBudgetForRequest, answerWithOrderedSources } =
  requireTranspiledTs('src/main/engine/study-chat-format.ts')
const report = existsSync(values.session) ? readJson(values.session) : {
  createdAt: new Date().toISOString(), model: values.model,
  method: 'Experimental pipeline: Gemma rewrite of follow-ups, fresh production hybrid retrieval, Gemma answer using current app prompts. Not the installed app router or GUI.', turns: []
}
if (report.model !== values.model) throw new Error('Session model does not match')
async function completion(request) {
  const started = performance.now()
  const response = await fetch('http://127.0.0.1:11434/api/chat', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request), signal: AbortSignal.timeout(300000)
  })
  if (!response.ok) throw new Error(await response.text())
  const rawResponse = await response.json()
  return { request, rawResponse, text: rawResponse.message.content,
    seconds: (performance.now() - started) / 1000 }
}
const started = performance.now()
const previous = report.turns.at(-1)
let query = values.question
let rewrite
if (previous) {
  rewrite = await completion({ ...promptRequest, model: values.model,
    messages: [
      promptRequest.messages.find((message) => message.role === 'system'),
      { role: 'user', content: previous.question },
      { role: 'assistant', content: previous.displayText },
      { role: 'user', content: values.question }
    ] })
  query = JSON.parse(rewrite.text).question
  if (typeof query !== 'string' || !query.trim()) {
    throw new Error(`No resolved question; not guessing a replacement: ${rewrite.text}`)
  }
}
const worker = spawn(resolve('app_runtime/python/bin/python'),
  [resolve('tests/experiments/buzzdb-library-worker.py'), resolve(values['user-data'])],
  { env: { ...process.env, TOKENSMITH_LOG_FILE: '', PYTHONUNBUFFERED: '1' } })
const workerExit = once(worker, 'close')
worker.stderr.pipe(process.stderr)
const lines = createInterface({ input: worker.stdout })[Symbol.asyncIterator]()
try {
  const searchStarted = performance.now()
  worker.stdin.write(JSON.stringify({ query, limit: 4,
    materials: state.materials.filter((material) => material.isActive && material.status === 'ready'),
    embeddingModels: [{ engine: 'ollama', role: 'embedder', ollamaModelName: 'nomic-embed-text' }] }) + '\n')
  const line = await lines.next()
  if (line.done) throw new Error('Library worker exited without returning sources')
  const search = JSON.parse(line.value)
  if (!search.ok) throw new Error(search.error)
  const searchSeconds = (performance.now() - searchStarted) / 1000
  const sources = search.result.sources
  const request = { prompt: query, messages: [], materials: [], retrievedSources: sources,
    model: { id: `ollama:${values.model}`, name: values.model, engine: 'ollama', source: 'ollama',
      role: 'generator', ollamaModelName: values.model, contextLength: 8192 },
    settings: {}, applicationSettings: { suggestionMode: 'off' },
    modelSettings: { contextLength: 8192, maxLength: 1024, temperature: 0 },
    conversationContextMode: 'standalone' }
  const answer = await completion({ model: values.model, messages: studyChatMessages(request),
    stream: false, think: false,
    options: { num_ctx: 8192, num_predict: 1024, temperature: 0, seed: 42 } })
  const display = answerWithOrderedSources(answer.text, sources)
  const turn = { question: values.question, query, rewrite, sources, vectorHits: search.vectorHits,
    retrieval: search.result, searchSeconds, budget: sourceContextBudgetForRequest(request),
    answer, displayText: display.text, totalSeconds: (performance.now() - started) / 1000 }
  report.turns.push(turn)
  report.updatedAt = new Date().toISOString()
  writeFileSync(values.session, JSON.stringify(report, null, 2))
  console.log(JSON.stringify({ question: turn.question, query, vectorHits: turn.vectorHits,
    sections: sources.map((source) => source.sectionHeader), answer: answer.text,
    answerSeconds: answer.seconds, totalSeconds: turn.totalSeconds,
    doneReason: answer.rawResponse.done_reason }, null, 2))
} finally {
  worker.stdin.end()
  await workerExit
}
