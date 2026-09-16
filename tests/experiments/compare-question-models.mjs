// Opt-in live evaluation. Requires the app index and installed Ollama models.
// node tests/experiments/compare-question-models.mjs --user-data PATH --out PATH
import { spawn } from 'node:child_process'
import { once } from 'node:events'
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { createInterface } from 'node:readline'
import { parseArgs } from 'node:util'
import { requireTranspiledTs } from '../unit/typescript/ts-module-loader.mjs'

const { values } = parseArgs({ options: {
  'user-data': { type: 'string' }, out: { type: 'string' },
  models: { type: 'string', default: 'llama3:latest,qwen3:1.7b,qwen3:0.6b' },
  phase: { type: 'string', default: 'all' }
} })
if (!values['user-data'] || !values.out) throw new Error('--user-data and --out are required')
const userData = resolve(values['user-data'])
const outputPath = resolve(values.out)
const modelNames = values.models.split(',')
const state = JSON.parse(readFileSync(resolve(userData, 'tokensmith-state.json'), 'utf8'))
const { routeRetrievalContext } = requireTranspiledTs('src/shared/chat-context.ts')
const { studyChatMessages, followUpSuggestionMessages, questionSuggestionMessages,
  parseFollowUpSuggestions, filterSuggestedQuestions } = requireTranspiledTs('src/main/engine/study-chat-format.ts')
const activeMaterials = state.materials.filter((m) => m.isActive && m.status === 'ready')
const embeddingModels = [{ id: 'ollama:nomic-embed-text', engine: 'ollama', source: 'ollama', role: 'embedder',
  ollamaModelName: 'nomic-embed-text' }]
const baseUrl = 'http://127.0.0.1:11434'
const report = { createdAt: new Date().toISOString(), generator: 'llama3:latest',
  searchMode: 'hybrid', embeddingModel: 'nomic-embed-text', models: {},
  cases: [], resolutions: [], suggestions: [], answers: [], starterSuggestions: [] }
mkdirSync(dirname(outputPath), { recursive: true })
function save() { writeFileSync(outputPath, JSON.stringify(report, null, 2)) }

const worker = spawn(resolve('app_runtime/python/bin/python'),
  [resolve('tests/experiments/buzzdb-library-worker.py'), userData],
  { env: { ...process.env, TOKENSMITH_LOG_FILE: '', PYTHONUNBUFFERED: '1' } })
const workerExit = once(worker, 'close')
const responses = createInterface({ input: worker.stdout })[Symbol.asyncIterator]()
let workerError = ''
worker.stderr.on('data', (data) => { workerError += data.toString() })
const searchCache = new Map()
async function search(query, limit = 4) {
  const key = JSON.stringify([query, limit])
  if (searchCache.has(key)) return searchCache.get(key)
  worker.stdin.write(JSON.stringify({ query, limit, materials: activeMaterials, embeddingModels }) + '\n')
  const line = await responses.next()
  if (line.done) throw new Error(`Library worker exited: ${workerError}`)
  const response = JSON.parse(line.value)
  if (!response.ok) throw new Error(response.error)
  searchCache.set(key, response.result.sources)
  return response.result.sources
}

async function completion(model, messages, { format, maxTokens = 256, temperature = 0 } = {}) {
  const start = performance.now()
  const response = await fetch(`${baseUrl}/api/chat`, { method: 'POST',
    headers: { 'Content-Type': 'application/json' }, signal: AbortSignal.timeout(180000),
    body: JSON.stringify({ model, messages, stream: false, think: false, format,
      options: { num_ctx: 8192, num_predict: maxTokens, temperature, seed: 42 } }) })
  if (!response.ok) throw new Error(`${model}: ${response.status} ${await response.text()}`)
  const result = await response.json()
  return { text: result.message.content, seconds: (performance.now() - start) / 1000,
    loadSeconds: (result.load_duration ?? 0) / 1e9,
    promptTokens: result.prompt_eval_count, outputTokens: result.eval_count,
    doneReason: result.done_reason }
}

function request(question, messages, sources) {
  return { prompt: question, messages, retrievedSources: sources, materials: activeMaterials,
    model: { id: 'ollama:llama3', name: 'llama3:latest', engine: 'ollama', source: 'ollama',
      role: 'generator', ollamaModelName: 'llama3:latest', contextLength: 8192 },
    settings: {}, applicationSettings: { suggestionMode: 'on', followUpSuggestionCount: 4 },
    modelSettings: { contextLength: 8192, maxLength: 1024, temperature: 0 },
    conversationContextMode: 'standalone' }
}
function history(question, answer, sources = []) {
  return [{ id: 'u1', role: 'user', text: question },
    { id: 'a1', role: 'assistant', text: answer, sources }]
}
async function seed(id, question) {
  const sources = await search(question)
  const answer = await completion(report.generator, studyChatMessages(request(question, [], sources)), { maxTokens: 768 })
  console.log(`seed ${id}: ${answer.seconds.toFixed(2)}s`)
  return { id, question, answer, sources }
}
const resolutionSchema = { type: 'object', additionalProperties: false,
  properties: { mode: { type: 'string', enum: ['standalone', 'contextual', 'clarify'] },
    resolvedQuestion: { type: 'string' } }, required: ['mode', 'resolvedQuestion'] }
const resolutionInstruction = [
  'Resolve a student question for document retrieval. Do not answer it.',
  'Use the conversation only to identify references, not as verified factual evidence.',
  'If the current question names its own subject and is understandable alone, use standalone and copy it unchanged.',
  'If its subject or intended comparison depends on the previous exchange, use contextual and write a self-contained question.',
  'Preserve the requested task, comparison direction, qualifications, and concrete example identifiers. Correct obvious typos.',
  'Do not add explanations, assumptions, factual claims, or new technical concepts.',
  'If multiple antecedents are equally plausible, use clarify and write a short clarification question.',
  'Return only JSON matching the supplied schema.'
].join('\n')
async function resolveQuestion(model, item) {
  const generated = await completion(model, [
    { role: 'system', content: resolutionInstruction },
    { role: 'user', content: JSON.stringify({ previousQuestion: item.history[0].text,
      previousAnswer: item.history[1].text, currentQuestion: item.question }) }
  ], { format: resolutionSchema })
  let parsed
  try { parsed = JSON.parse(generated.text) } catch { /* Retain invalid output as a failure. */ }
  const valid = parsed && ['standalone', 'contextual', 'clarify'].includes(parsed.mode) &&
    typeof parsed.resolvedQuestion === 'string' && parsed.resolvedQuestion.trim().length > 0
  const subjectPresent = valid && (!item.subject || new RegExp(item.subject, 'i').test(parsed.resolvedQuestion))
  const routingAndSubjectCheck = Boolean(valid && parsed.mode === item.expectedMode && subjectPresent)
  return { caseId: item.id, model, ...generated, parsed, routingAndSubjectCheck }
}

try {
  for (const model of modelNames) {
    const response = await fetch(`${baseUrl}/api/show`, { method: 'POST',
      headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ model }) })
    if (!response.ok) throw new Error(`Model not installed: ${model}`)
    const data = await response.json()
    report.models[model] = { details: data.details, parameters: data.parameters }
  }
  const rehashQuestion = 'How does rehashing impact performance when dealing with large datasets or high concurrency in BuzzDB?'
  const logRows = readFileSync(resolve(userData, 'logs/tokensmith.log'), 'utf8').trim().split('\n').map(JSON.parse)
  const logIndex = logRows.findLastIndex((r) => r.event === 'chat_request_context' && r.prompt === rehashQuestion)
  if (logIndex < 0) throw new Error('The reported rehashing exchange was not found in the app log.')
  const logAnswer = logRows.slice(logIndex + 1).find((r) => r.event === 'chat_response_context')
  const rehash = { id: 'rehash', question: rehashQuestion, answer: { text: logAnswer.text },
    sources: await search(rehashQuestion), origin: 'reported-app-exchange' }
  const lru = await seed('lru', 'How does LRU keep hot pages in the buffer?')
  const slots = await seed('slots', 'How does a slotted page keep record identifiers stable during compaction?')
  const vectors = await seed('vectors', 'How does vectorized execution reduce per-tuple overhead?')
  const btree = await seed('btree', 'Why are B+ trees efficient for range scans?')
  const pinning = await seed('pinning', 'What is pinning a page?')
  report.seeds = [rehash, lru, slots, vectors, btree, pinning]
  const caseFrom = (id, seed, question, expectedMode, subject) =>
    ({ id, question, expectedMode, subject, history: history(seed.question, seed.answer.text, seed.sources) })
  report.cases = [
    caseFrom('rehash_example', rehash, 'Can you give me a small example?', 'contextual', 'rehash'),
    caseFrom('lru_example', lru, 'Can you give me a small example?', 'contextual', 'LRU|least.recent'),
    caseFrom('lru_elaborate', lru, 'Elaborate on that', 'contextual', 'LRU|least.recent'),
    caseFrom('slotted_typo', slots, 'can you show how thse stay the same when the records move?', 'contextual', 'slot|identif'),
    caseFrom('vector_simd', vectors, 'How does SIMD build on that idea?', 'contextual', 'vector|batch'),
    caseFrom('hash_contrast_typo', btree, 'does that mean we should always prefer it over hasing?', 'contextual', 'hash'),
    caseFrom('pinning_why', pinning, 'why is that needed?', 'contextual', 'pin'),
    caseFrom('short_topic_switch', lru, 'What is durability?', 'standalone', 'durability'),
    caseFrom('long_topic_switch', rehash, 'How does write-ahead logging protect committed transactions after a crash?', 'standalone', 'write.ahead'),
    { id: 'ambiguous_comparison', question: 'Why is it better?', expectedMode: 'clarify',
      history: history('Which indexes are available?', 'A hash index and a B+ tree index are available.') },
    { id: 'preserve_example_ids', question: 'Why access A and B again before the scan?', expectedMode: 'contextual', subject: '\\bA\\b.*\\bB\\b',
      history: history('Show how 2Q handles a scan.', 'Access pages A, B, A, B to promote A and B. Then scan C, D, E once.') }
  ]
  for (const item of report.cases) {
    const start = performance.now()
    const choice = await routeRetrievalContext(item.question, item.history, { limit: 4, turnCount: 1, carriedSourceLimit: 2, search })
    item.baseline = { ...choice, seconds: (performance.now() - start) / 1000 }
    console.log(`baseline ${item.id}: ${choice.mode}`)
  }
  save()
  if (['all', 'resolver'].includes(values.phase)) {
    for (const model of modelNames) {
      await completion(model, [{ role: 'user', content: 'Return OK.' }], { maxTokens: 4 })
      for (const item of report.cases) {
        const result = await resolveQuestion(model, item)
        report.resolutions.push(result); save()
        console.log(`${model} ${item.id}: routing/subject check=${result.routingAndSubjectCheck} ${result.seconds.toFixed(2)}s ${result.text}`)
      }
    }
  }
  if (['all', 'suggestions'].includes(values.phase)) {
    for (const model of modelNames) {
      for (const item of [rehash, slots, vectors]) {
        const req = request(item.question, [], item.sources)
        const result = await completion(model, followUpSuggestionMessages(req, item.answer.text), { maxTokens: 160, temperature: 0.2 })
        const questions = filterSuggestedQuestions(parseFollowUpSuggestions(result.text, 8), [item.question], 4)
        report.suggestions.push({ seedId: item.id, model, ...result, questions }); save()
        console.log(`suggestions ${model} ${item.id}: ${JSON.stringify(questions)}`)
      }
      // Same distributed evidence sample for every model; starter source sampling is not under test here.
      const starterSources = [lru, slots, vectors, btree].map((s) => s.sources[0])
      const result = await completion(model, questionSuggestionMessages(request('', [], starterSources)), { maxTokens: 160, temperature: 0.2 })
      report.starterSuggestions.push({ model, ...result, questions: filterSuggestedQuestions(parseFollowUpSuggestions(result.text, 8), [], 4) })
      save()
    }
  }
  if (values.phase === 'all') {
    // Evaluate real answers for the reported failure and two fresh student follow-ups.
    for (const item of report.cases.filter((c) => ['rehash_example', 'slotted_typo', 'vector_simd'].includes(c.id))) {
      for (const model of ['baseline', ...modelNames]) {
        const resolved = report.resolutions.find((r) => r.model === model && r.caseId === item.id)
        if (model !== 'baseline' && (!resolved?.parsed || resolved.parsed.mode === 'clarify')) continue
        const query = model === 'baseline' ? item.baseline.query : resolved.parsed.resolvedQuestion
        const sources = model === 'baseline' ? item.baseline.sources : await search(query)
        const req = request(query, [], sources)
        if (model === 'baseline') {
          Object.assign(req, { prompt: item.question, answerPrompt: item.baseline.answerPrompt,
            retrievalQuery: query, conversationContextMode: item.baseline.mode, messages: item.history })
        }
        const answer = await completion(report.generator, studyChatMessages(req), { maxTokens: 768 })
        report.answers.push({ caseId: item.id, resolver: model, query, sources, ...answer }); save()
        console.log(`answer ${item.id} ${model}: ${answer.seconds.toFixed(2)}s`)
      }
    }
    // Click the first actual suggestion from the answer model and ask it; no authored follow-up substitutes.
    for (const seedItem of [slots, vectors]) {
      const suggestion = report.suggestions.find((s) => s.model === report.generator && s.seedId === seedItem.id)?.questions[0]
      if (!suggestion) continue
      const item = caseFrom(`${seedItem.id}_suggestion_click`, seedItem, suggestion, 'contextual')
      const resolved = await resolveQuestion('qwen3:1.7b', item)
      if (!resolved.parsed || resolved.parsed.mode === 'clarify') {
        report.answers.push({ caseId: item.id, question: suggestion, resolution: resolved }); save(); continue
      }
      const query = resolved.parsed.resolvedQuestion
      const sources = await search(query)
      const answer = await completion(report.generator, studyChatMessages(request(query, [], sources)), { maxTokens: 768 })
      report.answers.push({ caseId: item.id, question: suggestion, resolution: resolved, query, sources, ...answer }); save()
      console.log(`clicked suggestion ${item.id}: ${suggestion}`)
    }
  }
  save()
  console.log(`Report: ${outputPath}`)
} finally {
  worker.stdin.end()
  await workerExit
}
