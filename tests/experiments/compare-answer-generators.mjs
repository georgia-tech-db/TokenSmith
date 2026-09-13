import assert from 'node:assert/strict'
import { createHash } from 'node:crypto'
import { readFileSync, writeFileSync } from 'node:fs'
import { parseArgs } from 'node:util'
import { requireTranspiledTs } from '../unit/typescript/ts-module-loader.mjs'

const { values } = parseArgs({ options: {
  original: { type: 'string' }, replays: { type: 'string' }, out: { type: 'string' },
  models: { type: 'string', default: 'llama3:latest,gemma4:e4b' }
} })
if (!values.original || !values.replays || !values.out) throw new Error('--original, --replays and --out are required')
const readJson = (path) => JSON.parse(readFileSync(path, 'utf8'))
const original = readJson(values.original)
const replays = readJson(values.replays)
const specs = readJson(new URL('./generator-review-cases.json', import.meta.url))
const { studyChatMessages, sourceContextBudgetForRequest, answerWithOrderedSources } =
  requireTranspiledTs('src/main/engine/study-chat-format.ts')
const options = { num_ctx: 8192, num_predict: 1024, temperature: 0, seed: 42 }
const models = values.models.split(',')
const cases = specs.map((spec) => {
  const source = spec.seedId ? original.seeds.find((entry) => entry.id === spec.seedId)
    : replays.find((entry) => entry.caseId === spec.replayCaseId && entry.resolver === spec.resolver)
  if (!source) throw new Error(`Missing source case ${spec.id}`)
  const question = spec.seedId ? source.question : source.query
  const request = { prompt: question, messages: [], materials: [], retrievedSources: source.sources,
    model: { id: 'comparison', name: 'comparison', engine: 'ollama', source: 'ollama',
      role: 'generator', contextLength: 8192 },
    settings: {}, applicationSettings: { suggestionMode: 'off' },
    modelSettings: { contextLength: 8192, maxLength: 1024, temperature: 0 },
    conversationContextMode: 'standalone' }
  const messages = studyChatMessages(request)
  const budget = sourceContextBudgetForRequest(request)
  assert.equal(budget.includedSourceCount, source.sources.length)
  assert.equal(budget.truncatedSourceCount, 0)
  return { ...spec, question, sources: source.sources, messages, budget,
    promptSha256: createHash('sha256').update(JSON.stringify(messages)).digest('hex'),
    historicalLlamaAnswer: spec.seedId ? source.answer.text : source.text }
})
const report = { createdAt: new Date().toISOString(),
  method: 'Fixed previously hybrid-retrieved sources; identical current app prompts for both generators. Review criteria are not sent to either model. No live retrieval or rewriting in this phase.',
  options, models: {}, cases, warmups: [], answers: [] }
const save = () => writeFileSync(values.out, JSON.stringify(report, null, 2))
async function api(path, body) {
  const response = await fetch(`http://127.0.0.1:11434${path}`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body), signal: AbortSignal.timeout(300000)
  })
  if (!response.ok) throw new Error(`${response.status}: ${await response.text()}`)
  return response.json()
}
for (const model of models) {
  const info = await api('/api/show', { model })
  report.models[model] = { details: info.details, parameters: info.parameters, capabilities: info.capabilities }
  const warmStarted = performance.now()
  const warmResponse = await api('/api/chat', { model,
    messages: [{ role: 'user', content: 'Return OK.' }], stream: false, think: false,
    options: { ...options, num_predict: 4 } })
  report.warmups.push({ model, seconds: (performance.now() - warmStarted) / 1000, rawResponse: warmResponse })
  save()
  for (const item of cases) {
    const request = { model, messages: item.messages, stream: false, think: false, options }
    const started = performance.now()
    const rawResponse = await api('/api/chat', request)
    const seconds = (performance.now() - started) / 1000
    const rawText = rawResponse.message.content
    const display = answerWithOrderedSources(rawText, item.sources)
    report.answers.push({ caseId: item.id, model, request, rawResponse, rawText,
      displayText: display.text, displaySourceIds: display.sources.map((source) => source.chunkRowid ?? source.id),
      seconds, loadSeconds: (rawResponse.load_duration ?? 0) / 1e9,
      doneReason: rawResponse.done_reason, promptTokens: rawResponse.prompt_eval_count,
      outputTokens: rawResponse.eval_count, promptSha256: item.promptSha256 })
    save()
    console.log(JSON.stringify({ caseId: item.id, model, seconds,
      doneReason: rawResponse.done_reason, text: rawText }))
  }
}
report.completedAt = new Date().toISOString()
save()
