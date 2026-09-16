import { readFileSync, writeFileSync } from 'node:fs'
import { parseArgs } from 'node:util'

const { values } = parseArgs({ options: {
  input: { type: 'string' }, previous: { type: 'string' },
  'prompt-from': { type: 'string' }, out: { type: 'string' }, model: { type: 'string' }
} })
if (!values.input || !values.previous || !values['prompt-from'] || !values.out) {
  throw new Error('--input, --previous, --prompt-from and --out are required')
}
const readJson = (path) => JSON.parse(readFileSync(path, 'utf8'))
const original = readJson(values.input)
const previous = readJson(values.previous)
const retry = readJson(values['prompt-from'])
const { messages: retryMessages, ...savedSettings } = retry.request
const settings = { ...savedSettings, model: values.model ?? savedSettings.model }
const comparisonModel = previous.settings?.model ?? savedSettings.model
const instruction = retryMessages.find((message) => message.role === 'system')?.content
if (!instruction || !original.cases.length) throw new Error('Missing prompt or cases')
const result = {
  createdAt: new Date().toISOString(), settings, instruction, comparisonModel,
  note: 'One real model call per original scenario. This evaluates rewriting, not retrieval or answer accuracy.',
  resolutions: []
}
const save = () => writeFileSync(values.out, JSON.stringify(result, null, 2))
async function generate(messages, overrides = {}) {
  const request = { ...settings, messages, ...overrides }
  const started = performance.now()
  const response = await fetch('http://127.0.0.1:11434/api/chat', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request), signal: AbortSignal.timeout(180000)
  })
  if (!response.ok) throw new Error(await response.text())
  const rawResponse = await response.json()
  return { request, rawResponse, seconds: (performance.now() - started) / 1000,
    loadSeconds: (rawResponse.load_duration ?? 0) / 1e9 }
}
result.warmup = await generate([{ role: 'user', content: 'Return OK.' }], {
  format: undefined, options: { ...settings.options, num_predict: 4 }
})
save()
for (const item of original.cases) {
  const response = await generate([
    { role: 'system', content: instruction },
    ...item.history.map((message) => ({ role: message.role, content: message.text })),
    { role: 'user', content: item.question }
  ])
  let parsed
  let parseError
  try { parsed = JSON.parse(response.rawResponse.message.content) }
  catch (error) { parseError = String(error) }
  const prior = previous.resolutions.find((entry) =>
    entry.caseId === item.id && (entry.model ?? previous.settings?.model) === comparisonModel)
  const entry = {
    caseId: item.id, currentQuestion: item.question, expectedMode: item.expectedMode,
    previousRewrite: prior?.question, question: parsed?.question, parseError, ...response
  }
  result.resolutions.push(entry)
  save()
  console.log(JSON.stringify({ caseId: item.id, current: item.question,
    previous: prior?.question, currentRewrite: parsed?.question,
    seconds: response.seconds, parseError }))
}
const times = result.resolutions.map((entry) => entry.seconds).sort((a, b) => a - b)
const middle = Math.floor(times.length / 2)
result.medianSeconds = times.length % 2 ? times[middle] : (times[middle - 1] + times[middle]) / 2
result.completedAt = new Date().toISOString()
save()
console.log(JSON.stringify({ cases: result.resolutions.length, medianSeconds: result.medianSeconds }))
