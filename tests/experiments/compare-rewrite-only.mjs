// Second experiment: isolate reference resolution from mode classification.
import { readFileSync, writeFileSync } from 'node:fs'
import { parseArgs } from 'node:util'

const { values } = parseArgs({ options: { input: { type: 'string' }, out: { type: 'string' } } })
if (!values.input || !values.out) throw new Error('--input and --out are required')
const original = JSON.parse(readFileSync(values.input, 'utf8'))
const result = { createdAt: new Date().toISOString(), resolutions: [], suggestions: [] }
const save = () => writeFileSync(values.out, JSON.stringify(result, null, 2))
const instruction = [
  'Rewrite the LAST student message into one self-contained question for document search.',
  'Use the previous exchange ONLY to replace missing references with their subjects.',
  'Keep the requested task, comparison direction, conditions and example identifiers. Correct obvious typos.',
  'Copy a message that is already understandable alone exactly, even if it changes topics.',
  'If the subject is ambiguous, return an empty question. Do not guess or answer.',
  'Return JSON with one string field: question.'
].join('\n')
const schema = { type: 'object', properties: { question: { type: 'string' } },
  required: ['question'], additionalProperties: false }
async function generate(model, messages, format, maxTokens = 256) {
  const start = performance.now()
  const response = await fetch('http://127.0.0.1:11434/api/chat', { method: 'POST',
    headers: { 'Content-Type': 'application/json' }, signal: AbortSignal.timeout(180000),
    body: JSON.stringify({ model, messages, format, stream: false, think: false,
      options: { num_ctx: 8192, num_predict: maxTokens, temperature: 0, seed: 42 } }) })
  if (!response.ok) throw new Error(await response.text())
  const data = await response.json()
  return { text: data.message.content, seconds: (performance.now() - start) / 1000,
    loadSeconds: (data.load_duration ?? 0) / 1e9, doneReason: data.done_reason,
    promptTokens: data.prompt_eval_count, outputTokens: data.eval_count }
}
for (const model of Object.keys(original.models)) {
  await generate(model, [{ role: 'user', content: 'Return OK.' }], undefined, 4)
  for (const item of original.cases) {
    const response = await generate(model, [
      { role: 'system', content: instruction },
      ...item.history.map((m) => ({ role: m.role, content: m.text })),
      { role: 'user', content: item.question }
    ], schema)
    let parsed
    try { parsed = JSON.parse(response.text) } catch { /* Invalid outputs remain failures. */ }
    const question = parsed?.question
    const topicOrCopyCheck = typeof question === 'string' && (item.expectedMode === 'clarify'
      ? question === '' : item.expectedMode === 'standalone'
        ? question === item.question : new RegExp(item.subject, 'i').test(question))
    result.resolutions.push({ caseId: item.id, model, ...response, question, topicOrCopyCheck }); save()
    console.log(`${model} ${item.id}: topic/copy check=${topicOrCopyCheck} ${response.seconds.toFixed(2)}s ${response.text}`)
  }
  for (const item of original.seeds.filter((s) => ['rehash', 'slots', 'vectors'].includes(s.id))) {
    const response = await generate(model, [
      { role: 'system', content: [
        'Write up to four natural next questions a curious undergraduate would ask about the latest answer.',
        'Explore something the answer mentions but has not explained: a mechanism, reason, example, or implementation.',
        'Name the subject. Use short, direct questions, at most 14 words each.',
        'Do not repeat the student question or ask again for the main explanation already given.',
        'Do not introduce new technical topics or assume unsupported facts.',
        'Return a JSON array of question strings. Fewer than four is fine.'
      ].join('\n') },
      { role: 'user', content: JSON.stringify({ question: item.question, answer: item.answer.text }) }
    ], { type: 'array', items: { type: 'string' }, maxItems: 4 })
    let questions
    try { questions = JSON.parse(response.text) } catch { /* Retain raw failures. */ }
    result.suggestions.push({ seedId: item.id, model, ...response, questions }); save()
    console.log(`suggestions ${model} ${item.id}: ${response.text}`)
  }
}
save()
