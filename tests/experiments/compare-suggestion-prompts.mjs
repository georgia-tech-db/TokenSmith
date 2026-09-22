import { readFileSync, writeFileSync, mkdirSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { parseArgs } from 'node:util'
import { requireTranspiledTs } from '../unit/typescript/ts-module-loader.mjs'

const { values } = parseArgs({ options: { 'user-data': { type: 'string' }, out: { type: 'string' } } })
if (!values['user-data'] || !values.out) throw new Error('--user-data and --out are required')
const state = JSON.parse(readFileSync(resolve(values['user-data'], 'tokensmith-state.json'), 'utf8'))
const rows = readFileSync(resolve(values['user-data'], 'logs/tokensmith.log'), 'utf8')
  .split('\n').filter(Boolean).flatMap((line) => { try { return [JSON.parse(line)] } catch { return [] } })
const { questionSuggestionMessages, followUpSuggestionMessages, questionSuggestionSchema,
  parseFollowUpSuggestions, filterSuggestedQuestions, suggestionMaxTokens } =
  requireTranspiledTs('src/main/engine/study-chat-format.ts')
const model = state.models.find((item) => item.id === state.selectedModelId)
if (model?.engine !== 'ollama') throw new Error('This replay requires the selected Ollama model')
const settings = state.settings.modelSettingsById[model.id] ?? state.settings.modelDefaults
const { defaultStarterQuestionPrompt, defaultSuggestedFollowUpPrompt } =
  requireTranspiledTs('src/shared/model-defaults.ts')
const request = { model, modelSettings: { ...settings,
  starterQuestionPrompt: defaultStarterQuestionPrompt,
  suggestedFollowUpPrompt: defaultSuggestedFollowUpPrompt }, settings: state.settings,
  applicationSettings: state.settings.application, materials: [], messages: [], retrievedSources: [] }
const cases = []
for (const row of rows.filter((item) => item.event === 'question_suggestion_request_context').slice(-3)) {
  cases.push({ kind: 'starter', originalTime: row.time,
    messages: [...row.modelMessages.slice(0, -1), questionSuggestionMessages(request).at(-1)] })
}
for (const row of rows.filter((item) => item.event === 'follow_up_suggestions').slice(-3)) {
  const index = rows.indexOf(row)
  const answer = rows.slice(index + 1).find((item) => item.event === 'chat_response_context')?.text
  if (!answer) throw new Error(`No logged answer following ${row.time}`)
  cases.push({ kind: 'followUp', originalTime: row.time, question: row.prompt, answer,
    previousRaw: row.rawResponse, previouslyShown: row.suggestions,
    messages: followUpSuggestionMessages({ ...request, prompt: row.prompt }, answer) })
}
const report = { createdAt: new Date().toISOString(), model: model.ollamaModelName,
  method: 'Real local model calls using the new production suggestion prompts and parser. Initial calls reuse the exact logged source context; follow-up calls use actual logged questions and answers. Not new retrieval, answer generation, or GUI runs.', results: [] }
for (const item of cases) {
  const body = { model: model.ollamaModelName, messages: item.messages,
    options: { num_ctx: settings.contextLength, num_predict: suggestionMaxTokens,
      temperature: Math.min(Math.max(settings.temperature, 0.2), 0.8),
      top_p: settings.topP, top_k: settings.topK, min_p: settings.minP, repeat_penalty: settings.repeatPenalty },
    format: questionSuggestionSchema(4), stream: false, think: false }
  const started = performance.now()
  const response = await fetch(`${model.ollamaBaseUrl || 'http://127.0.0.1:11434'}/api/chat`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
    signal: AbortSignal.timeout(180000)
  })
  if (!response.ok) throw new Error(await response.text())
  const raw = await response.json()
  const parsed = parseFollowUpSuggestions(raw.message.content, 8)
  const shown = filterSuggestedQuestions(parsed, item.question ? [item.question] : [], 4)
  const result = { ...item, request: body, raw, shown, seconds: (performance.now() - started) / 1000 }
  report.results.push(result)
  mkdirSync(dirname(resolve(values.out)), { recursive: true })
  writeFileSync(values.out, JSON.stringify(report, null, 2))
  console.log(JSON.stringify({ kind: item.kind, question: item.question, shown, seconds: result.seconds, doneReason: raw.done_reason }))
  if (raw.done_reason === 'length') process.exitCode = 1
}
