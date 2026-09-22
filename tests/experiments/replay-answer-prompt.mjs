import { readFileSync, writeFileSync, mkdirSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { parseArgs } from 'node:util'
import { requireTranspiledTs } from '../unit/typescript/ts-module-loader.mjs'

const { values } = parseArgs({ options: {
  'user-data': { type: 'string' }, out: { type: 'string' }, question: { type: 'string', multiple: true },
  'system-prompt': { type: 'string' }, 'skip-before': { type: 'boolean' },
  'instructions-last': { type: 'boolean' }
} })
if (!values['user-data'] || !values.out || !values.question?.length) {
  throw new Error('--user-data, --out, and at least one --question are required')
}
const state = JSON.parse(readFileSync(resolve(values['user-data'], 'tokensmith-state.json'), 'utf8'))
const rows = readFileSync(resolve(values['user-data'], 'logs/tokensmith.log'), 'utf8')
  .split('\n').filter(Boolean).flatMap((line) => { try { return [JSON.parse(line)] } catch { return [] } })
const { studyChatMessages, followUpSuggestionMessages, questionSuggestionSchema, suggestionMaxTokens,
  parseFollowUpSuggestions, filterSuggestedQuestions } = requireTranspiledTs('src/main/engine/study-chat-format.ts')
const { defaultSuggestedFollowUpPrompt } = requireTranspiledTs('src/shared/model-defaults.ts')
const model = state.models.find((item) => item.id === state.selectedModelId)
if (model?.engine !== 'ollama') throw new Error('This replay requires the selected Ollama model')
const settings = state.settings.modelSettingsById[model.id] ?? state.settings.modelDefaults
const report = { createdAt: new Date().toISOString(), model: model.ollamaModelName,
  method: 'Real model calls reusing logged hybrid sources, question, and reference exchange. Instruction changes are recorded in each request. Suggestions use the newly generated answer. Not new retrieval or GUI calls.', results: [] }

async function generate(messages, suggestions = false) {
  const request = { model: model.ollamaModelName, messages, stream: false, think: false,
    options: { num_ctx: settings.contextLength, num_predict: suggestions ? suggestionMaxTokens : settings.maxLength,
      temperature: suggestions ? 0.2 : settings.temperature, top_p: settings.topP, top_k: settings.topK,
      min_p: settings.minP, repeat_penalty: settings.repeatPenalty },
    ...(suggestions ? { format: questionSuggestionSchema(4) } : {}) }
  const started = performance.now()
  const response = await fetch(`${model.ollamaBaseUrl || 'http://127.0.0.1:11434'}/api/chat`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(request),
    signal: AbortSignal.timeout(180000)
  })
  if (!response.ok) throw new Error(await response.text())
  const raw = await response.json()
  if (raw.done_reason === 'length') process.exitCode = 1
  return { request, raw, text: raw.message.content, seconds: (performance.now() - started) / 1000 }
}

for (const question of values.question) {
  const row = rows.findLast((item) => item.event === 'chat_request_context' && item.prompt === question)
  if (!row) throw new Error(`No logged request for ${question}`)
  const request = { model, modelSettings: { ...settings, systemMessage: row.systemPrompt,
    suggestedFollowUpPrompt: defaultSuggestedFollowUpPrompt }, prompt: row.prompt, messages: [], materials: [],
    settings: state.settings, applicationSettings: state.settings.application, retrievedSources: row.sources }
  const systemMessage = values['system-prompt']
    ? { role: 'system', content: readFileSync(values['system-prompt'], 'utf8').trim() }
    : studyChatMessages(request).find((item) => item.role === 'system')
  const before = values['skip-before'] ? undefined : await generate(row.modelMessages)
  const updatedMessages = row.modelMessages.map((item) => item.role === 'system' ? systemMessage : { ...item })
  // Older logs put the old answer instructions before the material in the user message.
  const legacyInlineInstructionsReplaced = !updatedMessages.some((item) => item.role === 'system')
  if (legacyInlineInstructionsReplaced) {
    const user = updatedMessages.findLast((item) => item.role === 'user')
    const contextStart = user.content.indexOf('### Context:')
    if (contextStart < 0) throw new Error('Cannot identify the legacy material boundary')
    user.content = user.content.slice(contextStart)
    updatedMessages.unshift(systemMessage)
  }
  if (values['instructions-last']) {
    const user = updatedMessages.findLast((item) => item.role === 'user')
    const position = user.content.lastIndexOf('Question: ')
    user.content = `${user.content.slice(0, position)}### Answer instructions:\n${systemMessage.content}\n\n${user.content.slice(position)}`
  }
  const after = await generate(updatedMessages)
  const suggestions = await generate(followUpSuggestionMessages(request, after.text), true)
  suggestions.shown = filterSuggestedQuestions(parseFollowUpSuggestions(suggestions.text, 8), [question], 4)
  report.results.push({ question, originalTime: row.time, legacyInlineInstructionsReplaced, before, after, suggestions })
  mkdirSync(dirname(resolve(values.out)), { recursive: true })
  writeFileSync(values.out, JSON.stringify(report, null, 2))
  console.log(JSON.stringify({ question, before: before?.text, after: after.text,
    seconds: after.seconds, suggestions: suggestions.shown }))
}
