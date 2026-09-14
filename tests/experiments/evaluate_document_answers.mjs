// Real production rewrite/generation services and Python hybrid search; no mocked model calls.
import { readFileSync, writeFileSync } from 'node:fs'
import { spawn } from 'node:child_process'
import { createInterface } from 'node:readline'
import { resolve } from 'node:path'
import { parseArgs } from 'node:util'
import { requireTranspiledTs } from '../unit/typescript/ts-module-loader.mjs'

const { values } = parseArgs({ options: { input: { type: 'string' }, out: { type: 'string' }, organic: { type: 'boolean' } } })
if (!values.input || !values.out) throw new Error('--input and --out are required')
const input = JSON.parse(readFileSync(values.input, 'utf8'))
const { prepareRewrittenStudyChat } = requireTranspiledTs('src/shared/study-chat-pipeline.ts')
const { runOllamaStudyEngine, resolveOllamaChatQuestion } = requireTranspiledTs('src/main/engine/ollama-service.ts')
const { sourceContextBudgetForRequest } = requireTranspiledTs('src/main/engine/study-chat-format.ts')
const worker = spawn(resolve('app_runtime/python/bin/python'), [resolve('python_engine/tokensmith_engine.py')], { stdio: ['pipe', 'pipe', 'inherit'] })
const pending = new Map()
worker.on('error', error => {
  for (const request of pending.values()) { clearTimeout(request.timer); request.reject(error) }
  pending.clear()
})
createInterface({ input: worker.stdout }).on('line', line => {
  const event = JSON.parse(line)
  const request = pending.get(event.id)
  if (!request || event.ok === undefined) return
  clearTimeout(request.timer)
  pending.delete(event.id)
  if (event.ok) request.resolve(event.result)
  else request.reject(new Error(event.error))
})
worker.on('exit', code => {
  for (const request of pending.values()) { clearTimeout(request.timer); request.reject(new Error(`Worker exited: ${code}`)) }
  pending.clear()
})
let sequence = 0
function search(caseData, query) {
  const id = String(++sequence)
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => { pending.delete(id); reject(new Error('Search timed out')) }, 120000)
    pending.set(id, { resolve, reject, timer })
    worker.stdin.write(JSON.stringify({ id, command: 'search', payload: {
      userDataPath: input.profile, query, materials: [caseData.material], limit: 4,
      searchMode: 'hybrid', embeddingModels: [input.embedder], embeddingModel: input.embedder
    } }) + '\n')
  })
}
const results = { createdAt: new Date().toISOString(), model: input.model, cases: [] }
const save = () => writeFileSync(values.out, JSON.stringify(results, null, 2) + '\n')
try {
  for (const definition of input.cases) {
    const result = { id: definition.id, turns: [] }
    results.cases.push(result)
    if (definition.error) { result.error = definition.error; save(); continue }
    async function ask(question, history = []) {
      const start = performance.now()
      const request = {
        prompt: question, messages: history, materials: [definition.material], settings: {},
        model: { ...input.model, name: input.model.ollamaModelName, role: 'generator', status: 'ready' },
        modelSettings: { contextLength: 8192, maxLength: 1024, temperature: 0.7, topP: 0.4, topK: 40, minP: 0, repeatPenalty: 1.18 },
        applicationSettings: { suggestionMode: 'on', followUpSuggestionCount: 4 }
      }
      const prepared = await prepareRewrittenStudyChat(request, {
        resolve: resolveOllamaChatQuestion,
        search: async query => {
          const result = await search(definition, query)
          if (result.reason || !result.sources.length) throw new Error(result.reason || 'No sources returned')
          return result.sources
        }
      })
      const answer = prepared.request ? await runOllamaStudyEngine(prepared.request) : { text: prepared.resolution.clarification }
      const turn = { question, ...answer, resolution: prepared.resolution,
        rewriteMs: prepared.rewriteMs, searchMs: prepared.searchMs,
        seconds: (performance.now() - start) / 1000,
        budget: prepared.request && sourceContextBudgetForRequest(prepared.request) }
      result.turns.push(turn)
      save()
      console.log(definition.id, prepared.resolution.mode, turn.seconds.toFixed(1), question, '\n', answer.text, '\n')
      return turn
    }
    for (const question of definition.questions) await ask(question)
    if (values.organic) {
      let last = result.turns.at(-1)
      const history = [{ role: 'user', text: last.question }, { role: 'assistant', text: last.text }]
      for (let i = 0; i < 2; i++) {
        const question = last.followUpSuggestions?.[0]
        if (!question) break
        last = await ask(question, [...history])
        history.push({ role: 'user', text: last.question }, { role: 'assistant', text: last.text })
      }
    }
  }
} finally {
  worker.stdin.end()
  worker.kill()
  save()
}
