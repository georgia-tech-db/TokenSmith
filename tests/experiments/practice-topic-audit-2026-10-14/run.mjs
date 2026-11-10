// Opt-in live audit: production topic retrieval, Practice client, parser, and Ollama provider.
// Runs against an isolated SQLite snapshot; never changes the user's chats or library.
import Module from 'node:module'
import { EventEmitter } from 'node:events'
import { execFileSync } from 'node:child_process'
import { readFileSync, writeFileSync, mkdirSync, mkdtempSync, copyFileSync } from 'node:fs'
import { join, resolve } from 'node:path'
import { homedir, tmpdir } from 'node:os'
import { requireTranspiledTs } from '../../unit/typescript/ts-module-loader.mjs'

const outputDir = resolve('tests/experiments/practice-topic-audit-2026-10-14')
const stateDir = process.env.TOKENSMITH_AUDIT_STATE || join(homedir(), 'Library/Application Support/TokenSmith')
const state = JSON.parse(readFileSync(join(stateDir, 'tokensmith-state.json'), 'utf8'))
const saved = JSON.parse(readFileSync(join(stateDir, 'collections.json'), 'utf8')).find(c => c.name === 'BuzzDBBook' && c.loaded)
if (!saved) throw new Error('Open BuzzDBBook before running this audit.')
const model = state.models.find(m => m.id === state.practice.draftModelId)
const embedder = state.models.find(m => m.id === state.selectedEmbeddingModelId)
if (model?.engine !== 'ollama' || embedder?.engine !== 'ollama') throw new Error('This audit is local-only.')
const modelSettings = state.practice.sessions.find(s => s.modelId === model.id)?.modelSettings || state.settings.modelSettingsById[model.id]
const settings = { application: state.settings.application }
const root = mkdtempSync(join(tmpdir(), 'tokensmith-practice-topic-audit-'))
const library = join(root, 'BuzzDBBook')
mkdirSync(library)
copyFileSync(join(saved.path, 'collection.json'), join(library, 'collection.json'))
execFileSync('python3', ['-c', 'import sqlite3,sys; s=sqlite3.connect("file:"+sys.argv[1]+"?mode=ro",uri=True); d=sqlite3.connect(sys.argv[2]); s.backup(d); d.close(); s.close()', join(saved.path, 'tokensmith.sqlite'), join(library, 'tokensmith.sqlite')])
writeFileSync(join(root, 'collections.json'), JSON.stringify([{ ...saved, path: library }]))
const app = new EventEmitter()
Object.assign(app, { isPackaged: false, getAppPath: () => resolve('.'), getPath: () => root })
const originalLoad = Module._load
Module._load = function(name, ...args) {
  return name === 'electron' ? { app, BrowserWindow: { getAllWindows: () => [] } } : originalLoad.call(this, name, ...args)
}
const service = requireTranspiledTs('src/main/python/python-engine-service.ts')
const { runOllamaStudyEngine } = requireTranspiledTs('src/main/engine/ollama-service.ts')
const { createPracticeClient } = requireTranspiledTs('src/renderer/src/practice-client.ts')
const { parsePracticeQuestion, gradeTrueFalse } = requireTranspiledTs('src/shared/quiz.ts')
Module._load = originalLoad
const report = { startedAt: new Date().toISOString(), model: model.ollamaModelName, embedder: embedder.ollamaModelName,
  modelSettings, applicationSettings: settings.application, mode: 'Production topic retrieval and generation; isolated index snapshot; two consecutive questions per topic; no prompt edits between cases', topics: [], calls: [], runtime: [] }
const save = () => writeFileSync(join(outputDir, 'results.json'), JSON.stringify(report, null, 2))
const nativeFetch = globalThis.fetch
globalThis.fetch = async (url, options) => {
  if (String(url).endsWith('/api/chat')) {
    const body = JSON.parse(options.body)
    report.runtime.push({ model: body.model, think: body.think, options: body.options, stream: body.stream })
  }
  return nativeFetch(url, options)
}
let activeTopic, activeNumber
const bridge = { searchLibrary: service.searchLibraryWithPython, practiceSources: service.practiceSourcesWithPython,
  sendChatMessage: async request => {
    const start = performance.now()
    const call = { topic: activeTopic, questionNumber: activeNumber, prompt: request.prompt, inputSources: request.retrievedSources }
    report.calls.push(call)
    try {
      const reply = await runOllamaStudyEngine(request)
      call.reply = reply
      try { parsePracticeQuestion(reply.text, reply.sources, 'true_false'); call.validation = 'accepted' }
      catch (error) { call.validation = error.reason || error.message }
      return reply
    } catch (error) { call.error = error.message; throw error }
    finally { call.seconds = (performance.now() - start) / 1000; save() }
  }
}
try {
  await service.collections().initialize()
  const materials = await service.listIndexedMaterialsWithPython()
  const documents = await service.studyDocumentsWithPython()
  const client = createPracticeClient(bridge, model, settings, { materials, embeddingModels: [embedder] })
  for (const title of ['B+ trees', 'Volcano iterator model', 'Heap files', 'LRU replacement', 'Extendible hashing', 'Vectorized execution']) {
    activeTopic = title
    const entry = { title, questions: [] }
    report.topics.push(entry)
    const retrievalStart = performance.now()
    let session = { id: `audit-${title}`, modelId: model.id, modelSettings, documents,
      questions: [], topic: { title } }
    try {
      session = await client.prepareTopic(session, new AbortController().signal)
      entry.retrievalSeconds = (performance.now() - retrievalStart) / 1000
      entry.retrievedSources = session.topic.sources
      for (let n = 1; n <= 2; n++) {
        activeNumber = n
        const start = performance.now(), callStart = report.calls.length
        try {
          const question = await client.question(session, `audit-${title}-${n}`, new AbortController().signal)
          if (!question) throw new Error('No question returned')
          const trueFeedback = gradeTrueFalse(question, 'True'), falseFeedback = gradeTrueFalse(question, 'False')
          entry.questions.push({ number: n, seconds: (performance.now() - start) / 1000,
            calls: report.calls.length - callStart, question, trueFeedback, falseFeedback })
          session.questions.push(question)
          console.log(JSON.stringify({ topic: title, number: n, seconds: entry.questions.at(-1).seconds,
            calls: report.calls.length - callStart, question: question.text, key: question.correctAnswer }))
        } catch (error) {
          entry.questions.push({ number: n, seconds: (performance.now() - start) / 1000, calls: report.calls.length - callStart, error: error.message })
          console.log(JSON.stringify({ topic: title, number: n, error: error.message }))
        }
        save()
      }
    } catch (error) { entry.error = error.message; save(); console.log(JSON.stringify({ topic: title, error: error.message })) }
  }
} finally {
  report.finishedAt = new Date().toISOString()
  save()
  app.emit('before-quit')
}
console.log(`Saved live results to ${join(outputDir, 'results.json')}; semantic review is separate from parser validation.`)
