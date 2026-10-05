import { createHash } from 'node:crypto'
import { spawn } from 'node:child_process'
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { createInterface } from 'node:readline'
import { requireTranspiledTs } from '../unit/typescript/ts-module-loader.mjs'

const casesPath = 'tests/benchmarks/buzzdb_reasoning_cases.json'

// Only questions and actual preceding answers cross this boundary. Rubrics are
// attached to results afterwards, never to the generator request.
export async function runReasoningTurns(families, run, onResult = async () => {}) {
  const results = []
  for (const family of families) {
    const history = []
    let blockedBy
    for (const turn of family.turns) {
      const row = {
        id: `${family.id}.${turn.phase}`, question: turn.question,
        referenceAnswer: turn.referenceAnswer, criteria: turn.criteria,
        evidenceGroups: family.evidenceGroups,
        referenceExchange: history.length ? { question: history.at(-2).text, answer: history.at(-1).text } : undefined
      }
      if (blockedBy) {
        row.error = `Not run: preceding turn ${blockedBy} failed. No reference answer was substituted.`
        row.status = 'blocked'
      } else {
        try {
          Object.assign(row, await run({ prompt: turn.question, messages: structuredClone(history) }))
          if (row.error || !row.answer?.text?.trim()) throw new Error(row.error || 'No complete answer was returned.')
          row.status = 'completed'
          history.push({ role: 'user', text: turn.question }, { role: 'assistant', text: row.answer.text })
        } catch (error) {
          row.error = error.message
          row.status = 'error'
          blockedBy = row.id
        }
      }
      row.passed = row.status === 'completed'
      row.failures = row.error ? [row.error] : []
      row.reviewStatus = 'ungraded'
      results.push(row)
      await onResult(row, results)
    }
  }
  return results
}

function startRetrievalWorker() {
  const child = spawn(process.env.TOKENSMITH_BENCHMARK_PYTHON ?? 'python3',
    ['-m', 'tests.benchmarks.buzzdb_live_worker'], {
      cwd: process.cwd(), stdio: ['pipe', 'pipe', 'inherit'],
      env: { ...process.env, TOKENSMITH_RUN_EMBEDDING_BENCHMARK: '1' }
    })
  const pending = new Map()
  let sequence = 0
  let dead = false
  let readyResolve, readyReject
  const ready = new Promise((resolve, reject) => { readyResolve = resolve; readyReject = reject })
  const fail = error => {
    dead = true
    readyReject(error)
    for (const item of pending.values()) { clearTimeout(item.timer); item.reject(error) }
    pending.clear()
  }
  const startupTimer = setTimeout(() => { fail(new Error('Hybrid worker startup timed out.')); child.kill() }, 120_000)
  const lines = createInterface({ input: child.stdout })
  lines.on('line', line => {
    let message
    try { message = JSON.parse(line) } catch { fail(new Error(`Invalid hybrid worker output: ${line}`)); child.kill(); return }
    if (message.ready) { clearTimeout(startupTimer); readyResolve(message); return }
    const item = pending.get(message.id)
    if (!item) return
    pending.delete(message.id)
    clearTimeout(item.timer)
    message.ok ? item.resolve(message.result) : item.reject(new Error(message.error))
  })
  child.on('error', fail)
  child.stdin.on('error', fail)
  const closed = new Promise(resolve => child.on('close', code => {
    clearTimeout(startupTimer)
    lines.close()
    fail(new Error(`Hybrid worker exited (${code}).`))
    resolve()
  }))
  return {
    ready,
    call(payload) {
      if (dead) return Promise.reject(new Error('Hybrid worker is not running.'))
      const id = ++sequence
      return new Promise((resolve, reject) => {
        const timer = setTimeout(() => {
          pending.delete(id)
          reject(new Error('Hybrid worker request timed out.'))
          child.kill()
        }, 120_000)
        pending.set(id, { resolve, reject, timer })
        child.stdin.write(`${JSON.stringify({ ...payload, id })}\n`)
      })
    },
    async close() {
      clearTimeout(startupTimer)
      child.stdin.end()
      const timer = setTimeout(() => child.kill(), 5000)
      await closed
      clearTimeout(timer)
    }
  }
}

export function packedEvidenceFromCalls(calls, sources) {
  const final = calls.findLast(call => call.request?.messages && !call.request.format && call.request.options?.num_predict !== 1)
  if (!final) return []
  const prompt = final.request.messages.filter(message => message.role === 'user').map(message => message.content).join('\n')
  // Gemma E4B's production packer preserves complete source units. Test the
  // actual final request, not the preflight estimate or previous-answer text.
  const sourceRegion = prompt.split('### Context:')[1]?.split('\n\nQuestion:')[0] ?? ''
  return sources.filter(source => sourceRegion.includes(`Text: ${(source.context || source.excerpt).trim()}\n### End source unit`))
}

export function formatReasoningAnswers(suite) {
  const lines = [
    '# BuzzDB Live Reasoning Answers', '',
    `Model: ${suite.model}. ${suite.passed}/${suite.total} completed answers. **Not an answer-accuracy score.**`,
    'Real Nomic hybrid retrieval, production rewriting/packing/generation, and each chain\'s own new answers.',
    'Fixed diagnostic follow-ups, not adaptive student questions. Reference answers and criteria were withheld from the model.',
    'Suggestions disabled to measure the answer path. Timings include preflight; no retries replace failed answers.', '',
    ...suite.cases.flatMap(item => [
      `## ${item.id}`, '', `**Question:** ${item.question}`, '',
      `**Status:** ${item.status}. **Time:** ${((item.totalMs ?? 0) / 1000).toFixed(1)}s. **Review:** ${item.reviewStatus}.`, '',
      '**Actual answer:**', '', item.answer?.text || item.error || '(no answer)', '',
      '**Reference answer (not generated):**', '', item.referenceAnswer, '',
      '**Review criteria:**', ...item.criteria.map(criterion => `- ${criterion}`), '',
      `**Retrieved evidence:** ${item.retrievalEvidenceFailures?.length ? item.retrievalEvidenceFailures.join('; ') : item.retrievalEvidenceFailures ? 'All required groups found.' : 'Not evaluated.'}`, '',
      `**Packed evidence:** ${item.packedEvidenceFailures?.length ? item.packedEvidenceFailures.join('; ') : item.packedEvidenceFailures ? 'All required groups preserved.' : 'Not evaluated.'}`, '',
      '**Sources included in the final model request:**',
      ...(item.packedEvidence ?? []).map(source => `- ${source.chunkIds.join(', ')}: ${source.section}`), ''
    ])
  ]
  return lines.join('\n')
}

export async function runBuzzdbReasoningBenchmark() {
  const { prepareRewrittenStudyChat } = requireTranspiledTs('src/shared/study-chat-pipeline.ts')
  const { resolveOllamaChatQuestion, runOllamaStudyEngine } = requireTranspiledTs('src/main/engine/ollama-service.ts')
  const { modelAwareRetrievalLimit } = requireTranspiledTs('src/shared/retrieval-budget.ts')
  const families = JSON.parse(readFileSync(casesPath, 'utf8'))
  const baseUrl = (process.env.TOKENSMITH_BENCHMARK_OLLAMA_URL || 'http://127.0.0.1:11434').replace(/\/$/, '')
  const modelName = 'gemma4:e4b'
  const tags = await fetch(`${baseUrl}/api/tags`, { signal: AbortSignal.timeout(15_000) }).then(response => {
    if (!response.ok) throw new Error(`Ollama model listing: HTTP ${response.status}`)
    return response.json()
  })
  const installed = tags.models.find(model => model.name === modelName)
  if (!installed) throw new Error(`Install ${modelName} before running --live. No downloads are performed by this benchmark.`)
  const model = { id: 'benchmark-gemma', name: modelName, ollamaModelName: modelName,
    engine: 'ollama', source: 'ollama', role: 'generator', status: 'ready', ollamaBaseUrl: baseUrl, contextLength: 8192 }
  const modelSettings = { contextLength: 8192, maxLength: 1536, temperature: 0.7, topP: 0.4,
    topK: 40, minP: 0, repeatPenalty: 1.18, thinking: false, systemMessage: '' }
  const applicationSettings = { suggestionMode: 'off', searchMode: 'hybrid', explanationDepthEnabled: false, embeddingGpuEnabled: true }
  const base = { model, modelSettings, applicationSettings, materials: [{ id: '1', status: 'ready', isActive: true }],
    settings: { maxSources: 4, application: applicationSettings, modelDefaults: modelSettings, modelSettingsById: {} } }
  const limit = modelAwareRetrievalLimit(base.settings.maxSources, model, modelSettings)
  const directory = resolve(process.env.TOKENSMITH_BENCHMARK_LIVE_DIR ?? `tmp/buzzdb-reasoning-${new Date().toISOString().replaceAll(':', '-')}`)
  mkdirSync(dirname(directory), { recursive: true })
  mkdirSync(directory, { recursive: false })
  const files = [casesPath, 'tests/fixtures/buzzdb/buzzdb-book.tokensmith.md',
    'tests/benchmarks/buzzdb_reasoning.mjs', 'tests/benchmarks/buzzdb_live_worker.py',
    'src/main/engine/ollama-service.ts', 'src/main/engine/study-chat-format.ts',
    'src/shared/study-chat-pipeline.ts', 'src/shared/retrieval-budget.ts',
    'python_engine/tokensmith_engine.py', 'python_engine/tokensmith_store.py']
  const suite = { name: 'reasoning_answers', label: 'live reasoning answers (execution only; ungraded)',
    unit: 'completed answers', model: modelName, modelDigest: installed.digest, modelMetadata: installed,
    settings: modelSettings, retrievalLimit: limit, reviewStatus: 'ungraded',
    sourceHashes: Object.fromEntries(files.map(path => [path, createHash('sha256').update(readFileSync(path)).digest('hex')])),
    createdAt: new Date().toISOString(), outputDirectory: directory, passed: 0,
    total: families.reduce((count, family) => count + family.turns.length, 0), cases: [] }
  const save = () => {
    writeFileSync(`${directory}/results.json`, `${JSON.stringify(suite, null, 2)}\n`)
    writeFileSync(`${directory}/answers.md`, formatReasoningAnswers(suite))
  }
  save()
  const worker = startRetrievalWorker()
  const originalFetch = globalThis.fetch
  let calls = []
  let stage
  globalThis.fetch = async (...args) => {
    const started = performance.now()
    const request = args[1]?.body ? JSON.parse(args[1].body) : undefined
    const call = { stage, url: String(args[0]), request }
    calls.push(call)
    try {
      const response = await originalFetch(...args)
      call.response = await response.clone().json()
      call.httpStatus = response.status
      return response
    } catch (error) { call.error = error.message; throw error }
    finally { call.ms = performance.now() - started }
  }
  const evidence = sources => sources.map(source => ({
    chunkIds: source.sourceChunkIds ?? [source.tokensmithChunkId ?? source.chunkId],
    section: source.sectionHeader ?? '', context: source.context || source.excerpt
  }))
  try {
    suite.embeddingCache = (await worker.ready).cache
    await runReasoningTurns(families, async input => {
      calls = []
      const started = performance.now()
      const detail = {}
      try {
        stage = 'rewrite'
        const prepared = await prepareRewrittenStudyChat({ ...base, ...input }, {
          resolve: resolveOllamaChatQuestion,
          search: async query => {
            stage = 'retrieval'
            const result = await worker.call({ command: 'search', query, limit })
            detail.retrieval = result
            return result.sources
          }
        })
        Object.assign(detail, prepared)
        if (!prepared.request) throw new Error(`Clarification instead of answer: ${prepared.resolution.clarification}`)
        stage = 'generation'
        const generationStart = performance.now()
        try { detail.answer = await runOllamaStudyEngine(prepared.request) }
        finally { detail.generationMs = performance.now() - generationStart }
      } catch (error) { detail.error = error.message }
      return { ...detail, calls, totalMs: performance.now() - started }
    }, async (row, results) => {
      const sources = row.retrieval?.sources ?? []
      row.evidenceLabel = 'Retrieved evidence (real hybrid search)'
      row.evidence = evidence(sources)
      row.packedEvidence = evidence(packedEvidenceFromCalls(row.calls ?? [], sources))
      if (row.retrieval) {
        const check = items => worker.call({ command: 'validate', case: { evidenceGroups: row.evidenceGroups },
          chunkIds: items.flatMap(source => source.chunkIds), context: items.map(source => source.context).join('\n') })
        row.retrievalEvidenceFailures = await check(row.evidence)
        row.packedEvidenceFailures = await check(row.packedEvidence)
      }
      suite.cases = results
      suite.passed = results.filter(item => item.passed).length
      save()
      console.log(`${row.id}: ${row.status}, ${((row.totalMs ?? 0) / 1000).toFixed(1)}s; answer ungraded`)
    })
    return suite
  } finally {
    globalThis.fetch = originalFetch
    await worker.close()
    save()
    console.log(`Live answer report: ${directory}/answers.md`)
  }
}
