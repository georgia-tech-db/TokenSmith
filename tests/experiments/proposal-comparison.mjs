// Frozen evidence comparison through the production Ollama answer path.
// Run with Electron, or use Node with --dry-run to inspect requests without inference.
import { readFileSync, writeFileSync, mkdirSync, readdirSync, mkdtempSync } from 'node:fs'
import { tmpdir, platform, arch, totalmem, cpus } from 'node:os'
import { join, resolve } from 'node:path'
import { createHash } from 'node:crypto'
import { parseArgs } from 'node:util'
import { requireTranspiledTs } from '../unit/typescript/ts-module-loader.mjs'
const { values } = parseArgs({ options: { input: { type: 'string' }, out: { type: 'string' }, 'dry-run': { type: 'boolean' } } })
if (!values.input || !values.out) throw new Error('--input manifest.json --out NEW_DIRECTORY are required')
const raw = readFileSync(values.input, 'utf8'), manifest = JSON.parse(raw)
if (manifest.version !== 1 || !Array.isArray(manifest.cases) || !manifest.cases.length) throw new Error('Expected version 1 cases.')
const out = resolve(values.out); mkdirSync(out, { recursive: true })
if (readdirSync(out).length) throw new Error('Choose an empty output directory.')
const requests = []
const ids = new Set()
for (const [i, c] of manifest.cases.entries()) {
  if (!c.id || typeof c.prompt !== 'string' || c.model?.engine !== 'ollama' || !c.modelSettings) throw new Error('Each case requires id, prompt, Ollama model, settings and evidence.')
  if (ids.has(c.id)) throw new Error('Duplicate case ID.')
  ids.add(c.id)
  const conditions = ['baseline', 'expanded', 'reviewed'].flatMap(evidence => ['off', 'on'].map(reasoning => ({ evidence, reasoning })))
  for (const condition of i % 2 ? conditions.reverse() : conditions) {
    if (!Array.isArray(c.evidence?.[condition.evidence]) || !c.evidence[condition.evidence].length) throw new Error('Every case requires baseline, expanded, and reviewed source arrays.')
    requests.push({ id: c.id, condition, request: { prompt: c.prompt, messages: [], materials: [], settings: {},
      applicationSettings: { suggestionMode: 'off' }, model: c.model,
      modelSettings: { ...c.modelSettings, reasoningMode: condition.reasoning, thinking: condition.reasoning === 'on' },
      retrievedSources: c.evidence[condition.evidence] } })
  }
}
writeFileSync(join(out, 'manifest.json'), JSON.stringify({ input: manifest, inputSha256: createHash('sha256').update(raw).digest('hex'),
  method: 'Fixed supplied evidence; production source packing and Ollama answer generation. Reviewed evidence is a diagnostic ceiling. Excludes online retrieval and preparation; those costs must be measured separately.',
  memoryBytes: null, preparationSeconds: null, retrievalSeconds: null, dryRun: Boolean(values['dry-run']) }, null, 2))
if (values['dry-run']) {
  writeFileSync(join(out, 'requests.json'), JSON.stringify(requests, null, 2))
} else {
  const { app } = await import('electron')
  app.setPath('userData', mkdtempSync(join(tmpdir(), 'tokensmith-proposal-')))
  // Let the entry module finish: top-level awaiting readiness prevents Electron from becoming ready.
  void app.whenReady().then(async () => {
    try {
      const runtime = { platform: platform(), arch: arch(), totalMemoryBytes: totalmem(), cpu: cpus()[0]?.model,
        codeHashes: Object.fromEntries(['src/main/engine/ollama-service.ts','src/main/engine/study-chat-format.ts', 'tests/experiments/proposal-comparison.mjs']
          .map(path => [path, createHash('sha256').update(readFileSync(path)).digest('hex')])), servers: [], logDirectory: app.getPath('userData') }
      for (const base of new Set(requests.map(r => (r.request.model.ollamaBaseUrl || 'http://127.0.0.1:11434').replace(/\/$/, '')))) {
        try {
          const tags = await fetch(`${base}/api/tags`, { signal: AbortSignal.timeout(5000) }).then(r => r.json())
          const residency = await fetch(`${base}/api/ps`, { signal: AbortSignal.timeout(5000) }).then(r => r.json())
          runtime.servers.push({ base, models: tags.models?.filter(m => requests.some(r => r.request.model.ollamaModelName === m.name || `${r.request.model.ollamaModelName}:latest` === m.name)), residency })
        } catch (e) { runtime.servers.push({ base, error: String(e) }) }
      }
      writeFileSync(join(out, 'runtime.json'), JSON.stringify(runtime, null, 2))
      const { runOllamaStudyEngine } = requireTranspiledTs('src/main/engine/ollama-service.ts')
      const results = []
      for (const row of requests) {
        const start = performance.now()
        try { results.push({ ...row, result: await runOllamaStudyEngine(row.request), seconds: (performance.now()-start)/1000 }) }
        catch (e) { results.push({ ...row, error: String(e), seconds: (performance.now()-start)/1000 }) }
        writeFileSync(join(out, 'results.json'), JSON.stringify(results, null, 2))
      }
    } finally { app.quit() }
  }).catch(error => { console.error(error); app.exit(1) })
}
