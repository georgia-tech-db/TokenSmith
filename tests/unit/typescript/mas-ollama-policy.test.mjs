import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import vm from 'node:vm'
import { createRequire } from 'node:module'
import ts from 'typescript'
import { requireTranspiledTs } from './ts-module-loader.mjs'
const require = createRequire(import.meta.url)

test('MAS rejects Ollama mutations before side effects, but can discover served models', async () => {
  const exports = {}, calls = []
  const forbidden = () => { throw Error('Unexpected OS access') }
  const code = ts.transpileModule(readFileSync('src/main/engine/ollama-service.ts', 'utf8'), {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 }
  }).outputText
  vm.runInNewContext(code, { exports, process: { mas: true }, console, Buffer, setTimeout, clearTimeout, AbortController,
    fetch: async url => { calls.push(url); return { ok: true, json: async () => ({ models: [{ name: 'test-chat:latest' }] }) } },
    require(name) {
      if (name === '../distribution') return { isMacAppStore: () => true, assertOllamaManagementAllowed: () => { throw Error('Manage Ollama outside TokenSmith') } }
      if (name === 'electron') return { app: { getPath: forbidden }, shell: { openPath: forbidden, openExternal: forbidden }, BrowserWindow: { getAllWindows: () => [] } }
      if (name === 'node:child_process') return { spawn: forbidden }
      if (name === 'node:fs') return { existsSync: forbidden }
      if (name === '../../shared/ollama') return requireTranspiledTs('src/shared/ollama.ts')
      if (name.startsWith('.')) return {}
      return require(name)
    }
  })
  for (const name of ['openOllamaDownloadPage', 'openOllamaApp', 'startOllamaService', 'pullOllamaModel', 'cancelOllamaPullModel', 'deleteOllamaModel']) {
    await assert.rejects(async () => exports[name]('test-chat'), /outside TokenSmith/, name)
  }
  assert.equal(calls.length, 0)
  const status = await exports.getOllamaStatus()
  assert.equal(status.running, true); assert.equal(status.installedApp, false)
  assert.equal(status.models[0].name, 'test-chat:latest'); assert.equal(calls.length, 1)
})
