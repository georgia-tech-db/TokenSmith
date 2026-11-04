import assert from 'node:assert/strict'
import test from 'node:test'
import vm from 'node:vm'
import { EventEmitter } from 'node:events'
import { PassThrough } from 'node:stream'
import { createRequire } from 'node:module'
import { readFileSync, existsSync } from 'node:fs'
import { mkdtemp, readFile, rm } from 'node:fs/promises'
import { join, dirname, resolve } from 'node:path'
import { tmpdir } from 'node:os'
import ts from 'typescript'
import { requireTranspiledTs } from './ts-module-loader.mjs'

const require = createRequire(import.meta.url)
const { setCloudCredentialResolver } = requireTranspiledTs('src/main/engine/remote-model-secrets.ts')

test('indexing and search hydrate cloud models at the worker boundary, without logging secrets', async t => {
  const folder = await mkdtemp(join(tmpdir(), 'tokensmith-credential-bridge-'))
  t.after(async () => { setCloudCredentialResolver(() => undefined); await rm(folder, { recursive: true, force: true }) })
  const requests = []
  const spawn = () => {
    const child = new EventEmitter()
    child.stdout = new PassThrough()
    child.stderr = new PassThrough()
    const send = line => {
      const request = JSON.parse(line)
      requests.push(request)
      const result = request.command === 'index_material' ? { material: { id: 'book', status: 'ready' } } : { sources: [] }
      queueMicrotask(() => child.stdout.write(JSON.stringify({ id: request.id, ok: true, result }) + '\n'))
    }
    child.stdin = { write: send, end: send }
    return child
  }
  const app = Object.assign(new EventEmitter(), { isPackaged: false, getAppPath: () => process.cwd(), getPath: () => folder })
  const path = 'src/main/python/python-engine-service.ts'
  const exports = {}
  const code = ts.transpileModule(readFileSync(path, 'utf8'), {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 }
  }).outputText
  vm.runInNewContext(code, { exports, process, console, Buffer, setTimeout, clearTimeout, require(name) {
    if (name === 'electron') return { app, BrowserWindow: { getAllWindows: () => [] } }
    if (name === 'node:child_process') return { spawn }
    if (name === 'node:fs') return { ...require(name), existsSync: path => String(path).includes('app_runtime') || existsSync(path) }
    if (name.startsWith('.')) return requireTranspiledTs(resolve(dirname(path), `${name}.ts`))
    return require(name)
  } })
  const model = { id: 'cloud-embedder', engine: 'remote', role: 'embedder', connectionId: 'shared',
    providerId: 'custom', baseUrl: 'https://test.example/v1', remoteModelName: 'embed-test' }
  let key = 'test-key-only'
  setCloudCredentialResolver(model => model.connectionId === 'shared' ? key : undefined)
  await exports.indexMaterialWithPython('/test-book', model, 'book', { preparationModel: { ...model, role: 'generator' } })
  assert.equal(requests[0].payload.model.apiKey, key)
  assert.equal(requests[0].payload.preparationModel.apiKey, key)
  await exports.searchLibraryWithPython('Why?', [], 4, [model], 'hybrid')
  assert.equal(requests[1].payload.embeddingModels[0].apiKey, key)
  assert.equal(model.apiKey, undefined)
  key = 'replacement-test-key'
  await exports.searchLibraryWithPython('Again?', [], 4, [model], 'hybrid')
  assert.equal(requests[2].payload.embeddingModels[0].apiKey, key)
  key = undefined
  await exports.searchLibraryWithPython('Forgotten?', [], 4, [model], 'hybrid')
  assert.equal(requests[3].payload.embeddingModels[0].apiKey, undefined)
  const log = await readFile(join(folder, 'logs', 'tokensmith.log'), 'utf8')
  assert.equal(log.includes('test-key-only'), false)
  assert.equal(log.includes('replacement-test-key'), false)
})
