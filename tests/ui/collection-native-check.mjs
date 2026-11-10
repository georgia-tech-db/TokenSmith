import assert from 'node:assert/strict'
import { mkdtemp, mkdir, readFile, writeFile, rm, realpath } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'

// Built Electron app and real preload/IPC, with disposable app data and indexes.
// No document preparation, inference, or access to the user's Library/history.
const { _electron } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright')
const root = await realpath(await mkdtemp(join(tmpdir(), 'tokensmith-native-collections-')))
const userData = join(root, 'app')
await mkdir(userData)
const savedModel = { id: 'fixture', name: 'Fixture model', engine: 'ollama', role: 'generator', status: 'needsRuntime', ollamaModelName: 'fixture-only' }
await writeFile(join(userData, 'tokensmith-state.json'), JSON.stringify({
  version: 1, appVersion: '0.1.15', activeScreen: 'library', activeConversationId: 'legacy',
  conversations: [{ id: 'legacy', title: 'Legacy chat', messages: [{ id: 'legacy-message', role: 'user', text: 'Old question' }] }],
  materials: [{ id: 'legacy', title: 'Legacy library', path: '/legacy', status: 'ready' }],
  models: [savedModel], selectedEmbeddingModelId: '',
  settings: { application: { fontSize: 'large', suggestionMode: 'off', theme: 'light' }, modelSettingsById: {} }
}))
const bootstrap = join(root, 'bootstrap.cjs')
await writeFile(bootstrap, `const { app } = require('electron');
app.setAppPath(${JSON.stringify(resolve('.'))});
app.setPath('userData', ${JSON.stringify(userData)});
require(${JSON.stringify(resolve('out/main/index.js'))});
`)
let electron
const launch = async () => {
  electron = await _electron.launch({ executablePath: process.env.ELECTRON_PATH || resolve('node_modules/electron/dist/Electron.app/Contents/MacOS/Electron'), args: [bootstrap] })
  assert.equal(await electron.evaluate(({ app }) => app.getPath('userData')), userData)
  const page = await electron.firstWindow()
  await page.waitForFunction(() => window.tokensmith?.listCollections)
  return page
}
try {
  let page = await launch()
  const state = await page.evaluate(() => window.tokensmith.loadAppState())
  assert.equal(state.collectionFormatVersion, 1)
  assert.equal(state.materials.length, 0)
  assert.ok(state.conversations.every(c => c.id !== 'legacy'))
  assert.equal(state.settings.application.fontSize, 'large')
  assert.ok(state.models.some(m => m.id === savedModel.id && m.ollamaModelName === savedModel.ollamaModelName))
  const a = join(root, 'Databases'), b = join(root, 'Networks')
  await mkdir(a); await mkdir(b)
  const create = async (path, name) => page.evaluate(async ({ path, name }) => (await window.tokensmith.createCollection(path, name))[0], { path, name })
  const one = await page.evaluate(async () => (await window.tokensmith.createLibrary('Databases'))[0]), two = await create(b, 'Networks')
  assert.ok(one.path.startsWith(join(userData, 'libraries') + '/'))
  const before = await readFile(join(one.path, 'tokensmith.sqlite'))
  await page.evaluate(id => window.tokensmith.unloadCollection(id), one.id)
  assert.deepEqual(await readFile(join(one.path, 'tokensmith.sqlite')), before)
  let collections = await page.evaluate(() => window.tokensmith.listCollections())
  assert.deepEqual(collections.filter(c => c.loaded).map(c => c.id), [two.id])
  await electron.close(); electron = undefined
  page = await launch()
  collections = await page.evaluate(() => window.tokensmith.listCollections())
  assert.deepEqual(collections.filter(c => c.loaded).map(c => c.id), [two.id])
  assert.equal(collections.find(c => c.id === one.id).loaded, false)
  await page.evaluate(path => window.tokensmith.loadCollection(path), one.path)
  assert.equal((await page.evaluate(() => window.tokensmith.listCollections())).filter(c => c.loaded).length, 2)
  assert.deepEqual(await page.evaluate(() => window.tokensmith.listMaterials()), [])
  assert.deepEqual(await page.evaluate(() => window.tokensmith.studyDocuments()), [])
  console.log('Native Electron collections passed: real preload/IPC, legacy study-data reset with settings/models retained, creation, unload with byte-preserved index, restart state, and reload.')
} finally {
  await electron?.close()
  await rm(root, { recursive: true, force: true })
}
