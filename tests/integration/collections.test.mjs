import assert from 'node:assert/strict'
import Module from 'node:module'
import { EventEmitter, once } from 'node:events'
import { createServer } from 'node:http'
import { mkdtemp, mkdir, realpath, readFile, writeFile, rm, rename } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import { requireTranspiledTs } from '../unit/typescript/ts-module-loader.mjs'

const root = await realpath(await mkdtemp(join(tmpdir(), 'tokensmith-collection-integration-')))
const app = new EventEmitter()
Object.assign(app, { isPackaged: false, getAppPath: () => resolve('.'), getPath: () => join(root, 'app') })
const originalLoad = Module._load
Module._load = function (name, ...args) {
  if (name === 'electron') return { app, BrowserWindow: { getAllWindows: () => [] } }
  return originalLoad.call(this, name, ...args)
}
const service = requireTranspiledTs('src/main/python/python-engine-service.ts')
Module._load = originalLoad
const registry = service.collections()
let embeddings = 0
let held, release
const server = createServer(async (request, response) => {
  let text = ''; for await (const chunk of request) text += chunk
  const body = JSON.parse(text)
  assert.equal(request.url, '/v1/embeddings')
  embeddings++
  if (String(body.input).includes('hold-request')) { held?.(); await new Promise(resolve => { release = resolve }) }
  response.writeHead(200, { 'Content-Type': 'application/json' })
  response.end(JSON.stringify({ data: [{ embedding: [1, 0.5, 0.2, 0.1] }] }))
})
server.listen(0, '127.0.0.1'); await once(server, 'listening')
const model = { id: 'fixture-embedding', engine: 'remote', role: 'embedder', status: 'ready', name: 'Fixture embeddings',
  remoteModelName: 'fixture', providerId: 'fixture', baseUrl: `http://127.0.0.1:${server.address().port}/v1`, apiKey: 'fixture-only' }
const create = async (name, text) => {
  const path = join(root, name); await mkdir(path)
  const collection = (await registry.create(path, name))[0]
  const originals = join(path, 'documents'); await mkdir(originals)
  await writeFile(join(originals, 'notes.md'), `# ${name}\n\n${text.repeat(20)}`)
  const material = await service.indexMaterialWithPython(originals, model, `${collection.id}/new`, { preparation: { mode: 'basic', instructions: '', documentInstructions: {} } })
  return { collection, material, path, originals }
}
const refs = docs => docs.map(({ materialId, documentId }) => ({ materialId, documentId }))
const search = (query, materials, docs, mode = 'hybrid') => service.searchLibraryWithPython(query, materials, 4, [model], mode, { documents: refs(docs) })
try {
  await registry.initialize()
  const a = await create('Databases', 'Alpha B+ trees use ancestor paths to propagate splits. ')
  const b = await create('Networks', 'Beta networks use packets and routers to forward messages. ')
  let docs = await service.studyDocumentsWithPython()
  const left = docs.filter(d => d.collectionId === a.collection.id), right = docs.filter(d => d.collectionId === b.collection.id)
  assert.equal(left[0].documentId, right[0].documentId, 'Deliberately overlapping database-local document IDs')
  assert.notEqual(left[0].materialId, right[0].materialId)
  const both = await search('ancestor packets', [a.material, b.material], docs)
  assert.deepEqual(new Set(both.sources.map(s => s.materialId)), new Set([a.material.id, b.material.id]))
  assert.ok(both.retrievalChildren.some(span => span.stage === 'Query encoding'))
  assert.ok(both.retrievalChildren.every(span => Number.isFinite(span.durationMs) && span.durationMs >= 0))
  assert.deepEqual(new Set((await search('ancestor', [a.material], left)).sources.map(s => s.materialId)), new Set([a.material.id]))
  const source = both.sources.find(s => s.materialId === a.material.id)
  const resolved = await service.resolveSourceDocumentWithPython(source)
  assert.ok(resolved.path.endsWith('Databases/documents/notes.md'))
  const first = await service.practiceSourcesWithPython(refs(left), [], 0)
  assert.equal(first[0].materialId, a.material.id)
  const key = [first[0].materialId, first[0].documentId, first[0].sourceUnitId || first[0].chunkId || first[0].chunkRowid || first[0].excerpt].join(':')
  const nextSources = await service.practiceSourcesWithPython(refs(left), [key], 0)
  assert.ok(nextSources.every(s => [s.materialId, s.documentId, s.sourceUnitId || s.chunkId || s.chunkRowid || s.excerpt].join(':') !== key))
  await registry.saveTopics(refs(left), [{ title: 'B+ trees' }], registry.get(a.collection.id).indexRevision)
  assert.deepEqual(await registry.topics(refs(left)), [{ title: 'B+ trees' }])
  await writeFile(join(a.path, 'graph.bin'), 'optional graph remains intact')
  const databaseBefore = await readFile(join(a.path, 'tokensmith.sqlite'))
  const callsBefore = embeddings
  await registry.unload(a.collection.id)
  await assert.rejects(search('ancestor', [a.material], left), /closed/)
  await assert.rejects(service.resolveSourceDocumentWithPython(source), /closed/)
  assert.ok((await search('packets', [b.material], right, 'keyword')).sources.length)
  assert.equal((await service.studyDocumentsWithPython()).length, 1)
  await registry.load(a.path)
  assert.equal(embeddings, callsBefore, 'Loading/unloading never embeds or generates')
  assert.deepEqual(await readFile(join(a.path, 'tokensmith.sqlite')), databaseBefore)
  assert.equal(await readFile(join(a.path, 'graph.bin'), 'utf8'), 'optional graph remains intact')
  for (let i = 0; i < 5; i++) { await registry.unload(a.collection.id); await registry.load(a.path) }
  assert.equal((await search('ancestor', [a.material], left, 'keyword')).sources[0].materialId, a.material.id)

  // A real blocked embedding request is terminated only for its collection.
  const started = new Promise(resolve => { held = resolve })
  const pending = search('hold-request ancestor', [a.material], left).then(() => assert.fail('A cancelled search returned'), error => error)
  await started
  await registry.unload(a.collection.id)
  await registry.load(a.path)
  assert.match((await pending).message, /closed/)
  release(); held = undefined
  assert.ok((await search('packets', [b.material], right, 'keyword')).sources.length)
  assert.ok((await search('ancestor', [a.material], left, 'keyword')).sources.length)

  await registry.unload(a.collection.id)
  const movedCollection = join(root, 'Moved collection')
  await rename(a.path, movedCollection)
  await registry.load(movedCollection)
  a.path = movedCollection; a.originals = join(movedCollection, 'documents')
  assert.equal((await service.resolveSourceDocumentWithPython(source)).path, join(a.originals, 'notes.md'))
  const report = await service.preparationReportWithPython(a.originals, join(a.originals, 'notes.md'), a.material.id)
  assert.ok(report.documents[0].path.startsWith(a.originals))
  assert.ok(report.documents[0].chunks.length)

  // Missing original files do not prevent indexed retrieval. Locate files restores viewing.
  const moved = join(root, 'relocated-originals'); await rename(a.originals, moved)
  assert.ok((await search('ancestor', [a.material], left, 'keyword')).sources.length)
  await service.locateCollectionFolder(a.material.id, moved)
  assert.equal((await service.resolveSourceDocumentWithPython(source)).path, join(moved, 'notes.md'))

  // Unload while a preparation worker is waiting for inference.
  await writeFile(join(moved, 'new.md'), '# New\n\n' + 'hold-request new source evidence. '.repeat(20))
  const indexingStarted = new Promise(resolve => { held = resolve })
  const indexing = service.indexMaterialWithPython(moved, model, a.material.id, { preparation: { mode: 'basic', instructions: '', documentInstructions: {} } })
    .then(() => assert.fail('Cancelled indexing completed'), error => error)
  await indexingStarted
  await registry.unload(a.collection.id, () => service.cancelCollectionIndexing(a.collection.id))
  assert.match((await indexing).message, /cancelled/)
  release(); held = undefined
  assert.ok((await search('packets', [b.material], right, 'keyword')).sources.length)
  await registry.load(a.path)
  assert.deepEqual(await registry.topics(refs(left)), [], 'Index updates invalidate derived topics')
  await registry.unload(a.collection.id)
  await registry.unload(b.collection.id)

  // Independent files in the same directory must remain independent documents.
  const library = (await registry.createManaged('Sibling documents'))[0]
  const siblings = join(root, 'same-directory'); await mkdir(siblings)
  const firstPath = join(siblings, 'trees.md'), secondPath = join(siblings, 'volcano.txt')
  await writeFile(firstPath, '# Trees\n\n' + 'Leaf pages are linked for range scans. '.repeat(20))
  await writeFile(secondPath, 'Volcano iterators implement open, next, and close. '.repeat(20))
  const preparation = { mode: 'basic', instructions: '', documentInstructions: {} }
  const firstDocument = await service.indexMaterialWithPython(firstPath, model, `${library.id}/new-first`, { preparation })
  const secondDocument = await service.indexMaterialWithPython(secondPath, model, `${library.id}/new-second`, { preparation })
  let siblingsCatalog = await service.studyDocumentsWithPython()
  assert.equal(siblingsCatalog.length, 2, 'Each file appears exactly once')
  assert.deepEqual(siblingsCatalog.filter(d => d.materialId === firstDocument.id).map(d => d.path), [firstPath])
  assert.deepEqual(siblingsCatalog.filter(d => d.materialId === secondDocument.id).map(d => d.path), [secondPath])
  await service.indexMaterialWithPython(firstPath, model, firstDocument.id, { preparation })
  assert.equal((await service.studyDocumentsWithPython()).length, 2, 'Updating one file does not duplicate or hide its sibling')
  await service.removeMaterialWithPython(firstDocument.id, firstPath)
  siblingsCatalog = await service.studyDocumentsWithPython()
  assert.deepEqual(siblingsCatalog.map(d => d.path), [secondPath])
  const remaining = await search('Volcano next iterator', [secondDocument], siblingsCatalog)
  assert.ok(remaining.sources.length)
  assert.ok(remaining.sources.every(s => s.materialId === secondDocument.id && s.path === secondPath))
  assert.ok((await readFile(firstPath, 'utf8')).includes('Leaf pages'), 'Removal preserves original files')
  const callsAfterPrepare = embeddings
  await registry.unload(library.id); await registry.load(library.path)
  assert.equal(embeddings, callsAfterPrepare, 'Reopening a managed library requires no inference')
  assert.equal((await service.studyDocumentsWithPython())[0].materialId, secondDocument.id)
  await registry.unload(library.id)
  assert.deepEqual(await service.listIndexedMaterialsWithPython(), [])
  assert.deepEqual(await service.studyDocumentsWithPython(), [])
  console.log('Collections integration passed: real SQLite/FTS/vector indexes, overlapping IDs, scoped and combined retrieval, citations, Practice sources, saved topics, no-call loading, repeated reload, isolated search/index cancellation, original-file relocation, and unload-all.')
} finally {
  release?.()
  app.emit('before-quit')
  server.closeAllConnections(); server.close()
  await rm(root, { recursive: true, force: true })
}
