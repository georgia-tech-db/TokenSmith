import assert from 'node:assert/strict'
import { mkdtemp, mkdir, realpath, readFile, writeFile, rename, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import test from 'node:test'
import { requireTranspiledTs } from './ts-module-loader.mjs'
const { CollectionRegistry } = requireTranspiledTs('src/main/collections/collection-registry.ts')
const { localDocuments, qualifySource, mergeCollectionSources } = requireTranspiledTs('src/main/collections/collection-routing.ts')
const { collectionMaterialId } = requireTranspiledTs('src/shared/collections.ts')

async function fixture(t) {
  const root = await realpath(await mkdtemp(join(tmpdir(), 'tokensmith-collections-')))
  t.after(() => rm(root, { recursive: true, force: true }))
  const closed = [], validations = []
  const access = { create: async path => writeFile(join(path, 'tokensmith.sqlite'), 'test database'),
    validate: async path => { validations.push(path); await readFile(join(path, 'tokensmith.sqlite')) }, close: path => closed.push(path) }
  const registryPath = join(root, 'app', 'collections.json')
  const registry = new CollectionRegistry(registryPath, access)
  const a = join(root, 'A'), b = join(root, 'B')
  await mkdir(a); await mkdir(b)
  const one = (await registry.create(a, 'Databases'))[0]
  const two = (await registry.create(b, 'Networks'))[0]
  return { root, a, b, one, two, registry, registryPath, closed, validations, access }
}

test('unload and reload preserve files, independent collections, and qualified source identity', async t => {
  const f = await fixture(t)
  const mounted = f.registry.get(f.one.id)
  const source = { materialId: '1', documentId: 1, chunkId: '1', title: 'Passage', excerpt: 'Evidence', path: 'documents/book.md' }
  const left = qualifySource(f.registry, mounted, source)
  const right = qualifySource(f.registry, f.registry.get(f.two.id), source)
  assert.notEqual(left.materialId, right.materialId)
  assert.notEqual(left.path, right.path)
  assert.deepEqual(localDocuments(mounted, [{ materialId: left.materialId, documentId: 1 }]), [{ materialId: '1', documentId: 1 }])
  assert.throws(() => localDocuments(mounted, [{ materialId: right.materialId, documentId: 1 }]), /different collection/)
  await writeFile(join(f.a, 'graph.json'), '{"untouched":true}')
  const before = await readFile(join(f.a, 'tokensmith.sqlite'))
  await f.registry.unload(f.one.id)
  assert.throws(() => f.registry.get(f.one.id), /closed/)
  assert.equal(f.registry.get(f.two.id).id, f.two.id)
  assert.deepEqual(await readFile(join(f.a, 'tokensmith.sqlite')), before)
  assert.equal(await readFile(join(f.a, 'graph.json'), 'utf8'), '{"untouched":true}')
  assert.deepEqual(f.closed, [f.a])
  await f.registry.load(f.a)
  assert.throws(() => f.registry.current(mounted), /closed/, 'A stale request must not revive after reloading the same collection')
  assert.deepEqual(qualifySource(f.registry, f.registry.get(f.one.id), source), left)
  assert.deepEqual(mergeCollectionSources([[left], [right]], 2), [left, right])
})

test('restart restores loaded collections only and moved folders retain their identity', async t => {
  const f = await fixture(t)
  await f.registry.unload(f.one.id)
  const restarted = new CollectionRegistry(f.registryPath, f.access)
  await restarted.initialize()
  assert.deepEqual(restarted.loaded().map(c => c.id), [f.two.id])
  const moved = join(f.root, 'Moved A'); await rename(f.a, moved)
  await restarted.load(moved)
  assert.equal(restarted.get(f.one.id).path, moved)
  assert.equal(restarted.list().filter(c => c.id === f.one.id).length, 1)
  await rename(f.b, join(f.root, 'Missing B'))
  const missing = new CollectionRegistry(f.registryPath, f.access)
  await missing.initialize()
  assert.equal(missing.get(f.one.id).path, moved)
  assert.equal(missing.list().find(c => c.id === f.two.id).loaded, false)
  assert.ok(missing.list().find(c => c.id === f.two.id).error)
})

test('invalid loads, duplicate copies, and nonempty creates leave current mounts unchanged', async t => {
  const f = await fixture(t)
  const bad = join(f.root, 'bad'); await mkdir(bad)
  await assert.rejects(f.registry.load(bad))
  await writeFile(join(bad, 'collection.json'), JSON.stringify({ ...f.one, version: 500 }))
  await assert.rejects(f.registry.load(bad), /saved TokenSmith library/)
  await writeFile(join(bad, 'collection.json'), await readFile(join(f.a, 'collection.json')))
  await writeFile(join(bad, 'tokensmith.sqlite'), 'test')
  await assert.rejects(f.registry.load(bad), /already open/)
  await assert.rejects(f.registry.create(f.a, 'Overwrite'), /empty folder/)
  assert.equal(f.registry.loaded().length, 2)
  assert.throws(() => f.registry.sourcePath(f.registry.get(f.one.id), '../outside.md'), /leaves/)
})


test('unload waits for cancellation and worker exit before a queued reload', async t => {
  const f = await fixture(t)
  let releaseWorker, releaseCancellation
  const workerClosed = new Promise(resolve => { releaseWorker = resolve })
  const cancelled = new Promise(resolve => { releaseCancellation = resolve })
  const lifecycle = []
  const registry = new CollectionRegistry(f.registryPath, { ...f.access,
    validate: async path => { lifecycle.push('validate'); await f.access.validate(path) },
    close: async () => { lifecycle.push('closing'); await workerClosed; lifecycle.push('closed') } })
  await registry.initialize()
  lifecycle.length = 0
  const unloading = registry.unload(f.one.id, async () => { lifecycle.push('cancelling'); await cancelled })
  const loading = registry.load(f.a)
  while (!lifecycle.length) await new Promise(resolve => setTimeout(resolve, 1))
  assert.throws(() => registry.get(f.one.id), /closed/)
  assert.deepEqual(lifecycle, ['cancelling'])
  releaseCancellation()
  while (!lifecycle.includes('closing')) await new Promise(resolve => setTimeout(resolve, 1))
  assert.deepEqual(lifecycle, ['cancelling', 'closing'])
  releaseWorker()
  await unloading; await loading
  assert.deepEqual(lifecycle, ['cancelling', 'closing', 'closed', 'validate'])
})

test('locating source folders changes only the local registry and persists across restart', async t => {
  const f = await fixture(t)
  const target = join(f.root, 'originals'); await mkdir(target)
  const manifestBefore = await readFile(join(f.a, 'collection.json'))
  const id = collectionMaterialId(f.one.id, '1')
  await f.registry.locateFolder(id, '/old/course', target)
  assert.equal(f.registry.sourcePath(f.registry.get(f.one.id), '/old/course/sub/book.pdf', '1'), join(target, 'sub/book.pdf'))
  assert.equal(f.registry.sourcePath(f.registry.get(f.two.id), '/old/course/sub/book.pdf', '1'), '/old/course/sub/book.pdf')
  assert.deepEqual(await readFile(join(f.a, 'collection.json')), manifestBefore)
  const restarted = new CollectionRegistry(f.registryPath, f.access); await restarted.initialize()
  assert.equal(restarted.sourcePath(restarted.get(f.one.id), '/old/course/book.md', '1'), join(target, 'book.md'))
})

test('managed libraries allocate separate storage and close/reopen without touching documents', async t => {
  const f = await fixture(t)
  const library = (await f.registry.createManaged('  BuzzDBBook  '))[0]
  assert.equal(library.name, 'BuzzDBBook')
  assert.ok(library.path.startsWith(join(f.root, 'app', 'libraries') + '/'))
  await assert.rejects(f.registry.createManaged('  buzzdbbook  '), /already exists/)
  const duplicate = (await f.registry.createManaged('BuzzDBBook (2)'))[0]
  assert.notEqual(duplicate.id, library.id)
  assert.notEqual(duplicate.path, library.path)
  const before = await readFile(join(library.path, 'tokensmith.sqlite'))
  await f.registry.unload(library.id)
  await assert.rejects(f.registry.createManaged('BuzzDBBook'), /already exists/)
  await f.registry.load(library.path)
  assert.deepEqual(await readFile(join(library.path, 'tokensmith.sqlite')), before)
  await assert.rejects(f.registry.createManaged(' '), /library name/)
  const { readdir } = await import('node:fs/promises')
  const namesBefore = await readdir(join(f.root, 'app', 'libraries'))
  const broken = new CollectionRegistry(f.registryPath, { ...f.access, create: async path => {
    await writeFile(join(path, 'partial.sqlite'), 'partial'); throw new Error('Disk unavailable')
  } })
  await assert.rejects(broken.createManaged('Failed creation'), /Disk unavailable/)
  assert.deepEqual(await readdir(join(f.root, 'app', 'libraries')), namesBefore)
})

test('topic inventory checkpoints survive reopening and reject stale writes without touching legacy topics', async t => {
  const f = await fixture(t)
  const revision = f.registry.get(f.one.id).indexRevision
  const value = { version: 3, modelId: 'gemma4:26b', collectionId: f.one.id, indexRevision: revision, batches: {}, topics: [], complete: false, completed: 0, total: 3 }
  await f.registry.saveTopicInventory(value,revision)
  assert.deepEqual(await f.registry.topicInventory(f.one.id),value)
  await f.registry.revise(f.registry.get(f.one.id))
  await assert.rejects(f.registry.saveTopicInventory(value,revision),/changed/)
  assert.deepEqual(await f.registry.topicInventory(f.one.id),value,'Old checkpoints remain reusable by content fingerprint')
  await f.registry.unload(f.one.id)
  await f.registry.load(f.a)
  assert.deepEqual(await f.registry.topicInventory(f.one.id),value)
})

test('delete library removes loaded or closed entries, invalidates requests, and preserves files', async t => {
  const f = await fixture(t)
  const mounted = f.registry.get(f.one.id)
  await writeFile(join(f.a, 'original.pdf'), 'original document')
  let canceled = false
  await f.registry.remove(f.one.id, () => { canceled = true })
  assert.equal(canceled, true)
  assert.throws(() => f.registry.current(mounted), /closed/)
  assert.equal(await readFile(join(f.a, 'original.pdf'), 'utf8'), 'original document')
  assert.equal(await readFile(join(f.a, 'tokensmith.sqlite'), 'utf8'), 'test database')
  assert.deepEqual(f.registry.list().map(c => c.id), [f.two.id])
  await f.registry.unload(f.two.id)
  await f.registry.remove(f.two.id)
  const restarted = new CollectionRegistry(f.registryPath, f.access)
  await restarted.initialize()
  assert.deepEqual(restarted.list(), [])
})

for (const action of ['unload','remove']) test(`${action} persistence failure keeps the library mounted and does not cancel its work`,async t=>{
 const f=await fixture(t), mounted=f.registry.get(f.one.id)
 const saved=f.registryPath+'.saved'
 await rename(f.registryPath,saved);await mkdir(f.registryPath)
 let stopped=false
 await assert.rejects(f.registry[action](f.one.id,()=>{stopped=true}))
 assert.equal(stopped,false);assert.equal(f.registry.get(f.one.id),mounted)
 assert.equal(f.registry.list().length,2);assert.deepEqual(f.closed,[])
 await rm(f.registryPath,{recursive:true});await rename(saved,f.registryPath)
 await f.registry[action](f.one.id,()=>{stopped=true})
 assert.equal(stopped,true)
})

test('delete cancels the target only and invalidates stale requests before cleanup',async t=>{
 const f=await fixture(t), old=f.registry.get(f.one.id), other=f.registry.get(f.two.id)
 await f.registry.remove(f.one.id,()=>{
  assert.throws(()=>f.registry.current(old),/closed/)
  f.registry.current(other)
 })
 assert.deepEqual(f.closed,[f.a]);assert.equal(f.registry.get(f.two.id),other)
 await f.registry.remove(f.one.id,()=>assert.fail('Repeated delete must not cancel another library'))
 assert.deepEqual(f.closed,[f.a])
})

test('export snapshots serialize revision changes and release the lock after failure',async t=>{
 const f=await fixture(t), item=f.registry.get(f.one.id), revision=item.indexRevision
 let release, entered
 const ready=new Promise(resolve=>{entered=resolve}), gate=new Promise(resolve=>{release=resolve})
 const snapshot=f.registry.snapshot(item.id,async current=>{assert.equal(current,item);entered();await gate;assert.equal(current.indexRevision,revision);throw Error('Export failed')})
 await ready
 const failure=assert.rejects(snapshot,/Export failed/)
 const change=f.registry.revise(item)
 assert.equal(item.indexRevision,revision)
 release();await failure;await change
 assert.notEqual(item.indexRevision,revision)
})
