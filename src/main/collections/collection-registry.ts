import { randomUUID } from 'node:crypto'
import { mkdir, readFile, readdir, realpath, rename, rm, stat, writeFile } from 'node:fs/promises'
import { dirname, isAbsolute, join, relative, resolve, sep } from 'node:path'
import type { Collection, CollectionManifest, SavedCollectionTopics } from '../../shared/collections'
import { collectionMaterialId, splitMaterialId } from '../../shared/collections'
import type { StudyDocumentRef } from '../../shared/study-scope'
import type { PracticeTopic } from '../../shared/practice'

const manifestName = 'collection.json'
interface Entry { id: string; path: string; name: string; loaded: boolean; mappings?: Record<string, { from: string; to: string }> }
export interface MountedCollection extends Collection { loaded: true; epoch: number }
interface StoreAccess {
  validate: (path: string) => Promise<void>
  create: (path: string) => Promise<void>
  close: (path: string) => void | Promise<void>
}
const inside = (root: string, path: string) => { const rel = relative(root, path); return rel === '' || (!rel.startsWith(`..${sep}`) && rel !== '..' && !isAbsolute(rel)) }

async function readJson(path: string, maxBytes = 1_000_000): Promise<unknown> {
  if ((await stat(path)).size > maxBytes) throw new Error('The library information is too large.')
  return JSON.parse(await readFile(path, 'utf8'))
}
async function writeJson(path: string, value: unknown) {
  const temporary = `${path}.${randomUUID()}.tmp`
  try {
    await writeFile(temporary, JSON.stringify(value, null, 2), 'utf8')
    await rename(temporary, path)
  } finally { await rm(temporary, { force: true }) }
}
function manifest(value: unknown): CollectionManifest {
  const m = value as Partial<CollectionManifest> | null
  if (!m || m.format !== 'tokensmith-collection' || m.version !== 1) throw new Error('Choose a saved TokenSmith library. To add documents, use New library.')
  if (typeof m.id !== 'string' || !/^[a-zA-Z0-9_-]{8,80}$/.test(m.id) || typeof m.name !== 'string' || !m.name.trim() || m.name.length > 200 ||
      typeof m.indexRevision !== 'string' || !m.indexRevision || m.indexRevision.length > 100) throw new Error('The library information is incomplete.')
  return { format: m.format, version: 1, id: m.id, name: m.name, indexRevision: m.indexRevision,
    ...(typeof m.sourceBasePath === 'string' ? { sourceBasePath: m.sourceBasePath } : {}) }
}

export class CollectionRegistry {
  private entries: Entry[] = []
  private mounts = new Map<string, MountedCollection>()
  private errors = new Map<string, string>()
  private sequence = 0
  private queue: Promise<unknown> = Promise.resolve()
  constructor(private registryPath: string, private store: StoreAccess) {}
  private serial<T>(work: () => Promise<T>): Promise<T> {
    const result = this.queue.then(work)
    this.queue = result.catch(() => undefined)
    return result
  }
  private async persist(entries = this.entries) {
    await mkdir(resolve(this.registryPath, '..'), { recursive: true })
    await writeJson(this.registryPath, entries)
  }
  async initialize() {
    try {
      const saved = await readJson(this.registryPath)
      if (Array.isArray(saved)) this.entries = saved.filter(e => e && typeof e.id === 'string' && typeof e.path === 'string' && typeof e.name === 'string')
    } catch (error) { if ((error as NodeJS.ErrnoException).code !== 'ENOENT') throw error }
    for (const entry of this.entries) {
      if (!entry.loaded) continue
      try { const item = await this.open(entry.path, entry.id); this.mounts.set(item.id, item) }
      catch (error) { entry.loaded = false; this.errors.set(entry.id, (error as Error).message) }
    }
  }
  list(): Collection[] {
    return this.entries.map<Collection>(entry => this.mounts.get(entry.id) ?? { format: 'tokensmith-collection', version: 1, id: entry.id,
      name: entry.name, path: entry.path, indexRevision: '', loaded: false, error: this.errors.get(entry.id) })
      .map(({ ...item }) => item)
  }
  loaded(): MountedCollection[] { return [...this.mounts.values()] }
  get(id: string): MountedCollection {
    const item = this.mounts.get(id)
    if (!item) throw new Error('This library is closed. Open it from Libraries to continue.')
    return item
  }
  current(item: MountedCollection): void {
    if (this.mounts.get(item.id)?.epoch !== item.epoch) throw new Error('The library was closed. Open it from Libraries to continue.')
  }
  forMaterial(id: string): { collection: MountedCollection; localId: string } {
    const parts = splitMaterialId(id)
    return { collection: this.get(parts.collectionId), localId: parts.localId }
  }
  private async open(path: string, expectedId?: string): Promise<MountedCollection> {
    const root = await realpath(path)
    let raw: unknown
    try { raw = await readJson(join(root, manifestName)) }
    catch (error) {
      if ((error as NodeJS.ErrnoException).code === 'ENOENT' || error instanceof SyntaxError) throw new Error('Choose a saved TokenSmith library. To add documents, use New library.')
      throw error
    }
    const m = manifest(raw)
    if (expectedId && m.id !== expectedId) throw new Error('This location now contains a different library. Open it again from Libraries.')
    if (this.loaded().some(item => item.path === root && item.id !== m.id)) throw new Error('This location is already open as another library. Close it first.')
    const existing = this.mounts.get(m.id)
    if (existing) {
      if (existing.path !== root) throw new Error('Another copy of this library is already open. Close it first.')
      return existing
    }
    try { await this.store.validate(root) }
    catch (error) { await this.store.close(root); throw error }
    return { ...m, path: root, loaded: true, epoch: ++this.sequence }
  }
  load(path: string): Promise<Collection[]> {
    return this.serial(async () => {
      const item = await this.open(path)
      const old = this.entries.find(entry => entry.id === item.id)
      const entry: Entry = { ...old, id: item.id, name: item.name, path: item.path, loaded: true }
      const next = [entry, ...this.entries.filter(value => value.id !== item.id)]
      try { await this.persist(next) }
      catch (error) { if (!this.mounts.has(item.id)) await this.store.close(item.path); throw error }
      this.entries = next; this.mounts.set(item.id, item); this.errors.delete(item.id)
      return this.list()
    })
  }
  create(path: string, name: string): Promise<Collection[]> {
    return this.serial(async () => {
      const root = await realpath(path)
      if (!name.trim() || name.trim().length > 200) throw new Error('Enter a library name of up to 200 characters.')
      if ((await readdir(root)).some(file => file !== '.DS_Store')) throw new Error('Choose an empty folder for the new collection.')
      const m: CollectionManifest = { format: 'tokensmith-collection', version: 1, id: randomUUID(), name: name.trim(), indexRevision: randomUUID(), sourceBasePath: root }
      const item: MountedCollection = { ...m, path: root, loaded: true, epoch: ++this.sequence }
      const next = [{ id: m.id, path: root, name: m.name, loaded: true }, ...this.entries]
      try {
        await this.store.create(root)
        await writeJson(join(root, manifestName), m)
        await this.persist(next)
      } catch (error) { await this.store.close(root); throw error }
      this.entries = next; this.mounts.set(item.id, item)
      return this.list()
    })
  }
  async createManaged(name: string): Promise<Collection[]> {
    if (!name.trim() || name.trim().length > 200) throw new Error('Enter a library name of up to 200 characters.')
    const path = join(dirname(this.registryPath), 'libraries', randomUUID())
    await mkdir(path, { recursive: true })
    try { return await this.create(path, name) }
    catch (error) {
      // Only this freshly allocated app-owned directory can be removed on failure.
      await rm(path, { recursive: true, force: true })
      throw error
    }
  }
  unload(id: string, onUnload: () => void | Promise<void> = () => {}): Promise<Collection[]> {
    return this.serial(async () => {
      const item = this.get(id)
      const next = this.entries.map(entry => entry.id === id ? { ...entry, loaded: false } : entry)
      await this.persist(next)
      this.mounts.delete(id); this.entries = next
      try { await onUnload() }
      finally { await this.store.close(item.path) }
      return this.list()
    })
  }
  async revise(item: MountedCollection) {
    return this.serial(async () => {
      this.current(item)
      const raw = await readJson(join(item.path, manifestName)) as CollectionManifest
      const m = manifest(raw)
      m.indexRevision = randomUUID()
      await writeJson(join(item.path, manifestName), { ...raw, indexRevision: m.indexRevision })
      item.indexRevision = m.indexRevision
    })
  }
  async locateFolder(materialId: string, from: string, path: string) {
    return this.serial(async () => {
      const { collection, localId } = this.forMaterial(materialId)
      const to = await realpath(path)
      if (!(await stat(to)).isDirectory()) throw new Error('Choose the folder containing the original documents.')
      const next = this.entries.map(entry => entry.id === collection.id ? { ...entry,
        mappings: { ...entry.mappings, [localId]: { from, to } } } : entry)
      await this.persist(next); this.entries = next
    })
  }
  sourcePath(item: MountedCollection, path: string, localMaterialId?: string): string {
    this.current(item)
    const mapping = localMaterialId && this.entries.find(e => e.id === item.id)?.mappings?.[localMaterialId]
    const original = isAbsolute(path) ? path : resolve(item.path, path)
    if (mapping && inside(mapping.from, original)) return resolve(mapping.to, relative(mapping.from, original))
    const portable = path.replaceAll('\\', '/')
    const base = item.sourceBasePath?.replaceAll('\\', '/').replace(/\/$/, '')
    if (base && (portable === base || portable.startsWith(`${base}/`))) return resolve(item.path, portable.slice(base.length + 1))
    if (isAbsolute(path)) return path
    const target = resolve(item.path, path)
    if (!inside(item.path, target)) throw new Error('The source path leaves the collection folder.')
    return target
  }
  originalPath(item: MountedCollection, path: string, localMaterialId: string): string {
    const mapping = this.entries.find(e => e.id === item.id)?.mappings?.[localMaterialId]
    if (mapping && inside(mapping.to, path)) return resolve(mapping.from, relative(mapping.to, path))
    if (item.sourceBasePath && inside(item.path, path)) return resolve(item.sourceBasePath, relative(item.path, path))
    return path
  }
  async topics(documents: StudyDocumentRef[]): Promise<PracticeTopic[]> {
    const keys = new Set(documents.map(d => `${d.materialId}:${d.documentId}`))
    const ids = [...new Set(documents.map(d => splitMaterialId(d.materialId).collectionId))]
    const result: PracticeTopic[] = []
    for (const id of ids) {
      const item = this.get(id)
      try {
        const saved = await readJson(join(item.path, 'topics.json')) as SavedCollectionTopics
        this.current(item)
        if (saved.version !== 1 || saved.indexRevision !== item.indexRevision || !Array.isArray(saved.topics)) continue
        for (const topic of saved.topics) {
          if (typeof topic.title !== 'string' || !topic.title.trim() || topic.title.length > 100 || !Array.isArray(topic.documents) || !topic.documents.length) continue
          if (topic.documents.every(d => keys.has(`${collectionMaterialId(id, d.materialId)}:${d.documentId}`))) result.push({ title: topic.title })
        }
      } catch (error) { this.current(item) /* Missing or malformed optional data never prevents basic search. */ }
    }
    return [...new Map(result.map(t => [t.title.toLowerCase(), t])).values()].slice(0, 6)
  }
  async saveTopics(documents: StudyDocumentRef[], topics: PracticeTopic[], expectedRevision: string) {
    return this.serial(async () => {
      const ids = [...new Set(documents.map(d => splitMaterialId(d.materialId).collectionId))]
      if (ids.length !== 1) return
      const item = this.get(ids[0])
      if (item.indexRevision !== expectedRevision) return
      const refs = documents.map(d => ({ ...d, materialId: splitMaterialId(d.materialId).localId }))
      let previous: SavedCollectionTopics['topics'] = []
      try {
        const saved = await readJson(join(item.path, 'topics.json')) as SavedCollectionTopics
        if (saved.version !== 1 || !Array.isArray(saved.topics)) return
        const key = (docs: StudyDocumentRef[]) => docs.map(d => `${d.materialId}:${d.documentId}`).sort().join('|')
        if (saved.indexRevision === item.indexRevision) previous = saved.topics.filter(t => Array.isArray(t.documents) && key(t.documents) !== key(refs))
      } catch (error) { if ((error as NodeJS.ErrnoException).code !== 'ENOENT') return }
      const saved: SavedCollectionTopics = { version: 1, indexRevision: item.indexRevision,
        topics: [...topics.filter(t => t.title.trim() && t.title.length <= 100).slice(0, 6).map(t => ({ title: t.title, documents: refs })), ...previous].slice(0, 200) }
      await writeJson(join(item.path, 'topics.json'), saved)
    })
  }
}
