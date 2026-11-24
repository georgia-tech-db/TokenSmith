import { mkdir, readFile, rename, writeFile } from 'node:fs/promises'
import { dirname } from 'node:path'

/** Retain user-selected file grants across launches of the sandboxed app. */
export class SecurityScopedFiles {
  private bookmarks = new Map<string, string>()
  private access = new Map<string, () => void>()
  private pending: Promise<void> = Promise.resolve()

  constructor(private file: string, private startAccess: (bookmark: string) => () => void,
    private report: (path: string, error: unknown) => void = () => {}) {}

  async restore(): Promise<void> {
    let records: unknown
    try { records = JSON.parse(await readFile(this.file, 'utf8')) }
    catch (error) { if ((error as NodeJS.ErrnoException).code !== 'ENOENT') this.report(this.file, error); return }
    if (!Array.isArray(records)) return
    for (const record of records) {
      if (!Array.isArray(record) || record.length !== 2 || !record.every(value => typeof value === 'string')) continue
      const [path, bookmark] = record
      try { this.access.set(path, this.startAccess(bookmark)); this.bookmarks.set(path, bookmark) }
      catch (error) { this.report(path, error) } // A stale grant must not prevent other libraries from opening.
    }
  }

  remember(paths: string[], bookmarks: string[]): Promise<void> {
    const operation = this.pending.then(async () => {
      for (let i = 0; i < paths.length; i++) {
        const bookmark = bookmarks[i]
        if (!bookmark) throw new Error('File access was not granted. Please select the file or folder again.')
        if (this.bookmarks.get(paths[i]) === bookmark) continue
        const stop = this.startAccess(bookmark)
        this.access.get(paths[i])?.()
        this.access.set(paths[i], stop)
        this.bookmarks.set(paths[i], bookmark)
      }
      await mkdir(dirname(this.file), { recursive: true })
      await writeFile(`${this.file}.tmp`, JSON.stringify([...this.bookmarks]), { mode: 0o600 })
      await rename(`${this.file}.tmp`, this.file)
    })
    this.pending = operation.catch(() => {})
    return operation
  }

  close(): void {
    for (const stop of this.access.values()) stop()
    this.access.clear()
  }
}
