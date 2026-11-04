import { mkdir, writeFile, rename } from 'node:fs/promises'
import { dirname } from 'node:path'

// Serialize snapshots in arrival order; a failed write must not poison later saves.
export function createStateWriter() {
  let pending: Promise<void> = Promise.resolve()
  return (path: string, json: string) => {
    const write = pending.catch(() => {}).then(async () => {
      await mkdir(dirname(path), { recursive: true })
      const temporary = `${path}.next`
      await writeFile(temporary, json, 'utf8')
      await rename(temporary, path)
    })
    pending = write
    return write
  }
}
