import { app } from 'electron'
import { rm } from 'node:fs/promises'
import { join } from 'node:path'
import type { LocalModel } from '../../shared/app-state'

function localModelDirectory(): string {
  return join(app.getPath('userData'), 'models')
}

function assertSafeFilename(filename: string): void {
  if (!filename || filename.includes('/') || filename.includes('\\')) {
    throw new Error('Model filename is invalid.')
  }
}

function localModelPath(filename: string): string {
  assertSafeFilename(filename)
  return join(localModelDirectory(), filename)
}

export async function removeLocalModelFile(model: LocalModel): Promise<void> {
  if (model.engine === 'remote' || model.engine === 'ollama' || model.source === 'remote' || model.source === 'ollama') {
    return
  }

  const filename = model.filename ?? model.path?.split(/[\\/]/).pop()
  if (!filename) {
    throw new Error('Model filename is missing.')
  }

  await Promise.all([
    rm(localModelPath(filename), { force: true }),
    rm(`${localModelPath(filename)}.part`, { force: true })
  ])
}
