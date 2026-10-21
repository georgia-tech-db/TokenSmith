import assert from 'node:assert/strict'
import { spawnSync } from 'node:child_process'
import { copyFile, mkdir, mkdtemp, readFile, rm, symlink, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { dirname, join } from 'node:path'
import test from 'node:test'

for (const incompatibleBinary of ['Electron', 'Python']) {
  test(`macOS packaging rejects incompatible ${incompatibleBinary} before changing an existing package`, {
    skip: process.platform !== 'darwin'
  }, async () => {
    const root = await mkdtemp(join(tmpdir(), 'tokensmith mac package '))
    try {
      const script = join(root, 'scripts', 'package-mac.mjs')
      const binaries = {
        Electron: join(root, 'node_modules', 'electron', 'dist', 'Electron.app', 'Contents', 'MacOS', 'Electron'),
        Python: join(root, 'app_runtime', 'python', 'bin', 'python')
      }
      await mkdir(dirname(script), { recursive: true })
      await copyFile(new URL('../../../scripts/package-mac.mjs', import.meta.url), script)
      await writeFile(join(root, 'package.json'), JSON.stringify({ version: '0.1.15' }))
      for (const [name, binary] of Object.entries(binaries)) {
        await mkdir(dirname(binary), { recursive: true })
        if (name === incompatibleBinary) {
          await writeFile(binary, '#!/bin/sh\nexit 0\n', { mode: 0o755 })
        } else {
          await symlink(process.execPath, binary)
        }
      }
      const existingPackage = join(root, 'release', 'mac', 'existing-package')
      await mkdir(dirname(existingPackage), { recursive: true })
      await writeFile(existingPackage, 'preserve this package')

      const result = spawnSync(process.execPath, [script], { encoding: 'utf8', timeout: 30_000 })
      assert.equal(result.status, 1, result.error?.message || result.stderr)
      assert.ok(result.stderr.includes(binaries[incompatibleBinary]), result.stderr)
      assert.ok(result.stderr.includes(`does not support macOS ${process.arch}`), result.stderr)
      assert.equal(await readFile(existingPackage, 'utf8'), 'preserve this package')
    } finally {
      await rm(root, { recursive: true, force: true })
    }
  })
}
