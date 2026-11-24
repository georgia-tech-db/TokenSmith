import assert from 'node:assert/strict'
import { test } from 'node:test'
import { mkdtemp, readFile, writeFile, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { macDistribution } from '../../../scripts/mac-distribution.mjs'
import { requireTranspiledTs } from './ts-module-loader.mjs'
const { SecurityScopedFiles } = requireTranspiledTs('src/main/security-scoped-files.ts')
const { assertOllamaManagementAllowed, isMacAppStore } = requireTranspiledTs('src/main/distribution.ts')

test('MAS signing fails closed without credentials; unsigned and DMG channels stay separate', async t => {
  const dir = await mkdtemp(join(tmpdir(), 'mas-signing-')); t.after(() => rm(dir, { recursive: true, force: true }))
  assert.equal(macDistribution([], {}).channel, 'mac')
  assert.equal(macDistribution(['--mas-unsigned'], {}).channel, 'mas-unsigned')
  assert.throws(() => macDistribution(['--mas'], {}), /requires/)
  assert.throws(() => macDistribution(['--mas', '--mas-unsigned'], {}), /not both/)
  assert.throws(() => macDistribution(['--mas'], { TOKENSMITH_SKIP_CODESIGN: '1' }), /cannot skip/)
  const profile = join(dir, 'test.provisionprofile'); await writeFile(profile, 'test fixture')
  const env = { TOKENSMITH_MAS_IDENTITY: 'Apple Development: Test', TOKENSMITH_MAS_PROFILE: profile }
  assert.equal(macDistribution(['--mas'], env).channel, 'mas-development')
  assert.throws(() => macDistribution(['--mas'], { ...env, TOKENSMITH_MAS_TYPE: 'distribution' }), /INSTALLER/)
  assert.throws(() => macDistribution(['--mas'], { ...env, TOKENSMITH_MAS_BUNDLE_ID: 'invalid/id' }), /Invalid/)
})

test('MAS management policy uses Electron runtime, never environment overrides', () => {
  const before = process.mas
  try {
    process.mas = false; assert.doesNotThrow(assertOllamaManagementAllowed)
    process.mas = true; assert.equal(isMacAppStore(), true)
    assert.throws(assertOllamaManagementAllowed, /outside TokenSmith/)
  } finally { if (before === undefined) delete process.mas; else process.mas = before }
})

test('selected file access persists, restores independently, and releases replaced grants', async t => {
  const dir = await mkdtemp(join(tmpdir(), 'mas-files-')); t.after(() => rm(dir, { recursive: true, force: true }))
  const file = join(dir, 'selected.json'), started = [], stopped = [], errors = []
  const start = token => { if (token === 'expired') throw Error('stale'); started.push(token); return () => stopped.push(token) }
  const access = new SecurityScopedFiles(file, start)
  await access.restore()
  await access.remember(['/book'], ['one'])
  await access.remember(['/book'], ['one'])
  await access.remember(['/book'], ['two'])
  assert.deepEqual(started, ['one', 'two']); assert.deepEqual(stopped, ['one'])
  await assert.rejects(access.remember(['/no-grant'], []), /select/)
  await access.remember(['/folder'], ['three'])
  assert.deepEqual(JSON.parse(await readFile(file, 'utf8')), [['/book', 'two'], ['/folder', 'three']])
  access.close(); assert.deepEqual(stopped, ['one', 'two', 'three'])
  await writeFile(file, JSON.stringify([['/bad', 'expired'], ['/good', 'four']]))
  const restored = new SecurityScopedFiles(file, start, path => errors.push(path))
  await restored.restore(); assert.deepEqual(errors, ['/bad']); assert.equal(started.at(-1), 'four')
  restored.close(); assert.equal(stopped.at(-1), 'four')
})
