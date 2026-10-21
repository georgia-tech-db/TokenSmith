import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import test from 'node:test'
import { standaloneRuntime } from '../../../scripts/python-runtime-manifest.mjs'

test('selects the pinned Linux x64 runtime', () => {
  const runtime = standaloneRuntime('linux', 'x64')

  assert.equal(runtime.pythonVersion, '3.12.14')
  assert.match(runtime.filename, /x86_64-unknown-linux-gnu-install_only_stripped\.tar\.gz$/)
  assert.match(runtime.url, /releases\/download\/20260901\//)
  assert.match(runtime.sha256, /^[a-f0-9]{64}$/)
})

test('selects the pinned Intel Mac runtime without changing the Apple Silicon Python version', () => {
  const runtime = standaloneRuntime('darwin', 'x64')
  assert.equal(runtime.key, 'darwin-x64')
  assert.equal(runtime.pythonVersion, standaloneRuntime('darwin', 'arm64').pythonVersion)
  assert.match(runtime.filename, /x86_64-apple-darwin-install_only_stripped\.tar\.gz$/)
  assert.match(runtime.url, /cpython-3\.12\.14%2B20260901-x86_64-apple-darwin/)
  assert.equal(runtime.sha256, '65b195c9cedc1fef6767f044f9822069adbd1bd9204d424ece4628776fdc04bb')
})

test('selects platform-specific runtimes', () => {
  assert.match(standaloneRuntime('darwin', 'arm64').filename, /aarch64-apple-darwin/)
  assert.match(standaloneRuntime('win32', 'x64').filename, /x86_64-pc-windows-msvc/)
})

test('rejects unsupported targets instead of using system Python', () => {
  assert.throws(
    () => standaloneRuntime('linux', 'riscv64'),
    /does not provide a Python runtime for linux-riscv64/
  )
})

test('runtime Python dependencies are exactly pinned', async () => {
  const requirements = await readFile(new URL('../../../requirements-runtime.txt', import.meta.url), 'utf8')
  const entries = requirements
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line && !line.startsWith('#'))

  assert.ok(entries.length > 0)
  for (const entry of entries) {
    assert.match(entry, /^[A-Za-z0-9_.-]+==[A-Za-z0-9_.!+*-]+$/)
  }
})
