import assert from 'node:assert/strict'
import { execFile } from 'node:child_process'
import { mkdtemp, mkdir, readFile, realpath, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import test from 'node:test'
import { promisify } from 'node:util'

const run = promisify(execFile)
const loaderUrl = new URL('./ts-module-loader.mjs', import.meta.url).href

test('parallel test processes isolate generated modules and preserve dependency source maps', async t => {
  const root = await realpath(await mkdtemp(join(tmpdir(), 'tokensmith-test-loader-')))
  t.after(() => rm(root, { recursive: true, force: true }))
  await mkdir(join(root, 'src'))
  const sources = {
    'dependency.ts': 'export const value: number = 42; export const generatedPath = __filename;',
    'entry.ts': "import { value, generatedPath } from './dependency'; export const answer = value; export const paths = [__filename, generatedPath];"
  }
  for (const [name, source] of Object.entries(sources)) {
    await writeFile(join(root, 'src', name), source)
  }
  const script = `
    import assert from 'node:assert/strict';
    import { requireTranspiledTs } from ${JSON.stringify(loaderUrl)};
    const first = requireTranspiledTs('src/entry.ts');
    assert.equal(first.answer, 42);
    assert.equal(requireTranspiledTs('src/entry.ts'), first);
    console.log(JSON.stringify({ pid: process.pid, paths: first.paths }));
  `
  // Fixture modules are outside the repo and must not enter its coverage report.
  const env = { ...process.env }
  delete env.NODE_V8_COVERAGE
  const results = await Promise.allSettled([0, 1].map(() => run(process.execPath,
    ['--input-type=module', '--eval', script], { cwd: root, env, timeout: 30_000 })))
  const workers = results.map(result => {
    assert.equal(result.status, 'fulfilled', result.reason?.message)
    assert.equal(result.value.stderr, '')
    return JSON.parse(result.value.stdout)
  })
  assert.notEqual(workers[0].pid, workers[1].pid)
  assert.notEqual(workers[0].paths[0], workers[1].paths[0])
  assert.notEqual(workers[0].paths[1], workers[1].paths[1])
  for (const worker of workers) {
    for (const [index, name] of ['entry.ts', 'dependency.ts'].entries()) {
      const path = join(root, '.coverage-ts-runtime', String(worker.pid), 'src', name)
      assert.equal(worker.paths[index], path)
      const generated = await readFile(path, 'utf8')
      const encodedMap = generated.match(/sourceMappingURL=data:application\/json;base64,(\S+)/)?.[1]
      assert.ok(encodedMap, 'generated modules must retain inline source maps')
      const map = JSON.parse(Buffer.from(encodedMap, 'base64').toString('utf8'))
      assert.equal(resolve(map.sourceRoot, map.sources[0]), join(root, 'src', name))
      assert.deepEqual(map.sourcesContent, [sources[name]])
    }
  }
})
