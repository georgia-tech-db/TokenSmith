import assert from 'node:assert/strict'
import { spawn } from 'node:child_process'
import { existsSync, readdirSync } from 'node:fs'
import { mkdtemp, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { delimiter, dirname, join, resolve } from 'node:path'
import { once } from 'node:events'

const rootDir = resolve(new URL('../..', import.meta.url).pathname)
const workerPath = join(rootDir, 'python_engine', 'tokensmith_engine.py')
const bundledPythonPath = process.platform === 'win32'
  ? join(rootDir, 'app_runtime', 'python', 'python.exe')
  : join(rootDir, 'app_runtime', 'python', 'bin', 'python')
if (!existsSync(bundledPythonPath)) {
  throw new Error('TokenSmith app_runtime/python was not found. Run npm run setup:python-runtime first.')
}
const pythonPath = bundledPythonPath
const configuredTimeoutMs = Number(process.env.TOKENSMITH_TEST_WORKER_TIMEOUT_MS ?? 180_000)
const workerTimeoutMs = Number.isFinite(configuredTimeoutMs) && configuredTimeoutMs > 0 ? configuredTimeoutMs : 180_000

function appPythonEnv() {
  if (process.platform === 'win32') {
    return {
      PYTHONIOENCODING: 'utf-8'
    }
  }

  const runtimeRoot = dirname(dirname(resolve(pythonPath)))
  const libRoot = join(runtimeRoot, 'lib')
  const pythonLibName = existsSync(libRoot)
    ? readdirSync(libRoot).find((name) => /^python3\.\d+$/.test(name))
    : undefined
  const pythonLib = pythonLibName ? join(libRoot, pythonLibName) : join(libRoot, 'python3.10')
  const sitePackages = join(pythonLib, 'site-packages')
  const pythonPathParts = []
  const hasStdlib = existsSync(join(pythonLib, 'encodings', '__init__.py'))

  if (hasStdlib) {
    pythonPathParts.push(pythonLib)
  }
  if (existsSync(sitePackages)) {
    pythonPathParts.push(sitePackages)
  }

  return {
    ...(hasStdlib ? { PYTHONHOME: runtimeRoot } : {}),
    ...(pythonPathParts.length ? { PYTHONPATH: pythonPathParts.join(delimiter) } : {}),
    DYLD_FALLBACK_LIBRARY_PATH: libRoot,
    LD_LIBRARY_PATH: libRoot,
    PYTHONIOENCODING: 'utf-8'
  }
}

function escapePdfText(text) {
  return text.replaceAll('\\', '\\\\').replaceAll('(', '\\(').replaceAll(')', '\\)')
}

function createToyPdfBuffer(text) {
  const stream = [
    'BT',
    '/F1 12 Tf',
    '72 720 Td',
    '14 TL',
    ...text
      .split('\n')
      .flatMap((line, index) => [
        index === 0 ? `(${escapePdfText(line)}) Tj` : `(${escapePdfText(line)}) Tj`
      ])
      .flatMap((line, index, lines) => (index === lines.length - 1 ? [line] : [line, 'T*'])),
    'ET'
  ].join('\n')
  const objects = [
    '1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n',
    '2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n',
    '3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>\nendobj\n',
    '4 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n',
    `5 0 obj\n<< /Length ${Buffer.byteLength(stream, 'latin1')} >>\nstream\n${stream}\nendstream\nendobj\n`
  ]
  let pdf = '%PDF-1.4\n'
  const offsets = [0]

  for (const object of objects) {
    offsets.push(Buffer.byteLength(pdf, 'latin1'))
    pdf += object
  }

  const xrefOffset = Buffer.byteLength(pdf, 'latin1')
  pdf += `xref\n0 ${objects.length + 1}\n`
  pdf += '0000000000 65535 f \n'
  for (const offset of offsets.slice(1)) {
    pdf += `${String(offset).padStart(10, '0')} 00000 n \n`
  }
  pdf += `trailer\n<< /Size ${objects.length + 1} /Root 1 0 R >>\nstartxref\n${xrefOffset}\n%%EOF\n`

  return Buffer.from(pdf, 'latin1')
}

class PythonWorker {
  constructor() {
    this.nextId = 1
    this.pending = new Map()
    this.buffer = ''
    this.stderr = ''
    this.process = spawn(pythonPath, [workerPath], {
      cwd: rootDir,
      env: {
        ...process.env,
        ...appPythonEnv()
      },
      stdio: ['pipe', 'pipe', 'pipe']
    })
    this.process.stdout.on('data', (chunk) => this.handleStdout(chunk))
    this.process.stderr.on('data', (chunk) => {
      this.stderr += chunk.toString('utf8')
    })
  }

  handleStdout(chunk) {
    this.buffer += chunk.toString('utf8')
    const lines = this.buffer.split(/\r?\n/)
    this.buffer = lines.pop() ?? ''
    for (const line of lines) {
      if (!line.trim()) continue
      const message = JSON.parse(line)
      const pending = this.pending.get(message.id)
      if (!pending) continue
      if (message.progress) continue
      this.pending.delete(message.id)
      if (message.ok) {
        pending.resolve(message.result)
      } else {
        pending.reject(new Error(message.error ?? 'Worker request failed'))
      }
    }
  }

  request(command, payload, timeoutMs = workerTimeoutMs) {
    const id = String(this.nextId++)
    this.process.stdin.write(`${JSON.stringify({ id, command, payload })}\n`)
    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        this.pending.delete(id)
        reject(new Error(`Timed out waiting for ${command}: ${this.stderr}`))
      }, timeoutMs)
      this.pending.set(id, {
        resolve: (value) => {
          clearTimeout(timeout)
          resolve(value)
        },
        reject: (error) => {
          clearTimeout(timeout)
          reject(error)
        }
      })
    })
  }

  async close() {
    this.process.stdin.end()
    await once(this.process, 'close')
  }
}

const tempDir = await mkdtemp(join(tmpdir(), 'tokensmith-pdf-integration-'))
const pdfPath = join(tempDir, 'database-systems-toy.pdf')
const pdfText = [
  'Database Systems Study Note.',
  'A primary key uniquely identifies each row in a relational table.',
  'Normalization reduces duplicated data and update anomalies.',
  'Third normal form removes transitive dependencies between non-key attributes.',
  'Transactions should preserve ACID properties: atomicity, consistency, isolation, and durability.'
].join('\n')

await writeFile(pdfPath, createToyPdfBuffer(pdfText))

async function runTest() {
const worker = new PythonWorker()

try {
  const health = await worker.request('health', {})
  assert.equal(health.ok, true)
  assert.equal(health.engine, 'python')

  const preview = await worker.request('preview_cleaning', {
    path: pdfPath,
    cleaningProfileId: 'course'
  })

  assert.equal(preview.document.kind, 'pdf')
  assert.equal(preview.document.pageCount, 1)
  assert.match(preview.rawPages.map((page) => page.text).join('\n'), /Third normal form/i)
  assert.match(preview.cleanedPages.map((page) => page.text).join('\n'), /transitive dependencies/i)

  assert.equal(health.supports.some(value => value.includes('gguf')), false)
  await assert.rejects(worker.request('chat', {}), /Unknown command/)
  console.log('Python PDF preview and worker integration test passed.')
} finally {
  await worker.close()
}
}

await runTest()
