import assert from 'node:assert/strict'
import { spawn } from 'node:child_process'
import { existsSync } from 'node:fs'
import { mkdtemp, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { dirname, join, resolve } from 'node:path'
import { once } from 'node:events'

const rootDir = resolve(new URL('../..', import.meta.url).pathname)
const workerPath = join(rootDir, 'python_engine', 'tokensmith_engine.py')
const bundledPythonPath = join(rootDir, 'app_runtime', 'python', 'bin', 'python')
const venvPythonPath = process.platform === 'win32'
  ? join(rootDir, '.venv', 'Scripts', 'python.exe')
  : join(rootDir, '.venv', 'bin', 'python')
const pythonPath =
  process.env.TOKENSMITH_TEST_PYTHON ??
  process.env.TOKENSMITH_PYTHON ??
  (existsSync(venvPythonPath) ? venvPythonPath : null) ??
  (existsSync(bundledPythonPath) ? bundledPythonPath : 'python3')
const embeddingPath = process.env.TOKENSMITH_TEST_EMBEDDING_MODEL_PATH
const chatModelPath = process.env.TOKENSMITH_TEST_CHAT_MODEL_PATH
const requireGgufIntegration = process.env.TOKENSMITH_REQUIRE_GGUF_INTEGRATION === '1'
const configuredTimeoutMs = Number(process.env.TOKENSMITH_TEST_WORKER_TIMEOUT_MS ?? (requireGgufIntegration ? 900_000 : 180_000))
const workerTimeoutMs = Number.isFinite(configuredTimeoutMs) && configuredTimeoutMs > 0 ? configuredTimeoutMs : 180_000

function appPythonEnv() {
  if (pythonPath !== bundledPythonPath) {
    return {
      PYTHONIOENCODING: 'utf-8'
    }
  }

  const runtimeRoot = dirname(dirname(pythonPath))
  const libRoot = join(runtimeRoot, 'lib')
  const pythonLib = join(libRoot, 'python3.10')
  const sitePackages = join(pythonLib, 'site-packages')

  return {
    PYTHONHOME: runtimeRoot,
    PYTHONPATH: `${pythonLib}:${sitePackages}`,
    DYLD_FALLBACK_LIBRARY_PATH: libRoot,
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
const userDataPath = join(tempDir, 'user-data')
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

  if (!embeddingPath || !existsSync(embeddingPath)) {
    if (requireGgufIntegration) {
      throw new Error('TOKENSMITH_TEST_EMBEDDING_MODEL_PATH must point to a GGUF embedder for GGUF integration tests.')
    }
    console.log('Python PDF preview integration test passed. Skipping GGUF indexing; set TOKENSMITH_TEST_EMBEDDING_MODEL_PATH to run it.')
    process.exitCode = 0
    return
  }

  const embedder = {
    id: 'test-embedder',
    name: 'Test Embedder',
    role: 'embedder',
    engine: 'python',
    status: 'ready',
    path: embeddingPath,
    embeddingPath
  }

  const indexResult = await worker.request('index_material', {
    path: pdfPath,
    userDataPath,
    model: embedder
  })
  const material = indexResult.material

  assert.equal(material.status, 'ready')
  assert.equal(material.kind, 'pdf')
  assert.ok(material.wordCount >= 30, `expected PDF text extraction, got ${material.wordCount} words`)
  assert.ok(material.chunkCount >= 1, `expected chunks, got ${material.chunkCount}`)

  const question = 'What does third normal form remove?'
  const searchResult = await worker.request('search', {
    query: question,
    materials: [material],
    limit: 2,
    userDataPath,
    embeddingModels: [embedder]
  })

  assert.ok(searchResult.sources.length >= 1, 'expected at least one retrieved source')
  assert.match(searchResult.sources[0].excerpt, /transitive dependencies/i)

  if (!chatModelPath || !existsSync(chatModelPath)) {
    if (requireGgufIntegration) {
      throw new Error('TOKENSMITH_TEST_CHAT_MODEL_PATH must point to a GGUF chat model for GGUF integration tests.')
    }
    console.log('Python PDF GGUF indexing/search integration test passed.')
    process.exitCode = 0
    return
  }

  const chatResult = await worker.request('chat', {
    prompt: question,
    messages: [],
    materials: [material],
    model: {
      id: 'test-chat-model',
      name: 'Test Chat Model',
      role: 'chat',
      engine: 'python',
      status: 'ready',
      path: chatModelPath
    },
    settings: {
      maxSources: 2
    },
    userDataPath,
    embeddingModels: [embedder]
  })

  assert.equal(chatResult.engineId, 'tokensmith')
  assert.ok(chatResult.sources.length >= 1, 'expected chat sources')
  assert.match(`${chatResult.text}\n${chatResult.sources[0].excerpt}`, /transitive dependencies/i)

  console.log('Python PDF integration test passed.')
} finally {
  await worker.close()
}
}

await runTest()
