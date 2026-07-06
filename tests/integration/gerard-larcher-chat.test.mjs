import assert from 'node:assert/strict'
import { spawn } from 'node:child_process'
import { existsSync } from 'node:fs'
import { mkdtemp } from 'node:fs/promises'
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
const explicitPdfPath = process.env.TOKENSMITH_GERARD_PDF_PATH
const pdfPath = explicitPdfPath ?? '/Users/aj/Desktop/Gérard Larcher - Wikipedia.pdf'
const embeddingPath =
  process.env.TOKENSMITH_TEST_EMBEDDING_MODEL_PATH ??
  join(rootDir, 'bundled_models', 'embedders', 'nomic-embed-text-v1.5.f16.gguf')
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

class PythonWorker {
  constructor(logPath) {
    this.nextId = 1
    this.pending = new Map()
    this.buffer = ''
    this.stderr = ''
    this.process = spawn(pythonPath, [workerPath], {
      cwd: rootDir,
      env: {
        ...process.env,
        ...appPythonEnv(),
        TOKENSMITH_LOG_FILE: logPath
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
    return new Promise((resolvePromise, reject) => {
      const timeout = setTimeout(() => {
        this.pending.delete(id)
        reject(new Error(`Timed out waiting for ${command}`))
      }, timeoutMs)
      this.pending.set(id, {
        resolve: (value) => {
          clearTimeout(timeout)
          resolvePromise(value)
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

if (!existsSync(pdfPath) && explicitPdfPath) {
  throw new Error(`Expected Gérard Larcher PDF at ${pdfPath}`)
}

if (!existsSync(pdfPath)) {
  if (requireGgufIntegration) {
    throw new Error(`TOKENSMITH_GERARD_PDF_PATH must point to a real Gérard Larcher PDF for GGUF integration tests. Default path was ${pdfPath}`)
  }
  console.log(`Skipping Gérard Larcher PDF chat test; fixture not found at ${pdfPath}`)
  process.exit(0)
}

if (!existsSync(embeddingPath)) {
  if (requireGgufIntegration) {
    throw new Error('TOKENSMITH_TEST_EMBEDDING_MODEL_PATH must point to a GGUF embedder for the Gérard Larcher test.')
  }
  console.log('Skipping Gérard Larcher PDF chat test; GGUF embedder not configured.')
  process.exit(0)
}

const tempDir = await mkdtemp(join(tmpdir(), 'tokensmith-gerard-chat-'))
const userDataPath = join(tempDir, 'user-data')
const logPath = join(tempDir, 'tokensmith.log')
const embedder = {
  id: 'test-embedder',
  name: 'Test Embedder',
  role: 'embedder',
  engine: 'python',
  status: 'ready',
  path: embeddingPath,
  embeddingPath
}
const worker = new PythonWorker(logPath)

try {
  const indexResult = await worker.request('index_material', {
    path: pdfPath,
    userDataPath,
    model: embedder
  })
  const material = indexResult.material

  assert.equal(material.status, 'ready')
  assert.equal(material.kind, 'pdf')
  assert.ok(material.wordCount >= 600, `expected substantial PDF extraction, got ${material.wordCount}`)
  assert.ok(material.chunkCount >= 5, `expected multiple chunks, got ${material.chunkCount}`)

  const chatResult = await worker.request('chat', {
    prompt: 'What office did Larcher assume on 1 October 2014?',
    messages: [],
    materials: [material],
    model: {
      id: 'llama-3-2-3b-instruct',
      name: 'Llama 3.2 3B Instruct',
      engine: 'python',
      status: 'ready',
      embeddingPath
    },
    settings: {
      maxSources: 3
    },
    userDataPath,
    embeddingModels: [embedder]
  })

  assert.equal(chatResult.engineId, 'tokensmith')
  assert.ok(chatResult.sources.length >= 1, 'expected chat to retrieve at least one source')
  assert.match(chatResult.text, /local model is required/i)
  assert.equal(chatResult.sources[0].retrievalMode, 'vector')
  assert.match(chatResult.sources.map((source) => source.title).join('\n'), /Larcher/i)

  console.log('Gérard Larcher PDF chat test passed.')
} finally {
  await worker.close()
}
