import { getQuickJS } from 'quickjs-emscripten'

export const javascriptLimits = {
  codeBytes: 8_192,
  outputBytes: 8_192,
  timeoutMs: 1_000,
  memoryBytes: 16 * 1024 * 1024,
  stackBytes: 512 * 1024
} as const

export type JavascriptExecution =
  | { ok: true; result: string; latencyMs: number }
  | { ok: false; error: string; latencyMs: number; partialOutput?: string }

function boundedError(error: unknown): string {
  const message = error instanceof Error ? error.message : String(error)
  return Buffer.from(message, 'utf8').subarray(0, javascriptLimits.outputBytes).toString('utf8')
}

function executionDiagnostic(error: unknown, code: string): string {
  const details = error && typeof error === 'object' ? error as Record<string, unknown> : {}
  const name = typeof details.name === 'string' ? details.name.slice(0, 120) : 'Error'
  const message = boundedError(details.message ?? error).slice(0, 2000)
  const stack = typeof details.stack === 'string' ? details.stack.slice(0, javascriptLimits.outputBytes) : ''
  const location = stack.match(/javascript_interpret\.js:(\d+)(?::(\d+))?/)
  const line = location ? Number(location[1]) : typeof details.lineNumber === 'number' ? details.lineNumber : undefined
  const column = location?.[2] ? Number(location[2]) : undefined
  const lines = code.split(/\r?\n/)
  if (!line || !Number.isInteger(line) || line < 1 || line > lines.length) return boundedError(`${name}: ${message}`)
  return boundedError(`${name} at line ${line}${column ? `, column ${column}` : ''}: ${message}\n${line}: ${lines[line - 1].slice(0, 400)}`)
}

export async function executeIsolatedJavascript(code: string): Promise<JavascriptExecution> {
  const started = performance.now()
  if (Buffer.byteLength(code, 'utf8') > javascriptLimits.codeBytes) {
    return { ok: false, error: `Code exceeds ${javascriptLimits.codeBytes} bytes.`, latencyMs: performance.now() - started }
  }
  if (!code.trim()) {
    return { ok: false, error: 'Code is empty.', latencyMs: performance.now() - started }
  }

  const quickJS = await getQuickJS()
  const runtime = quickJS.newRuntime()
  runtime.setMemoryLimit(javascriptLimits.memoryBytes)
  runtime.setMaxStackSize(javascriptLimits.stackBytes)
  const deadline = performance.now() + javascriptLimits.timeoutMs
  runtime.setInterruptHandler(() => performance.now() > deadline)
  const context = runtime.newContext()
  try {
    const output: string[] = []
    let outputBytes = 0
    let outputExceeded = false
    let nonFiniteOutput = false
    // This bounded output callback is the only host function exposed to QuickJS.
    const consoleObject = context.newObject()
    const logFunction = context.newFunction('log', (...args) => {
      const line = args.map((arg) => {
        if (context.typeof(arg) === 'number' && !Number.isFinite(context.getNumber(arg))) {
          nonFiniteOutput = true
        }
        const value = context.dump(arg)
        return typeof value === 'string' ? value : JSON.stringify(value) ?? String(value)
      }).join(' ')
      outputBytes += Buffer.byteLength(line, 'utf8') + (output.length > 0 ? 1 : 0)
      if (outputBytes > javascriptLimits.outputBytes) outputExceeded = true
      else output.push(line)
      return context.undefined
    })
    context.setProp(consoleObject, 'log', logFunction)
    context.setProp(context.global, 'console', consoleObject)
    logFunction.dispose()
    consoleObject.dispose()

    const evaluation = context.evalCode(code, 'javascript_interpret.js')
    if (evaluation.error) {
      const error = context.dump(evaluation.error)
      evaluation.error.dispose()
      return {
        ok: false,
        error: performance.now() > deadline ? 'JavaScript execution timed out.' : executionDiagnostic(error, code),
        latencyMs: performance.now() - started
      }
    }

    const nonFiniteReturn = context.typeof(evaluation.value) === 'number' &&
      !Number.isFinite(context.getNumber(evaluation.value))
    const value = context.dump(evaluation.value)
    evaluation.value.dispose()
    if (outputExceeded) {
      return { ok: false, error: `Output exceeds ${javascriptLimits.outputBytes} bytes.`, latencyMs: performance.now() - started }
    }
    if (nonFiniteReturn || (typeof value === 'number' && !Number.isFinite(value))) {
      return { ok: false, error: 'JavaScript output contains NaN or Infinity. Check the inputs, loop bounds, and divisor.',
        latencyMs: performance.now() - started, ...(output.length ? { partialOutput: output.join('\n') } : {}) }
    }
    if (value === undefined && output.length === 0) {
      return { ok: false, error: 'JavaScript must produce a finite, defined result.', latencyMs: performance.now() - started }
    }
    const returned = value === undefined ? '' : typeof value === 'string' ? value : JSON.stringify(value) ?? String(value)
    const result = [output.join('\n'), returned].filter(Boolean).join(output.length > 0 && returned ? '\nResult: ' : '')
    if (Buffer.byteLength(result, 'utf8') > javascriptLimits.outputBytes) {
      return { ok: false, error: `Output exceeds ${javascriptLimits.outputBytes} bytes.`, latencyMs: performance.now() - started }
    }
    if (nonFiniteOutput || /(?:^|[^\p{L}\p{N}_])(?:NaN|[-+]?Infinity)(?=$|[^\p{L}\p{N}_])/u.test(result)) {
      return { ok: false, error: 'JavaScript output contains NaN or Infinity. Check the inputs, loop bounds, and divisor.',
        latencyMs: performance.now() - started, partialOutput: result }
    }
    return { ok: true, result, latencyMs: performance.now() - started }
  } catch (error) {
    return { ok: false, error: boundedError(error), latencyMs: performance.now() - started }
  } finally {
    context.dispose()
    runtime.dispose()
  }
}
