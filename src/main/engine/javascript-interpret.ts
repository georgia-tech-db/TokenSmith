import type { StudyChatMessage } from './study-chat-format'
import type { JavascriptExecution } from './javascript-executor'
import { verifiedSeriesAnswerMatches, verifiedSeriesFacts, verifiedSeriesFromQuestion, verifiedSeriesOutputMatches } from './verified-series'
import { verifiedCalculationAnswerMatches, verifiedCalculationFromQuestion, verifiedCalculationOutputMatches, verifiedCalculationTeachingFacts } from './verified-calculation'

export const javascriptToolSchema = {
  type: 'object',
  properties: {
    tool: { type: 'string', enum: ['javascript_interpret'] },
    code: { type: 'string' }
  },
  required: ['tool', 'code'],
  additionalProperties: false
} as const

export const javascriptChoiceSchema = {
  type: 'object',
  properties: {
    route: { type: 'string', enum: ['answer', 'javascript_interpret'] },
    content: { type: 'string' }
  },
  required: ['route', 'content'],
  additionalProperties: false
} as const

export const maxExecutionRetries = 2

export interface JavascriptInterpretTrace {
  attempts: Array<{ code: string; execution: JavascriptExecution; retryCount: number; incompleteResult?: boolean; outputIssue?: string }>
  inferenceCount: number
  retryCount: number
  latencyMs: number
  inferenceLatenciesMs: number[]
  directAnswer?: boolean
  rejectedFinalAnswer?: string
  unsupportedFinalNumbers?: number[]
  emptyFinalAnswer?: boolean
  finalMissingRequestedList?: boolean
  finalMissingComputedList?: boolean
  finalOutputMismatch?: boolean
  verifiedSeriesFallback?: boolean
  verifiedSeriesOperation?: string
  verifiedCalculationFallback?: boolean
  verifiedCalculationOperation?: string
}

function numericValues(text: string): number[] {
  return Array.from(text.matchAll(/(?<![\p{L}\p{N}.])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?![\p{L}\p{N}])/gu),
    (match) => Number(match[0].replaceAll(',', '')))
}

export function formatComputedResult(result: string, code?: string): string {
  const output = `**Computed result**\n\n${result.trim().split(/\r?\n/).join('  \n')}`
  if (!code?.trim()) return output
  const longestBacktickRun = Math.max(0, ...(code.match(/`+/g) ?? []).map((run) => run.length))
  const fence = '`'.repeat(Math.max(3, longestBacktickRun + 1))
  return `${output}\n\n**Calculation (executed JavaScript)**\n\n${fence}js\n${code.trim()}\n${fence}`
}

export function unsupportedFinalNumbers(answer: string, question: string, result: string): number[] {
  const supported = new Set(numericValues(`${question}\n${result}`))
  // Markdown list/section counters describe layout, not quantitative claims.
  const claims = answer
    .replace(/^([ \t]*(?:#{1,6}[ \t]*)?)Step[ \t]+\d+(?=[ \t]*[:.)]|[ \t]+[A-Za-z])/gim, '$1Step')
    .replace(/^([ \t]*(?:#{1,6}[ \t]*)?)\d+[.)](?=[ \t]+)/gm, '$1')
  return [...new Set(numericValues(claims).filter((value) => !supported.has(value)))]
}

export function normalizeMathCommands(answer: string): string {
  // Some local models double-escape TeX commands inside math fences. KaTeX
  // reads those as line breaks, so fix only delimited math and leave prose alone.
  const singleSlashCommands = (math: string) => math.replace(/\\\\(?=[A-Za-z])/g, '\\')
  return answer
    .replace(/\$\$([\s\S]*?)\$\$/g, (_match, math: string) => `$$${singleSlashCommands(math)}$$`)
    .replace(/\\\(([\s\S]*?)\\\)/g, (_match, math: string) => `\\(${singleSlashCommands(math)}\\)`)
    .replace(/\\\[([\s\S]*?)\\\]/g, (_match, math: string) => `\\[${singleSlashCommands(math)}\\]`)
}

export function missingRequestedList(question: string, result: string): boolean {
  if (!/\b(?:list|enumerate)\b/i.test(question)) return false
  if (numericValues(result).length <= 1 && !/[,;\n]|\[[^\]]+\]/.test(result)) return true

  const countMatch = question.match(/\b(?:list|enumerate)\s+(?:the\s+)?(two|three|four|five|six|seven|eight|nine|ten|\d+)\b/i)
  const countNames: Record<string, number> = { two: 2, three: 3, four: 4, five: 5, six: 6, seven: 7, eight: 8, nine: 9, ten: 10 }
  const expectedCount = countMatch ? countNames[countMatch[1].toLowerCase()] ?? Number(countMatch[1]) : undefined
  const numericArray = result.match(/\[((?:\s*-?\d+(?:\.\d+)?\s*,)*\s*-?\d+(?:\.\d+)?\s*)\]/)
  if (expectedCount && numericArray) {
    return numericArray[1].split(',').length < expectedCount
  }
  return false
}

export function outputPrecisionIssue(question: string, result: string): string | undefined {
  if (!/\b(?:round\b[^.?!]*\b(?:cents?|two decimal places)|to two decimal places)\b/i.test(question)) return undefined
  const moneyOnly = /\bround\b[^.?!]*\b(?:cents?|money|monetary)\b/i.test(question) &&
    !/\bto two decimal places\b/i.test(question)
  for (const match of result.matchAll(/(?<![\d.])-?\d+\.\d{3,}(?!\d)/g)) {
    const next = result.slice((match.index ?? 0) + match[0].length).trimStart()[0]
    if (moneyOnly && next === '%') continue
    return 'The question requests values rounded to cents or two decimal places, but the output contains a longer decimal.'
  }
  return undefined
}

export function missingComputedList(answer: string, result: string): boolean {
  const answerCounts = new Map<number, number>()
  for (const value of numericValues(answer)) answerCounts.set(value, (answerCounts.get(value) ?? 0) + 1)
  for (const line of result.split(/\r?\n/)) {
    if (!/,\s*\d/.test(line)) continue
    const requiredCounts = new Map<number, number>()
    for (const value of numericValues(line)) requiredCounts.set(value, (requiredCounts.get(value) ?? 0) + 1)
    if (requiredCounts.size === 0) continue
    for (const [value, count] of requiredCounts) {
      if ((answerCounts.get(value) ?? 0) < count) return true
    }
  }
  return false
}

export async function runJavascriptInterpret(
  initialMessages: StudyChatMessage[],
  question: string,
  infer: (messages: StudyChatMessage[], format?: Record<string, unknown>, maxTokens?: number) => Promise<string>,
  execute: (code: string) => Promise<JavascriptExecution>,
  onTrace: (trace: JavascriptInterpretTrace) => void,
  sourceExcerpts: string[] = [],
  routeMode: 'tool' | 'choice' = 'choice'
): Promise<string> {
  const started = performance.now()
  const forcedSeries = routeMode === 'tool' ? verifiedSeriesFromQuestion(question) : undefined
  const forcedCalculation = routeMode === 'tool' ? verifiedCalculationFromQuestion(question, sourceExcerpts) : undefined
  const forcedFacts = forcedSeries ? verifiedSeriesFacts(forcedSeries)
    : forcedCalculation ? verifiedCalculationTeachingFacts(forcedCalculation) : ''
  const messages: StudyChatMessage[] = [
    { role: 'system', content: routeMode === 'tool' ? [
      'You have one computation tool named javascript_interpret. For the next response, emit only a JSON tool call with {"tool":"javascript_interpret","code":"..."}.',
      'Write self-contained synchronous JavaScript. Return a value with the final expression or print it with console.log(...). Console output is captured as the tool result. Use a separate console.log call for each output line; do not put newline escapes inside quoted JavaScript strings. Do not use imports, require, fetch, process, or Node APIs.',
      'Use the source material and current question to choose the correct values and population. Check signs, loop bounds, divisors, and requested rounding before calling the tool. Make the tool output include every requested computed value, including each item in any requested list and intermediate totals used for aggregates, with labels and units. A single aggregate is insufficient when a list is requested. You may repair a failed or incomplete execution when feedback is returned.',
      forcedFacts ? `Independently verified inputs and method for this question:\n${forcedFacts}\nCheck your JavaScript output against these values.` : ''
    ].join(' ') : [
      'You may use one isolated JavaScript computation tool. Choose whether it is needed for this question. Return exactly one JSON object with route and content.',
      'Set route to answer for source fact lookup, identifying or listing facts, counting items already named in a source, or one simple arithmetic operation. Content is then the complete student-facing Markdown answer, including every requested item and count.',
      'For questions asking for multiple derived numeric values or an aggregate such as a sum, average, percentage, weighted result, invoice total, or conversion from supplied numbers, set route to javascript_interpret even if you could calculate mentally. Content is then raw, self-contained synchronous JavaScript. Do not wrap code in Markdown fences; never leave content empty.',
      'JavaScript must print every requested computed item and final value with labels and units. Use a separate console.log call for each output line. Check signs, loop bounds, divisors, and rounding. Do not use imports, require, fetch, process, or Node APIs. Use source material for facts; never invent source facts in code.'
    ].join(' ') },
    ...initialMessages
  ]
  const trace: JavascriptInterpretTrace = { attempts: [], inferenceCount: 0, retryCount: 0, latencyMs: 0, inferenceLatenciesMs: [] }

  async function timedInference(format?: Record<string, unknown>, maxTokens?: number): Promise<string> {
    const inferenceStarted = performance.now()
    trace.inferenceCount += 1
    try {
      return await infer(messages, format, maxTokens)
    } finally {
      trace.inferenceLatenciesMs.push(performance.now() - inferenceStarted)
    }
  }

  try {
    const firstResponse = await timedInference(
      (routeMode === 'tool' ? javascriptToolSchema : javascriptChoiceSchema) as unknown as Record<string, unknown>, 1024)
    const choice = JSON.parse(firstResponse) as { route?: unknown; content?: unknown; tool?: unknown; code?: unknown }
    if (routeMode === 'choice' && choice.route === 'answer') {
      if (typeof choice.content !== 'string' || !choice.content.trim()) {
        throw new Error('The model chose a direct answer but returned no answer text.')
      }
      trace.directAnswer = true
      return choice.content.trim()
    }
    const firstCode = choice.route === 'javascript_interpret' ? choice.content
      : choice.tool === 'javascript_interpret' ? choice.code : undefined
    if (typeof firstCode !== 'string') throw new Error('Expected an answer or a javascript_interpret tool call.')
    const verifiedSeries = forcedSeries ?? verifiedSeriesFromQuestion(question)
    const verifiedCalculation = forcedCalculation ?? verifiedCalculationFromQuestion(question, sourceExcerpts)
    trace.verifiedSeriesOperation = verifiedSeries?.operation
    trace.verifiedCalculationOperation = verifiedCalculation?.operation
    const verifiedInstruction = verifiedSeries
      ? `Independently verified inputs and method for this question:\n${verifiedSeriesFacts(verifiedSeries)}`
      : verifiedCalculation
        ? `Independently verified inputs and method for this question:\n${verifiedCalculationTeachingFacts(verifiedCalculation)}`
        : ''
    const verifiedFinalInstruction = verifiedSeries
      ? `Authoritative verified inputs, formula, substitutions, intermediate results, units, and conclusion:\n${verifiedSeriesFacts(verifiedSeries)}\n\nWrite a clear student-facing explanation in Markdown. Explain the formula, show how the supplied inputs enter each step, and explain how the final mean follows from the verified changes. Prefer readable arithmetic; use KaTeX only with valid math delimiters and single-backslash commands. Keep every verified value and unit unchanged. Do not recalculate, omit requested values, or introduce unsupported numbers or assumptions. You are writing the final answer, not another tool call.`
      : verifiedCalculation
        ? `Authoritative verified inputs, method, intermediate results, units, and conclusion:\n${verifiedCalculationTeachingFacts(verifiedCalculation)}\n\nWrite a clear student-facing explanation in Markdown. Explain the method or formula, use the verified inputs to walk through the intermediate steps, and state the verified conclusion. Prefer readable arithmetic; use KaTeX only with valid math delimiters and single-backslash commands. Keep every verified value and unit unchanged. Do not recalculate, omit requested values, or introduce unsupported numbers or assumptions. You are writing the final answer, not another tool call.`
        : ''
    for (let attempt = 0; attempt <= maxExecutionRetries; attempt += 1) {
      const response = attempt === 0 ? firstResponse
        : await timedInference(javascriptToolSchema as unknown as Record<string, unknown>, 1024)
      let code = ''
      let execution: JavascriptExecution
      try {
        const call = attempt === 0 ? { tool: 'javascript_interpret', code: firstCode }
          : JSON.parse(response) as { tool?: unknown; code?: unknown }
        if (call.tool !== 'javascript_interpret' || typeof call.code !== 'string') {
          throw new Error('Expected a javascript_interpret call with a code string.')
        }
        const trimmed = call.code.trim()
        const fenced = trimmed.match(/^```(?:javascript|js)?\s*\n([\s\S]*?)\n```$/i)
        code = fenced ? fenced[1] : trimmed
        execution = await execute(code)
      } catch (error) {
        execution = { ok: false, error: error instanceof Error ? error.message : String(error), latencyMs: 0 }
      }

      const outputIssue = execution.ok
        ? missingRequestedList(question, execution.result)
          ? 'The output does not include every requested list item.'
          : outputPrecisionIssue(question, execution.result) ??
            (verifiedSeries && !verifiedSeriesOutputMatches(verifiedSeries, execution.result)
              ? 'The executed values do not match the independently calculated adjacent results and aggregate.' :
              verifiedCalculation && !verifiedCalculationOutputMatches(verifiedCalculation, execution.result)
                ? 'The executed values do not match the source-grounded calculation.' : undefined)
        : undefined
      const incompleteResult = Boolean(outputIssue)
      trace.attempts.push({ code, execution, retryCount: attempt,
        ...(incompleteResult ? { incompleteResult: true, outputIssue } : {}) })
      trace.retryCount = attempt
      messages.push({ role: 'assistant', content: response })
      const failedOutput = !execution.ok && execution.partialOutput
        ? `\nOutput before failure:\n${execution.partialOutput}`
        : ''
      const failedListHint = !execution.ok && execution.partialOutput &&
        missingRequestedList(question, execution.partialOutput)
        ? '\nThe numeric list has fewer items than requested. Check the loop bounds and whether the final input pair was processed.'
        : ''
      const failedStringHint = !execution.ok && /unexpected end of string/i.test(execution.error)
        ? '\nIf you need multiple output lines, use separate console.log calls instead of a newline inside a quoted string.'
        : ''
      messages.push({ role: 'user', content: execution.ok
        ? incompleteResult
          ? `javascript_interpret result:\n${execution.result}\n\n${outputIssue}\n${verifiedInstruction} Generate a revised javascript_interpret call that prints every requested item, any intermediate values used for the aggregate, and the aggregate with the requested formatting. Return only the JSON tool call.`
          : verifiedFinalInstruction
            ? `javascript_interpret result:\n${execution.result}\n\n${verifiedFinalInstruction}`
          : `javascript_interpret result:\n${execution.result}\n\nThe tool succeeded. Your final answer must repeat the tool result verbatim and contain no other text. Do not explain, reformat, omit values, or call the tool again.`
        : `javascript_interpret error: ${execution.error}${failedOutput}${failedListHint}${failedStringHint}\n${verifiedInstruction}` })

      if (execution.ok && !incompleteResult) {
        if (verifiedFinalInstruction) {
          const finalAnswer = normalizeMathCommands(await timedInference(undefined, 768))
          if (verifiedSeries) {
            const unsupported = unsupportedFinalNumbers(finalAnswer, question, verifiedSeriesFacts(verifiedSeries))
            if (finalAnswer.trim() && unsupported.length === 0 &&
              verifiedSeriesAnswerMatches(verifiedSeries, finalAnswer)) return finalAnswer.trim()
            trace.rejectedFinalAnswer = finalAnswer
            trace.unsupportedFinalNumbers = unsupported.length ? unsupported : undefined
            trace.verifiedSeriesFallback = true
            return verifiedSeriesFacts(verifiedSeries)
          }
          if (verifiedCalculation) {
            const supportedFacts = `${verifiedCalculationTeachingFacts(verifiedCalculation)}\n${(verifiedCalculation.allowedExplanatoryNumbers ?? []).join(', ')}`
            const unsupported = unsupportedFinalNumbers(finalAnswer, `${question}\n${sourceExcerpts.join('\n')}`, supportedFacts)
            if (finalAnswer.trim() && unsupported.length === 0 &&
              verifiedCalculationAnswerMatches(verifiedCalculation, finalAnswer)) return finalAnswer.trim()
            trace.rejectedFinalAnswer = finalAnswer
            trace.unsupportedFinalNumbers = unsupported.length ? unsupported : undefined
            trace.verifiedCalculationFallback = true
            return verifiedCalculationTeachingFacts(verifiedCalculation)
          }
        }
        const finalAnswer = await timedInference(undefined, 256)
        if (!finalAnswer.trim()) {
          trace.emptyFinalAnswer = true
          return formatComputedResult(execution.result, code)
        }
        const unsupported = unsupportedFinalNumbers(finalAnswer, question, execution.result)
        if (unsupported.length > 0) {
          trace.rejectedFinalAnswer = finalAnswer
          trace.unsupportedFinalNumbers = unsupported
          return formatComputedResult(execution.result, code)
        }
        if (missingRequestedList(question, finalAnswer)) {
          trace.rejectedFinalAnswer = finalAnswer
          trace.finalMissingRequestedList = true
          return formatComputedResult(execution.result, code)
        }
        if (missingComputedList(finalAnswer, execution.result)) {
          trace.rejectedFinalAnswer = finalAnswer
          trace.finalMissingComputedList = true
          return formatComputedResult(execution.result, code)
        }
        if (finalAnswer.trim() !== execution.result.trim()) {
          trace.rejectedFinalAnswer = finalAnswer
          trace.finalOutputMismatch = true
          return formatComputedResult(execution.result, code)
        }
        return formatComputedResult(finalAnswer, code)
      }
      if (attempt < maxExecutionRetries && !incompleteResult) {
        messages.push({ role: 'user', content: 'Repair the JavaScript and call javascript_interpret again. Return only the JSON tool call.' })
      }
    }
    const lastExecution = trace.attempts.at(-1)?.execution
    if (verifiedSeries) {
      trace.verifiedSeriesFallback = true
      return verifiedSeriesFacts(verifiedSeries)
    }
    if (verifiedCalculation) {
      trace.verifiedCalculationFallback = true
      return verifiedCalculationTeachingFacts(verifiedCalculation)
    }
    if (trace.attempts.at(-1)?.incompleteResult) {
      throw new Error(`javascript_interpret produced incomplete output after ${maxExecutionRetries + 1} attempts: ${trace.attempts.at(-1)?.outputIssue}`)
    }
    throw new Error(`javascript_interpret failed after ${maxExecutionRetries + 1} attempts: ${lastExecution && !lastExecution.ok ? lastExecution.error : 'unknown error'}`)
  } finally {
    trace.latencyMs = performance.now() - started
    onTrace(trace)
  }
}
