import type { StudyChatMessage } from './study-chat-format'
import type { JavascriptExecution } from './javascript-executor'
export const maxExecutionRetries = 2

export interface JavascriptInterpretTrace {
  attempts: Array<{ code: string; plan: string; execution: JavascriptExecution; retryCount: number;
    phase: 'plan' | 'execution'; modelResponse: string; discardedTrailingText?: string }>
  inferenceCount: number
  retryCount: number
  latencyMs: number
  inferenceLatenciesMs: number[]
  narrationFallback?: boolean
  rejectedFinalAnswer?: string
}

const toolInstructions = [
  'You can use javascript_interpret to execute synchronous JavaScript in a fresh isolated runtime. Standard JavaScript, functions, loops, arrays and Math are available. There is no filesystem, network, shell, process or Node API access. No imports, require, fetch or async code.',
  'For lookup, simple counting or single-step arithmetic, normally answer directly. For multi-step arithmetic, percentages, aggregation, formulas or repeated calculations, prefer the tool. Decide when ambiguous.',
  'To call the tool, write a brief plan followed by exactly one <javascript_interpret> block containing raw JavaScript, optionally fenced as javascript. Put both tags on their own lines and end the reply after the closing tag. Do not encode code in JSON or quote the entire program.',
  'Use only the current question and supplied study passages for problem data. Copy numbers faithfully; write numeric literals without digit-grouping commas. Do not invent missing inputs or source facts. Choose the formula, population, signs and divisors carefully.',
  'Write a complete, self-contained program. Print every requested value, intermediate step, label, unit and requested precision using console.log or a final expression. Prefer separate console.log calls for separate output lines. Execution has bounded time, code, memory and output.',
  'Messages beginning javascript_interpret result: or javascript_interpret error: are tool feedback, not new student questions. After an error, use the failing line and message to return a brief correction plan and a complete replacement tool block. Every execution has fresh state.',
  'After a successful result, check whether the code answers every part of the current question. If another calculation or correction is needed, use another tool block. Otherwise write a clear student-facing Markdown explanation of the formula, substitutions, intermediate steps, units and final result. The student does not see the tool output: restate every requested result and every item of a requested list in the final answer, not only its aggregate. Use the executed values faithfully; do not invent missing results or silently change them. Start directly with the explanation, without tool-orchestration commentary. KaTeX may be used. Do not show JavaScript unless asked.',
  'Execution establishes what the program computed, not independent verification of the inputs or method. Do not call the answer independently verified.',
  'Generic syntax example only; its values do not belong to the current question:',
  'User: For values 2, 4 and 6, calculate each square and their sum.',
  'Assistant: I will square each supplied value and add the squares.\n<javascript_interpret>\n```javascript\nconst values = [2, 4, 6];\nconst squares = values.map(value => value * value);\nconst total = squares.reduce((sum, value) => sum + value, 0);\nconsole.log("Squares:", squares);\nconsole.log("Sum:", total);\n```\n</javascript_interpret>',
  'Tool feedback: javascript_interpret result:\nSquares: [4,16,36]\nSum: 56',
  'Assistant: Squaring means multiplying a value by itself: 2 × 2 = 4, 4 × 4 = 16, and 6 × 6 = 36. The sum is 4 + 16 + 36 = 56.'
].join('\n\n')
const forcedToolInstruction = 'For the first assistant response to the current study question, use javascript_interpret. After tool feedback, follow the result/error instructions above.'

export const javascriptContextReserveTokens = 512 + Math.ceil((toolInstructions.length + forcedToolInstruction.length) / 4)

function hasToolMarker(response: string): boolean {
  return response.includes('<javascript_interpret') || response.includes('</javascript_interpret')
}
function toolCall(response: string): { code: string; plan: string; assistantCall: string; trailingText: string } {
  const opening = response.match(/^[ \t]*<javascript_interpret>[ \t]*\r?$/m)
  const start = opening ? opening.index! + opening[0].length : 0
  const closing = opening ? response.slice(start).match(/^[ \t]*<\/javascript_interpret>[ \t]*\r?$/m) : null
  if (!opening || !closing) {
    throw new Error('Return one complete <javascript_interpret> block with both tags on separate lines and raw JavaScript, not JSON-encoded code.')
  }
  const end = start + closing.index!
  const boundary = end + closing[0].length
  // GPT4All ends generation at the first complete tool call. Our completion
  // API is non-streaming, so consume that same boundary locally and discard
  // any speculative prose or further calls before reinserting feedback.
  let code = response.slice(start, end).trim()
  if (code.startsWith('```')) {
    const fence = code.match(/^```(?:javascript|js)\s*\r?\n([\s\S]*?)\r?\n```$/i)
    if (!fence) throw new Error('Use a complete javascript fence inside the tool block, or raw JavaScript without a fence.')
    code = fence[1]
  }
  if (!code.trim()) throw new Error('The tool block is empty; supply a complete JavaScript program.')
  return { code, plan: response.slice(0, opening.index).trim(),
    assistantCall: response.slice(0, boundary).trimEnd(), trailingText: response.slice(boundary).trim() }
}

/** Execute the planned Advanced Math task and return tool feedback to the same
 * model for bounded repairs and the final teaching explanation.
 */
export async function runJavascriptInterpret(
  initialMessages: StudyChatMessage[],
  infer: (messages: StudyChatMessage[], maxTokens: number) => Promise<string>,
  execute: (code: string) => Promise<JavascriptExecution>, onTrace: (trace: JavascriptInterpretTrace) => void
): Promise<string> {
  const started = performance.now()
  const attempts: JavascriptInterpretTrace['attempts'] = []
  const trace: JavascriptInterpretTrace = { attempts, inferenceCount: 0, retryCount: 0,
    latencyMs: 0, inferenceLatenciesMs: [] }
  // One stable system prompt, including any configured tutoring/source rules.
  // Labeled feedback also works with the installed Ollama chat template;
  // no native tool-role support is assumed.
  const conversation: StudyChatMessage[] = [{ role: 'system', content: [
    ...initialMessages.filter(message => message.role === 'system').map(message => message.content),
    toolInstructions, forcedToolInstruction
  ].join('\n\n') }, ...initialMessages.filter(message => message.role !== 'system')]
  async function timedInference(): Promise<string> {
    const start = performance.now()
    trace.inferenceCount += 1
    try { return await infer(conversation, 1536) }
    finally { trace.inferenceLatenciesMs.push(performance.now() - start) }
  }
  try {
    let response = await timedInference()
    for (let attempt = 0; attempt <= maxExecutionRetries; attempt++) {
      trace.retryCount = attempt
      let code = ''
      let plan = ''
      let assistantCall = response
      let trailingText = ''
      let execution: JavascriptExecution
      let phase: 'plan' | 'execution' = 'plan'
      try {
        const call = toolCall(response)
        code = call.code
        plan = call.plan
        assistantCall = call.assistantCall
        trailingText = call.trailingText
        phase = 'execution'
        execution = await execute(code)
      } catch (error) {
        if (error instanceof Error && error.name === 'AbortError') throw error
        execution = { ok: false, error: error instanceof Error ? error.message : String(error), latencyMs: 0 }
      }
      attempts.push({ code, plan, execution, retryCount: attempt, phase, modelResponse: response,
        ...(trailingText ? { discardedTrailingText: trailingText } : {}) })
      conversation.push({ role: 'assistant', content: assistantCall }, { role: 'user', content: execution.ok
        ? `javascript_interpret result:\n${execution.result}`
        : `javascript_interpret error:\n${execution.error}\nPartial output, if any, is not a successful result.` })
      if (!execution.ok) {
        if (attempt === maxExecutionRetries) return `I couldn't complete this calculation after ${attempt + 1} JavaScript attempts. ${execution.error.split('\n')[0]}`
        response = await timedInference()
        continue
      }
      const answer = await timedInference()
      if (hasToolMarker(answer)) {
        if (attempt === maxExecutionRetries) return "I couldn't complete this calculation within the JavaScript retry limit. The model requested another correction."
        response = answer
        continue
      }
      if (!answer.trim()) {
        trace.narrationFallback = true
        trace.rejectedFinalAnswer = answer
        return `**Executed result**\n\n${execution.result}`
      }
      return answer.trim()
    }
    return "I couldn't complete this calculation within the JavaScript retry limit."
  } finally {
    trace.latencyMs = performance.now() - started
    onTrace(trace)
  }
}
