import type { StudyChatMessage } from './study-chat-format'
import type { JavascriptExecution } from './javascript-executor'
export const maxExecutionRetries = 2

export interface JavascriptInterpretTrace {
  attempts: Array<{ code: string; plan: string; execution: JavascriptExecution; retryCount: number;
    phase: 'plan' | 'execution'; modelResponse: string; discardedTrailingText?: string; recoveredTransport?: boolean;
    reusedExecution?: boolean }>
  inferenceCount: number
  retryCount: number
  latencyMs: number
  inferenceLatenciesMs: number[]
  narrationFallback?: boolean
  rejectedFinalAnswer?: string
  narrationIssues?: string[]
  validationLatencyMs?: number
  informationRequest?: string
  unchangedCorrections?: number
  teachingOutput?: boolean
  teachingIssues?: string[]
  omittedTeachingSteps?: number
}

const toolInstructions = [
  'Advanced Math has two phases: calculate with javascript_interpret, then teach from its execution output. On the initial turn, return a brief plan and one tool call, or needs_information if inputs are unresolved. After successful execution, normally give the final explanation; do not restart the task or repeat the program.',
  'The tool runs synchronous JavaScript in a fresh isolated runtime: functions, loops, arrays, Math and console.log are available. No filesystem, network, shell, process, Node APIs, imports, require, fetch or async code. Time, memory, code and output are bounded.',
  'Tool syntax: <javascript_interpret> on its own line, raw JavaScript (optionally fenced as javascript), then </javascript_interpret> on its own line. End the reply there. Never JSON-escape the program, simulate tool feedback or answer using unexecuted calculations.',
  'Use only the current question and supplied passages for data. Copy numbers faithfully without digit-grouping commas in literals. Preserve observation order and multiplicity, including repeated values. Distinguish independent observations from incompatible versions of the same observations. Do not pool conflicting versions or invent authority. If required inputs or assumptions are unresolved, return <needs_information>A concise clarification explaining what is missing or conflicting.</needs_information>. Compute separate scenarios only when requested.',
  'Choose the method, signs and divisors carefully. Calculate EVERY requested individual result and aggregate. For teaching, finish with one console.log({method, inputs, steps, results}). method is a short explanation including the formula; inputs contains the supplied data; steps is an array of strings showing each substitution and its computed result; results is an object with descriptive labels and formatted values, including units and requested precision. Build step strings from the same variables used in the calculation, not handwritten answers. In substitutions, print the actual numeric operands, not only variable names or positional labels. Preserve parentheses and operator precedence. Include the aggregate steps, their intermediate totals and counts/divisors, as well as the individual steps. Print every requested result in results. Use source labels or positions; never invent dates or other names. This object is output data, not JSON-escaped tool code.',
  'javascript_interpret error: is execution feedback. Correct the reported failing line and return a complete replacement program; state what changed. Partial output is not a successful result. Every attempt has fresh state.',
  'javascript_interpret result: is completed execution feedback, not a new student question. If evidence selection is unresolved, use needs_information and withhold the result. If a requested result is missing or an input/method is wrong, return a genuinely changed replacement program, explaining the correction. Otherwise write the final student-facing Markdown explanation now.',
  'In the final explanation, teach why the printed method is used, show the printed substitutions, restate every requested result and list item, and conclude with the executed result. Keep the printed definition/formula intact. Describe operand order consistently with the displayed substitutions; do not reverse the meaning in prose. Copy each numerical substitution from the feedback; your job is to explain it, not derive new arithmetic. Execution output is authoritative: do not recalculate or change its values, signs, precision or supported units. Use the question/source labels only; otherwise refer to positions. Do not claim independent verification. KaTeX may be used; escape currency dollars as \\$ to distinguish them from math delimiters. Start with the explanation, without orchestration commentary. Do not show JavaScript unless asked.',
  'Generic syntax example only; these values are not current question data:',
  'User: For values 2, 4 and 6, calculate each square, their sum and their mean.',
  'Assistant: I will multiply each value by itself, add the squares, and divide the sum by their count.\n<javascript_interpret>\nconst values = [2, 4, 6];\nconst squares = values.map(value => value * value);\nconst total = squares.reduce((sum, value) => sum + value, 0);\nconst count = squares.length;\nconst mean = (total / count).toFixed(2);\nconst steps = values.map((value, i) => value + " × " + value + " = " + squares[i]);\nsteps.push(squares.join(" + ") + " = " + total);\nsteps.push("(" + squares.join(" + ") + ") ÷ " + count + " = " + total + " ÷ " + count + " = " + mean);\nconsole.log({method: "Square each value, add the squares, then divide by the count for their mean.", inputs: values, steps, results: {squares, sum: total, count, mean}});\n</javascript_interpret>',
  'Tool feedback: javascript_interpret result:\nMethod: Square each value, add the squares, then divide by the count for their mean.\nInputs: 2, 4, 6\nSteps:\n2 × 2 = 4\n4 × 4 = 16\n6 × 6 = 36\n4 + 16 + 36 = 56\n(4 + 16 + 36) ÷ 3 = 56 ÷ 3 = 18.67\nResults:\nSquares: 4, 16, 36\nSum: 56\nCount: 3\nMean: 18.67',
  'Assistant: Squaring multiplies a value by itself: 2 × 2 = 4, 4 × 4 = 16, and 6 × 6 = 36. Adding all three squares gives 4 + 16 + 36 = 56. Their mean is the sum divided by the three results: 56 ÷ 3 ≈ 18.67.'
].join('\n\n')

export const javascriptContextReserveTokens = 512 + Math.ceil(toolInstructions.length / 4)

function hasToolMarker(response: string): boolean {
  return response.includes('<javascript_interpret') || response.includes('</javascript_interpret') || /^```(?:javascript|js)\b/im.test(response)
}
function informationRequest(response: string): string | undefined {
  const match = response.trim().match(/<needs_information>\s*([\s\S]+?)\s*<\/needs_information>$/)
  return match && !hasToolMarker(response) ? match[1].trim() || undefined : undefined
}

function toolCall(response: string): { code: string; plan: string; assistantCall: string; trailingText: string; recoveredTransport?: boolean } {
  const opening = response.match(/^[ \t]*<javascript_interpret>[ \t]*\r?$/m)
  const start = opening ? opening.index! + opening[0].length : 0
  const closing = opening ? response.slice(start).match(/^[ \t]*<\/javascript_interpret>[ \t]*\r?$/m) : null
  if (!opening && !response.includes('<javascript_interpret')) {
    // Recover complete, explicitly JavaScript fences, including a program split
    // across fences. Never execute prose or model-invented feedback. An opening
    // tool tag without a closing tag remains an incomplete call, not recovery.
    const boundary = response.search(/<\/javascript_interpret>|^\s*(?:Tool feedback:|javascript_interpret (?:result|error):|Assistant:)/im)
    const head = boundary < 0 ? response : response.slice(0, boundary)
    const fences = [...head.matchAll(/^```(?:javascript|js)[ \t]*\r?\n([\s\S]*?)^```[ \t]*\r?$/gim)]
    const fenceCount = (head.match(/^```/gm) ?? []).length
    if (fences.length && fenceCount === fences.length * 2) {
      const code = fences.map(fence => fence[1].trim()).join('\n')
      const plan = head.slice(0, fences[0].index).trim()
      const end = fences.at(-1)!.index! + fences.at(-1)![0].length
      if (code) return { code, plan, assistantCall: `${plan}\n<javascript_interpret>\n${code}\n</javascript_interpret>`.trim(),
        trailingText: response.slice(end).trim(), recoveredTransport: true }
    }
  }
  if (!opening || !closing) {
    throw new Error('Return one complete raw JavaScript tool block, not prose or JSON. Format:\n<javascript_interpret>\nconsole.log(/* computed results */);\n</javascript_interpret>\nIf inputs are missing or conflicting, use <needs_information>Explain what is needed.</needs_information>.')
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

const numericToken = /(?<![\p{L}\p{N}_.])[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?(?![\p{L}\p{N}_])/giu
function numericText(text: string): string {
  return text.replace(/[−–]/g, '-').replace(/(?<=\d),(?=\d{3}(?:\D|$))/g, '')
}
function numbers(text: string): number[] {
  return [...numericText(text).matchAll(numericToken)].map(match => Number(match[0])).filter(Number.isFinite)
}
function matchesNumber(left: number, right: number): boolean {
  // Permit displayed rounding to cents/two decimals, never a one-cent change.
  return Math.abs(left - right) < 0.005 + Number.EPSILON * Math.max(1, Math.abs(left), Math.abs(right))
}

function balancedExpression(part: string): string {
  let text = part.trim().replace(/\.$/, '')
  // A prose parenthesis can wrap an entire equality rather than a single
  // operand. Remove only those unmatched boundary parentheses, not arithmetic.
  let balance = [...text].reduce((sum, char) => sum + (char === '(' ? 1 : char === ')' ? -1 : 0), 0)
  while (balance > 0 && text.startsWith('(')) { text = text.slice(1).trim(); balance-- }
  while (balance > 0 && text.endsWith('(')) { text = text.slice(0, -1).trim(); balance-- }
  while (balance < 0 && text.endsWith(')')) { text = text.slice(0, -1).trim(); balance++ }
  return text
}

function resultNumbers(result: string): number[] {
  try { return numbers(JSON.stringify(JSON.parse(result))) } catch { /* Plain console output. */ }
  return result.split('\n').flatMap(line => {
    // Labels and formula operands are not separate result claims. Require the
    // displayed right-hand result, or the values in a plain result/list line.
    const value = line.slice(line.indexOf(':') + 1).trim()
    const rhs = value.includes('=') ? value.slice(value.lastIndexOf('=') + 1).trim() : value
    if (/[*/×÷^]|\d\s*[+−-]\s*\d/.test(rhs)) return []
    return numbers(rhs)
  })
}

function requestedPrecision(question: string): number | undefined {
  const words = ['zero', 'one', 'two', 'three', 'four', 'five', 'six']
  const match = question.match(/\b(\d|zero|one|two|three|four|five|six) decimal places?\b/i)
  const count = match ? (/^\d$/.test(match[1]) ? Number(match[1]) : words.indexOf(match[1].toLowerCase())) : undefined
  return count !== undefined && count <= 6 ? count : undefined
}

interface TeachingOutput {
  method: string
  inputs: unknown
  steps: string[]
  results: Record<string, unknown>
}

function teachingOutput(result: string): TeachingOutput | undefined {
  try {
    const value = JSON.parse(result)
    const results = value?.results ?? value?.result
    if (value && typeof value.method === 'string' && value.method.trim() && 'inputs' in value &&
      Array.isArray(value.steps) && value.steps.every((step: unknown) => typeof step === 'string' && step.trim()) &&
      results && typeof results === 'object' && !Array.isArray(results) && Object.keys(results).length) return { ...value, results }
  } catch { /* Plain output remains supported; do not reject a working program for its layout. */ }
  return undefined
}

function displayLines(value: unknown, label = ''): string[] {
  if (Array.isArray(value)) {
    return value.every(item => typeof item !== 'object' || item === null)
      ? [`${label}${value.map(String).join(', ')}`] : value.flatMap(child => displayLines(child, label))
  }
  if (value && typeof value === 'object') return Object.entries(value).flatMap(([key, child]) =>
    displayLines(child, `${label}${key.replace(/_/g, ' ')}: `))
  return [`${label}${String(value)}`]
}

function teachingText(output: TeachingOutput): string {
  return [
    `**Method:** ${output.method}`,
    '**Inputs:**\n' + displayLines(output.inputs).map(line => `- ${line}`).join('\n'),
    ...(output.steps.length ? ['**Worked steps:**\n' + output.steps.map(step => `- ${step}`).join('\n')] : []),
    '**Results:**\n' + displayLines(output.results).map(line => `- ${line}`).join('\n')
  ].join('\n\n').replace(/\$(?=\d)/g, '\\$')
}

function arithmeticClaims(answer: string, labels: string[]): string[][] {
  let text = numericText(answer)
    // Keep each math span separate from surrounding prose numbers/parentheses.
    .replace(/\\\(([\s\S]*?)\\\)|\\\[([\s\S]*?)\\\]|\$\$([\s\S]*?)\$\$/g,
      (_match, inline: string, display: string, dollars: string) => `\n${inline ?? display ?? dollars}\n`)
    .replace(/\*\*|__/g, '').replace(/^\s*(?:[-*]\s+|\d+[.)]\s+)/gm, '')
    .replace(/\\(?:left|right|,|;|!)/g, '').replace(/\\(?:times|cdot)/g, '*').replace(/\\div/g, '/')
    .replace(/\\%/g, '%')
    .replace(/[×·]/g, '*').replace(/÷/g, '/').replace(/\\text\{[^{}]*\}/g, '')
  for (let depth = 0; depth < 16 && /\\frac/.test(text); depth++) {
    const next = text.replace(/\\frac\{([^{}]+)\}\{([^{}]+)\}/g, '(($1)/($2))')
    if (next === text) break
    text = next
  }
  text = text.replace(/\\[$()[\]]|[$`{}]/g, '').replace(/(?<=\d)\s*°[\p{L}]+/gu, '')
  // Only literal numeric arithmetic is checked, not algebra, prose, or a chosen
  // textbook formula. Restricted characters are evaluated solely in QuickJS.
  return text.split('\n').flatMap(line => [...line.matchAll(
    /[+(\-\d][\d.\s()+*/%^-]*(?:[=≈]\s*[+(\-\d][\d.\s()+*/%^-]*)+/g
  )].map(match => {
    const parts = match[0].split(/[=≈]/).map(balancedExpression)
    // In "Reading 2 = 18 - 12 = 6", the 2 belongs to the prose label.
    // Start with the arithmetic RHS, retaining all arithmetic equality checks.
    const lhs = (line.slice(0, match.index) + parts[0]).trim().toLowerCase()
    if (parts.length > 1 && /^[-+]?\d+(?:\.\d+)?$/.test(parts[0]) &&
      labels.some(label => lhs.endsWith(label.toLowerCase()))) parts.shift()
    return parts
  }))
    .filter(parts => parts.length > 1 && parts.every(part => /\d/.test(part)) && parts.some(part => /[*/^%]|\d\s*[+-]/.test(part)))
}

function outputFacts(result: string): Array<[string, number]> {
  const facts: Array<[string, number]> = []
  function scalar(text: string): number | undefined {
    const match = numericText(text).match(/^\s*[$€£¥]?\s*([-+]?\d+(?:\.\d+)?)(?:\s*(?:%|°?[\p{L} ]+))?\s*[.!]?$/u)
    return match ? Number(match[1]) : undefined
  }
  function line(text: string): void {
    const match = text.match(/^\s*(?:[-*]\s+)?([^:=\n]+)\s*[:=]\s*(.+)$/)
    if (!match || !/\p{L}/u.test(match[1])) return
    const value = scalar(match[2].split(/[=≈]/).at(-1)!)
    if (value !== undefined) facts.push([numericText(match[1].trim()), value])
  }
  function collect(value: unknown, name = ''): void {
    if (typeof value === 'number' && name) facts.push([numericText(name), value])
    else if (typeof value === 'string' && name) {
      const number = scalar(value)
      if (number !== undefined) facts.push([numericText(name), number])
    } else if (value && typeof value === 'object' && !Array.isArray(value)) {
      for (const [key, child] of Object.entries(value)) collect(child, key.replace(/_/g, ' '))
    }
  }
  const teaching = teachingOutput(result)
  if (teaching) { collect(teaching.results); teaching.steps.forEach(line) }
  else try { collect(JSON.parse(result)) } catch { result.split('\n').forEach(line) }
  return facts
}

function contradictsNamedResult(answer: string, result: string): boolean {
  const facts = outputFacts(result)
  const text = numericText(answer).replace(/\*\*|__|`|\\[()[\]]|\$\$/g, '').replace(/\\([$€£¥])/g, '$1')
  return facts.some(([label, value]) => {
    // Compare an output's own label, including the final value of a substitution
    // chain. This binds printed positional labels without defining their meaning.
    if (facts.some(([other, number]) => other === label && number !== value)) return false
    const escaped = label.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
    const pattern = new RegExp(`(?<![\\p{L}\\p{N}])${escaped}[ \\t]*(?::|=|\\bis\\b|\\bequals\\b)[ \\t]*([^\\n]+)`, 'gimu')
    return [...text.matchAll(pattern)].some(match => {
      const assertion = match[1].split(/;|\.(?=\s+\p{L})/u)[0]
      const rhs = assertion.split(/[=≈]/).at(-1)!.trim()
      const asserted = rhs.match(/^[$€£¥]?\s*([-+]?\d+(?:\.\d+)?)(?=[ \t]*(?:[%°\p{L}.,;]|$))/u)
      return asserted && !matchesNumber(Number(asserted[1]), value)
    })
  })
}

async function checkNarration(answer: string, result: string, evidence: string, code: string,
  execute: (code: string) => Promise<JavascriptExecution>, question: string
): Promise<{ answer: string; issues: string[]; invalidClaims: string[]; verifiedClaims: string[] }> {
  // Restore explicit trailing-zero precision from output, without rewriting
  // non-equivalent numbers or imposing a layout/heading/template on narration.
  const displays = [...numericText(result).matchAll(numericToken)].map(match => match[0])
    .filter(value => /\.\d*0$/.test(value))
  answer = answer.replace(new RegExp(numericToken.source.replace('[-+]?', '[-+−]?'), numericToken.flags), (value, offset: number) => {
    if (answer[offset - 1] === ',' || answer[offset + value.length] === ',') return value
    if (/^\s*[)\]}+*/×÷^−-]/.test(answer.slice(offset + value.length))) return value
    return displays.find(display => Number(numericText(value)) === Number(display) && display.length > value.length) ?? value
  })
  const precision = requestedPrecision(question)
  if (precision !== undefined) {
    // Formatting a displayed percentage does not change a computation. Leave
    // the percent multiplier inside a formula alone.
    answer = answer.replace(/(?<![\p{L}\p{N}_.])[-+−]?\d+(?:\.\d+)?\s*%/gu, (display, offset: number) => {
      if (/[×*]\s*$/.test(answer.slice(0, offset))) return display
      const value = Number(numericText(display.replace('%', '').trim()))
      return `${value.toFixed(precision)}%`
    })
  }
  const issues: string[] = []
  const invalidClaims: string[] = []
  const verifiedClaims: string[] = []
  const teaching = teachingOutput(result)
  const resultValues = teaching ? JSON.stringify(teaching.results) : result
  const claims = arithmeticClaims(answer, outputFacts(result).map(([label]) => label))
  const derived: number[] = []
  const supportedOperands = numbers(evidence + '\n' + code + '\n' + result)
  if (claims.length > 64) {
    issues.push('Too many arithmetic claims to check within the validation limit.')
    invalidClaims.push(...claims.map(claim => claim.join(' = ')))
  }
  else if (claims.length) {
    const raw = claims.map(parts => parts.map(part => part.replace(/%/g, '').replace(/\^/g, '**')))
    const scaled = claims.map(parts => parts.map(part => part.replace(/(\d+(?:\.\d*)?)%/g, '($1/100)').replace(/\^/g, '**')))
    const validation = await execute(`console.log(JSON.stringify([${[...raw, ...scaled].map(parts =>
      `[${parts.map(part => `(${part})`).join(',')}]`).join(',')}]))`)
    if (!validation.ok) {
      issues.push('A numeric equation could not be checked: ' + validation.error.split('\n')[0])
      invalidClaims.push(...claims.map(claim => claim.join(' = ')))
    }
    else {
      try {
        const values: number[][] = JSON.parse(validation.result)
        for (let i = 0; i < claims.length; i++) {
          const consistent = (row: number[], scaled: boolean) => row.every((value, index) => {
            if (!Number.isFinite(value)) return false
            if (index === 0) return true
            const decimalPlaces = claims[i][index].match(/^[-+]?\d*\.(\d+)%?$/)?.[1].length ?? 2
            const tolerance = Math.min(0.005, 0.5 * 10 ** -decimalPlaces) /
              (scaled && claims[i][index].includes('%') ? 100 : 1)
            return Math.abs(row[index - 1] - value) < tolerance + Number.EPSILON * Math.max(1, Math.abs(value))
          })
          const rawValid = consistent(values[i], false)
          const scaledValid = consistent(values[i + claims.length], true)
          if (!rawValid && !scaledValid) {
            issues.push('An arithmetic equation contradicts its operands: ' + claims[i].join(' = '))
            invalidClaims.push(claims[i].join(' = '))
          } else {
            // Retain checked numerical work for fallback only when its first
            // expression uses supplied/printed operands or a previously checked
            // intermediate. Never salvage arbitrary model-invented inputs.
            if (numbers(claims[i][0]).every(value => supportedOperands.some(candidate => matchesNumber(value, candidate)))) {
              verifiedClaims.push(claims[i].join(' = '))
              if (rawValid) supportedOperands.push(...values[i])
              if (scaledValid) supportedOperands.push(...values[i + claims.length])
            }
            if (rawValid) derived.push(...values[i])
            if (scaledValid) derived.push(...values[i + claims.length])
            // A checked equality chain also justifies its literal intermediate
            // operands, e.g. (a-b)*c = d*c = result. Merely collecting the final
            // evaluations wrongly rejected d even when the entire chain agreed.
            derived.push(...numbers(claims[i].join(' ')))
          }
        }
      } catch { issues.push('Arithmetic validation did not return usable numeric results.') }
    }
  }
  const visible = answer.replace(/^\s*\d+[.)]\s+/gm, '')
  const presented = numbers(visible)
  const computed = numbers(result)
  if (contradictsNamedResult(answer, result)) issues.push('A named result contradicts execution output.')
  if (resultNumbers(resultValues).some(value => !presented.some(candidate => matchesNumber(value, candidate)))) {
    issues.push('The explanation omits or changes a value from execution output.')
  }
  const arrayCounts: number[] = []
  for (const text of [resultValues, code]) for (const match of text.matchAll(/\[[^\[\]]*\]/g)) {
    try {
      const value: unknown = JSON.parse(match[0])
      if (!Array.isArray(value)) continue
      arrayCounts.push(value.length)
      if (text === resultValues && value.length && value.every(item => typeof item === 'number' ||
        typeof item === 'string' && /^[-+]?\d+(?:\.\d+)?$/.test(item))) {
        let position = 0
        for (const candidate of presented) if (matchesNumber(candidate, Number(value[position]))) position++
        if (position < value.length) issues.push('An ordered numeric list from execution output is incomplete or reordered.')
      }
    } catch { /* Not a JSON array literal. */ }
  }
  const allowed = [...computed, ...numbers(evidence), ...numbers(code), ...derived, ...arrayCounts]
  if (presented.some(value => !allowed.some(candidate => matchesNumber(value, candidate)))) {
    issues.push('The explanation introduces a numeric value absent from the evidence or checked calculation.')
  }
  // Dollar math delimiters are not currency. An unpaired/escaped currency
  // marker without supporting evidence/output must not turn scores into money.
  const currencyText = answer.replace(/(?<!\\)\$\$[\s\S]*?\$\$|(?<!\\)\$[^$\n]+(?<!\\)\$/g, '')
  for (const symbol of ['$', '€', '£', '¥']) {
    if (currencyText.includes(symbol) && !(evidence + result).includes(symbol)) {
      issues.push('The explanation introduces an unsupported currency unit.')
    }
  }
  if (result.includes('%') && !/%|\bpercent(?:age)?\b/i.test(answer)) {
    issues.push('The explanation omits the percentage unit from execution output.')
  }
  if (answer.includes('%') && !/%|\bpercent(?:age)?\b/i.test(evidence + result)) {
    issues.push('The explanation introduces an unsupported percentage unit.')
  }
  return { answer, issues, invalidClaims, verifiedClaims }
}

function executedFallback(result: string, question = '', verifiedClaims: string[] = []): string {
  // Teaching details come from execution output, never inferred from a prompt
  // template or recalculated from the question after narration fails.
  const teaching = teachingOutput(result)
  const precision = requestedPrecision(question)
  function formatted(value: unknown): unknown {
    if (precision === undefined) return value
    if (typeof value === 'number') return Number.isInteger(value) ? value : value.toFixed(precision)
    if (typeof value === 'string') {
      const match = value.match(/^(\s*[$€£¥]?\s*)([-+]?\d+(?:\.\d+)?)(\s*(?:%|°?[\p{L} ]+)?\s*)$/u)
      return match && /[.%$€£¥]/.test(value) ? `${match[1]}${Number(match[2]).toFixed(precision)}${match[3]}` : value
    }
    if (Array.isArray(value)) return value.map(formatted)
    if (value && typeof value === 'object') return Object.fromEntries(Object.entries(value).map(([key, child]) => [key, formatted(child)]))
    return value
  }
  if (teaching) {
    const printed = new Set(teaching.steps.flatMap(step => arithmeticClaims(step, outputFacts(result).map(([label]) => label)))
      .map(claim => claim.join(' = ')))
    const steps = [...teaching.steps, ...verifiedClaims.filter(claim => !printed.has(claim)).map(claim => `\`${claim}\``)]
    return teachingText({ ...teaching, steps, results: formatted(teaching.results) as TeachingOutput['results'] })
  }
  let text = result
  try { text = displayLines(JSON.parse(result)).join('\n') } catch {
    // Readably display array output while preserving each string's precision.
    text = text.replace(/\[[^\[\]\n]*\]/g, array => {
      try {
        const value = JSON.parse(array)
        return Array.isArray(value) && value.every(item => ['number', 'string'].includes(typeof item))
          ? value.map(String).join(', ') : array
      } catch { return array }
    })
  }
  const worked = verifiedClaims.length ? `**Worked calculations:**\n\n${verifiedClaims.map(claim => `- \`${claim}\``).join('\n')}\n\n` : ''
  return worked + `### Calculation results\n\n${text.replace(/\$(?=\d)/g, '\\$').split('\n').map(line => `- ${line}`).join('\n')}`
}

/** Execute the planned Advanced Math task and return tool feedback to the same
 * model for bounded repairs and the final teaching explanation.
 */
export async function runJavascriptInterpret(
  initialMessages: StudyChatMessage[],
  infer: (messages: StudyChatMessage[], maxTokens: number) => Promise<string>,
  execute: (code: string) => Promise<JavascriptExecution>, onTrace: (trace: JavascriptInterpretTrace) => void,
  question?: string
): Promise<string> {
  const started = performance.now()
  const attempts: JavascriptInterpretTrace['attempts'] = []
  const trace: JavascriptInterpretTrace = { attempts, inferenceCount: 0, retryCount: 0,
    latencyMs: 0, inferenceLatenciesMs: [] }
  const evidence = initialMessages.filter(message => message.role !== 'system').map(message => message.content).join('\n')
  // One stable system prompt, including any configured tutoring/source rules.
  // Labeled feedback also works with the installed Ollama chat template;
  // no native tool-role support is assumed.
  const conversation: StudyChatMessage[] = [{ role: 'system', content: [
    ...initialMessages.filter(message => message.role === 'system').map(message => message.content),
    toolInstructions
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
      const information = informationRequest(response)
      if (information) { trace.informationRequest = information; return information }
      let code = ''
      let plan = ''
      let assistantCall = response
      let trailingText = ''
      let recoveredTransport = false
      let reusedExecution = false
      let execution: JavascriptExecution
      let phase: 'plan' | 'execution' = 'plan'
      try {
        const call = toolCall(response)
        code = call.code
        plan = call.plan
        assistantCall = call.assistantCall
        trailingText = call.trailingText
        recoveredTransport = call.recoveredTransport ?? false
        phase = 'execution'
        const previous = attempts.find(entry => entry.execution.ok && entry.code.replace(/\r\n/g, '\n').trim() === code.replace(/\r\n/g, '\n').trim())
        reusedExecution = !!previous
        if (previous) {
          execution = previous.execution
          trace.unchangedCorrections = (trace.unchangedCorrections ?? 0) + 1
        } else execution = await execute(code)
      } catch (error) {
        if (error instanceof Error && error.name === 'AbortError') throw error
        execution = { ok: false, error: error instanceof Error ? error.message : String(error), latencyMs: 0 }
        // Do not reinsert fabricated results even when transport is malformed.
        assistantCall = response.split(/^\s*(?:Tool feedback:|javascript_interpret (?:result|error):|Assistant:)/im)[0].trim()
      }
      attempts.push({ code, plan, execution, retryCount: attempt, phase, modelResponse: response,
        ...(trailingText ? { discardedTrailingText: trailingText } : {}), ...(recoveredTransport ? { recoveredTransport } : {}),
        ...(reusedExecution ? { reusedExecution } : {}) })
      let narrationResult = execution.ok ? execution.result : ''
      let feedbackResult = narrationResult
      const teaching = execution.ok ? teachingOutput(execution.result) : undefined
      if (teaching) {
        trace.teachingOutput = true
        const checkedAt = performance.now()
        const checked = await checkNarration(teachingText(teaching), narrationResult, evidence, code, execute, question ?? '')
        if (teaching.steps.some(step => /\b(?:undefined|null|NaN|Infinity)\b/.test(step))) {
          checked.issues.push('A printed teaching step contains an unresolved operand.')
        }
        trace.validationLatencyMs = (trace.validationLatencyMs ?? 0) + performance.now() - checkedAt
        if (checked.issues.length) {
          trace.teachingIssues = checked.issues
          // Preserve independently consistent steps when another printed step
          // is wrong. Recheck the remaining teaching data before using it.
          const badClaims = new Set(checked.invalidClaims)
          const labels = outputFacts(narrationResult).map(([label]) => label)
          const safeText = (text: string) => !/\b(?:undefined|null|NaN|Infinity)\b/.test(text) &&
            !arithmeticClaims(text, labels).some(claim => badClaims.has(claim.join(' = ')))
          const steps = teaching.steps.filter(safeText)
          const safeTeaching = { ...teaching, steps,
            method: safeText(teaching.method) ? teaching.method : 'Follow the executed substitutions below.' }
          let retained = false
          if (steps.length !== teaching.steps.length || safeTeaching.method !== teaching.method) {
            const recheckedAt = performance.now()
            const candidate = JSON.stringify(safeTeaching)
            const rechecked = await checkNarration(teachingText(safeTeaching), candidate, evidence, code, execute, question ?? '')
            trace.validationLatencyMs += performance.now() - recheckedAt
            if (!rechecked.issues.length) {
              retained = true
              narrationResult = candidate
              feedbackResult = rechecked.answer + '\nInconsistent printed steps were withheld. Use only the steps shown above.'
              trace.omittedTeachingSteps = (trace.omittedTeachingSteps ?? 0) + teaching.steps.length - steps.length
            }
          }
          if (!retained) {
            // Never teach from uncheckable or contradictory printed arithmetic.
            narrationResult = JSON.stringify(teaching.results)
            feedbackResult = narrationResult + '\nThe printed teaching steps failed numeric consistency checks. Use only these results, or supply a corrected program with consistent steps.'
          }
        } else feedbackResult = checked.answer
      }
      conversation.push({ role: 'assistant', content: assistantCall }, { role: 'user', content: execution.ok
        ? [`javascript_interpret result:\n${feedbackResult}`,
          ...(reusedExecution ? ['Your replacement program is unchanged. It has not been executed again. Repeating it cannot supply missing outputs. Either explain the existing results or identify the missing requested output and change the program to print it.'] : []),
          'Execution is complete. Do not repeat the program. Write a student-facing explanation: introduce the printed method/formula, walk through the printed substitutions, explain the aggregate step, and state every requested result with its units. Copy the numerical steps as given; explain their purpose in your own words without doing new arithmetic. Do not return only a result list. Use only supplied names; otherwise use positions. Check the original evidence for incompatible versions and omitted observations, including repeated values. If inputs remain unresolved, use needs_information and withhold the result. Only call the tool again for an identified omission or error, with a changed program that addresses it.'
        ].join('\n\n')
        : `javascript_interpret error:\n${execution.error}\nPartial output, if any, is not a successful result.` })
      if (!execution.ok) {
        if (attempt === maxExecutionRetries) return `I couldn't complete this calculation after ${attempt + 1} JavaScript attempts. ${execution.error.split('\n')[0]}`
        response = await timedInference()
        continue
      }
      const answer = await timedInference()
      const informationAfterExecution = informationRequest(answer)
      if (informationAfterExecution) { trace.informationRequest = informationAfterExecution; return informationAfterExecution }
      if (hasToolMarker(answer)) {
        if (attempt === maxExecutionRetries) {
          // Do not present a partial computation as a complete answer. Preserve
          // diagnostic output only when the model repeats the identical program;
          // a different correction may invalidate the previous result.
          let unchanged = false
          try { unchanged = toolCall(answer).code.trim() === code.trim() } catch { /* Incomplete correction. */ }
          return "I couldn't finish all requested results and their explanation within the JavaScript retry limit." +
            (unchanged ? `\n\nThe unchanged program produced the following partial output; this is not a complete answer.\n\n${executedFallback(narrationResult, question)}` : '')
        }
        response = answer
        continue
      }
      if (!answer.trim()) {
        trace.narrationFallback = true
        trace.rejectedFinalAnswer = answer
        trace.narrationIssues = ['The final explanation was empty.']
        return executedFallback(narrationResult, question)
      }
      const validationStarted = performance.now()
      const checked = await checkNarration(answer.trim(), narrationResult, evidence, code, execute,
        question ?? initialMessages.filter(message => message.role === 'user').at(-1)?.content ?? '')
      trace.validationLatencyMs = (trace.validationLatencyMs ?? 0) + performance.now() - validationStarted
      if (checked.issues.length) {
        trace.narrationFallback = true
        trace.rejectedFinalAnswer = answer
        trace.narrationIssues = checked.issues
        return executedFallback(narrationResult, question, checked.verifiedClaims)
      }
      return checked.answer
    }
    return "I couldn't complete this calculation within the JavaScript retry limit."
  } finally {
    trace.latencyMs = performance.now() - started
    onTrace(trace)
  }
}
