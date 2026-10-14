import type { OllamaStatus } from './ollama'

export const advancedMathModelName = 'qwen2.5-coder:7b'
export const advancedMathModelLabel = 'Qwen2.5-Coder 7B'

export function advancedMathModelInstalled(status: OllamaStatus | null): boolean {
  return status?.running === true && status.models.some(model => model.name.toLowerCase() === advancedMathModelName)
}

// An offline server cannot establish that a model was removed. Preserve the
// preference on connection failure, but clear it after a confirmed removal.
export function reconcileAdvancedMathEnabled(enabled: boolean, status: OllamaStatus | null): boolean {
  return enabled && !(status?.running && !advancedMathModelInstalled(status))
}
export type AdvancedMathBucket = 'chat' | 'math' | 'ambiguous'

export function normalizeAdvancedMathEnabled(value: unknown): boolean {
  return value === true
}

// Intent fast paths only: these select a specialist, never a formula, population
// or expected result. Unclear requests go to the default model's classifier.
export function advancedMathBucket(question: string): AdvancedMathBucket {
  const text = question.trim()
  if (/^(?:hi|hello|hey|thanks|thank you|goodbye|bye)[.!?]*$/i.test(text)) return 'chat'
  const number = String.raw`[-+]?(?:\d+(?:,\d{3})*(?:\.\d+)?|\.\d+)`
  if (new RegExp(String.raw`^(?:(?:what(?:'s| is)|calculate|compute|evaluate)\s+)?${number}\s*(?:[+×*÷/−-]|plus|minus|times|multiplied by|divided by)\s*${number}\s*[?.!]?\s*$`, 'i').test(text)) return 'chat'
  const computation = /\b(?:calculate|compute|evaluate|solve|derive|convert)\b/i.test(text)
  const numbers = text.match(/\d+(?:,\d{3})*(?:\.\d+)?/g) ?? []
  const metric = /\b(?:mean|average|median|weighted|percentage|percent|consecutive|adjacent|variance|standard deviation|subtotal|tax|sum|total|ratio|rate of change)\b/i.test(text)
  const appliedCalculation = metric && (computation || numbers.length >= 2 || /\b(?:consecutive|adjacent)\b/i.test(text))
  const additionalCalculation = /\b(?:then|and)\s+(?:calculate|compute|evaluate|solve|derive|convert)\b/i.test(text)
  // An explanation can accompany a calculation. Concept-only requests still
  // stay on chat; adding teaching instructions must not override data/task intent.
  if (/\b(?:explain|describe|define|meaning|definition|formula)\b/i.test(text) &&
    (!computation && !appliedCalculation || /^(?:explain|describe|define)\b/i.test(text) && numbers.length < 2 && !additionalCalculation && !/\b(?:using|from)\b.*\b(?:data|values|table|measurements)\b/i.test(text))) return 'chat'
  const lookup = /\baccording to\b/i.test(text) || /(?:^|[,;]\s*)(?:which|who|when|where|identify|name|list|how many|count)\b/i.test(text)
  if (lookup && !metric && !computation) return 'chat'
  if (!metric && !computation && numbers.length < 2 && /^what (?:is|are)\b/i.test(text)) return 'chat'
  if (!metric && !computation && numbers.length < 2 && /^(?:why|how (?:does|do|can))\b/i.test(text)) return 'chat'
  if (!computation && numbers.length === 0 && /^what(?:'s| is) (?:a|an)\b/i.test(text)) return 'chat'
  if (new RegExp(String.raw`^(?:(?:what(?:'s| is)|calculate|compute|evaluate)\s+)?${number}(?:\s*[+×*÷/−-]\s*${number}){2,}\s*[?.!]?\s*$`, 'i').test(text)) return 'math'
  if (appliedCalculation) return 'math'
  if (computation && /\b(?:then|and then|followed by)\b/i.test(text)) return 'math'
  return 'ambiguous'
}

export const advancedMathRoutingInstruction = [
  'Also choose answer_route: chat or math, and give a short route_reason. Decide task intent only, not the formula or numeric answer.',
  'Use chat for factual lookup, listing or counting source facts, a single arithmetic operation, or explaining a mathematical concept/formula without calculating a result.',
  'Use math for multi-step computation, derived aggregates, percentages, formulas applied to data, or repeated calculations. Numbers or mathematical words alone do not establish a computation request.',
  'Resolve referents using the supplied conversation or selected passage. If the student task remains genuinely ambiguous, use mode clarify and ask a short question about the task, not which model to use. Do not manufacture problem data.',
  'Preserve the separate reasoning classification. Return only JSON containing mode, query, clarification, reasoning, answer_route, and route_reason.'
].join('\n')
