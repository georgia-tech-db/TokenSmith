export type SeriesOperation = 'difference' | 'percentage-change'

export interface VerifiedSeries {
  operation: SeriesOperation
  label: string
  inputs: number[]
  changes: number[]
  total: number
  includeTotal: boolean
  mean: number
  precision: number
}

const numberToken = /(?<![\p{L}\p{N}])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?![\p{L}\p{N}])/gu

function numericSequence(question: string): number[] | undefined {
  const inputText = question.replace(/\b(?:to|at)\s+2\s+decimal\s+places\b/gi, 'to two decimal places')
  const tokens = [...inputText.matchAll(numberToken)]
  if (tokens.length < 3 || tokens.length > 32) return undefined
  // Accept one explicit ordered list. Other numbers make the input population ambiguous.
  for (let index = 1; index < tokens.length; index += 1) {
    const previous = tokens[index - 1]
    const between = inputText.slice((previous.index ?? 0) + previous[0].length, tokens[index].index)
    if (!/^[\s,;]*(?:and\s+)?$/i.test(between)) return undefined
  }
  const values = tokens.map((token) => Number(token[0].replaceAll(',', '')))
  return values.every(Number.isFinite) ? values : undefined
}

export function verifiedSeriesFromQuestion(question: string): VerifiedSeries | undefined {
  const percentage = /\bconsecutive\s+percentage\s+changes?\b/i.test(question)
  const difference = /\bconsecutive\s+(?:changes?|gaps?|differences?)\b|\b(?:gaps?|differences?)\s+between\s+consecutive\b/i.test(question)
  if (!percentage && !difference) return undefined
  if (!/\b(?:average|mean)\b/i.test(question)) return undefined
  const inputs = numericSequence(question)
  if (!inputs) return undefined
  const operation: SeriesOperation = percentage ? 'percentage-change' : 'difference'
  if (operation === 'percentage-change' && inputs.slice(0, -1).some((value) => value === 0)) return undefined
  const changes = inputs.slice(1).map((next, index) => operation === 'percentage-change'
    ? (next - inputs[index]) / inputs[index] * 100
    : next - inputs[index])
  const total = changes.reduce((sum, value) => sum + value, 0)
  const mean = total / changes.length
  if (!changes.every(Number.isFinite) || !Number.isFinite(mean)) return undefined
  const precision = /\b(?:two|2)\s+decimal\s+places\b/i.test(question) ? 2 : 0
  const label = percentage ? 'Consecutive percentage changes'
    : /\bgaps?\b/i.test(question) ? 'Consecutive gaps' : 'Consecutive changes'
  const includeTotal = /\b(?:their|its)\s+total\b|\btotal\s+(?:of\s+)?(?:consecutive\s+)?(?:changes?|gaps?|differences?)\b/i.test(question)
  return { operation, label, inputs, changes, total, includeTotal, mean, precision }
}

function renderedNumber(value: number, precision: number): string {
  return precision ? value.toFixed(precision) : Number.isInteger(value) ? String(value) : String(Number(value.toFixed(2)))
}

export function verifiedSeriesFacts(series: VerifiedSeries): string {
  const unit = series.operation === 'percentage-change' ? '%' : ''
  const total = series.includeTotal ? `\n\nTotal change: ${renderedNumber(series.total, series.precision)}${unit}` : ''
  const method = series.operation === 'percentage-change'
    ? 'For each pair, subtract the earlier count from the later count, divide by the earlier count, then multiply by 100.'
    : 'For each pair, subtract the earlier value from the later value.'
  const approximate = series.precision === 0 && !Number.isInteger(series.mean) &&
    Math.abs(Number(renderedNumber(series.mean, series.precision)) - series.mean) > 1e-9 ? 'approximately ' : ''
  const factor = series.operation === 'percentage-change' ? ' × 100' : ''
  const formula = series.operation === 'percentage-change'
    ? '(later value − earlier value) ÷ earlier value × 100; report the result as a percentage'
    : 'later value − earlier value'
  const substitutions = series.changes.map((change, index) => {
    const earlier = series.inputs[index]
    const later = series.inputs[index + 1]
    const expression = series.operation === 'percentage-change'
      ? `(${later} − ${earlier}) ÷ ${earlier}${factor}`
      : `${later} − ${earlier}`
    return `- ${earlier} to ${later}: ${expression} = ${renderedNumber(change, series.precision)}${unit}`
  })
  const meanSum = renderedNumber(series.total, series.precision)
  return `**Method:** ${method} Formula: ${formula}\n\n**Verified steps:**\n${substitutions.join('\n')}${total}\n\n**Mean:** Add the ${series.changes.length} changes (${meanSum}${unit}) and divide by ${series.changes.length}. Arithmetic mean: ${approximate}${renderedNumber(series.mean, series.precision)}${unit}.`
}

export function verifiedSeriesOutputMatches(series: VerifiedSeries, result: string): boolean {
  const numbers = [...result.matchAll(numberToken)].map((token) => Number(token[0].replaceAll(',', '')))
  let position = 0
  for (const expected of [...series.changes, ...(series.includeTotal ? [series.total] : []), series.mean]) {
    while (position < numbers.length && Math.abs(numbers[position] - expected) > 0.005) position += 1
    if (position >= numbers.length) return false
    position += 1
  }
  return true
}

export function verifiedSeriesAnswerMatches(series: VerifiedSeries, answer: string): boolean {
  const normalized = answer.replaceAll('\\%', '%')
  const unit = series.operation === 'percentage-change' ? '%' : ''
  let position = 0
  for (const value of [...series.changes, ...(series.includeTotal ? [series.total] : []), series.mean]) {
    const expected = `${renderedNumber(value, series.precision)}${unit}`
    position = normalized.indexOf(expected, position)
    if (position < 0) return false
    position += expected.length
  }
  const conclusions = normalized.matchAll(/\b(?:mean|average)(?:\s+[A-Za-z]+){0,3}\s*(?::|=|\bis\b)\s*(?:approximately\s+|about\s+)?([-+]?\d+(?:\.\d+)?)/gi)
  for (const match of conclusions) {
    const actual = Number(match[1])
    if (Math.abs(actual - series.mean) > 0.005) return false
  }
  return true
}
