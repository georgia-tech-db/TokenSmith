/** Small, source-grounded calculation contracts for common quantitative questions.
 * A contract is used only when its operands and requested operation are explicit.
 * The JavaScript tool remains available for other questions, without a claim of verification.
 */

export interface VerifiedCalculation {
  operation: string
  facts: string
  method: string
  expectedOutput: number[]
  requiredNumbers: number[]
  requiredUnits: string[]
  allowedExplanatoryNumbers?: number[]
  conclusionTerms: string[]
  conclusionValue: number
  claimVerdict?: 'correct' | 'incorrect'
}

const numberToken = /(?<![\p{L}\p{N}])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?![\p{L}\p{N}])/gu

function numbers(text: string): number[] {
  return [...text.matchAll(numberToken)].map((match) => Number(match[0].replaceAll(',', '')))
}

function valueMatches(actual: number, expected: number): boolean {
  return Math.abs(actual - expected) <= Math.max(0.005, Math.abs(expected) * 0.00001)
}

function containsNumbers(text: string, expected: number[]): boolean {
  const available = numbers(text)
  return expected.every((value) => {
    const index = available.findIndex((candidate) => valueMatches(candidate, value))
    if (index < 0) return false
    available.splice(index, 1)
    return true
  })
}

function orderedNumbers(text: string, expected: number[]): boolean {
  const actual = numbers(text)
  let position = 0
  for (const value of actual) {
    if (valueMatches(value, expected[position])) position += 1
    if (position === expected.length) return true
  }
  return expected.length === 0
}

function money(value: number): string {
  return `$${value.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`
}
function cents(value: number): number { return Math.round((value + Number.EPSILON) * 100) / 100 }
function readableNumber(value: number): string {
  if (Math.abs(value - Math.round(value)) < 1e-9) return String(Math.round(value))
  return String(Number(value.toFixed(2)))
}

function invoice(question: string, source: string): VerifiedCalculation | undefined {
  if (!/\b(?:invoice|line totals?|subtotal)\b/i.test(question)) return undefined
  const items = [...source.matchAll(/\b(\d+)\s+([a-z][a-z\s-]*?)\s+at\s+\$([\d,]+(?:\.\d+)?)\s+each\b/gi)]
    .map((match) => ({ quantity: Number(match[1]), name: match[2].trim().replace(/^and\s+/i, ''), price: Number(match[3].replaceAll(',', '')) }))
  const taxMatch = source.match(/(?:sales\s+)?tax\s+is\s+(\d+(?:\.\d+)?)%/i) ??
    question.match(/\b(\d+(?:\.\d+)?)%\s+tax\b/i)
  const taxRate = taxMatch ? Number(taxMatch[1]) : NaN
  if (!items.length || !Number.isFinite(taxRate) || items.some((item) => !item.name || item.quantity <= 0 || item.price < 0)) return undefined
  const totals = items.map((item) => cents(item.quantity * item.price))
  const subtotal = cents(totals.reduce((sum, value) => sum + value, 0))
  const tax = cents(subtotal * taxRate / 100)
  const total = cents(subtotal + tax)
  const lines = items.map((item, index) => `${item.name}: ${item.quantity} × ${money(item.price)} = ${money(totals[index])}`)
  return {
    operation: 'invoice-total',
    facts: `${lines.join('\n')}\nSubtotal: ${totals.map(money).join(' + ')} = ${money(subtotal)}.\nTax: ${money(subtotal)} × ${taxRate}% = ${money(tax)} (rounded to cents).\nFinal total: ${money(subtotal)} + ${money(tax)} = ${money(total)}.`,
    method: 'Multiply each quantity by its unit price for the line total. Add the line totals for the subtotal. Multiply the subtotal by the tax rate, round tax to cents, then add it to the subtotal.',
    expectedOutput: [...totals, subtotal, tax, total],
    requiredNumbers: items.flatMap((item, index) => [item.quantity, item.price, totals[index]]).concat([taxRate, subtotal, tax, total]),
    requiredUnits: [...totals, subtotal, tax, total].map(money),
    allowedExplanatoryNumbers: [taxRate / 100, ...items.map((_, index) => index + 1)],
    conclusionTerms: ['final total'], conclusionValue: total
  }
}

function weighted(question: string, source: string): VerifiedCalculation | undefined {
  if (!/\bweighted\b/i.test(question) || !/\baverage\b/i.test(question)) return undefined
  const sections = [...source.matchAll(/\bSection\s+([A-Z])\s+has\s+(\d+)\s+(?:students?\s+)?(?:with\s+an?\s+average\s+(?:score\s+)?of|averaging)\s+([\d.]+)/gi)]
    .map((match) => ({ name: match[1], size: Number(match[2]), score: Number(match[3]) }))
  if (sections.length < 2 || sections.some((section) => section.size <= 0 || !Number.isFinite(section.score))) return undefined
  const products = sections.map((section) => section.size * section.score)
  const totalStudents = sections.reduce((sum, section) => sum + section.size, 0)
  const totalPoints = products.reduce((sum, value) => sum + value, 0)
  const average = totalPoints / totalStudents
  if (!Number.isFinite(average)) return undefined
  return {
    operation: 'weighted-mean',
    facts: `${sections.map((section, index) => `Section ${section.name}: ${section.size} students × ${section.score} = ${products[index]} weighted points.`).join('\n')}\nOverall weighted average: (${totalPoints} points ÷ ${totalStudents} students) = ${average}.`,
    method: 'Multiply each section size by its average score. Add those weighted points and divide by the total number of students.',
    expectedOutput: [...products, average],
    requiredNumbers: sections.flatMap((section, index) => [section.size, section.score, products[index]]).concat(average),
    requiredUnits: [], conclusionTerms: ['weighted average'], conclusionValue: average
  }
}

function quarterly(question: string, source: string): VerifiedCalculation | undefined {
  if (!/\bconsecutive\s+percentage\s+changes?\b/i.test(question)) return undefined
  const counts = [...source.matchAll(/\b(Q\d+)\s*:\s*([\d,]+)\b/gi)]
    .map((match) => ({ label: match[1], value: Number(match[2].replaceAll(',', '')) }))
  if (counts.length < 3 || counts.slice(0, -1).some((count) => count.value === 0)) return undefined
  const changes = counts.slice(1).map((count, index) => (count.value - counts[index].value) / counts[index].value * 100)
  const mean = changes.reduce((sum, value) => sum + value, 0) / changes.length
  if (!changes.every(Number.isFinite) || !Number.isFinite(mean)) return undefined
  const pairs = changes.map((change, index) =>
    `${counts[index].label} to ${counts[index + 1].label}: (${counts[index + 1].value} − ${counts[index].value}) ÷ ${counts[index].value} × 100 = ${change.toFixed(2)}%`)
  const changeSum = changes.reduce((sum, value) => sum + value, 0)
  return {
    operation: 'adjacent-percentage-change',
    facts: `Subscriber counts: ${counts.map((count) => `${count.label} ${count.value.toLocaleString('en-US')}`).join(', ')}.\nConsecutive percentage changes:\n${pairs.map((pair) => `- ${pair}`).join('\n')}\nSum of changes: ${changeSum.toFixed(2)} percentage points. Arithmetic mean: ${changeSum.toFixed(2)} ÷ ${changes.length} = ${mean.toFixed(2)}%.`,
    method: 'For each adjacent pair, use (later count − earlier count) ÷ earlier count × 100 and report the result as a percentage. Add the signed percentage changes and divide by the number of changes for their arithmetic mean.',
    expectedOutput: [...changes.map((value) => cents(value)), cents(mean)],
    requiredNumbers: counts.map((count) => count.value).concat(changes.map(cents), cents(mean)),
    requiredUnits: [...changes, mean].map((value) => `${value.toFixed(2)}%`),
    allowedExplanatoryNumbers: [100, changes.length, cents(changeSum)],
    conclusionTerms: ['arithmetic mean', 'average percentage'], conclusionValue: cents(mean)
  }
}

function speed(question: string, source: string): VerifiedCalculation | undefined {
  if (!/\baverage\s+speed\b/i.test(question)) return undefined
  const segments = [...source.matchAll(/\bSegment\s+(\d+)\s+covered\s+([\d.]+)\s+miles?\s+in\s+([\d.]+)\s+hours?\b/gi)]
    .map((match) => ({ label: match[1], distance: Number(match[2]), time: Number(match[3]) }))
  if (!segments.length || segments.some((segment) => segment.time <= 0 || segment.distance < 0)) return undefined
  const distance = segments.reduce((sum, segment) => sum + segment.distance, 0)
  const time = segments.reduce((sum, segment) => sum + segment.time, 0)
  const average = distance / time
  const averageDisplay = readableNumber(average)
  const approximation = Math.abs(Number(averageDisplay) - average) > 1e-9 ? 'approximately ' : ''
  return {
    operation: 'average-speed',
    facts: `${segments.map((segment) => `Segment ${segment.label}: ${readableNumber(segment.distance)} miles in ${readableNumber(segment.time)} ${segment.time === 1 ? 'hour' : 'hours'}.`).join('\n')}\nTotal distance: ${readableNumber(distance)} miles. Total time: ${readableNumber(time)} hours. Average speed: ${readableNumber(distance)} ÷ ${readableNumber(time)} = ${approximation}${averageDisplay} miles per hour.`,
    method: 'Add all segment distances and all segment times separately. Average speed is total distance divided by total time.',
    expectedOutput: [distance, time, average],
    requiredNumbers: segments.flatMap((segment) => [segment.distance, segment.time]).concat([distance, time, average]),
    requiredUnits: ['miles', 'hours', 'miles per hour'],
    conclusionTerms: ['average speed'], conclusionValue: average
  }
}

function breakEven(question: string, source: string): VerifiedCalculation | undefined {
  if (!/\bbreak[ -]even\b/i.test(question)) return undefined
  const fixed = source.match(/\bfixed\s+cost\s+is\s+\$([\d,]+(?:\.\d+)?)/i)
  const price = source.match(/\bselling\s+price\s+is\s+\$([\d,]+(?:\.\d+)?)/i)
  const variable = source.match(/\bvariable\s+cost\s+is\s+\$([\d,]+(?:\.\d+)?)/i)
  if (!fixed || !price || !variable) return undefined
  const values = [fixed, price, variable].map((match) => Number(match[1].replaceAll(',', '')))
  const [fixedCost, sellingPrice, variableCost] = values
  const margin = sellingPrice - variableCost
  if (fixedCost < 0 || margin <= 0) return undefined
  const units = Math.ceil(fixedCost / margin)
  return {
    operation: 'break-even',
    facts: `Fixed cost: ${money(fixedCost)}. Selling price: ${money(sellingPrice)} per unit. Variable cost: ${money(variableCost)} per unit. Contribution margin: ${money(sellingPrice)} − ${money(variableCost)} = ${money(margin)} per unit. Break-even quantity: ceil(${money(fixedCost)} ÷ ${money(margin)}) = ${units} units.`,
    method: 'Subtract variable cost from selling price to get contribution margin per unit. Divide fixed cost by that margin and round up to a whole unit.',
    expectedOutput: [margin, units],
    requiredNumbers: [fixedCost, sellingPrice, variableCost, margin, units],
    requiredUnits: [money(fixedCost), money(sellingPrice), money(variableCost), money(margin)],
    conclusionTerms: ['break-even quantity', 'break-even point', 'units needed to break even'], conclusionValue: units
  }
}

function expectedValue(question: string, source: string): VerifiedCalculation | undefined {
  if (!/\bexpected\s+value\b/i.test(question)) return undefined
  const cases = [...source.matchAll(/\$([\d,]+(?:\.\d+)?)\s+with\s+probability\s+(\d+(?:\.\d+)?)/gi)]
    .map((match) => ({ payoff: Number(match[1].replaceAll(',', '')), probability: Number(match[2]) }))
  if (cases.length < 2 || cases.some((item) => item.probability < 0 || item.probability > 1) ||
    Math.abs(cases.reduce((sum, item) => sum + item.probability, 0) - 1) > 0.000001) return undefined
  const contributions = cases.map((item) => item.payoff * item.probability)
  const expected = contributions.reduce((sum, value) => sum + value, 0)
  return {
    operation: 'expected-value',
    facts: `${cases.map((item, index) => `Payoff ${money(item.payoff)} at probability ${item.probability.toFixed(2)} contributes ${money(contributions[index])}.`).join('\n')}\nExpected value: ${contributions.map(money).join(' + ')} = ${money(expected)}.`,
    method: 'Multiply each possible payoff by its probability, then add those contributions to get expected value.',
    expectedOutput: [...contributions, expected],
    requiredNumbers: cases.flatMap((item, index) => [item.payoff, item.probability, contributions[index]]).concat(expected),
    requiredUnits: [money(expected)],
    allowedExplanatoryNumbers: cases.map((item) => item.probability * 100),
    conclusionTerms: ['expected value'], conclusionValue: expected
  }
}

function temperature(question: string): VerifiedCalculation | undefined {
  if (!/\b(?:convert|conversion)\b/i.test(question) || !/\bCelsius\b/i.test(question)) return undefined
  const fahrenheit = [...question.matchAll(/([-+]?\d+(?:\.\d+)?)\s*°\s*F\b/gi)].map((match) => Number(match[1]))
  if (!fahrenheit.length) return undefined
  const celsius = fahrenheit.map((value) => (value - 32) * 5 / 9)
  const mean = celsius.reduce((sum, value) => sum + value, 0) / celsius.length
  if (!Number.isFinite(mean)) return undefined
  const rounded = (value: number) => Number(value.toFixed(2))
  return {
    operation: 'fahrenheit-celsius',
    facts: `${fahrenheit.map((value, index) => `${value}°F = ${rounded(celsius[index]).toFixed(2)}°C`).join('; ')}.\nAverage Celsius temperature: ${rounded(mean).toFixed(2)}°C.\nFormula: (°F − 32) × 5 ÷ 9; then average the Celsius values.`,
    method: 'For each Fahrenheit value, subtract 32 and multiply by 5 ÷ 9. Add the converted Celsius values and divide by the number of values.',
    expectedOutput: [...celsius.map(rounded), rounded(mean)],
    requiredNumbers: celsius.map(rounded).concat(rounded(mean)),
    requiredUnits: ['°C'],
    allowedExplanatoryNumbers: [1.8, 5, 9, rounded(celsius.reduce((sum, value) => sum + value, 0)), celsius.length],
    conclusionTerms: ['average'], conclusionValue: rounded(mean)
  }
}

function meanClaim(question: string): VerifiedCalculation | undefined {
  const match = question.match(/\b(?:arithmetic\s+)?mean\s+of\s+([\d,\s.and-]+?)\s+is\s+([-+]?\d+(?:\.\d+)?)\b/i)
  if (!match || !/\bclaim\b/i.test(question)) return undefined
  const values = numbers(match[1])
  const claimed = Number(match[2])
  if (values.length < 2 || !Number.isFinite(claimed)) return undefined
  const sum = values.reduce((total, value) => total + value, 0)
  const mean = sum / values.length
  const verdict = valueMatches(mean, claimed) ? 'correct' : 'incorrect'
  return {
    operation: 'mean-claim',
    facts: `${values.join(' + ')} = ${sum}; ${sum} ÷ ${values.length} = ${mean}.\nArithmetic mean: ${mean}.\nThe claim of ${claimed} is ${verdict}.`,
    method: 'Add the values, divide by how many values there are, then compare that mean with the claimed number.',
    expectedOutput: [sum, mean],
    requiredNumbers: [...values, sum, mean],
    requiredUnits: [], conclusionTerms: ['mean'], conclusionValue: mean, claimVerdict: verdict
  }
}

export function verifiedCalculationFromQuestion(question: string, sourceExcerpts: string[] = []): VerifiedCalculation | undefined {
  const fromQuestion = temperature(question) ?? meanClaim(question) ??
    invoice(question, question) ?? weighted(question, question) ?? quarterly(question, question) ??
    speed(question, question) ?? breakEven(question, question) ?? expectedValue(question, question)
  if (fromQuestion) return fromQuestion
  const candidates = sourceExcerpts.flatMap((source) => {
    const calculation = invoice(question, source) ?? weighted(question, source) ?? quarterly(question, source) ??
      speed(question, source) ?? breakEven(question, source) ?? expectedValue(question, source)
    return calculation ? [calculation] : []
  })
  // Never assemble operands across retrieval hits or choose among conflicting records.
  return candidates.length === 1 ? candidates[0] : undefined
}

export function verifiedCalculationOutputMatches(calculation: VerifiedCalculation, output: string): boolean {
  return orderedNumbers(output, calculation.expectedOutput)
}

export function verifiedCalculationTeachingFacts(calculation: VerifiedCalculation): string {
  return `**Method:** ${calculation.method}\n\n**Verified inputs and steps:**\n${calculation.facts}`
}

export function verifiedCalculationAnswerMatches(calculation: VerifiedCalculation, answer: string): boolean {
  const normalized = answer.replaceAll('\\%', '%').replaceAll('\\$', '$')
  if (!containsNumbers(normalized, calculation.requiredNumbers)) return false
  if (calculation.claimVerdict) {
    const lower = normalized.toLowerCase()
    const rejectsClaim = /\b(?:incorrect|wrong|false|not correct|isn't correct|no)\b/.test(lower)
    const acceptsClaim = /\b(?:correct|true|yes)\b/.test(lower) && !/\b(?:not|isn't)\s+correct\b/.test(lower)
    if (calculation.claimVerdict === 'incorrect' ? !rejectsClaim || /\bclaim\s+(?:is\s+)?correct\b/.test(lower)
      : !acceptsClaim || /\bclaim\s+(?:is\s+)?(?:incorrect|wrong|false)\b/.test(lower)) return false
  }
  const amountsWithCurrency = [...normalized.matchAll(/\$\s*(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?/g)]
    .map((match) => Number(match[0].replaceAll(/[$,\s]/g, '')))
  if (!calculation.requiredUnits.every((unit) => unit.startsWith('$')
    ? amountsWithCurrency.some((amount) => valueMatches(amount, Number(unit.slice(1).replaceAll(',', ''))))
    : unit === 'miles per hour' ? /\b(?:miles\s+per\s+hour|mph)\b/i.test(normalized)
      : unit === '°C' ? /(?:°\s*C|degrees?\s+Celsius)\b/i.test(normalized)
        : normalized.toLowerCase().includes(unit.toLowerCase()))) return false

  const lines = normalized.split(/\r?\n/)
  let foundConclusion = false
  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index]
    const lowered = line.toLowerCase()
    const termPositions = calculation.conclusionTerms.map((term) => lowered.indexOf(term)).filter((position) => position >= 0)
    if (!termPositions.length) continue
    if (calculation.operation === 'mean-claim' && /\bclaim\b/i.test(line)) continue
    if (/^\s*(?:to find|to calculate)\b/i.test(line)) continue
    if (calculation.operation === 'expected-value' && /\bcontribution\b/i.test(line) &&
      !/\btotal expected value\b/i.test(line)) continue
    const relevant = line.slice(Math.min(...termPositions))
    const marker = relevant.match(/(?::|=|\bis\b)/i)
    if (!marker) continue
    const afterMarker = relevant.slice((marker.index ?? 0) + marker[0].length)
    const lastEquals = Math.max(afterMarker.lastIndexOf('='), afterMarker.lastIndexOf('≈'), afterMarker.lastIndexOf('\\approx'))
    const completedFormula = lastEquals >= 0 ? numbers(afterMarker.slice(lastEquals + 1)).at(-1) : undefined
    const directValue = afterMarker.match(/^\s*(?:\*|\\\(|\{)*\s*(?:approximately|about|roughly)?\s*\$?\s*([-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)/i)
    const statedValue = completedFormula ?? (/[÷×+*/−]/.test(afterMarker) ? undefined
      : directValue ? Number(directValue[1].replaceAll(',', '')) : undefined)
    if (statedValue === undefined) continue
    if (!valueMatches(statedValue, calculation.conclusionValue)) return false
    foundConclusion = true
  }
  if (foundConclusion) return true
  // Accept a differently worded conclusion when its final numerical result is correct.
  const lastValue = numbers(normalized).at(-1)
  return lastValue !== undefined && valueMatches(lastValue, calculation.conclusionValue)
}
