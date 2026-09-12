import type { ChatSource, LocalModel, ModelRuntimeSettings } from '../../shared/app-state'
import type { EngineChatRequest, EngineQuestionSuggestionRequest } from '../../shared/engine'
import {
  defaultFollowUpSuggestionCount,
  defaultSuggestedFollowUpPrompt,
  minFollowUpSuggestionCount
} from '../../shared/model-defaults'

export type StudyChatMessage = { role: 'system' | 'user' | 'assistant'; content: string }

const estimatedCharsPerToken = 4
const defaultModelContextTokens = 2048
const defaultAutoContextCapTokens = 8192
const minModelContextTokens = 512
const maxModelContextTokens = 32768
const defaultAnswerReserveTokens = 768
const minAnswerReserveTokens = 256
const maxAnswerReserveTokens = 1024
const minSourceTextTokens = 80
const sourceContextInstructions = [
  'Use the context below only when it is relevant to the question.',
  'Answer directly for a student with enough detail to teach the concept. Use only relevant evidence and keep the answer scoped to the user\'s question.',
  'Explain mechanism and consequence; for yes/no, comparison, or judgment questions, start with the conclusion, name the comparison target, and state the workload or condition behind the trade-off.',
  'Do not overstate with words like always, faster, or better unless the context gives that condition.',
  'Do not quote the context before answering. Do not mention context labels, source labels, excerpt labels, locators, or page numbers.',
  'If the context does not contain the answer, say that plainly.'
]

function sourceContextInstructionText(): string {
  return sourceContextInstructions.join('\n')
}

export interface SourceContextBudget {
  modelContextTokens: number
  answerReserveTokens: number
  safetyMarginTokens: number
  fixedPromptTokens: number
  sourceBudgetTokens: number
  usedSourceTokens: number
  estimatedPromptTokens: number
  includedSourceCount: number
  truncatedSourceCount: number
}

interface SourceContextOptions {
  prompt?: string
  model?: LocalModel
  modelSettings?: Partial<ModelRuntimeSettings>
  includeBudget?: boolean
  includeInstructions?: boolean
}

export function estimateTokens(text: string): number {
  return Math.ceil(text.length / estimatedCharsPerToken)
}

function clampNumber(value: unknown, fallback: number, min: number, max: number): number {
  const numericValue = typeof value === 'number' ? value : Number(value)
  if (!Number.isFinite(numericValue)) {
    return fallback
  }

  return Math.max(min, Math.min(max, Math.round(numericValue)))
}

function configuredContextLength(settings?: Partial<ModelRuntimeSettings>): number {
  return clampNumber(settings?.contextLength, defaultModelContextTokens, minModelContextTokens, maxModelContextTokens)
}

export function effectiveContextLength(model?: LocalModel, settings?: Partial<ModelRuntimeSettings>): number {
  const configured = configuredContextLength(settings)
  const discovered = clampNumber(model?.contextLength, 0, 0, maxModelContextTokens)
  if (discovered > 0) {
    return Math.min(discovered, Math.max(configured, defaultAutoContextCapTokens))
  }

  return configured
}

export function modelAwareRuntimeSettings(
  request: Pick<EngineChatRequest | EngineQuestionSuggestionRequest, 'model' | 'modelSettings'>
): ModelRuntimeSettings | undefined {
  if (!request.modelSettings) {
    return undefined
  }

  return {
    ...request.modelSettings,
    contextLength: effectiveContextLength(request.model, request.modelSettings)
  }
}

function answerReserveTokens(settings?: Partial<ModelRuntimeSettings>): number {
  return clampNumber(settings?.maxLength, defaultAnswerReserveTokens, minAnswerReserveTokens, maxAnswerReserveTokens)
}

function safetyMarginTokens(modelContextTokens: number): number {
  return clampNumber(Math.ceil(modelContextTokens * 0.05), 256, 128, 512)
}

const queryStopWords = new Set([
  'a',
  'an',
  'and',
  'are',
  'as',
  'at',
  'be',
  'by',
  'can',
  'do',
  'does',
  'for',
  'from',
  'how',
  'in',
  'is',
  'it',
  'of',
  'on',
  'or',
  'that',
  'the',
  'this',
  'to',
  'was',
  'what',
  'when',
  'where',
  'which',
  'who',
  'why',
  'with'
])

function queryTerms(text: string): string[] {
  const normalized = text
    .normalize('NFKD')
    .replace(/[\u0300-\u036f]/g, '')
    .toLowerCase()
  const tokens = normalized.match(/[a-z0-9][a-z0-9+#.-]*/g) ?? []
  const terms = tokens
    .flatMap((token) => [token, token.replace(/[^a-z0-9]+/g, '')])
    .filter((term) => term.length >= 3 && !queryStopWords.has(term))

  return Array.from(new Set(terms))
}

function bestTermOffset(text: string, terms: string[]): number {
  const lowered = text.toLowerCase()
  const offsets = terms
    .map((term) => lowered.indexOf(term.toLowerCase()))
    .filter((offset) => offset >= 0)

  return offsets.length > 0 ? Math.min(...offsets) : 0
}

function matchingTermOffsets(text: string, terms: string[]): number[] {
  const lowered = text.toLowerCase()
  const offsets: number[] = []

  for (const term of terms) {
    const normalizedTerm = term.toLowerCase()
    let offset = lowered.indexOf(normalizedTerm)
    let matchCount = 0
    while (offset >= 0 && matchCount < 12) {
      offsets.push(offset)
      offset = lowered.indexOf(normalizedTerm, offset + normalizedTerm.length)
      matchCount += 1
    }
  }

  return offsets.sort((left, right) => left - right)
}

function clipSourceText(text: string, maxChars: number, terms: string[]): string {
  if (text.length <= maxChars) {
    return text
  }

  const offsets = matchingTermOffsets(text, terms)
  const firstOffset = offsets[0]
  const lastOffset = offsets.at(-1)
  if (
    maxChars >= 700 &&
    firstOffset !== undefined &&
    lastOffset !== undefined &&
    firstOffset < maxChars * 0.2 &&
    lastOffset > maxChars
  ) {
    const bridge = '...\n...\n'
    const headChars = Math.floor((maxChars - bridge.length) * 0.55)
    const tailChars = Math.max(0, maxChars - bridge.length - headChars)
    const tailStart = Math.max(
      headChars,
      Math.min(lastOffset - Math.floor(tailChars * 0.35), text.length - tailChars)
    )
    const suffix = tailStart + tailChars < text.length ? '...' : ''
    return `${text.slice(0, headChars).trim()}${bridge}${text.slice(tailStart, tailStart + tailChars).trim()}${suffix}`
  }

  const center = bestTermOffset(text, terms)
  const halfWindow = Math.floor(maxChars / 2)
  const start = Math.max(0, Math.min(center - halfWindow, text.length - maxChars))
  const end = Math.min(text.length, start + maxChars)
  const prefix = start > 0 ? '...' : ''
  const suffix = end < text.length ? '...' : ''
  return `${prefix}${text.slice(start, end).trim()}${suffix}`
}

function sourceText(source: ChatSource, maxTokens?: number, terms: string[] = []): { text: string; truncated: boolean } {
  const text = (source.context || source.excerpt).trim()
  if (!maxTokens) {
    return { text, truncated: false }
  }

  const maxChars = Math.max(0, maxTokens * estimatedCharsPerToken)
  if (text.length <= maxChars) {
    return { text, truncated: false }
  }

  return { text: clipSourceText(text, maxChars, terms), truncated: true }
}

function sourcePrefix(source: ChatSource): string {
  const collection = source.collectionName || source.documentTitle || source.title || 'Library'
  const path = source.path || source.title || ''
  const section = source.sectionHeader ? `Section: ${source.sectionHeader}\n` : ''
  return `Collection: ${collection}\nPath: ${path}\n${section}Text: `
}

function emptyBudget(options?: SourceContextOptions): SourceContextBudget {
  const modelContextTokens = effectiveContextLength(options?.model, options?.modelSettings)
  const answerReserve = answerReserveTokens(options?.modelSettings)
  const safetyMargin = safetyMarginTokens(modelContextTokens)
  const fixedPromptTokens = estimateTokens([
    options?.modelSettings?.systemMessage,
    sourceContextInstructions.join('\n'),
    options?.prompt ? `Question: ${options.prompt}` : ''
  ].filter(Boolean).join('\n\n'))

  return {
    modelContextTokens,
    answerReserveTokens: answerReserve,
    safetyMarginTokens: safetyMargin,
    fixedPromptTokens,
    sourceBudgetTokens: Math.max(0, modelContextTokens - answerReserve - safetyMargin - fixedPromptTokens),
    usedSourceTokens: 0,
    estimatedPromptTokens: fixedPromptTokens,
    includedSourceCount: 0,
    truncatedSourceCount: 0
  }
}

export function packSourceContext(
  sources: ChatSource[],
  options: SourceContextOptions = {}
): { context: string; budget: SourceContextBudget } {
  const budget = emptyBudget(options)
  if (sources.length === 0) {
    return { context: '', budget }
  }

  const terms = queryTerms(options.prompt ?? '')
  const blocks: string[] = []
  let usedSourceTokens = 0
  let truncatedSourceCount = 0

  for (const source of sources) {
    const prefix = sourcePrefix(source)
    if (!options.includeBudget) {
      const unbudgetedText = sourceText(source)
      blocks.push(`${prefix}${unbudgetedText.text}`)
      continue
    }

    const remainingTokens = budget.sourceBudgetTokens - usedSourceTokens
    const prefixTokens = estimateTokens(prefix)
    const textBudgetTokens = remainingTokens - prefixTokens - 4
    if (textBudgetTokens < minSourceTextTokens) {
      break
    }

    const clipped = sourceText(source, textBudgetTokens, terms)
    const block = `${prefix}${clipped.text}`
    const blockTokens = estimateTokens(block)
    if (blockTokens > remainingTokens) {
      break
    }

    blocks.push(block)
    usedSourceTokens += blockTokens
    if (clipped.truncated) {
      truncatedSourceCount += 1
    }
  }

  const finalBudget = {
    ...budget,
    usedSourceTokens,
    estimatedPromptTokens: budget.fixedPromptTokens + usedSourceTokens,
    includedSourceCount: blocks.length,
    truncatedSourceCount
  }

  if (blocks.length === 0) {
    return { context: '', budget: finalBudget }
  }

  return {
    context: [
      ...(options.includeInstructions === false ? [] : [...sourceContextInstructions, '']),
      '### Context:',
      ...blocks
    ].join('\n'),
    budget: finalBudget
  }
}

export function sourceContext(sources: ChatSource[], options: SourceContextOptions = {}): string {
  return packSourceContext(sources, options).context
}

export function sourceContextBudgetForRequest(request: EngineChatRequest | EngineQuestionSuggestionRequest): SourceContextBudget {
  const prompt = 'prompt' in request
    ? request.answerPrompt ?? request.prompt
    : request.modelSettings?.suggestedFollowUpPrompt ?? ''
  return packSourceContext(request.retrievedSources ?? [], {
    prompt,
    model: request.model,
    modelSettings: request.modelSettings,
    includeBudget: true
  }).budget
}

const citationLabelPattern = '(?:source|excerpt|passage|context|citation|evidence|reference)'

function firstReferencedSourceIndex(text: string, sourceCount: number): number | undefined {
  const citationPattern = new RegExp(`\\b${citationLabelPattern}\\s+(\\d+)\\b`, 'gi')
  for (const match of text.matchAll(citationPattern)) {
    const index = Number.parseInt(match[1], 10) - 1
    if (index >= 0 && index < sourceCount) {
      return index
    }
  }

  return undefined
}

function normalizeForSourceMatch(text: string): string {
  return text.toLowerCase().replace(/[^\p{L}\p{N}]+/gu, ' ').replace(/\s+/g, ' ').trim()
}

function firstQuotedContextSourceIndex(text: string, sources: ChatSource[]): number | undefined {
  const match = text.match(/^\s*["“]([^"”]{40,1000})["”]/)
  const quotedText = normalizeForSourceMatch(match?.[1] ?? '')
  if (!quotedText) {
    return undefined
  }

  return sources.findIndex((source) => {
    const excerpt = normalizeForSourceMatch(source.excerpt)
    return excerpt.includes(quotedText) || quotedText.includes(excerpt)
  })
}

function stripSourceNumberPhrases(text: string): string {
  const citationPrefix = new RegExp(`^\\s*(?:according to|as (?:stated|noted|shown|reported) in|per|from)\\s+(?:the\\s+)?${citationLabelPattern}\\s+\\d+\\s*,?\\s*`, 'i')
  const citationSubject = new RegExp(`^\\s*(?:the\\s+)?${citationLabelPattern}\\s+\\d+\\s+(?:states|says|notes|indicates|mentions|reports)\\s+(?:that\\s+)?`, 'i')
  const bracketCitation = new RegExp(`\\s*[([](?:${citationLabelPattern})\\s+\\d+[)\\]]`, 'gi')
  const inlineCitationPrefix = new RegExp(`\\b(?:according to|as (?:stated|noted|shown|reported) in|per|from)\\s+(?:the\\s+)?${citationLabelPattern}\\s+\\d+\\s*,?\\s*`, 'gi')
  const genericContextPrefix = /^\s*(?:so,\s*)?(?:according to|as (?:stated|noted|shown|reported) in|from|based on)\s+(?:this|the)\s+(?:excerpt|context|source|text)\s*,?\s*/i
  const inlineGenericContextPrefix = /\b(?:so,\s*)?(?:according to|as (?:stated|noted|shown|reported) in|from|based on)\s+(?:this|the)\s+(?:excerpt|context|source|text)\s*,?\s*/gi
  const quotedContextPreamble = /^\s*["“][^"”]{40,1000}["”]\s*(?:so,\s*)?(?:according to|as (?:stated|noted|shown|reported) in|from|based on)\s+(?:this|the)\s+(?:excerpt|context|source|text)\s*,?\s*/i
  const leakedInstructionSentence = /\s*[^.!?]*(?:source numbers?|excerpt labels?|excerpts?\s+label(?:led|ed)|phrases like\s+["']?according to)[^.!?]*[.!?]/gi

  return text
    .replace(quotedContextPreamble, '')
    .replace(citationPrefix, '')
    .replace(citationSubject, '')
    .replace(genericContextPrefix, '')
    .replace(bracketCitation, '')
    .replace(inlineCitationPrefix, '')
    .replace(inlineGenericContextPrefix, '')
    .replace(leakedInstructionSentence, '')
    .replace(/\s{2,}/g, ' ')
    .trim()
}

export function answerWithOrderedSources(text: string, sources: ChatSource[]): { text: string; sources: ChatSource[] } {
  const referencedSourceIndex =
    firstReferencedSourceIndex(text, sources.length) ?? firstQuotedContextSourceIndex(text, sources)
  const orderedSources =
    referencedSourceIndex === undefined || referencedSourceIndex < 0
      ? sources
      : [
          sources[referencedSourceIndex],
          ...sources.slice(0, referencedSourceIndex),
          ...sources.slice(referencedSourceIndex + 1)
        ]

  return {
    text: stripSourceNumberPhrases(text),
    sources: orderedSources
  }
}

export function studyChatMessages(request: EngineChatRequest): StudyChatMessage[] {
  const modelSettings = (modelAwareRuntimeSettings(request) ?? request.modelSettings) as
    | Partial<ModelRuntimeSettings>
    | undefined
  const configuredSystemMessage = modelSettings?.systemMessage?.trim()
  const answerPrompt = (request.answerPrompt ?? request.prompt).trim()
  const context = sourceContext(request.retrievedSources ?? [], {
    prompt: answerPrompt,
    model: request.model,
    modelSettings,
    includeBudget: true,
    includeInstructions: false
  })
  const userContent = context ? `${context}\n\nQuestion: ${answerPrompt}` : answerPrompt
  const messages: StudyChatMessage[] = []
  const systemMessage = [
    configuredSystemMessage,
    context ? sourceContextInstructionText() : ''
  ].filter(Boolean).join('\n\n')

  if (systemMessage) {
    messages.push({ role: 'system', content: systemMessage })
  }

  messages.push({ role: 'user', content: userContent })
  return messages
}

export function shouldGenerateFollowUps(request: EngineChatRequest): boolean {
  return request.applicationSettings?.suggestionMode !== 'off'
}

export function followUpSuggestionCount(request: EngineChatRequest): number {
  return questionSuggestionCount(request.applicationSettings)
}

export function questionSuggestionCount(applicationSettings?: EngineChatRequest['applicationSettings']): number {
  if (applicationSettings?.suggestionMode === 'off') {
    return 0
  }

  const count = Number(applicationSettings?.followUpSuggestionCount)
  if (!Number.isFinite(count)) {
    return defaultFollowUpSuggestionCount
  }

  return count <= minFollowUpSuggestionCount ? minFollowUpSuggestionCount : defaultFollowUpSuggestionCount
}

export function questionSuggestionMessages(request: EngineQuestionSuggestionRequest): StudyChatMessage[] {
  const modelSettings = modelAwareRuntimeSettings(request) ?? request.modelSettings
  const systemMessage = modelSettings?.systemMessage?.trim()
  const count = questionSuggestionCount(request.applicationSettings)
  const suggestionPrompt = formatFollowUpInstruction(
    modelSettings?.suggestedFollowUpPrompt?.trim() || defaultFollowUpPrompt(),
    count
  )
  const context = sourceContext(request.retrievedSources ?? [], {
    prompt: suggestionPrompt,
    model: request.model,
    modelSettings,
    includeBudget: true
  })
  const messages: StudyChatMessage[] = []

  if (systemMessage) {
    messages.push({ role: 'system', content: systemMessage })
  }

  if (context) {
    messages.push({ role: 'user', content: context })
  }

  messages.push({ role: 'user', content: suggestionPrompt })

  return messages
}

function stripSuggestionPrefix(value: string): string {
  return value
    .trim()
    .replace(/^[-*\u2022\s]+/, '')
    .replace(/^\d+[.)]\s*/, '')
    .replace(/^["']|["']$/g, '')
    .trim()
}

export function parseFollowUpSuggestions(text: string, limit: number): string[] {
  const suggestions: string[] = []

  try {
    const parsed = JSON.parse(text)
    if (Array.isArray(parsed)) {
      for (const item of parsed) {
        const suggestion = stripSuggestionPrefix(String(item))
        if (suggestion.endsWith('?')) {
          suggestions.push(suggestion)
        }
      }
    }
  } catch {
    // Plain text suggestions are common across model providers.
  }

  if (suggestions.length === 0) {
    const questionMatches = text.match(/\b(?:What|Where|How|Why|When|Who|Which|Whose|Whom)\b[^?\n]*\?/g) ?? []
    suggestions.push(...questionMatches.map(stripSuggestionPrefix))
  }

  if (suggestions.length === 0) {
    suggestions.push(
      ...text
        .split('\n')
        .map(stripSuggestionPrefix)
        .filter((line) => line.endsWith('?'))
    )
  }

  return Array.from(new Set(suggestions.filter((suggestion) => suggestion.length > 0 && suggestion.length <= 180))).slice(0, limit)
}

export function formatFollowUpInstruction(prompt: string, count: number): string {
  return prompt.includes('{count}')
    ? prompt.replaceAll('{count}', String(count))
    : `Generate ${count} suggested follow-up question${count === 1 ? '' : 's'}.\n${prompt}`
}

export function defaultFollowUpPrompt(): string {
  return defaultSuggestedFollowUpPrompt
}
