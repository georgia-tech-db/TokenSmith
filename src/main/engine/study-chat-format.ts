import type { ChatSource, ExplanationDepth, LocalModel, ModelRuntimeSettings } from '../../shared/app-state'
import type { EngineChatRequest, EngineQuestionSuggestionRequest } from '../../shared/engine'
import { trimReferenceExchange } from '../../shared/study-chat-pipeline'
import {
  defaultFollowUpSuggestionCount,
  normalizeStarterQuestionPrompt,
  defaultSuggestedFollowUpPrompt,
  normalizeSuggestedFollowUpPrompt,
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
  'You are a tutor helping an undergraduate understand the current question. Open with a direct answer or the condition needed to make a judgment. Do not narrate your evidence handling: avoid phrases such as "the context does not say", "the provided material shows", or "based on the context". Discuss the subject itself.',
  'Use the supplied study material as evidence to answer in your own words. When the answer is implicit, connect the relevant details and explain the inference. Explain the cause of an outcome, not merely that it happened; a quotation or restatement alone is not an explanation. Reasoning from the source belongs in the main answer. Only additions from outside knowledge go under a "General background" heading; use well-established knowledge and do not attribute it to the collection.',
  'Each source unit belongs to a specific work, section, example, or table. Identify the unit the question concerns. Keep its subjects and claims distinct from those of other units, even within one document. Do not substitute another unit\'s speaker, data, assumptions, or conclusion. Connect different units only when the question calls for it, naming the distinction.',
  'Ground interpretations in the actual details and distinguish interpretation from explicit statements. For numerical material, retain units, dates, variable definitions, and assumptions. Distinguish reasoning and illustrative examples from measured results. Never invent implementation details, benchmark numbers, or guarantees. Evidence about one object or workload does not establish the same claim for another.',
  'If a specific judgment needs missing measurements or requirements, identify that uncertainty briefly and explain the relevant trade-off. Do not replace the explanation with a statement about missing context, and do not append unrelated limitations.',
  'Match the student\'s request: for code, include the relevant fenced code or clearly labeled illustrative code; for an example, work through a small example. Explain how the mechanism works and why it matters, with enough detail to teach the idea. For comparisons or judgments, state the conclusion and its conditions without overstating them.',
  'Use short paragraphs and compact lists where helpful. Do not repeat background already explained. Do not mention source labels, locators, or page numbers, and do not end by asking whether the student wants more detail.'
]

function sourceContextInstructionText(): string {
  return sourceContextInstructions.join('\n')
}

// Shapes how far an answer unpacks an idea, never what it is allowed to claim: the
// grounding rules above still apply. 'standard' is empty so the default answer stays
// byte-identical to what it was before this setting existed.
const explanationDepthInstructions: Record<ExplanationDepth, string> = {
  simple: [
    'Explanation depth: the student is meeting this idea for the first time.',
    'Lead with the core idea in plain language, and define any technical term you cannot avoid the first time it appears.',
    'Ground the idea in one concrete example or familiar comparison, preferring an example the study material already uses.',
    'Stay on the single main idea: leave out edge cases, secondary conditions, and formal notation unless the question asks for them.',
    'Simplify without distorting. Where a simplification would leave a false impression, add one short sentence naming the limit instead of making an inaccurate claim.',
    'Brevity is not the goal; aim for an explanation the student could restate in their own words.'
  ].join(' '),
  standard: '',
  detailed: [
    'Explanation depth: the student already has a working understanding of this idea and wants to sharpen it.',
    'Use the precise terminology the study material uses, and state the conditions, assumptions, and limits under which each claim holds.',
    'Explain why the mechanism works and which trade-off or failure case it exists to address, not only what it does.',
    'Where the material supports it, separate this concept from the adjacent one it is most often confused with and name the distinguishing property.',
    'Depth must come from the evidence: do not pad the answer with length, restatement, speculation, or invented specifics.'
  ].join(' ')
}

function explanationDepthInstruction(depth?: ExplanationDepth): string {
  return (depth && explanationDepthInstructions[depth]) || ''
}

// The picker only reaches the prompt while the student has the setting switched on,
// so a depth left over from an earlier session cannot quietly shape answers.
function requestExplanationDepth(
  applicationSettings?: EngineChatRequest['applicationSettings']
): ExplanationDepth | undefined {
  return applicationSettings?.explanationDepthEnabled ? applicationSettings.explanationDepth : undefined
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
  evidenceQuery?: string
  referenceText?: string
  model?: LocalModel
  modelSettings?: Partial<ModelRuntimeSettings>
  includeBudget?: boolean
  includeInstructions?: boolean
  explanationDepth?: ExplanationDepth
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
  return `### Source unit\nCollection: ${collection}\nPath: ${path}\n${section}Text: `
}

function emptyBudget(options?: SourceContextOptions): SourceContextBudget {
  const modelContextTokens = effectiveContextLength(options?.model, options?.modelSettings)
  const answerReserve = answerReserveTokens(options?.modelSettings)
  const safetyMargin = safetyMarginTokens(modelContextTokens)
  const fixedPromptTokens = estimateTokens([
    options?.modelSettings?.systemMessage,
    sourceContextInstructions.join('\n'),
    explanationDepthInstruction(options?.explanationDepth),
    options?.referenceText,
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

  const terms = queryTerms(options.evidenceQuery ?? options.prompt ?? '')
  const blocks: string[] = []
  let usedSourceTokens = 0
  let truncatedSourceCount = 0

  for (const source of sources) {
    const prefix = sourcePrefix(source)
    const suffix = '\n### End source unit\n'
    if (!options.includeBudget) {
      const unbudgetedText = sourceText(source)
      blocks.push(`${prefix}${unbudgetedText.text}${suffix}`)
      continue
    }

    const remainingTokens = budget.sourceBudgetTokens - usedSourceTokens
    const prefixTokens = estimateTokens(prefix + suffix)
    const textBudgetTokens = remainingTokens - prefixTokens - 4
    if (textBudgetTokens < minSourceTextTokens) {
      break
    }

    const clipped = sourceText(source, textBudgetTokens, terms)
    const block = `${prefix}${clipped.text}${suffix}`
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
    : formatFollowUpInstruction(suggestionPromptFor(request.modelSettings, 'starter'), questionSuggestionCount(request.applicationSettings))
  return packSourceContext(request.retrievedSources ?? [], {
    prompt,
    ...('prompt' in request ? {
      evidenceQuery: request.retrievalQuery,
      referenceText: chatReferenceText(request),
      explanationDepth: requestExplanationDepth(request.applicationSettings)
    } : {}),
    model: request.model,
    modelSettings: 'prompt' in request ? request.modelSettings : { ...request.modelSettings, maxLength: suggestionMaxTokens },
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

function chatReferenceText(request: EngineChatRequest): string {
  if (request.conversationContextMode !== 'contextual' || !request.referenceExchange) return ''
  const historyTokens = Math.min(1536, Math.floor(effectiveContextLength(request.model, request.modelSettings) / 4))
  const exchange = trimReferenceExchange(request.referenceExchange, historyTokens * estimatedCharsPerToken)
  return [
    '### Previous exchange (reference context, not factual evidence):',
    'Use this only to identify what the current question refers to and preserve example identifiers. It may contain mistakes. Do not treat the previous answer as evidence for factual claims.',
    JSON.stringify(exchange)
  ].join('\n')
}

export function studyChatMessages(request: EngineChatRequest): StudyChatMessage[] {
  const modelSettings = (modelAwareRuntimeSettings(request) ?? request.modelSettings) as
    | Partial<ModelRuntimeSettings>
    | undefined
  const configuredSystemMessage = modelSettings?.systemMessage?.trim()
  const answerPrompt = (request.referenceExchange ? request.prompt : request.answerPrompt ?? request.prompt).trim()
  const referenceText = chatReferenceText(request)
  const context = sourceContext(request.retrievedSources ?? [], {
    prompt: answerPrompt,
    evidenceQuery: request.retrievalQuery,
    referenceText,
    model: request.model,
    modelSettings,
    includeBudget: true,
    includeInstructions: false,
    explanationDepth: requestExplanationDepth(request.applicationSettings)
  })
  const userContent = context || referenceText
    ? [referenceText, context, `Question: ${answerPrompt}`].filter(Boolean).join('\n\n')
    : answerPrompt
  const messages: StudyChatMessage[] = []
  const systemMessage = [
    configuredSystemMessage,
    context || referenceText ? sourceContextInstructionText() : '',
    explanationDepthInstruction(requestExplanationDepth(request.applicationSettings))
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

export type SuggestionPromptKind = 'followUp' | 'starter'

export function suggestionPromptFor(
  modelSettings: Partial<ModelRuntimeSettings> | undefined,
  kind: SuggestionPromptKind
): string {
  return kind === 'starter'
    ? normalizeStarterQuestionPrompt(modelSettings?.starterQuestionPrompt)
    : normalizeSuggestedFollowUpPrompt(modelSettings?.suggestedFollowUpPrompt)
}

export const suggestionMaxTokens = 384

export function questionSuggestionSchema(count: number): Record<string, unknown> {
  return { type: 'array', items: { type: 'string' }, maxItems: count }
}

export function questionSuggestionMessages(request: EngineQuestionSuggestionRequest): StudyChatMessage[] {
  const modelSettings = { ...(modelAwareRuntimeSettings(request) ?? request.modelSettings), maxLength: suggestionMaxTokens }
  const systemMessage = modelSettings?.systemMessage?.trim()
  const count = questionSuggestionCount(request.applicationSettings)
  const kind: SuggestionPromptKind = request.messages.length === 0 ? 'starter' : 'followUp'
  const suggestionPrompt = formatFollowUpInstruction(
    suggestionPromptFor(modelSettings, kind),
    count
  )
  const context = sourceContext(request.retrievedSources ?? [], {
    prompt: suggestionPrompt,
    model: request.model,
    modelSettings,
    includeBudget: true,
    includeInstructions: false
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

function clippedSuggestionAnswer(answer: string): string {
  const normalized = answer.trim()
  const maxChars = 4000
  if (normalized.length <= maxChars) {
    return normalized
  }

  return `${normalized.slice(0, maxChars).trim()}...`
}

export function followUpSuggestionMessages(request: EngineChatRequest, answer: string): StudyChatMessage[] {
  const modelSettings = modelAwareRuntimeSettings(request) ?? request.modelSettings
  const systemMessage = modelSettings?.systemMessage?.trim()
  const count = followUpSuggestionCount(request)
  const suggestionPrompt = formatFollowUpInstruction(
    suggestionPromptFor(modelSettings, 'followUp'),
    count
  )
  const messages: StudyChatMessage[] = []

  if (systemMessage) {
    messages.push({ role: 'system', content: systemMessage })
  }

  messages.push({
    role: 'user',
    content: [
      `Current question:\n${request.prompt.trim()}`,
      `Latest answer:\n${clippedSuggestionAnswer(answer)}`,
      `Already asked (avoid repeats):\n${request.messages.filter((message) => message.role === 'user').slice(-8).map((message) => message.text).join('\n')}`,
      suggestionPrompt
    ].join('\n\n')
  })

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

const metaSuggestionPattern =
  /\b(?:(?:based on|according to|from)\s+(?:the\s+)?(?:given\s+|provided\s+)?(?:answer|context|course material|document|documents|excerpt|excerpts|material|materials|passage|passages|question|source|sources|text)|in\s+(?:the\s+)?(?:given|provided)\s+(?:answer|context|course material|document|documents|excerpt|excerpts|material|materials|passage|passages|question|source|sources|text))\b/i
const awkwardSuggestionPattern =
  /\b(?:can you think of|given answer|given question|provided context|provided excerpt|provided source|usually confuses people|what part of this|what parts of this)\b/i
const followUpQuestionPattern =
  /\b(?:Are|Can|Could|Do|Does|How|Is|Should|What|When|Where|Which|Who|Whom|Whose|Why|Would)\b[^?\n]*\?/g
const startsWithQuestionPattern =
  /^(?:Are|Can|Could|Do|Does|How|Is|Should|What|When|Where|Which|Who|Whom|Whose|Why|Would)\b/i
const suggestionSimilarityStopWords = new Set([
  'a',
  'about',
  'after',
  'an',
  'and',
  'are',
  'as',
  'be',
  'can',
  'could',
  'did',
  'do',
  'does',
  'for',
  'from',
  'how',
  'in',
  'is',
  'it',
  'its',
  'me',
  'of',
  'on',
  'or',
  'the',
  'that',
  'this',
  'to',
  'was',
  'what',
  'when',
  'where',
  'which',
  'why',
  'with',
  'would',
  'you'
])

function normalizeSuggestionKey(suggestion: string): string {
  return suggestion
    .normalize('NFKD')
    .replace(/[\u0300-\u036f]/g, '')
    .toLowerCase()
    .replace(/[^\p{L}\p{N}]+/gu, ' ')
    .trim()
}

function isUsefulSuggestion(suggestion: string): boolean {
  return suggestion.length > 0 &&
    suggestion.length <= 180 &&
    suggestion.endsWith('?') &&
    !metaSuggestionPattern.test(suggestion) &&
    !awkwardSuggestionPattern.test(suggestion)
}

function comparableQuestionTokens(question: string): Set<string> {
  const tokens = normalizeSuggestionKey(question)
    .split(/\s+/)
    .filter((token) =>
      token.length > 1 &&
      !suggestionSimilarityStopWords.has(token) &&
      !/^\d+$/.test(token)
    )
  return new Set(tokens)
}


function questionOverlap(left: string, right: string): number {
  const leftTokens = comparableQuestionTokens(left)
  const rightTokens = comparableQuestionTokens(right)
  const smallerSize = Math.min(leftTokens.size, rightTokens.size)
  if (smallerSize < 3) {
    return 0
  }

  let matches = 0
  for (const token of leftTokens) {
    if (rightTokens.has(token)) {
      matches += 1
    }
  }
  return matches / smallerSize
}

function isRepeatedQuestion(suggestion: string, referenceQuestions: string[]): boolean {
  const suggestionKey = normalizeSuggestionKey(suggestion)
  return referenceQuestions.some((question) => {
    const referenceKey = normalizeSuggestionKey(question)
    return referenceKey.length > 0 &&
      (suggestionKey === referenceKey || questionOverlap(suggestion, question) >= 0.85)
  })
}

export function filterSuggestedQuestions(
  suggestions: string[],
  referenceQuestions: string[],
  limit: number
): string[] {
  const filtered: string[] = []
  const seen = new Set<string>()
  if (limit <= 0) {
    return filtered
  }

  for (const suggestion of suggestions) {
    const key = normalizeSuggestionKey(suggestion)
    if (!key || seen.has(key) || !isUsefulSuggestion(suggestion) || isRepeatedQuestion(suggestion, referenceQuestions)) {
      continue
    }
    seen.add(key)
    filtered.push(suggestion)
    if (filtered.length >= limit) {
      break
    }
  }

  return filtered
}


export function parseFollowUpSuggestions(text: string, limit: number): string[] {
  if (limit <= 0) {
    return []
  }
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
    for (const rawLine of text.split('\n')) {
      const line = stripSuggestionPrefix(rawLine)
      if (!line.endsWith('?')) {
        continue
      }
      if ((line.match(/\?/g) ?? []).length === 1) {
        suggestions.push(line)
        continue
      }
      const lineQuestions = line.match(followUpQuestionPattern) ?? []
      if (startsWithQuestionPattern.test(line)) {
        if (lineQuestions[0]) {
          suggestions.push(stripSuggestionPrefix(lineQuestions[0]))
        }
      } else {
        suggestions.push(...lineQuestions.map(stripSuggestionPrefix))
      }
    }
  }

  if (suggestions.length === 0) {
    const questionMatches = text.match(followUpQuestionPattern) ?? []
    suggestions.push(...questionMatches.map(stripSuggestionPrefix))
  }

  const dedupedSuggestions: string[] = []
  const seen = new Set<string>()

  for (const suggestion of suggestions) {
    if (!isUsefulSuggestion(suggestion)) {
      continue
    }
    const key = normalizeSuggestionKey(suggestion)
    if (!key || seen.has(key)) {
      continue
    }
    seen.add(key)
    dedupedSuggestions.push(suggestion)
    if (dedupedSuggestions.length >= limit) {
      break
    }
  }

  return dedupedSuggestions
}

export function formatFollowUpInstruction(prompt: string, count: number): string {
  return prompt.includes('{count}')
    ? prompt.replaceAll('{count}', String(count))
    : `Generate ${count} suggested follow-up question${count === 1 ? '' : 's'}.\n${prompt}`
}

export function defaultFollowUpPrompt(): string {
  return defaultSuggestedFollowUpPrompt
}
