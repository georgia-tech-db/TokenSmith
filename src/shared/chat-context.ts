import type { ChatMessage, ChatSource } from './app-state'
import type { ConversationContextMode } from './engine'

export interface RetrievalContextOptions {
  turnCount?: number
  carriedSourceLimit?: number
}

export interface RetrievalContext {
  query: string
  carriedSources: ChatSource[]
  mode: ConversationContextMode
  answerPrompt: string
  anchorTerms: string[]
  currentAnchorTerms: string[]
  focusAnchorTerms: string[]
  isContextualFollowUp: boolean
}

export interface QuestionProfile {
  tokens: string[]
  anchorTerms: string[]
  hasReference: boolean
  hasExternalReference: boolean
  specificity: number
}

export interface RetrievalChoice {
  mode: ConversationContextMode
  query: string
  answerPrompt: string
  sources: ChatSource[]
  standaloneQuality: number
  contextualQuality?: number
}

export interface RetrievalRouteOptions extends RetrievalContextOptions {
  limit: number
  search: (query: string, limit?: number) => Promise<ChatSource[]>
  onContextualSearch?: () => void
}

const genericQueryTerms = new Set([
  'a',
  'about',
  'above',
  'after',
  'again',
  'always',
  'also',
  'an',
  'and',
  'answer',
  'are',
  'as',
  'ask',
  'be',
  'because',
  'been',
  'before',
  'being',
  'bad',
  'best',
  'better',
  'but',
  'by',
  'can',
  'cant',
  'chapter',
  'cannot',
  'could',
  'describe',
  'did',
  'do',
  'does',
  'doing',
  'done',
  'dont',
  'each',
  'elaborate',
  'explain',
  'for',
  'from',
  'give',
  'had',
  'has',
  'have',
  'he',
  'her',
  'here',
  'him',
  'his',
  'how',
  'important',
  'in',
  'into',
  'is',
  'it',
  'its',
  'just',
  'me',
  'mean',
  'more',
  'need',
  'needed',
  'of',
  'on',
  'one',
  'ok',
  'or',
  'other',
  'our',
  'over',
  'previous',
  'prefer',
  'preferable',
  'preferred',
  'prefers',
  'purpose',
  'same',
  'she',
  'should',
  'so',
  'specific',
  'than',
  'that',
  'the',
  'their',
  'them',
  'then',
  'there',
  'these',
  'they',
  'thing',
  'this',
  'those',
  'through',
  'to',
  'use',
  'used',
  'uses',
  'using',
  'was',
  'we',
  'were',
  'what',
  'when',
  'where',
  'which',
  'who',
  'why',
  'with',
  'work',
  'would',
  'worse',
  'worst',
  'you',
  'your'
])

const referenceTerms = new Set([
  'above',
  'it',
  'its',
  'previous',
  'same',
  'that',
  'there',
  'their',
  'them',
  'these',
  'they',
  'this',
  'those'
])
const localPronounReferenceTerms = new Set(['it', 'its'])
const externalPronounPrepositions = new Set([
  'about',
  'above',
  'against',
  'before',
  'behind',
  'below',
  'beside',
  'between',
  'for',
  'from',
  'into',
  'like',
  'near',
  'of',
  'on',
  'over',
  'than',
  'to',
  'under',
  'with',
  'without'
])
const minimumStandaloneAnchorTerms = 2
const weakStandaloneCoverage = 0.5
const contextualImprovementMargin = 0.2
const defaultContextTermLimit = 10
const contextualCandidateMultiplier = 3
const maxContextualCandidateLimit = 24

function compactText(text: string): string {
  return text.replace(/\s+/g, ' ').trim()
}

function compactPreviousAnswer(text: string): string {
  const answer = compactText(text)
  if (answer.length <= 360) {
    return answer
  }
  return `${answer.slice(0, 357).trim()}...`
}

function recentCompletedTurns(messages: ChatMessage[], turnCount: number): Array<{ user: ChatMessage; assistant: ChatMessage }> {
  const turns: Array<{ user: ChatMessage; assistant: ChatMessage }> = []

  for (let assistantIndex = messages.length - 1; assistantIndex >= 0 && turns.length < turnCount; assistantIndex -= 1) {
    const assistant = messages[assistantIndex]
    if (assistant.role !== 'assistant') {
      continue
    }

    for (let userIndex = assistantIndex - 1; userIndex >= 0; userIndex -= 1) {
      const user = messages[userIndex]
      if (user.role === 'user') {
        turns.unshift({ user, assistant })
        assistantIndex = userIndex
        break
      }
    }
  }

  return turns
}

function normalizeForTerms(text: string): string {
  return text
    .normalize('NFKD')
    .replace(/[\u0300-\u036f]/g, '')
    .toLowerCase()
}

function tokenizeForTerms(text: string): string[] {
  const normalized = normalizeForTerms(text)
  const tokens = normalized.match(/[a-z0-9][a-z0-9+#.-]*/g) ?? []
  const expanded: string[] = []

  for (const token of tokens) {
    expanded.push(token)

    const compact = token.replace(/[^a-z0-9]+/g, '')
    if (compact && compact !== token) {
      expanded.push(compact)
    }

    if (token.includes('-')) {
      expanded.push(...token.split('-').filter(Boolean))
    }
  }

  return Array.from(new Set(expanded))
}

function isSingleEditOrTransposition(left: string, right: string): boolean {
  if (left === right || Math.abs(left.length - right.length) > 1) {
    return left === right
  }

  if (left.length === right.length) {
    let mismatchCount = 0
    let firstMismatch = -1
    for (let index = 0; index < left.length; index += 1) {
      if (left[index] === right[index]) {
        continue
      }
      mismatchCount += 1
      if (firstMismatch < 0) {
        firstMismatch = index
      }
      if (mismatchCount > 2) {
        return false
      }
    }

    return mismatchCount <= 1 ||
      (
        mismatchCount === 2 &&
        firstMismatch + 1 < left.length &&
        left[firstMismatch] === right[firstMismatch + 1] &&
        left[firstMismatch + 1] === right[firstMismatch]
      )
  }

  const shorter = left.length < right.length ? left : right
  const longer = left.length < right.length ? right : left
  let shortIndex = 0
  let longIndex = 0
  let skipped = false

  while (shortIndex < shorter.length && longIndex < longer.length) {
    if (shorter[shortIndex] === longer[longIndex]) {
      shortIndex += 1
      longIndex += 1
      continue
    }
    if (skipped) {
      return false
    }
    skipped = true
    longIndex += 1
  }

  return true
}

function referenceTermForToken(token: string): string | undefined {
  if (referenceTerms.has(token)) {
    return token
  }

  if (genericQueryTerms.has(token) || !/^[a-z]+$/.test(token) || token.length < 4 || token.length > 8) {
    return undefined
  }

  return Array.from(referenceTerms).find((referenceTerm) =>
    referenceTerm.length >= 4 && isSingleEditOrTransposition(token, referenceTerm)
  )
}

function isAnchorTerm(term: string): boolean {
  if (referenceTermForToken(term)) {
    return false
  }

  if (genericQueryTerms.has(term)) {
    return false
  }

  if (/^\d+(?:\.\d+)*\.?$/.test(term)) {
    return false
  }

  return term.length >= 3 || /[0-9+#]/.test(term)
}

function anchorTermsForText(text: string, limit = defaultContextTermLimit): string[] {
  const terms: string[] = []
  const seen = new Set<string>()

  for (const token of tokenizeForTerms(text)) {
    if (!isAnchorTerm(token) || seen.has(token)) {
      continue
    }

    seen.add(token)
    terms.push(token)
    if (terms.length >= limit) {
      break
    }
  }

  return terms
}

function termVariants(term: string): string[] {
  const variants = new Set([term])

  if (/^[a-z]{4,}ies$/.test(term)) {
    variants.add(`${term.slice(0, -3)}y`)
  }
  if (/^[a-z]{4,}ing$/.test(term)) {
    variants.add(term.slice(0, -3))
  }
  if (/^[a-z]{4,}ed$/.test(term)) {
    variants.add(term.slice(0, -2))
  }
  if (/^[a-z]{4,}es$/.test(term)) {
    variants.add(term.slice(0, -2))
  }
  if (/^[a-z]{4,}s$/.test(term)) {
    variants.add(term.slice(0, -1))
  }

  return Array.from(variants).filter((variant) => variant.length >= 3)
}

function tokenSetForText(text: string): Set<string> {
  return new Set(tokenizeForTerms(text).flatMap(termVariants))
}

function termMatchesTokens(term: string, tokens: Set<string>): boolean {
  return termVariants(term).some((variant) => tokens.has(variant))
}

function countMatches(terms: string[], tokens: Set<string>): number {
  return terms.filter((term) => termMatchesTokens(term, tokens)).length
}

function overlapRatio(terms: string[], tokens: Set<string>): number {
  return terms.length === 0 ? 0 : countMatches(terms, tokens) / terms.length
}

function looksLikeStrongLocalAnchor(term: string): boolean {
  return /[0-9+#.]/.test(term)
}

function localPronounHasAntecedent(tokens: string[], referenceIndex: number): boolean {
  if (referenceIndex <= 0 || externalPronounPrepositions.has(tokens[referenceIndex - 1])) {
    return false
  }

  const previousAnchors = tokens.slice(0, referenceIndex).filter(isAnchorTerm)
  return previousAnchors.length >= 2 || previousAnchors.some(looksLikeStrongLocalAnchor)
}

function hasExternalReference(tokens: string[]): boolean {
  for (let index = 0; index < tokens.length; index += 1) {
    const token = tokens[index]
    const referenceTerm = referenceTermForToken(token)
    if (!referenceTerm) {
      continue
    }

    if (!localPronounReferenceTerms.has(referenceTerm) || !localPronounHasAntecedent(tokens, index)) {
      return true
    }
  }

  return false
}

export function profileQuestion(prompt: string, limit = defaultContextTermLimit): QuestionProfile {
  const tokens = tokenizeForTerms(prompt)
  const anchorTerms = anchorTermsForText(prompt, limit)
  const hasReference = tokens.some((token) => Boolean(referenceTermForToken(token)))
  const externalReference = hasExternalReference(tokens)

  return {
    tokens,
    anchorTerms,
    hasReference,
    hasExternalReference: externalReference,
    specificity: anchorTerms.length
  }
}

function hasOnlyGenericQuestionTerms(profile: QuestionProfile): boolean {
  return profile.tokens.length > 0 && profile.tokens.every((token) => !isAnchorTerm(token))
}

function isWeakContextualFollowUp(profile: QuestionProfile): boolean {
  return profile.hasExternalReference ||
    (profile.anchorTerms.length < minimumStandaloneAnchorTerms && hasOnlyGenericQuestionTerms(profile))
}

function sourceSearchText(source: ChatSource): string {
  return [
    source.sectionHeader,
    ...(source.queryTerms ?? []),
    ...(source.keywordTerms ?? []),
    source.excerpt,
    source.context
  ]
    .filter(Boolean)
    .join(' ')
}

function normalizedSearchTermsForSources(sources: ChatSource[]): string[] {
  const terms = sources.flatMap((source) => [
    ...(source.queryTerms ?? []),
    ...(source.keywordTerms ?? [])
  ])
  return Array.from(new Set(terms.flatMap(tokenizeForTerms).filter(isAnchorTerm)))
}

function anchorTermsWithSourceCorrections(anchorTerms: string[], sources: ChatSource[]): string[] {
  const searchTerms = normalizedSearchTermsForSources(sources)
  if (searchTerms.length === 0) {
    return anchorTerms
  }

  const correctedTerms = anchorTerms.map((term) =>
    searchTerms.find((candidate) => candidate !== term && isSingleEditOrTransposition(term, candidate)) ?? term
  )
  return Array.from(new Set(correctedTerms))
}

function sourcePosition(source: ChatSource): number | undefined {
  const rowid = Number(source.chunkRowid)
  if (Number.isFinite(rowid)) {
    return rowid
  }

  const chunkId = String(source.chunkId ?? '')
  const chunkIdMatch = chunkId.match(/(\d+)(?!.*\d)/)
  if (chunkIdMatch) {
    return Number(chunkIdMatch[1])
  }

  if (typeof source.lineFrom === 'number') {
    return source.lineFrom
  }

  if (typeof source.pageStart === 'number') {
    return source.pageStart
  }

  return undefined
}

function sameSourceDocument(left: ChatSource, right: ChatSource): boolean {
  return ['materialId', 'documentId', 'path', 'documentTitle'].some((field) => {
    const leftValue = left[field as keyof ChatSource]
    const rightValue = right[field as keyof ChatSource]
    return leftValue !== undefined && rightValue !== undefined && String(leftValue) === String(rightValue)
  })
}

function normalizedSourceSection(source: ChatSource): string | undefined {
  const section = compactText(source.sectionHeader ?? '').toLowerCase()
  return section || undefined
}

function sourceDistance(left: ChatSource, right: ChatSource): number | undefined {
  if (!sameSourceDocument(left, right)) {
    return undefined
  }

  const leftPosition = sourcePosition(left)
  const rightPosition = sourcePosition(right)
  if (leftPosition !== undefined && rightPosition !== undefined) {
    return Math.abs(leftPosition - rightPosition)
  }

  if (typeof left.pageStart === 'number' && typeof right.pageStart === 'number') {
    return Math.abs(left.pageStart - right.pageStart)
  }

  return undefined
}

function focusScoreForSource(source: ChatSource, turn: { user: ChatMessage; assistant: ChatMessage }, sourceIndex: number): number {
  const questionTerms = profileQuestion(turn.user.text).anchorTerms
  const answerTerms = anchorTermsForText(turn.assistant.text, 6)
  const tokens = tokenSetForText(sourceSearchText(source))
  const questionOverlap = overlapRatio(questionTerms, tokens)
  const answerOverlap = overlapRatio(answerTerms, tokens)
  const orderBoost = Math.max(0, 1 - sourceIndex * 0.05)

  return questionOverlap * 12 + countMatches(questionTerms, tokens) * 2 + answerOverlap * 2 + orderBoost
}

function focusSourcesForTurns(
  turns: Array<{ user: ChatMessage; assistant: ChatMessage }>,
  limit: number,
  forceSingleFocus: boolean
): ChatSource[] {
  const focusLimit = forceSingleFocus ? 1 : limit
  if (focusLimit <= 0) {
    return []
  }

  const rankedSources = turns.flatMap((turn) =>
    (turn.assistant.sources ?? []).map((source, sourceIndex) => ({
      source,
      score: focusScoreForSource(source, turn, sourceIndex)
    }))
  )

  return rankedSources
    .sort((left, right) => right.score - left.score)
    .map((ranked) => ranked.source)
    .filter((source, index, sources) =>
      sources.findIndex((candidate) => sourceKey(candidate) === sourceKey(source)) === index
    )
    .slice(0, focusLimit)
}

function focusAnchorTermsForTurns(turns: Array<{ user: ChatMessage; assistant: ChatMessage }>, focusSources: ChatSource[]): string[] {
  return anchorTermsForText(
    [
      ...turns.map((turn) => compactText(turn.user.text)),
      focusSources.map(sourceSearchText).join(' ')
    ].join(' '),
    defaultContextTermLimit
  )
}

function sourceFocusContinuity(source: ChatSource, focusSources: ChatSource[], focusTerms: string[]): number {
  if (focusSources.length === 0) {
    return 0
  }

  const sourceTokens = tokenSetForText(sourceSearchText(source))
  let continuity = overlapRatio(focusTerms, sourceTokens)

  for (const focusSource of focusSources) {
    if (sourceKey(source) === sourceKey(focusSource)) {
      continuity = Math.max(continuity, 1)
      continue
    }

    const section = normalizedSourceSection(source)
    const focusSection = normalizedSourceSection(focusSource)
    if (section && focusSection && section === focusSection) {
      continuity = Math.max(continuity, 0.9)
    }

    const distance = sourceDistance(source, focusSource)
    if (distance !== undefined) {
      if (distance <= 6) {
        continuity = Math.max(continuity, 0.8)
      } else if (distance <= 20) {
        continuity = Math.max(continuity, 0.45)
      }
    }

    if (
      typeof source.pageStart === 'number' &&
      typeof focusSource.pageStart === 'number' &&
      Math.abs(source.pageStart - focusSource.pageStart) <= 1
    ) {
      continuity = Math.max(continuity, 0.75)
    }
  }

  return continuity
}

function sourceListFocusContinuity(
  sources: ChatSource[],
  focusSources: ChatSource[],
  focusTerms: string[],
  sourceLimit = 3
): number {
  return sources
    .slice(0, Math.max(1, sourceLimit))
    .reduce(
      (best, source) => Math.max(best, sourceFocusContinuity(source, focusSources, focusTerms)),
      0
    )
}

function contextSourceContinuity(sources: ChatSource[], context: RetrievalContext): number {
  const focusTerms = context.focusAnchorTerms.length > 0 ? context.focusAnchorTerms : context.anchorTerms
  return sourceListFocusContinuity(sources, context.carriedSources, focusTerms)
}

export function sourceAnchorCoverage(sources: ChatSource[], anchorTerms: string[], sourceLimit = 3): number {
  if (anchorTerms.length === 0 || sources.length === 0) {
    return 0
  }

  const correctedAnchorTerms = anchorTermsWithSourceCorrections(
    anchorTerms,
    sources.slice(0, Math.max(1, sourceLimit))
  )
  const sourceTokens = new Set(
    sources
      .slice(0, Math.max(1, sourceLimit))
      .flatMap((source) => tokenizeForTerms(sourceSearchText(source)))
  )
  const matches = correctedAnchorTerms.filter((term) => termMatchesTokens(term, sourceTokens))
  return matches.length / correctedAnchorTerms.length
}

export function buildContextualRetrievalContext(
  prompt: string,
  messages: ChatMessage[],
  options: RetrievalContextOptions = {}
): RetrievalContext | undefined {
  const query = compactText(prompt)
  const turnCount = Math.max(1, options.turnCount ?? 1)
  const carriedSourceLimit = Math.max(0, options.carriedSourceLimit ?? 2)
  const turns = recentCompletedTurns(messages, turnCount)
  const currentProfile = profileQuestion(query)
  const isContextualFollowUp = isWeakContextualFollowUp(currentProfile)

  if (turns.length === 0) {
    return undefined
  }

  const carriedSources = focusSourcesForTurns(turns, carriedSourceLimit, isContextualFollowUp)
  const previousQuestions = turns.map((turn) => compactText(turn.user.text)).filter(Boolean)
  const currentTerms = currentProfile.anchorTerms
  const focusAnchorTerms = focusAnchorTermsForTurns(turns, carriedSources)
  const anchorTerms = Array.from(new Set([...currentTerms, ...focusAnchorTerms])).slice(0, defaultContextTermLimit)
  const contextualQuery = anchorTerms.length > 0
    ? anchorTerms.join(' ')
    : [...previousQuestions, query].join('\n')
  const previousAnswerLines = isContextualFollowUp
    ? turns
        .map((turn) => compactPreviousAnswer(turn.assistant.text))
        .filter(Boolean)
        .map((answer) => `Previous answer: ${answer}`)
    : []
  const answerPrompt = [
    ...turns.flatMap((turn, index) => {
      const lines = [`Previous question: ${compactText(turn.user.text)}`]
      const answer = previousAnswerLines[index]
      if (answer) {
        lines.push(answer)
      }
      return lines
    }),
    `Current question: ${query}`
  ].join('\n')

  return {
    query: contextualQuery,
    carriedSources,
    mode: 'contextual',
    answerPrompt,
    anchorTerms,
    currentAnchorTerms: currentTerms,
    focusAnchorTerms,
    isContextualFollowUp
  }
}

export function shouldTryContextualRetrieval(
  prompt: string,
  messages: ChatMessage[],
  standaloneSources: ChatSource[]
): boolean {
  const turns = recentCompletedTurns(messages, 1)
  if (turns.length === 0) {
    return false
  }

  const profile = profileQuestion(prompt)
  const coverage = sourceAnchorCoverage(standaloneSources, profile.anchorTerms)
  const weakReferringQuestion = profile.hasExternalReference && profile.anchorTerms.length < minimumStandaloneAnchorTerms
  const standaloneHasTopic = profile.anchorTerms.length >= minimumStandaloneAnchorTerms ||
    (!profile.hasExternalReference && profile.anchorTerms.length > 0)
  const standaloneIsGrounded = !weakReferringQuestion && standaloneHasTopic && coverage >= weakStandaloneCoverage
  const focusSources = focusSourcesForTurns(turns, 2, true)
  const focusTerms = focusAnchorTermsForTurns(turns, focusSources)
  const standaloneFocusContinuity = sourceListFocusContinuity(standaloneSources, focusSources, focusTerms)
  const standaloneDriftedFromFocus = profile.hasExternalReference && standaloneFocusContinuity < weakStandaloneCoverage

  if (!profile.hasExternalReference) {
    return hasOnlyGenericQuestionTerms(profile) && !standaloneIsGrounded
  }

  return (
    !standaloneIsGrounded ||
    standaloneDriftedFromFocus ||
    (profile.hasExternalReference && profile.anchorTerms.length <= defaultContextTermLimit / 2) ||
    weakReferringQuestion
  )
}

export function chooseRetrievalContext(
  prompt: string,
  standaloneSources: ChatSource[],
  contextualContext?: RetrievalContext,
  contextualSources: ChatSource[] = [],
  limit = 4
): RetrievalChoice {
  const standaloneProfile = profileQuestion(prompt)
  const standaloneQuality = sourceAnchorCoverage(standaloneSources, standaloneProfile.anchorTerms)
  const standaloneChoice: RetrievalChoice = {
    mode: 'standalone',
    query: compactText(prompt),
    answerPrompt: compactText(prompt),
    sources: standaloneSources.slice(0, limit),
    standaloneQuality
  }

  if (!contextualContext) {
    return standaloneChoice
  }

  const mergedContextualSources = selectContextualSources(contextualContext, contextualSources, limit)
  const contextualQuality = sourceAnchorCoverage(mergedContextualSources, contextualContext.anchorTerms)
  const weakReferringQuestion =
    standaloneProfile.hasExternalReference && standaloneProfile.anchorTerms.length < minimumStandaloneAnchorTerms
  const standaloneHasTopic = standaloneProfile.anchorTerms.length >= minimumStandaloneAnchorTerms ||
    (!standaloneProfile.hasExternalReference && standaloneProfile.anchorTerms.length > 0)
  const standaloneIsGrounded =
    !weakReferringQuestion &&
    standaloneHasTopic &&
    standaloneQuality >= weakStandaloneCoverage
  const contextualImproves = contextualQuality >= standaloneQuality + contextualImprovementMargin
  const referringQuestion = standaloneProfile.hasExternalReference || contextualContext.isContextualFollowUp
  const contextualCandidateQuestion = weakReferringQuestion || referringQuestion
  const standaloneFocusContinuity = contextSourceContinuity(standaloneSources, contextualContext)
  const contextualFocusContinuity = contextSourceContinuity(mergedContextualSources, contextualContext)
  const contextualIsGrounded = contextualQuality >= weakStandaloneCoverage ||
    contextualFocusContinuity >= weakStandaloneCoverage ||
    (weakReferringQuestion && mergedContextualSources.length > 0)
  const standaloneDriftedFromFocus = referringQuestion && standaloneFocusContinuity < weakStandaloneCoverage
  const contextualPreservesFocus =
    contextualFocusContinuity >= weakStandaloneCoverage &&
    contextualFocusContinuity >= standaloneFocusContinuity + contextualImprovementMargin

  if (((weakReferringQuestion || standaloneDriftedFromFocus || (contextualCandidateQuestion && !standaloneIsGrounded)) && contextualIsGrounded) ||
    (referringQuestion && contextualIsGrounded && contextualPreservesFocus) ||
    (contextualCandidateQuestion && contextualIsGrounded && contextualImproves)) {
    return {
      mode: 'contextual',
      query: contextualContext.query,
      answerPrompt: contextualContext.answerPrompt,
      sources: mergedContextualSources,
      standaloneQuality,
      contextualQuality
    }
  }

  return {
    ...standaloneChoice,
    contextualQuality
  }
}

export async function routeRetrievalContext(
  prompt: string,
  messages: ChatMessage[],
  options: RetrievalRouteOptions
): Promise<RetrievalChoice> {
  const query = compactText(prompt)
  const standaloneSources = await options.search(query, options.limit)
  const contextualContext = shouldTryContextualRetrieval(query, messages, standaloneSources)
    ? buildContextualRetrievalContext(query, messages, {
        turnCount: options.turnCount,
        carriedSourceLimit: options.carriedSourceLimit
      })
    : undefined
  let contextualSources: ChatSource[] = []

  if (contextualContext) {
    options.onContextualSearch?.()
    contextualSources = await options.search(contextualContext.query, contextualCandidateLimit(options.limit))
  }

  return chooseRetrievalContext(
    query,
    standaloneSources,
    contextualContext,
    contextualSources,
    options.limit
  )
}

function contextualCandidateLimit(limit: number): number {
  return Math.min(
    maxContextualCandidateLimit,
    Math.max(limit, limit * contextualCandidateMultiplier)
  )
}

function sourceKey(source: ChatSource): string {
  const stableChunkId = source.chunkId ?? source.chunkRowid
  if (stableChunkId !== undefined && stableChunkId !== null && String(stableChunkId).trim()) {
    const locationKey = source.path ?? source.documentId ?? source.materialId ?? source.sourceId
    return [
      locationKey,
      stableChunkId
    ]
      .map((part) => String(part ?? ''))
      .join('|')
  }

  if (source.sourceId !== undefined && source.sourceId !== null && String(source.sourceId).trim()) {
    return String(source.sourceId)
  }

  return [
    source.materialId,
    source.documentId,
    source.path,
    source.pageStart,
    source.pageEnd,
    source.excerpt.slice(0, 120)
  ]
    .map((part) => String(part ?? ''))
    .join('|')
}

export function mergeChatSources(primary: ChatSource[], secondary: ChatSource[], limit: number): ChatSource[] {
  const seen = new Set<string>()
  const merged: ChatSource[] = []

  for (const source of [...primary, ...secondary]) {
    const key = sourceKey(source)
    if (seen.has(key)) {
      continue
    }

    seen.add(key)
    merged.push(source)
    if (merged.length >= limit) {
      break
    }
  }

  return merged
}

interface ContextualSourceScore {
  source: ChatSource
  score: number
  isFocusSource: boolean
  isFocusNeighborhood: boolean
  isCurrentQuestionMatch: boolean
  clusterKey: string
}

function sourceClusterKey(source: ChatSource): string {
  const documentKey = [
    source.materialId,
    source.documentId,
    source.path,
    source.documentTitle
  ]
    .map((part) => String(part ?? ''))
    .join('|')
  const section = normalizedSourceSection(source)

  if (section) {
    return `${documentKey}|section:${section}`
  }

  const position = sourcePosition(source)
  if (position !== undefined) {
    return `${documentKey}|chunk:${Math.floor(position / 4)}`
  }

  return `${documentKey}|source:${sourceKey(source)}`
}

function contextualScoreForSource(source: ChatSource, context: RetrievalContext, candidateIndex: number): ContextualSourceScore {
  const focusSource = context.carriedSources[0]
  const focusKeys = new Set(context.carriedSources.map(sourceKey))
  const isFocusSource = focusKeys.has(sourceKey(source))
  const sourceTokens = tokenSetForText(sourceSearchText(source))
  const focusTerms = anchorTermsWithSourceCorrections(
    context.focusAnchorTerms.length > 0 ? context.focusAnchorTerms : context.anchorTerms,
    [source]
  )
  const currentTerms = anchorTermsWithSourceCorrections(context.currentAnchorTerms, [source])
  const focusOverlap = overlapRatio(focusTerms, sourceTokens)
  const currentOverlap = overlapRatio(currentTerms, sourceTokens)
  const currentMatches = countMatches(currentTerms, sourceTokens)
  const section = normalizedSourceSection(source)
  const focusSection = focusSource ? normalizedSourceSection(focusSource) : undefined
  const sameSection = Boolean(section && focusSection && section === focusSection)
  const distance = focusSource ? sourceDistance(source, focusSource) : undefined
  const nearbyByChunk = distance !== undefined && distance <= 6
  const relatedByChunk = distance !== undefined && distance <= 20
  const nearbyByPage =
    focusSource &&
    typeof source.pageStart === 'number' &&
    typeof focusSource.pageStart === 'number' &&
    Math.abs(source.pageStart - focusSource.pageStart) <= 1
  const candidateTerms = anchorTermsForText([source.sectionHeader, source.excerpt].filter(Boolean).join(' '), 16)
  const acceptedTerms = new Set([...focusTerms, ...currentTerms, ...context.anchorTerms])
  const newTermRatio = candidateTerms.length === 0
    ? 0
    : candidateTerms.filter((term) => !acceptedTerms.has(term)).length / candidateTerms.length
  const textAligned = focusOverlap >= 0.65 && newTermRatio <= 0.55
  const isFocusNeighborhood = isFocusSource || sameSection || nearbyByChunk || Boolean(nearbyByPage) || textAligned
  const isCurrentQuestionMatch = currentTerms.length > 0 &&
    (currentOverlap >= 0.3 || currentMatches >= 2)
  const orderPenalty = candidateIndex * 0.02
  let score = focusOverlap * 10 + currentOverlap * 6 + currentMatches * 1.5 - newTermRatio * 2 - orderPenalty

  if (isFocusSource) {
    score += 100
  }
  if (sameSection) {
    score += 8
  }
  if (nearbyByChunk) {
    score += Math.max(0, 7 - (distance ?? 0))
  } else if (relatedByChunk) {
    score += 2
  }
  if (nearbyByPage) {
    score += 3
  }
  if (focusSource && sameSourceDocument(source, focusSource)) {
    score += 1
  }

  return {
    source,
    score,
    isFocusSource,
    isFocusNeighborhood,
    isCurrentQuestionMatch,
    clusterKey: sourceClusterKey(source)
  }
}

function selectContextualSources(context: RetrievalContext, contextualSources: ChatSource[], limit: number): ChatSource[] {
  if (!context.isContextualFollowUp || context.carriedSources.length === 0) {
    return mergeChatSources(context.carriedSources, contextualSources, limit)
  }

  const uniqueSources = mergeChatSources(context.carriedSources, contextualSources, context.carriedSources.length + contextualSources.length)
  const rankedSources = uniqueSources
    .map((source, index) => contextualScoreForSource(source, context, index))
    .sort((left, right) => right.score - left.score)
  const focusNeighborhoodSources = rankedSources.filter((ranked) => ranked.isFocusNeighborhood)
  const currentQuestionSources = rankedSources.filter((ranked) =>
    !ranked.isFocusNeighborhood && ranked.isCurrentQuestionMatch
  )
  const offFocusSources = rankedSources.filter((ranked) =>
    !ranked.isFocusNeighborhood && !ranked.isCurrentQuestionMatch
  )
  const focusSourceBucket = focusNeighborhoodSources.filter((ranked) => ranked.isFocusSource)
  const otherFocusNeighborhoodSources = focusNeighborhoodSources.filter((ranked) => !ranked.isFocusSource)
  const sourceBuckets = context.currentAnchorTerms.length > 0
    ? [
        currentQuestionSources,
        focusSourceBucket.slice(0, 1),
        otherFocusNeighborhoodSources,
        focusSourceBucket.slice(1),
        offFocusSources
      ]
    : [focusNeighborhoodSources, offFocusSources]
  const selected: ChatSource[] = []
  const selectedKeys = new Set<string>()
  const clusterCounts = new Map<string, number>()
  let offFocusCount = 0
  const maxOffFocusSources = context.currentAnchorTerms.length > 0 ? 2 : 1

  for (const ranked of sourceBuckets.flat()) {
    if (selected.length >= limit) {
      break
    }

    const key = sourceKey(ranked.source)
    if (selectedKeys.has(key)) {
      continue
    }

    const clusterCount = clusterCounts.get(ranked.clusterKey) ?? 0
    if (!ranked.isFocusSource && clusterCount >= 2) {
      continue
    }

    if (!ranked.isFocusSource && clusterCount > 0 && !ranked.isFocusNeighborhood) {
      continue
    }

    if (!ranked.isFocusNeighborhood) {
      const allowedOffFocusSources = ranked.isCurrentQuestionMatch ? maxOffFocusSources : 1
      if (offFocusCount >= allowedOffFocusSources) {
        continue
      }
      offFocusCount += 1
    }

    selectedKeys.add(key)
    clusterCounts.set(ranked.clusterKey, clusterCount + 1)
    selected.push(ranked.source)
  }

  return selected
}
