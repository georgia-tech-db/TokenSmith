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
}

export interface QuestionProfile {
  tokens: string[]
  anchorTerms: string[]
  hasReference: boolean
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
  search: (query: string) => Promise<ChatSource[]>
  onContextualSearch?: () => void
}

const genericQueryTerms = new Set([
  'a',
  'about',
  'above',
  'after',
  'again',
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
  'by',
  'can',
  'could',
  'describe',
  'did',
  'do',
  'does',
  'doing',
  'done',
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
  'me',
  'mean',
  'more',
  'needed',
  'of',
  'on',
  'or',
  'other',
  'our',
  'previous',
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
  'to',
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
  'their',
  'them',
  'these',
  'they',
  'this',
  'those'
])
const minimumStandaloneAnchorTerms = 2
const weakStandaloneCoverage = 0.5
const contextualImprovementMargin = 0.2
const defaultContextTermLimit = 10

function compactText(text: string): string {
  return text.replace(/\s+/g, ' ').trim()
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

function isAnchorTerm(term: string): boolean {
  if (genericQueryTerms.has(term)) {
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

export function profileQuestion(prompt: string, limit = defaultContextTermLimit): QuestionProfile {
  const tokens = tokenizeForTerms(prompt)
  const anchorTerms = anchorTermsForText(prompt, limit)
  const hasReference = tokens.some((token) => referenceTerms.has(token))

  return {
    tokens,
    anchorTerms,
    hasReference,
    specificity: anchorTerms.length
  }
}

function sourceSearchText(source: ChatSource): string {
  return [
    source.sectionHeader,
    source.excerpt,
    source.context
  ]
    .filter(Boolean)
    .join(' ')
}

export function sourceAnchorCoverage(sources: ChatSource[], anchorTerms: string[], sourceLimit = 3): number {
  if (anchorTerms.length === 0 || sources.length === 0) {
    return 0
  }

  const sourceTokens = new Set(
    sources
      .slice(0, Math.max(1, sourceLimit))
      .flatMap((source) => tokenizeForTerms(sourceSearchText(source)))
  )
  const matches = anchorTerms.filter((term) => sourceTokens.has(term))
  return matches.length / anchorTerms.length
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

  if (turns.length === 0) {
    return undefined
  }

  const carriedSources = turns
    .flatMap((turn) => turn.assistant.sources ?? [])
    .slice(0, carriedSourceLimit)
  const previousQuestions = turns.map((turn) => compactText(turn.user.text)).filter(Boolean)
  const previousSourceText = carriedSources.map(sourceSearchText).join(' ')
  const currentTerms = profileQuestion(query).anchorTerms
  const contextTerms = anchorTermsForText(
    [...previousQuestions, previousSourceText].join(' '),
    defaultContextTermLimit
  )
  const anchorTerms = Array.from(new Set([...currentTerms, ...contextTerms])).slice(0, defaultContextTermLimit)
  const contextualQuery = anchorTerms.length > 0
    ? anchorTerms.join(' ')
    : [...previousQuestions, query].join('\n')
  const answerPrompt = [
    ...previousQuestions.map((question) => `Previous question: ${question}`),
    `Current question: ${query}`
  ].join('\n')

  return {
    query: contextualQuery,
    carriedSources,
    mode: 'contextual',
    answerPrompt,
    anchorTerms
  }
}

export function shouldTryContextualRetrieval(
  prompt: string,
  messages: ChatMessage[],
  standaloneSources: ChatSource[]
): boolean {
  if (recentCompletedTurns(messages, 1).length === 0) {
    return false
  }

  const profile = profileQuestion(prompt)
  const coverage = sourceAnchorCoverage(standaloneSources, profile.anchorTerms)
  const weakReferringQuestion = profile.hasReference && profile.anchorTerms.length < minimumStandaloneAnchorTerms
  const standaloneHasTopic = profile.anchorTerms.length >= minimumStandaloneAnchorTerms ||
    (!profile.hasReference && profile.anchorTerms.length > 0)
  const standaloneIsGrounded = !weakReferringQuestion && standaloneHasTopic && coverage >= weakStandaloneCoverage

  return (
    !standaloneIsGrounded ||
    (profile.hasReference && profile.anchorTerms.length < defaultContextTermLimit / 2) ||
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

  const mergedContextualSources = mergeChatSources(contextualContext.carriedSources, contextualSources, limit)
  const contextualQuality = sourceAnchorCoverage(mergedContextualSources, contextualContext.anchorTerms)
  const weakReferringQuestion =
    standaloneProfile.hasReference && standaloneProfile.anchorTerms.length < minimumStandaloneAnchorTerms
  const standaloneHasTopic = standaloneProfile.anchorTerms.length >= minimumStandaloneAnchorTerms ||
    (!standaloneProfile.hasReference && standaloneProfile.anchorTerms.length > 0)
  const standaloneIsGrounded =
    !weakReferringQuestion &&
    standaloneHasTopic &&
    standaloneQuality >= weakStandaloneCoverage
  const contextualIsGrounded = contextualQuality >= weakStandaloneCoverage ||
    (weakReferringQuestion && mergedContextualSources.length > 0)
  const contextualImproves = contextualQuality >= standaloneQuality + contextualImprovementMargin

  if (((weakReferringQuestion || !standaloneIsGrounded) && contextualIsGrounded) ||
    (contextualIsGrounded && contextualImproves)) {
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
  const standaloneSources = await options.search(query)
  const contextualContext = shouldTryContextualRetrieval(query, messages, standaloneSources)
    ? buildContextualRetrievalContext(query, messages, {
        turnCount: options.turnCount,
        carriedSourceLimit: options.carriedSourceLimit
      })
    : undefined
  let contextualSources: ChatSource[] = []

  if (contextualContext) {
    options.onContextualSearch?.()
    contextualSources = await options.search(contextualContext.query)
  }

  return chooseRetrievalContext(
    query,
    standaloneSources,
    contextualContext,
    contextualSources,
    options.limit
  )
}

function sourceKey(source: ChatSource): string {
  return [
    source.materialId,
    source.documentId,
    source.chunkRowid ?? source.chunkId,
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
