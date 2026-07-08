import type { ChatMessage, ChatSource } from './app-state'

export interface RetrievalContextOptions {
  turnCount?: number
  carriedSourceLimit?: number
  maxAnswerChars?: number
}

export interface RetrievalContext {
  query: string
  carriedSources: ChatSource[]
  isContextualFollowUp: boolean
}

function compactText(text: string): string {
  return text.replace(/\s+/g, ' ').trim()
}

function truncateText(text: string, maxChars: number): string {
  const compact = compactText(text)
  return compact.length <= maxChars ? compact : `${compact.slice(0, Math.max(0, maxChars - 1)).trim()}…`
}

export function isContextualFollowUpPrompt(prompt: string): boolean {
  const normalized = compactText(prompt).toLowerCase().replace(/[?.!]+$/g, '')
  if (!normalized) {
    return false
  }

  if (
    /^(?:be\s+)?more\s+specific$/.test(normalized) ||
    /^(?:explain|elaborate|expand)(?:\s+(?:that|this|it|more))?$/.test(normalized) ||
    /^(?:tell me more|go deeper|more details|give(?: me)? more details|be precise|be clearer)$/.test(normalized) ||
    /^(?:why|how so|what do you mean)$/.test(normalized) ||
    /^what about\b/.test(normalized)
  ) {
    return true
  }

  const words = normalized.split(/\s+/)
  return words.length <= 6 && /\b(?:that|this|it|they|them|those|above|previous|same)\b/.test(normalized)
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

export function buildRetrievalContext(
  prompt: string,
  messages: ChatMessage[],
  options: RetrievalContextOptions = {}
): RetrievalContext {
  const query = compactText(prompt)
  const turnCount = Math.max(1, options.turnCount ?? 1)
  const carriedSourceLimit = Math.max(0, options.carriedSourceLimit ?? 2)
  const maxAnswerChars = Math.max(120, options.maxAnswerChars ?? 900)
  const isContextualFollowUp = isContextualFollowUpPrompt(query)
  const turns = isContextualFollowUp ? recentCompletedTurns(messages, turnCount) : []

  if (turns.length === 0) {
    return { query, carriedSources: [], isContextualFollowUp: false }
  }

  const carriedSources = turns
    .flatMap((turn) => turn.assistant.sources ?? [])
    .slice(0, carriedSourceLimit)
  const contextLines = turns.flatMap((turn) => [
    `Previous question: ${compactText(turn.user.text)}`,
    `Previous answer: ${truncateText(turn.assistant.text, maxAnswerChars)}`
  ])

  return {
    query: [...contextLines, `Follow-up: ${query}`].join('\n'),
    carriedSources,
    isContextualFollowUp
  }
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
