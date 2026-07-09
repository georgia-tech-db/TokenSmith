import type { ChatSource, ModelRuntimeSettings } from '../../shared/app-state'
import type { EngineChatRequest, EngineQuestionSuggestionRequest } from '../../shared/engine'
import {
  defaultFollowUpSuggestionCount,
  defaultSuggestedFollowUpPrompt,
  minFollowUpSuggestionCount
} from '../../shared/model-defaults'

export type StudyChatMessage = { role: 'system' | 'user' | 'assistant'; content: string }

export function sourceContext(sources: ChatSource[]): string {
  if (sources.length === 0) {
    return ''
  }

  return [
    'Use the context below only when it is relevant to the question.',
    'Answer directly. Do not quote the context before answering. Do not mention context labels, source labels, excerpt labels, locators, or page numbers.',
    'If the context does not contain the answer, say that plainly.',
    '',
    '### Context:',
    ...sources.map((source) => {
      const collection = source.collectionName || source.documentTitle || source.title || 'Library'
      const path = source.path || source.title || ''
      const section = source.sectionHeader ? `Section: ${source.sectionHeader}\n` : ''
      return `Collection: ${collection}\nPath: ${path}\n${section}Text: ${source.excerpt}`
    })
  ].join('\n')
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
  const modelSettings = request.modelSettings as Partial<ModelRuntimeSettings> | undefined
  const systemMessage = modelSettings?.systemMessage?.trim()
  const context = sourceContext(request.retrievedSources ?? [])
  const userContent = context ? `${context}\n\nQuestion: ${request.prompt}` : request.prompt
  const messages: StudyChatMessage[] = []

  if (systemMessage) {
    messages.push({ role: 'system', content: systemMessage })
  }

  for (const message of request.messages.slice(-12)) {
    const content = message.text.trim()
    if (!content) {
      continue
    }

    messages.push({ role: message.role, content })
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
  const systemMessage = request.modelSettings?.systemMessage?.trim()
  const count = questionSuggestionCount(request.applicationSettings)
  const suggestionPrompt = formatFollowUpInstruction(
    request.modelSettings?.suggestedFollowUpPrompt?.trim() || defaultFollowUpPrompt(),
    count
  )
  const context = sourceContext(request.retrievedSources ?? [])
  const messages: StudyChatMessage[] = []

  if (systemMessage) {
    messages.push({ role: 'system', content: systemMessage })
  }

  for (const message of request.messages.slice(-12)) {
    const content = message.text.trim()
    if (!content) {
      continue
    }

    messages.push({ role: message.role, content })
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
