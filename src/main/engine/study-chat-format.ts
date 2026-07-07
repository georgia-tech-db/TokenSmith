import type { ChatSource, CourseMaterial, ModelRuntimeSettings } from '../../shared/app-state'
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
    [
      'Use the following course text when it is relevant. If the course text does not contain the answer, say that plainly.',
      'Answer directly. TokenSmith shows evidence separately after your response.'
    ].join(' '),
    ...sources.map((source) => {
      const title = source.title || 'Untitled PDF'
      const locator = source.locator ? ` (${source.locator})` : ''
      return `PDF: ${title}${locator}\n${source.excerpt}`
    })
  ].join('\n\n')
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

function stripSourceNumberPhrases(text: string): string {
  const citationPrefix = new RegExp(`^\\s*(?:according to|as (?:stated|noted|shown|reported) in|per|from)\\s+(?:the\\s+)?${citationLabelPattern}\\s+\\d+\\s*,?\\s*`, 'i')
  const citationSubject = new RegExp(`^\\s*(?:the\\s+)?${citationLabelPattern}\\s+\\d+\\s+(?:states|says|notes|indicates|mentions|reports)\\s+(?:that\\s+)?`, 'i')
  const bracketCitation = new RegExp(`\\s*[([](?:${citationLabelPattern})\\s+\\d+[)\\]]`, 'gi')
  const inlineCitationPrefix = new RegExp(`\\b(?:according to|as (?:stated|noted|shown|reported) in|per|from)\\s+(?:the\\s+)?${citationLabelPattern}\\s+\\d+\\s*,?\\s*`, 'gi')
  const leakedInstructionSentence = /\s*[^.!?]*(?:source numbers?|excerpt labels?|excerpts?\s+label(?:led|ed)|phrases like\s+["']?according to)[^.!?]*[.!?]/gi

  return text
    .replace(citationPrefix, '')
    .replace(citationSubject, '')
    .replace(bracketCitation, '')
    .replace(inlineCitationPrefix, '')
    .replace(leakedInstructionSentence, '')
    .replace(/\s{2,}/g, ' ')
    .trim()
}

export function answerWithOrderedSources(text: string, sources: ChatSource[]): { text: string; sources: ChatSource[] } {
  const referencedSourceIndex = firstReferencedSourceIndex(text, sources.length)
  const orderedSources =
    referencedSourceIndex === undefined
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

function cleanMaterialTitle(title?: string): string {
  return (title ?? '')
    .replace(/\.(pdf|md|markdown|txt)$/i, '')
    .replace(/\s+/g, ' ')
    .trim()
}

function materialSuggestionContext(materials: CourseMaterial[]): string {
  const materialLines = materials
    .slice(0, 8)
    .map((material) => {
      const title = cleanMaterialTitle(material.title) || material.title || 'Untitled PDF'
      const detail = [material.pageCount ? `${material.pageCount} pages` : '', material.chunkCount ? `${material.chunkCount} chunks` : '']
        .filter(Boolean)
        .join(', ')
      return detail ? `- ${title} (${detail})` : `- ${title}`
    })

  if (materialLines.length === 0) {
    return ''
  }

  const remainingCount = Math.max(0, materials.length - materialLines.length)
  return [
    'Loaded PDFs:',
    ...materialLines,
    remainingCount > 0 ? `- ${remainingCount} more PDF${remainingCount === 1 ? '' : 's'}` : ''
  ]
    .filter(Boolean)
    .join('\n')
}

export function questionSuggestionMessages(request: EngineQuestionSuggestionRequest): StudyChatMessage[] {
  const systemMessage = request.modelSettings?.systemMessage?.trim()
  const count = questionSuggestionCount(request.applicationSettings)
  const suggestionPrompt = formatFollowUpInstruction(
    request.modelSettings?.suggestedFollowUpPrompt?.trim() || defaultFollowUpPrompt(),
    count
  )
  const context = sourceContext(request.retrievedSources ?? [])
  const materialContext = materialSuggestionContext(request.materials)
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

  messages.push({
    role: 'user',
    content: [
      request.messages.length === 0
        ? 'The user has loaded PDFs but has not asked a first question yet.'
        : 'Suggest questions the user can ask next.',
      materialContext,
      context,
      'Use the same suggestion prompt below. Return only the questions.',
      suggestionPrompt
    ]
      .filter(Boolean)
      .join('\n\n')
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
