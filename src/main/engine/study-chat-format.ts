import type { ChatSource, ModelRuntimeSettings } from '../../shared/app-state'
import type { EngineChatRequest } from '../../shared/engine'
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
    'Use these excerpts when they are relevant. If the excerpts do not contain the answer, say that plainly.',
    ...sources.map((source, index) => {
      const title = source.title || `Source ${index + 1}`
      const locator = source.locator ? ` (${source.locator})` : ''
      return `Source ${index + 1}: ${title}${locator}\n${source.excerpt}`
    })
  ].join('\n\n')
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
  if (request.applicationSettings?.suggestionMode === 'off') {
    return 0
  }

  const count = Number(request.applicationSettings?.followUpSuggestionCount)
  if (!Number.isFinite(count)) {
    return defaultFollowUpSuggestionCount
  }

  return count <= minFollowUpSuggestionCount ? minFollowUpSuggestionCount : defaultFollowUpSuggestionCount
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
