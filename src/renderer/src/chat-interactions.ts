import type { ApplicationSettings, ChatMessage } from '@shared/app-state'
import type { EngineChatRequest } from '@shared/engine'
import { lastChatExchange } from '../../shared/study-chat-pipeline'

export function addQuoteToDraft(draft: string, selection: string): string {
  const text = selection.trim()
  if (!text) return draft
  const quote = text.split(/\r?\n/).map((line) => `> ${line}`).join('\n')
  return `${quote}\n\n${draft}`
}

// Use the selected answer's own question and context, never the latest chat turn.
export function questionForAnswer(messages: ChatMessage[], answerId: string): ChatMessage | undefined {
  const answerIndex = messages.findIndex((message) => message.id === answerId && message.role === 'assistant')
  if (answerIndex < 0) return undefined
  return messages.slice(0, answerIndex).reverse().find((message) => message.role === 'user')
}

export function canExplainSimpler(message: ChatMessage): boolean {
  return message.role === 'assistant' && Boolean(message.text.trim()) &&
    Boolean(message.sources?.length) && message.conversationContextMode !== 'clarify' &&
    message.explanationDepth !== 'simple' && (!message.kind || message.kind === 'chat')
}

export function simplerExplanationSettings(settings: ApplicationSettings): ApplicationSettings {
  return { ...settings, explanationDepthEnabled: true, explanationDepth: 'simple',
    suggestionMode: 'off', followUpSuggestionCount: 0 }
}

export function simplerExplanationRequest(
  messages: ChatMessage[], answerId: string,
  context: Pick<EngineChatRequest, 'model' | 'materials' | 'settings' | 'modelSettings'>
): EngineChatRequest | undefined {
  const answerIndex = messages.findIndex(message => message.id === answerId)
  const answer = messages[answerIndex]
  const question = questionForAnswer(messages, answerId)
  if (!answer || !question || !canExplainSimpler(answer) || answer.simplerExplanation) return undefined
  const questionIndex = messages.findIndex(message => message.id === question.id)
  // Older conversations have no saved request metadata. Recover only the exchange
  // preceding this question; a later exchange may be about something else entirely.
  const previous = answer.conversationContextMode !== 'standalone'
    ? lastChatExchange(messages.slice(0, questionIndex)) : undefined
  const saved = answer.answerContext ?? {
    prompt: question.text,
    conversationContextMode: previous ? 'contextual' as const : 'standalone' as const,
    referenceExchange: previous
  }
  return {
    ...context, ...saved, messages: messages.slice(0, answerIndex),
    answerToSimplify: answer.text,
    retrievedSources: answer.sources,
    applicationSettings: simplerExplanationSettings(context.settings.application)
  }
}

// Editing keeps the question's identity but invalidates all answers after it.
export function replaceQuestion(messages: ChatMessage[], messageId: string, text: string): ChatMessage[] | undefined {
  const index = messages.findIndex((message) => message.id === messageId && message.role === 'user')
  if (index < 0 || !text.trim()) return undefined
  return [...messages.slice(0, index), { id: messageId, role: 'user', text: text.trim() }]
}
