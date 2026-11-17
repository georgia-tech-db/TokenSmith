import type { ApplicationSettings, ChatMessage, ChatSelectedPassage, LocalModel } from '@shared/app-state'
import type { EngineChatRequest } from '@shared/engine'
import { lastChatExchange } from '../../shared/study-chat-pipeline'
import { answerReasoningSettings } from '../../shared/reasoning'

export interface ChatDraft {
  text: string
  selectedPassage?: ChatSelectedPassage
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
    message.explanationDepth !== 'simple'
}

export function canRetryWithReasoning(message: ChatMessage, model?: LocalModel): boolean {
  return message.role === 'assistant' && Boolean(message.text.trim()) &&
    message.conversationContextMode !== 'clarify' &&
    Boolean(message.answerContext) && !message.reasoningAnswer &&
    message.reasoning?.used === false && message.reasoning.supported === true &&
    model?.engine === 'ollama' && model.status === 'ready' && model.id === message.reasoning.modelId
}

export function reasoningRetryRequest(
  messages: ChatMessage[], answerId: string,
  context: Pick<EngineChatRequest, 'model' | 'materials' | 'settings' | 'modelSettings'>
): EngineChatRequest | undefined {
  const answerIndex = messages.findIndex(message => message.id === answerId)
  const answer = messages[answerIndex]
  const question = questionForAnswer(messages, answerId)
  if (!answer || !question || !canRetryWithReasoning(answer, context.model)) return undefined
  const saved = answer.answerContext!
  const modelSettings = saved.modelSettings ?? context.modelSettings
  if (!modelSettings) return undefined
  // Re-answer the same task with its saved evidence, not the old answer or later turns.
  return {
    ...context, ...saved,
    messages: messages.slice(0, messages.findIndex(message => message.id === question.id)),
    retrievedSources: answer.sources ?? [],
    reasoning: true,
    modelSettings: answerReasoningSettings({ ...modelSettings, reasoningMode: 'on' }, true),
    applicationSettings: { ...(saved.applicationSettings ?? context.settings.application),
      suggestionMode: context.settings.application.suggestionMode,
      followUpSuggestionCount: context.settings.application.followUpSuggestionCount }
  }
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
    ...context, selectedPassage: question.selectedPassage, ...saved, messages: messages.slice(0, answerIndex),
    answerToSimplify: answer.text,
    retrievedSources: answer.sources,
    modelSettings: context.modelSettings,
    applicationSettings: simplerExplanationSettings(context.settings.application)
  }
}

// Editing keeps the question's identity but invalidates all answers after it.
export function replaceQuestion(messages: ChatMessage[], messageId: string, text: string, selectedPassage?: ChatSelectedPassage): ChatMessage[] | undefined {
  const index = messages.findIndex((message) => message.id === messageId && message.role === 'user')
  if (index < 0 || !text.trim()) return undefined
  return [...messages.slice(0, index), { id: messageId, role: 'user', text: text.trim(), ...(selectedPassage ? { selectedPassage } : {}) }]
}
