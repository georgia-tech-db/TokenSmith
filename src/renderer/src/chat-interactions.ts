import type { ApplicationSettings, ChatMessage, ChatSelectedPassage, PinnedRunningExample } from '@shared/app-state'
import type { EngineChatRequest } from '@shared/engine'
import { questionWithSelection } from '../../shared/chat-selection'
import { lastChatExchange } from '../../shared/study-chat-pipeline'

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
    ...context, selectedPassage: question.selectedPassage, ...saved, messages: messages.slice(0, answerIndex),
    answerToSimplify: answer.text,
    retrievedSources: answer.sources,
    applicationSettings: simplerExplanationSettings(context.settings.application)
  }
}

// Editing keeps the question's identity but invalidates all answers after it.
export function replaceQuestion(messages: ChatMessage[], messageId: string, text: string, selectedPassage?: ChatSelectedPassage): ChatMessage[] | undefined {
  const index = messages.findIndex((message) => message.id === messageId && message.role === 'user')
  if (index < 0 || !text.trim()) return undefined
  return [...messages.slice(0, index), { id: messageId, role: 'user', text: text.trim(), ...(selectedPassage ? { selectedPassage } : {}) }]
}

export const runningExampleSavedText =
  'Saved as your running example. I will use it in later answers in this chat when a question refers to it.'

export function isRunningExampleMessage(message: ChatMessage): boolean {
  return message.kind === 'runningExample' || message.kind === 'runningExampleSaved'
}

// Saved examples are memory, not conversation: keep them away from the question rewriter.
export function messagesForModel(messages: ChatMessage[]): ChatMessage[] {
  return messages.filter((message) => !isRunningExampleMessage(message))
}

// Saving stores the student's text verbatim and confirms it; nothing is sent to a model.
export function saveRunningExample(text: string, ids: { user: string; assistant: string }): { messages: ChatMessage[]; pin: PinnedRunningExample } {
  return {
    messages: [
      { id: ids.user, role: 'user', text: text.trim(), kind: 'runningExample' },
      { id: ids.assistant, role: 'assistant', text: runningExampleSavedText, kind: 'runningExampleSaved' }
    ],
    pin: { messageId: ids.user }
  }
}

// Resolve against one conversation's messages at send time; anything unexpected means no pin.
export function resolveRunningExample(messages: ChatMessage[], pin?: PinnedRunningExample): string | undefined {
  if (!pin) return undefined
  const saved = messages.find((message) => message.id === pin.messageId)
  return saved?.role === 'user' && saved.kind === 'runningExample' && saved.text.trim() ? saved.text : undefined
}

// A pin survives a history change only if its message is still there unchanged.
export function pinAfterHistoryChange(
  pin: PinnedRunningExample | undefined, messages: ChatMessage[], editedMessageId?: string
): PinnedRunningExample | undefined {
  if (!pin || editedMessageId === pin.messageId) return undefined
  return resolveRunningExample(messages, pin) ? pin : undefined
}
