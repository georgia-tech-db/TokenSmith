import type { ApplicationSettings, ChatMessage, SuggestionMode } from '@shared/app-state'

export function addQuoteToDraft(draft: string, selection: string): string {
  const text = selection.trim()
  if (!text) return draft
  const quote = text.split(/\r?\n/).map((line) => `> ${line}`).join('\n')
  return `${quote}\n\n${draft}`
}

// Re-explaining reuses the answer's own evidence, so it needs the question that produced it.
export function questionForAnswer(messages: ChatMessage[], answerId: string): ChatMessage | undefined {
  const answerIndex = messages.findIndex((message) => message.id === answerId)
  if (answerIndex < 0) return undefined
  return messages.slice(0, answerIndex).reverse().find((message) => message.role === 'user')
}

// "Explain simpler" reads as another suggested follow-up, so it lives and dies with that
// setting. It is also pointless on an answer that is already simple, and re-simplifying a
// quiz question would defeat the quiz.
export function canExplainSimpler(message: ChatMessage, suggestionMode: SuggestionMode): boolean {
  return suggestionMode !== 'off' &&
    message.role === 'assistant' &&
    message.explanationDepth !== 'simple' &&
    (!message.kind || message.kind === 'chat')
}

// Settings for one retelling. The depth gate exists to stop a stale picker value from
// quietly shaping answers, so an explicit click has to switch it on: the chip is offered
// alongside the follow-ups, whether or not the student uses the depth picker at all.
export function simplerExplanationSettings(settings: ApplicationSettings): ApplicationSettings {
  return { ...settings, explanationDepthEnabled: true, explanationDepth: 'simple' }
}

// Editing keeps the question's identity but invalidates all answers after it.
export function replaceQuestion(messages: ChatMessage[], messageId: string, text: string): ChatMessage[] | undefined {
  const index = messages.findIndex((message) => message.id === messageId && message.role === 'user')
  if (index < 0 || !text.trim()) return undefined
  return [...messages.slice(0, index), { id: messageId, role: 'user', text: text.trim() }]
}
