import type { ChatMessage, ChatSelectedPassage } from '@shared/app-state'

export interface ChatDraft {
  text: string
  selectedPassage?: ChatSelectedPassage
}

// Editing keeps the question's identity but invalidates all answers after it.
export function replaceQuestion(messages: ChatMessage[], messageId: string, text: string, selectedPassage?: ChatSelectedPassage): ChatMessage[] | undefined {
  const index = messages.findIndex((message) => message.id === messageId && message.role === 'user')
  if (index < 0 || !text.trim()) return undefined
  return [...messages.slice(0, index), { id: messageId, role: 'user', text: text.trim(), ...(selectedPassage ? { selectedPassage } : {}) }]
}
