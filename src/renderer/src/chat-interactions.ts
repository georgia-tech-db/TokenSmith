import type { ChatMessage } from '@shared/app-state'

export function addQuoteToDraft(draft: string, selection: string): string {
  const text = selection.trim()
  if (!text) return draft
  const quote = text.split(/\r?\n/).map((line) => `> ${line}`).join('\n')
  return `${quote}\n\n${draft}`
}

// Editing keeps the question's identity but invalidates all answers after it.
export function replaceQuestion(messages: ChatMessage[], messageId: string, text: string): ChatMessage[] | undefined {
  const index = messages.findIndex((message) => message.id === messageId && message.role === 'user')
  if (index < 0 || !text.trim()) return undefined
  return [...messages.slice(0, index), { id: messageId, role: 'user', text: text.trim() }]
}
