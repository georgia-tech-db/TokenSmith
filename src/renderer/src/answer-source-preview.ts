import type { ChatMessage, ChatSource } from '@shared/app-state'
import type { MarkdownSourceDocument, PdfSourceDocument } from '@shared/engine'
import { sourceNavigationList } from './source-navigation'

export type AnswerSourceDocument =
  | { kind: 'markdown'; document: MarkdownSourceDocument }
  | { kind: 'pdf'; document: PdfSourceDocument & { searchTerm?: string } }

export interface SourcePaneState {
  sources: ChatSource[]
  index: number
  question?: string
}

export function questionForSourceAnswer(messages: ChatMessage[], answer?: ChatMessage): string | undefined {
  if (!answer) return undefined
  if (answer.answerContext?.prompt) return answer.answerContext.prompt
  const index = messages.findIndex(message => message.id === answer.id)
  return messages.slice(0, Math.max(0, index)).reverse().find(message => message.role === 'user')?.text
}

// The pane contains retrieved text, so opening it needs no asynchronous document load.
// Completion preserves reading position; dismissal and manual selection reject late previews.
export function createSourcePane(publish: (value: SourcePaneState | null) => void) {
  let owner: string | null = null
  let state: SourcePaneState | null = null
  let manual = false
  const update = (value: SourcePaneState | null) => { state = value; publish(value) }
  return {
    begin(id: string) { owner = id; if (!manual) update(null) },
    open(id: string, sources: ChatSource[], question?: string) {
      if (owner !== id || manual || !sources.length) return false
      const navigation = sourceNavigationList(sources[0], sources)
      if (navigation.index < 0) return false
      update({ ...navigation, question })
      return true
    },
    select(source: ChatSource, sources: ChatSource[], question?: string) {
      const navigation = sourceNavigationList(source, sources)
      if (navigation.index < 0) return
      manual = true; owner = null
      update({ ...navigation, question })
    },
    move(direction: number) {
      if (!state || !state.sources[state.index + direction]) return
      manual = true; owner = null
      update({ ...state, index: state.index + direction })
    },
    finish(id: string) { if (owner === id) owner = null },
    dismissAutomatic() { if (!manual) { owner = null; update(null) } },
    dismiss() { owner = null; manual = false; update(null) }
  }
}
