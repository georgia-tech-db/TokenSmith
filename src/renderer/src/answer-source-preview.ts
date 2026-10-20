import type { MarkdownSourceDocument, PdfSourceDocument } from '@shared/engine'
import type { ChatMessage } from '@shared/app-state'

export type AnswerSourceDocument =
  | { kind: 'markdown'; document: MarkdownSourceDocument }
  | { kind: 'pdf'; document: PdfSourceDocument & { searchTerm?: string } }

export type SourceReadingStatus = 'preparing' | 'ready' | 'stopped'

export interface SourceReadingContext {
  question: string
  keptOpen: boolean
  status: SourceReadingStatus
}

// Historical answers retain their own question, rather than borrowing the latest one.
export function questionForSourceAnswer(messages: ChatMessage[], answer?: ChatMessage): string | undefined {
  if (!answer) return undefined
  if (answer.answerContext?.prompt) return answer.answerContext.prompt
  const answerIndex = messages.findIndex(message => message.id === answer.id)
  for (let index = answerIndex - 1; index >= 0; index--) {
    if (messages[index].role === 'user') return messages[index].text
  }
  return undefined
}

// A preview belongs to one pending answer. Late loads and completions must never
// reopen a dismissed preview or close a source the student opened manually.
export function createAnswerSourcePreview<T>(publish: (document: T | null) => void) {
  let owner: string | null = null
  let revision = 0
  let loadStarted = false
  let hasDocument = false
  let keptOpen = false
  const dismiss = () => {
    owner = null
    revision++
    loadStarted = false
    hasDocument = false
    keptOpen = false
    publish(null)
  }
  return {
    begin(requestId: string) {
      dismiss()
      owner = requestId
    },
    async open(requestId: string, load: () => Promise<T>) {
      if (owner !== requestId || loadStarted) return
      loadStarted = true
      const expectedRevision = revision
      try {
        const document = await load()
        if (owner === requestId && revision === expectedRevision) {
          hasDocument = true
          publish(document)
        }
      } catch {
        // A missing source must not interrupt or delay answer generation.
      }
    },
    keepOpen(requestId: string) {
      if (owner !== requestId || !hasDocument) return false
      keptOpen = true
      return true
    },
    finish(requestId: string) {
      if (owner !== requestId) return
      if (!keptOpen) dismiss()
      else {
        // Retain the mounted reader and scroll position, but reject late loads.
        owner = null
        revision++
      }
    },
    dismiss
  }
}
