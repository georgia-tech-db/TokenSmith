import type { MarkdownSourceDocument, PdfSourceDocument } from '@shared/engine'

export type AnswerSourceDocument =
  | { kind: 'markdown'; document: MarkdownSourceDocument }
  | { kind: 'pdf'; document: PdfSourceDocument & { searchTerm?: string } }

// A preview belongs to one pending answer. Late loads and manual dismissal must
// never reopen it, or close a source the student opened themselves.
export function createAnswerSourcePreview<T>(publish: (document: T | null) => void) {
  let owner: string | null = null
  let revision = 0
  let opened = false
  const dismiss = () => {
    owner = null
    revision++
    opened = false
    publish(null)
  }
  return {
    begin(requestId: string) {
      dismiss()
      owner = requestId
    },
    async open(requestId: string, load: () => Promise<T>) {
      if (owner !== requestId || opened) return
      opened = true
      const expectedRevision = revision
      try {
        const document = await load()
        if (owner === requestId && revision === expectedRevision) publish(document)
      } catch {
        // A missing source must not interrupt or delay answer generation.
      }
    },
    finish(requestId: string) { if (owner === requestId) dismiss() },
    dismiss
  }
}
