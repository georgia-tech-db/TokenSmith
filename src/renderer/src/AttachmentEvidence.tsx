import type { ChatMessage } from '@shared/app-state'

export function AttachmentEvidence({ message }: { message: ChatMessage }) {
  const attachment = message.answerContext?.attachment
  if (!attachment) return null
  const pages = attachment.kind === 'pdf'
    ? ` · ${attachment.pages.length === 1 ? 'PDF page' : 'PDF pages'} ${attachment.pages.map(page => page.page).join(', ')}`
    : ''

  return <details className="attachment-evidence">
    <summary>Attachment evidence · {attachment.name}{pages}</summary>
    <pre>{attachment.pages.map(page => attachment.kind === 'pdf'
      ? `PDF page ${page.page}\n${page.text}` : page.text).join('\n\n')}</pre>
  </details>
}
