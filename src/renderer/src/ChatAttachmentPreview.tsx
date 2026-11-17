import { FileText, Loader2, X } from 'lucide-react'
import type { ChatAttachment } from '@shared/chat-attachment'

export function ChatAttachmentPreview({ attachment, onRemove }: {
  attachment: ChatAttachment
  onRemove?: () => void
}) {
  return <div className="chat-attachment">
    <details>
      <summary><FileText size={15} aria-hidden="true" /><strong>{attachment.name}</strong>
        <span>{attachment.kind === 'pdf' ? `${attachment.pages.length} page${attachment.pages.length === 1 ? '' : 's'}` : attachment.kind === 'image' ? 'Screenshot' : 'Text'}</span>
      {attachment.extraction && <span>Read in {(attachment.extraction.durationMs / 1000).toFixed(2)}s
        {attachment.extraction.peakMemoryMiB !== undefined ? ` · ${Math.round(attachment.extraction.peakMemoryMiB)} MiB peak worker RAM` : ''}</span>}
      </summary>
      <pre>{attachment.pages.map(page => `${attachment.kind === 'pdf' ? `Page ${page.page}\n` : ''}${page.text}`).join('\n\n')}</pre>
    </details>
    {onRemove && <button type="button" className="message-action" onClick={onRemove} aria-label="Remove attachment" title="Remove attachment"><X size={16} aria-hidden="true" /></button>}
  </div>
}

export function PendingChatAttachment({ name, onRemove }: { name: string; onRemove: () => void }) {
  return <div className="chat-attachment" role="status" aria-live="polite" aria-busy="true">
    <Loader2 size={15} className="spin" aria-hidden="true" />
    <div className="attachment-pending-label"><strong>{name || 'Choosing file…'}</strong>
      {name && <span>Reading…</span>}
    </div>
    {name && <button type="button" className="message-action" onClick={onRemove} aria-label="Remove attachment" title="Remove attachment"><X size={16} aria-hidden="true" /></button>}
  </div>
}
