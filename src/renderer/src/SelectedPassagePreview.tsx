import { Quote, X } from 'lucide-react'
import type { ChatSelectedPassage } from '@shared/app-state'

export function SelectedPassagePreview({ passage, disabled, onRemove }: {
  passage: ChatSelectedPassage
  disabled?: boolean
  onRemove?: () => void
}) {
  return (
    <div className="selected-passage-preview" role="group" aria-label="Selected passage">
      <div className="selected-passage-heading">
        <span><Quote size={14} aria-hidden="true" /> {passage.role === 'assistant' ? 'From TokenSmith\'s answer' : 'From your question'}</span>
        {onRemove && <button type="button" className="selected-passage-remove" disabled={disabled}
          aria-label="Remove selected passage" title="Remove selected passage" onClick={onRemove}>
          <X size={16} aria-hidden="true" />
        </button>}
      </div>
      <blockquote className="selected-passage-text">{passage.text}</blockquote>
    </div>
  )
}
