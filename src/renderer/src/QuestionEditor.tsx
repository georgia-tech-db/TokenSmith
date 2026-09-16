import { useLayoutEffect, useRef, useState } from 'react'
import { SendHorizontal, X } from 'lucide-react'

export function QuestionEditor({ text, laterQuestions, disabled, onCancel, onSave }: {
  text: string
  laterQuestions: number
  disabled: boolean
  onCancel: () => void
  onSave: (text: string) => void
}) {
  const [draft, setDraft] = useState(text)
  const [confirmReplacement, setConfirmReplacement] = useState(false)
  const inputRef = useRef<HTMLTextAreaElement>(null)
  useLayoutEffect(() => {
    inputRef.current?.focus()
    inputRef.current?.setSelectionRange(text.length, text.length)
  }, [])
  useLayoutEffect(() => {
    const input = inputRef.current
    if (!input) return
    input.style.height = 'auto'
    input.style.height = `${Math.min(input.scrollHeight, 320)}px`
  }, [draft])

  return (
    <form className="question-editor" aria-label="Edit question" onSubmit={(event) => {
      event.preventDefault()
      if (!draft.trim() || disabled) return
      if (laterQuestions > 0 && !confirmReplacement) {
        setConfirmReplacement(true)
        return
      }
      onSave(draft)
    }}>
      <textarea ref={inputRef} aria-label="Edit question text" value={draft} disabled={disabled}
        onChange={(event) => { setDraft(event.target.value); setConfirmReplacement(false) }}
        onKeyDown={(event) => {
          if (event.nativeEvent.isComposing) return
          if (event.key === 'Escape') { event.preventDefault(); onCancel() }
          if (event.key === 'Enter' && (event.metaKey || event.ctrlKey)) {
            event.preventDefault()
            event.currentTarget.form?.requestSubmit()
          }
        }} />
      {laterQuestions > 0 && <p className="question-edit-warning" role={confirmReplacement ? 'alert' : undefined}>
        {confirmReplacement ? 'Replace' : 'Saving replaces'} the answer and {laterQuestions} later {laterQuestions === 1 ? 'question' : 'questions'}{confirmReplacement ? '? This cannot be undone.' : '.'}
      </p>}
      <div className="question-editor-actions">
        <button type="button" className="question-edit-cancel" onClick={onCancel}><X size={16} aria-hidden="true" /> Cancel</button>
        <button type="submit" className="question-edit-save" disabled={!draft.trim() || disabled}><SendHorizontal size={16} aria-hidden="true" /> {confirmReplacement ? 'Replace & resend' : 'Save & resend'}</button>
      </div>
    </form>
  )
}
