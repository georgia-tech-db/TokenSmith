import { useLayoutEffect, type ReactNode, type RefObject } from 'react'
import { SendHorizonal } from 'lucide-react'

export function MessageComposer({ value, onChange, inputRef, label, placeholder, submitLabel, disabled, submitDisabled, isActive, children }: {
  value: string
  onChange: (text: string) => void
  inputRef: RefObject<HTMLTextAreaElement | null>
  label: string
  placeholder: string
  submitLabel: string
  disabled: boolean
  submitDisabled: boolean
  isActive: boolean
  children?: ReactNode
}) {
  useLayoutEffect(() => {
    const input = inputRef.current
    if (!isActive || !input) return
    input.style.height = 'auto'
    input.style.height = `${Math.min(input.scrollHeight, 180)}px`
  }, [value, isActive, inputRef])

  return <div className="composer">
    {children}
    <textarea aria-label={label} rows={1} ref={inputRef} disabled={disabled} placeholder={placeholder} value={value}
      onChange={event => onChange(event.target.value)}
      onKeyDown={event => {
        if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing) {
          event.preventDefault()
          if (!submitDisabled) event.currentTarget.form?.requestSubmit()
        }
      }} />
    <button className="send-button" type="submit" disabled={submitDisabled} aria-label={submitLabel} title={submitLabel}>
      <SendHorizonal size={21} aria-hidden="true" />
    </button>
  </div>
}
