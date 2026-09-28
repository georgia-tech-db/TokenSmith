import { useEffect, useRef, useState } from 'react'
import { Check, ChevronDown, GraduationCap } from 'lucide-react'
import type { ExplanationDepth } from '@shared/app-state'

const depthChoices: Array<{ value: ExplanationDepth; label: string; hint: string }> = [
  { value: 'simple', label: 'Simple', hint: 'Plain language for a first pass' },
  { value: 'standard', label: 'Standard', hint: 'The usual explanation' },
  { value: 'detailed', label: 'In-Depth', hint: 'Conditions, trade-offs, and nuance' }
]

export function ChatDepthPicker({ depth, disabled, onSelect }: {
  depth: ExplanationDepth; disabled: boolean; onSelect: (depth: ExplanationDepth) => void
}) {
  const [open, setOpen] = useState(false)
  const root = useRef<HTMLDivElement>(null)
  const trigger = useRef<HTMLButtonElement>(null)
  const active = depthChoices.find(choice => choice.value === depth) ?? depthChoices[1]
  function close() { setOpen(false); trigger.current?.focus() }
  useEffect(() => {
    if (!open) return
    const outside = (event: PointerEvent) => { if (!root.current?.contains(event.target as Node)) setOpen(false) }
    document.addEventListener('pointerdown', outside)
    return () => document.removeEventListener('pointerdown', outside)
  }, [open])
  return <div className="chat-model-control chat-depth-control" ref={root} onBlur={event => {
    if (!event.currentTarget.contains(event.relatedTarget as Node | null)) setOpen(false)
  }} onKeyDown={event => {
    if (event.key === 'Escape' && open) { event.preventDefault(); event.stopPropagation(); close() }
    if (open && (event.key === 'ArrowDown' || event.key === 'ArrowUp')) {
      event.preventDefault()
      const items = [...(root.current?.querySelectorAll<HTMLElement>('.chat-model-menu button') || [])]
      const index = items.indexOf(document.activeElement as HTMLElement)
      items[(index + (event.key === 'ArrowDown' ? 1 : -1) + items.length) % items.length]?.focus()
    }
  }}>
    <button ref={trigger} className="chat-model-trigger" type="button" disabled={disabled}
      aria-label={`Explanation depth: ${active.label}`} aria-expanded={open} aria-controls="chat-depth-menu"
      title="How far answers unpack an idea"
      onClick={() => setOpen(!open)}>
      <GraduationCap size={17} />
      <span>{active.label}</span><ChevronDown size={16} />
    </button>
    {open && <div className="chat-model-menu" id="chat-depth-menu" aria-label="Explanation depth">
      <div className="chat-model-options">
        {depthChoices.map(choice => <button type="button" className="chat-model-option" key={choice.value}
          onClick={() => { close(); onSelect(choice.value) }}>
          <span><strong>{choice.label}</strong><small>{choice.hint}</small></span>
          {choice.value === active.value && <Check size={17} aria-label="Selected" />}
        </button>)}
      </div>
    </div>}
  </div>
}
