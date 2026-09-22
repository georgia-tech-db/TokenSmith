import { useEffect, useRef, useState } from 'react'
import { Check, ChevronDown, Cloud, Laptop, Plus, Search, Settings2 } from 'lucide-react'
import type { LocalModel } from '@shared/app-state'
import { isCloudGenerator } from '@shared/cloud-generators'

export function ChatModelPicker({ models, selectedModel, disabled, onSelect, onConnect, onManage }: {
  models: LocalModel[]; selectedModel?: LocalModel; disabled: boolean
  onSelect: (id: string) => void; onConnect: (model?: LocalModel) => void; onManage: () => void
}) {
  const [open, setOpen] = useState(false)
  const [query, setQuery] = useState('')
  const root = useRef<HTMLDivElement>(null)
  const trigger = useRef<HTMLButtonElement>(null)
  const search = useRef<HTMLInputElement>(null)
  const label = (model: LocalModel) => model.remoteModelName || model.ollamaModelName || model.name
  const choices = models.filter(model => isCloudGenerator(model) ||
    (model.engine === 'ollama' && model.role !== 'embedder' && model.status === 'ready'))
  const filtered = choices.filter(model => `${label(model)} ${model.providerName || ''}`.toLowerCase().includes(query.toLowerCase()))
  function close() { setOpen(false); trigger.current?.focus() }
  useEffect(() => {
    if (!open) return
    search.current?.focus()
    const outside = (event: PointerEvent) => { if (!root.current?.contains(event.target as Node)) setOpen(false) }
    document.addEventListener('pointerdown', outside)
    return () => document.removeEventListener('pointerdown', outside)
  }, [open])
  return <div className="chat-model-control" ref={root} onBlur={event => {
    if (!event.currentTarget.contains(event.relatedTarget as Node | null)) setOpen(false)
  }} onKeyDown={event => {
    if (event.key === 'Escape' && open) { event.preventDefault(); event.stopPropagation(); close() }
    if (open && (event.key === 'ArrowDown' || event.key === 'ArrowUp')) {
      event.preventDefault()
      const items = [...(root.current?.querySelectorAll<HTMLElement>('.chat-model-menu input, .chat-model-menu button') || [])]
      const index = items.indexOf(document.activeElement as HTMLElement)
      items[(index + (event.key === 'ArrowDown' ? 1 : -1) + items.length) % items.length]?.focus()
    }
  }}>
    <button ref={trigger} className="chat-model-trigger" type="button" disabled={disabled}
      aria-label="Choose chat model" aria-expanded={open} aria-controls="chat-model-menu"
      onClick={() => { setOpen(!open); setQuery('') }}>
      {selectedModel?.engine === 'remote' ? <Cloud size={17} /> : <Laptop size={17} />}
      <span>{selectedModel ? label(selectedModel) : 'Choose a model'}</span><ChevronDown size={16} />
    </button>
    {open && <div className="chat-model-menu" id="chat-model-menu" aria-label="Chat models">
      <label className="cloud-search"><Search size={16} /><input ref={search} type="search" aria-label="Find a chat model" placeholder="Find a model…" value={query} onChange={e => setQuery(e.target.value)} /></label>
      <div className="chat-model-options">
        {(['device', 'cloud'] as const).map(group => {
          const entries = filtered.filter(model => (model.engine === 'remote') === (group === 'cloud'))
          return entries.length > 0 && <div key={group}>
            <p className="cloud-eyebrow">{group === 'cloud' ? 'Cloud' : 'On this device'}</p>
            {entries.map(model => <button type="button" className="chat-model-option" key={model.id}
              aria-label={`${label(model)}${model.status !== 'ready' ? ', reconnect' : ''}`}
              onClick={() => { close(); model.status === 'ready' ? onSelect(model.id) : onConnect(model) }}>
              <span><strong>{label(model)}</strong><small>{model.status !== 'ready' ? 'Reconnect required' : model.providerName || 'Ollama'}</small></span>
              {selectedModel?.id === model.id && <Check size={17} aria-label="Selected" />}
            </button>)}
          </div>
        })}
        {!filtered.length && <p className="cloud-muted">{choices.length ? 'No matching models.' : 'Choose a cloud model to get started.'}</p>}
      </div>
      <div className="chat-model-menu-footer">
        <button type="button" onClick={() => { close(); onConnect() }}><Plus size={17} /><span>Connect a cloud model</span></button>
        <button type="button" onClick={() => { close(); onManage() }}><Settings2 size={16} /><span>Manage models</span></button>
      </div>
    </div>}
  </div>
}
