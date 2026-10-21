import { useEffect, useRef, useState } from 'react'
import { FileText, Pencil, Plus, Trash2 } from 'lucide-react'
import type { StudyMode } from '@shared/practice'

interface HistoryItem { id: string; title: string; detail?: string }

export function StudySidebar({ mode, activeId, groups, onNew, onSelect, onRename, onDelete, onClear, onViewLog }: {
  mode: StudyMode
  activeId?: string
  groups: { label: string; items: HistoryItem[] }[]
  onNew: () => void
  onSelect: (id: string) => void
  onRename: (id: string, title: string) => void
  onDelete: (id: string) => void
  onClear: () => void
  onViewLog: () => void
}) {
  const [editing, setEditing] = useState<{ id: string; title: string } | null>(null)
  const cancelRename = useRef(false)
  useEffect(() => { setEditing(null) }, [activeId])
  const practice = mode === 'practice'
  function commitRename() {
    const title = editing?.title.trim()
    if (editing && title && !cancelRename.current) onRename(editing.id, title)
    setEditing(null)
  }
  return <aside className="chat-sidebar study-sidebar" aria-label={practice ? 'Practice sessions' : 'Conversation list'}>
    <button className="new-chat-button" type="button" onClick={onNew}><Plus size={17} aria-hidden="true" /><span>{practice ? 'New practice' : 'New Chat'}</span></button>
    <div className="conversation-groups">
      {groups.filter(group => group.items.length).map(group => <section className="conversation-group" aria-label={group.label} key={group.label}>
        <h2>{group.label}</h2>
        <div className="conversation-list">{group.items.map(item => {
          const selected = item.id === activeId
          return <div className={`conversation-item ${selected ? 'is-selected' : ''}`} key={item.id}>
            {editing?.id === item.id ? <form className="conversation-rename-form" onSubmit={event => { event.preventDefault(); commitRename() }}>
              <input autoFocus aria-label={practice ? 'Practice name' : 'Chat name'} value={editing.title}
                onChange={event => setEditing({ id: item.id, title: event.target.value })} onBlur={commitRename}
                onKeyDown={event => {
                  if (event.key === 'Escape') { event.preventDefault(); cancelRename.current = true; setEditing(null) }
                }} />
            </form> : <button className="conversation-select" type="button" aria-current={selected ? 'true' : undefined} title={item.title} onClick={() => onSelect(item.id)}>
              <span>{item.title}</span>{item.detail && <small>{item.detail}</small>}
            </button>}
            {selected && editing?.id !== item.id && <div className="conversation-actions" aria-label={`${item.title} actions`}>
              <button className="conversation-action" type="button" aria-label={`Rename ${item.title}`} title={practice ? 'Rename practice' : 'Rename chat'}
                onClick={() => { cancelRename.current = false; setEditing({ id: item.id, title: item.title }) }}><Pencil size={17} aria-hidden="true" /></button>
              <button className="conversation-action" type="button" aria-label={`Delete ${item.title}`} title={practice ? 'Delete practice' : 'Delete chat'}
                onClick={() => onDelete(item.id)}><Trash2 size={18} aria-hidden="true" /></button>
            </div>}
          </div>
        })}</div>
      </section>)}
    </div>
    <div className="chat-sidebar-footer">
      <button className="view-log-button" type="button" onClick={onViewLog}><FileText size={15} aria-hidden="true" /><span>View Log</span></button>
      <button className="clear-chats-button" type="button" onClick={onClear}><Trash2 size={15} aria-hidden="true" /><span>{practice ? 'Clear Practice History' : 'Clear All Chats'}</span></button>
    </div>
  </aside>
}
