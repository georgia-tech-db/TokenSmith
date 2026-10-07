import { MessageSquare, NotebookPen } from 'lucide-react'
import type { StudyMode } from '@shared/practice'
import './practice.css'

export function StudyModeTabs({ mode, onChange }: { mode: StudyMode; onChange: (mode: StudyMode) => void }) {
  return <div className="study-mode-tabs" role="group" aria-label="Study mode">
    <button type="button" aria-pressed={mode === 'chat'} onClick={() => onChange('chat')}><MessageSquare size={16} aria-hidden="true" />Chat</button>
    <button type="button" aria-pressed={mode === 'practice'} onClick={() => onChange('practice')}><NotebookPen size={16} aria-hidden="true" />Practice</button>
  </div>
}
