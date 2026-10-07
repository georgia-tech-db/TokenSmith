import { useLayoutEffect, useRef, type ReactNode } from 'react'
import { Library, PanelLeft } from 'lucide-react'
import type { StudyMode } from '@shared/practice'
import { StudyModeTabs } from './StudyModeTabs'
import './study-workspace.css'

export function StudyHeader({ mode, onModeChange, historyOpen, onHistoryToggle, materialsOpen, onMaterialsToggle, count, children }: {
  mode: StudyMode; onModeChange: (mode: StudyMode) => void; historyOpen: boolean; onHistoryToggle: () => void
  materialsOpen: boolean; onMaterialsToggle: () => void; count: number; children: ReactNode
}) {
  const header = useRef<HTMLElement>(null)
  useLayoutEffect(() => {
    const element = header.current
    const frame = element?.closest<HTMLElement>('.study-frame')
    if (!element || !frame) return
    // Drawers start below the header even when its controls wrap or text size changes.
    const observer = new ResizeObserver(() => frame.style.setProperty('--study-header-height', `${element.getBoundingClientRect().height}px`))
    observer.observe(element)
    return () => observer.disconnect()
  }, [])
  return <header ref={header} className="study-topbar">
    <button className="icon-button subtle" type="button" aria-label="Toggle history" title="Toggle history" aria-expanded={historyOpen} onClick={onHistoryToggle}><PanelLeft size={19} /></button>
    <StudyModeTabs mode={mode} onChange={onModeChange} />
    <div className="study-model-controls">{children}</div>
    <button className="study-materials-toggle" type="button" aria-label="Toggle materials" title="Choose study documents" aria-expanded={materialsOpen} onClick={onMaterialsToggle}><Library size={17} /><span>Materials</span><span>{count}</span></button>
  </header>
}
