import { ChevronDown, Pin, X } from 'lucide-react'
import type { SourceReadingContext } from './answer-source-preview'
import { SourceNavigation, type SourceNavigationProps } from './SourceNavigation'

export interface SourceReadingControls extends SourceReadingContext {
  onKeepOpen?: () => void
}

interface SourceViewerHeaderProps {
  title: string
  path: string
  sectionHeader?: string
  location?: string
  details?: string
  navigation?: SourceNavigationProps
  reading?: SourceReadingControls
  autoFocusClose?: boolean
  onClose: () => void
}

export function sourceViewerHeading(title: string, path: string, sectionHeader?: string) {
  const filename = path.split(/[\\/]/).pop()?.replace(/\.(pdf|md|markdown|txt)$/i, '').replace(/_/g, ' ').trim()
  const documentTitle = title.trim() || filename || 'Source passage'
  const section = sectionHeader?.trim()
  return { heading: section || documentTitle, documentTitle, hasSection: Boolean(section) }
}

function SourceReadingActions({ reading, onClose, closeLabel, autoFocusClose }: {
  reading?: SourceReadingControls
  onClose: () => void
  closeLabel: string
  autoFocusClose?: boolean
}) {
  const statusLabel = reading?.status === 'ready' ? 'Answer ready'
    : reading?.status === 'stopped' ? 'Generation stopped' : undefined

  return <div className="source-viewer-actions">
    {statusLabel && <span className="source-viewer-status" role="status">{statusLabel}</span>}
    {reading?.keptOpen ? <span className="source-viewer-retained" title="This passage stays open until you return to chat">
      <Pin size={15} aria-hidden="true" /> Stays open
    </span> : reading?.onKeepOpen && <button type="button" className="source-viewer-button is-quiet"
      onClick={reading.onKeepOpen} title="Keep this passage open when the answer arrives">
      <Pin size={15} aria-hidden="true" /> Keep open
    </button>}
    {reading ? <button type="button" className="source-viewer-button" autoFocus={autoFocusClose} onClick={onClose}>
      <X size={16} aria-hidden="true" /> Return to chat
    </button> : <button type="button" className="icon-button subtle" aria-label={closeLabel} autoFocus={autoFocusClose} onClick={onClose}>
      <X size={18} aria-hidden="true" />
    </button>}
  </div>
}

function SourceQuestion({ question }: { question: string }) {
  return <details className="source-viewer-question">
    <summary>
      <span className="source-viewer-question-label">Your question</span>
      <span className="source-viewer-question-preview">{question}</span>
      <ChevronDown size={16} aria-hidden="true" />
    </summary>
    <p>{question}</p>
  </details>
}

export function SourceViewerHeader({ title, path, sectionHeader, location, details, navigation, reading, autoFocusClose, onClose }: SourceViewerHeaderProps) {
  const { heading, documentTitle, hasSection } = sourceViewerHeading(title, path, sectionHeader)
  const subtitle = [hasSection ? documentTitle : undefined, location].filter(Boolean).join(' · ')
  const closeLabel = /\.pdf$/i.test(path) ? 'Close PDF viewer' : 'Close Markdown viewer'

  return <header className="pdf-viewer-header source-viewer-header">
    <div className="source-viewer-heading">
      <strong>{heading}</strong>
      {subtitle && <span>{subtitle}</span>}
      {details && <details className="source-viewer-details">
        <summary>Source details</summary>
        <span>{details}</span>
      </details>}
    </div>
    <SourceNavigation navigation={navigation} />
    <SourceReadingActions reading={reading} onClose={onClose} closeLabel={closeLabel} autoFocusClose={autoFocusClose} />
    {reading?.question && <SourceQuestion question={reading.question} />}
  </header>
}
