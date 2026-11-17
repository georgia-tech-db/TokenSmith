import { ArrowLeft, ExternalLink, X } from 'lucide-react'
import type { ChatSource } from '@shared/app-state'
import type { SourcePaneState } from './answer-source-preview'
import { MessageText } from './MessageText'
import { SourceNavigation } from './SourceNavigation'
import { sourceEntryKey } from './source-navigation'
import './source-pane.css'

export function SourcePane({ state, open, error, loading, onMove, onClose, onDocuments, onOpenDocument }: {
  state: SourcePaneState; open: boolean; error?: string; loading: boolean
  onMove: (direction: number) => void; onClose: () => void; onDocuments: () => void
  onOpenDocument: (source: ChatSource) => void
}) {
  const source = state.sources[state.index]
  return <aside className="study-materials study-source-pane" aria-label="Source passage" hidden={!open}>
    <header>
      <button type="button" className="practice-link" onClick={onDocuments}><ArrowLeft size={16} />Sources and documents</button>
      <button type="button" className="icon-button subtle" aria-label="Close source pane" onClick={onClose}><X size={18} /></button>
    </header>
    <div className="source-pane-location">
      <strong>{source.sectionHeader || source.documentTitle || source.title}</strong>
      <small>{[source.documentTitle || source.title, source.locator].filter(Boolean).join(' · ')}</small>
      <SourceNavigation navigation={{ index: state.index, count: state.sources.length,
        onPrevious: () => onMove(-1), onNext: () => onMove(1) }} />
      {state.question && <details><summary>Related question</summary><p>{state.question}</p></details>}
    </div>
    <div key={sourceEntryKey(source)} className="source-pane-passage" role="region" aria-label="Passage text" tabIndex={0}>
      <MessageText text={source.context || source.excerpt} />
    </div>
    <footer>
      {error && <p role="alert">{error}</p>}
      <button type="button" className="study-action-button" disabled={loading} onClick={() => onOpenDocument(source)}><ExternalLink size={16} />{loading ? 'Opening document…' : 'Open document'}</button>
    </footer>
  </aside>
}
