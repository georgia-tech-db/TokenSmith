import { useEffect, useRef, useState } from 'react'
import { BookOpen, FileText } from 'lucide-react'
import type { ChatSource } from '@shared/app-state'
import type { PdfSourceThumbnail } from '@shared/engine'
import { canOpenSource, cleanMaterialTitle, isMarkdownSource, isPdfSource } from './source-display'

export const maxSourceTrayCards = 5

function sourceTrayKey(source: ChatSource, index: number) {
  return `${source.sourceId ?? source.chunkId ?? source.chunkRowid ?? source.path ?? source.title}-${index}`
}

function sourceFileTypeLabel(source: ChatSource) {
  if (isPdfSource(source)) return 'PDF'
  if (source.path?.toLowerCase().endsWith('.txt')) return 'TXT'
  return isMarkdownSource(source) ? 'MD' : 'TXT'
}

function sourcePageLabel(source: ChatSource) {
  if (source.pageStart && source.pageEnd && source.pageEnd > source.pageStart) return `Pages ${source.pageStart}-${source.pageEnd}`
  if (source.pageStart) return `Page ${source.pageStart}`
  if (isMarkdownSource(source) && source.lineFrom && source.lineTo && source.lineTo > source.lineFrom) return `Lines ${source.lineFrom}-${source.lineTo}`
  if (isMarkdownSource(source) && source.lineFrom) return `Line ${source.lineFrom}`
  return source.locator
}

export function SourceTray({ error, sources, onOpenSource }: {
  error?: string | null
  sources: ChatSource[]
  onOpenSource: (source: ChatSource) => void
}) {
  const [thumbnails, setThumbnails] = useState<Record<string, PdfSourceThumbnail>>({})
  const list = useRef<HTMLDivElement>(null)
  useEffect(() => {
    let canceled = false
    setThumbnails({})
    if (list.current) list.current.scrollTop = 0
    const bridge = window.tokensmith
    if (bridge?.getPdfThumbnailForSource) {
      sources.forEach((source, index) => {
        if (!isPdfSource(source)) return
        const key = sourceTrayKey(source, index)
        void bridge.getPdfThumbnailForSource(source).then(thumbnail => {
          if (!canceled) setThumbnails(current => ({ ...current, [key]: thumbnail }))
        }).catch(() => {
          // Thumbnail rendering is best-effort; the source card still opens the PDF.
        })
      })
    }
    return () => { canceled = true }
  }, [sources])

  if (!sources.length) return null
  return <section className="source-tray" aria-label="Sources">
    <h3 className="source-tray-header"><BookOpen size={15} aria-hidden="true" /><span>Sources</span></h3>
    <div ref={list} className="source-tray-list" role="region" aria-label="Source passages" tabIndex={0}>
      {sources.map((source, index) => {
        const canOpen = canOpenSource(source)
        const key = sourceTrayKey(source, index)
        const thumbnail = thumbnails[key]
        const contextLabel = [source.collectionName, source.sectionHeader].filter(Boolean).join(' · ')
        return <button className={`source-tray-card ${canOpen ? '' : 'is-text-only'}`} disabled={!canOpen}
          key={key} type="button" onClick={() => onOpenSource(source)}>
          <span className="source-media" aria-hidden="true">
            <span className={`source-thumbnail ${thumbnail ? 'has-image' : ''}`}>
              {thumbnail ? <img src={thumbnail.dataUrl} alt="" /> : <><FileText size={17} /><strong>{sourceFileTypeLabel(source)}</strong></>}
            </span>
          </span>
          <span className="source-tray-copy">
            <strong>{cleanMaterialTitle(source.documentTitle || source.title) || 'Source'}</strong>
            {contextLabel && <small>{contextLabel}</small>}
            <span>{source.excerpt}</span>
          </span>
          <span className="source-card-badges"><small className="source-page-label">{sourcePageLabel(source)}</small></span>
        </button>
      })}
    </div>
    {error && <p className="source-tray-error" role="alert">{error}</p>}
  </section>
}
