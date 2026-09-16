import { useCallback, useEffect, useLayoutEffect, useMemo, useRef } from 'react'
import Markdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import { LocateFixed, X } from 'lucide-react'
import type { MarkdownSourceDocument } from '@shared/engine'
import { rehypeSourceLines, resolveSourceLineRange, sourceAnchorIndex } from './markdown-source'

export function MarkdownSourceViewer({ viewer, onClose }: { viewer: MarkdownSourceDocument; onClose: () => void }) {
  const dialogRef = useRef<HTMLDialogElement>(null)
  const viewportRef = useRef<HTMLDivElement>(null)
  const range = useMemo(() => resolveSourceLineRange(viewer.text, viewer.lineFrom, viewer.lineTo, viewer.chunkText),
    [viewer.text, viewer.lineFrom, viewer.lineTo, viewer.chunkText])
  const subtitle = [viewer.sectionHeader, viewer.locator].filter(Boolean).join(' · ')
  const lineLabel = range ? range.to > range.from ? `Lines ${range.from}–${range.to}` : `Line ${range.from}` : ''

  const renderedDocument = useMemo(() => (
    <div className="message-text markdown-source-content">
      <Markdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[[rehypeSourceLines, { range }], [rehypeKatex, { trust: false, maxSize: 10 }]]}
        skipHtml
        urlTransform={(url) => /^(https?:\/\/|mailto:)/i.test(url) ? url : undefined}
        components={{
          a: ({ href, children }) => href
            ? <a href={href} target="_blank" rel="noopener noreferrer">{children}</a>
            : <span>{children}</span>,
          img: ({ alt }) => <span className="markdown-source-image">{alt}</span>,
          table: ({ node: _node, children, ...props }) => <div className="message-table"><table {...props}>{children}</table></div>
        }}
      >
        {viewer.text}
      </Markdown>
    </div>
  ), [viewer.text, range])

  const jumpToChunk = useCallback(() => {
    const viewport = viewportRef.current
    if (!viewport || !range) return
    const blocks = Array.from(viewport.querySelectorAll<HTMLElement>('[data-source-line-start]'))
    const index = sourceAnchorIndex(blocks.map((block) => ({
      from: Number(block.dataset.sourceLineStart), to: Number(block.dataset.sourceLineEnd)
    })), range.from)
    const target = blocks[index]
    if (!target) return
    viewport.scrollTo({ top: viewport.scrollTop + target.getBoundingClientRect().top - viewport.getBoundingClientRect().top - 28 })
  }, [range])

  useLayoutEffect(() => {
    const dialog = dialogRef.current
    dialog?.showModal()
    return () => dialog?.close()
  }, [])

  useLayoutEffect(() => {
    jumpToChunk()
  }, [jumpToChunk, viewer.text])

  useEffect(() => {
    const viewport = viewportRef.current
    if (!viewport) return
    let cancelled = false
    const cancel = () => { cancelled = true }
    viewport.addEventListener('wheel', cancel, { passive: true })
    viewport.addEventListener('touchstart', cancel, { passive: true })
    viewport.addEventListener('pointerdown', cancel)
    viewport.addEventListener('keydown', cancel)
    // Math fonts can change the heights of blocks above the selected passage.
    void window.document.fonts.ready.then(() => { if (!cancelled) jumpToChunk() })
    return () => {
      cancel()
      viewport.removeEventListener('wheel', cancel)
      viewport.removeEventListener('touchstart', cancel)
      viewport.removeEventListener('pointerdown', cancel)
      viewport.removeEventListener('keydown', cancel)
    }
  }, [jumpToChunk, viewer.text])

  return (
    <dialog ref={dialogRef} className="markdown-source-dialog" aria-label="Source Markdown viewer"
      onCancel={(event) => { event.preventDefault(); onClose() }}>
      <section className="markdown-viewer-panel">
        <header className="pdf-viewer-header">
          <div>
            <strong>{viewer.title}</strong>
            {subtitle && <span>{subtitle}</span>}
          </div>
          <button className="icon-button subtle" type="button" aria-label="Close Markdown viewer" onClick={onClose}>
            <X size={18} aria-hidden="true" />
          </button>
        </header>
        <div className="markdown-source-toolbar">
          <span>{range ? `Highlighted passage · ${lineLabel}` : 'Full document · Chunk position unavailable'}</span>
          <button type="button" className="markdown-source-jump" onClick={jumpToChunk} disabled={!range}>
            <LocateFixed size={16} aria-hidden="true" /> Back to chunk
          </button>
        </div>
        <div className="markdown-source-viewport" ref={viewportRef} tabIndex={0} role="region" aria-label="Full source document">
          {renderedDocument}
        </div>
      </section>
    </dialog>
  )
}
