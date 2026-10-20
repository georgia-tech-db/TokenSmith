import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState, useId } from 'react'
import Markdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import { ChevronDown, ChevronRight, LocateFixed } from 'lucide-react'
import type { MarkdownSourceDocument } from '@shared/engine'
import type { SourceNavigationProps } from './SourceNavigation'
import { SourceViewerHeader, type SourceReadingControls } from './SourceViewerHeader'
import { rehypeSourceHeadings, rehypeSourceLines, resolveSourceLineRange, sourceAnchorIndex } from './markdown-source'

interface MarkdownSourceViewerProps {
  viewer: MarkdownSourceDocument
  onClose: () => void
  automatic?: boolean
  navigation?: SourceNavigationProps
  reading?: SourceReadingControls
}

// Leave enough space above the passage to retain its surrounding heading.
const highlightedPassageScrollOffset = 56

export function MarkdownSourceViewer({ viewer, onClose, automatic = false, navigation, reading }: MarkdownSourceViewerProps) {
  const dialogRef = useRef<HTMLDialogElement>(null)
  const viewportRef = useRef<HTMLDivElement>(null)
  const tocId = useId()
  const [tocOpen, setTocOpen] = useState(() => {
    try { return localStorage.getItem('tokensmith-source-toc-open') === 'true' } catch { return false }
  })
  const [headings, setHeadings] = useState<Array<{ id: string; label: string; depth: number }>>([])
  const [activeHeading, setActiveHeading] = useState('')
  const userNavigated = useRef(false)
  const minimumHeadingDepth = headings.reduce((depth, heading) => Math.min(depth, heading.depth), 6)
  function toggleToc() {
    setTocOpen(value => {
      try { localStorage.setItem('tokensmith-source-toc-open', String(!value)) } catch { /* Optional preference. */ }
      return !value
    })
  }
  function jumpToHeading(id: string) {
    const viewport = viewportRef.current
    const heading = viewport?.querySelector<HTMLElement>(`[id="${id}"]`)
    if (!viewport || !heading) return
    userNavigated.current = true
    setActiveHeading(id)
    viewport.scrollTo({ top: viewport.scrollTop + heading.getBoundingClientRect().top - viewport.getBoundingClientRect().top - 24 })
    heading.focus({ preventScroll: true })
  }
  const range = useMemo(() => resolveSourceLineRange(viewer.text, viewer.lineFrom, viewer.lineTo, viewer.chunkText),
    [viewer.text, viewer.lineFrom, viewer.lineTo, viewer.chunkText])
  const lineLabel = range ? range.to > range.from ? `Lines ${range.from}–${range.to}` : `Line ${range.from}` : ''

  const renderedDocument = useMemo(() => (
    <div className="message-text markdown-source-content">
      <Markdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[[rehypeSourceLines, { range }], rehypeSourceHeadings, [rehypeKatex, { trust: false, maxSize: 10 }]]}
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

  const jumpToPassage = useCallback(() => {
    const viewport = viewportRef.current
    if (!viewport || !range) return
    const blocks = Array.from(viewport.querySelectorAll<HTMLElement>('[data-source-line-start]'))
    const index = sourceAnchorIndex(blocks.map((block) => ({
      from: Number(block.dataset.sourceLineStart), to: Number(block.dataset.sourceLineEnd)
    })), range.from)
    const target = blocks[index]
    if (!target) return
    viewport.scrollTo({ top: viewport.scrollTop + target.getBoundingClientRect().top - viewport.getBoundingClientRect().top - highlightedPassageScrollOffset })
  }, [range])

  useLayoutEffect(() => {
    const dialog = dialogRef.current
    if (automatic) dialog?.show()
    else dialog?.showModal()
    return () => dialog?.close()
  }, [automatic])

  useLayoutEffect(() => {
    userNavigated.current = false
    setActiveHeading('')
    setHeadings(Array.from(viewportRef.current?.querySelectorAll<HTMLElement>('[data-source-heading-depth]') ?? []).map(heading => ({
      id: heading.id, label: heading.dataset.sourceHeadingLabel || '', depth: Number(heading.dataset.sourceHeadingDepth)
    })))
    if (range) jumpToPassage()
    else viewportRef.current?.scrollTo({ top: 0 })
  }, [jumpToPassage, viewer.text, range])

  useEffect(() => {
    const viewport = viewportRef.current
    if (!viewport) return
    let cancelled = false
    const cancel = () => { cancelled = true; userNavigated.current = true }
    viewport.addEventListener('wheel', cancel, { passive: true })
    viewport.addEventListener('touchstart', cancel, { passive: true })
    viewport.addEventListener('pointerdown', cancel)
    viewport.addEventListener('keydown', cancel)
    // Math fonts can change the heights of blocks above the selected passage.
    void window.document.fonts.ready.then(() => { if (!cancelled && !userNavigated.current) jumpToPassage() })
    return () => {
      cancel()
      viewport.removeEventListener('wheel', cancel)
      viewport.removeEventListener('touchstart', cancel)
      viewport.removeEventListener('pointerdown', cancel)
      viewport.removeEventListener('keydown', cancel)
    }
  }, [jumpToPassage, viewer.text])

  return (
    <dialog ref={dialogRef} className="markdown-source-dialog" aria-label="Source Markdown viewer" aria-modal={!automatic}
      onKeyDown={event => { if (automatic && event.key === 'Escape') { event.preventDefault(); onClose() } }}
      onCancel={(event) => { event.preventDefault(); onClose() }}>
      <section className="markdown-viewer-panel">
        <SourceViewerHeader title={viewer.title} path={viewer.path} sectionHeader={viewer.sectionHeader}
          location={range ? undefined : 'Full document'}
          details={[viewer.locator, lineLabel].filter(Boolean).join(' · ')}
          navigation={navigation} reading={reading} onClose={onClose} />
        <div className="markdown-source-toolbar">
          <button type="button" className="markdown-source-jump is-quiet" aria-expanded={tocOpen} aria-controls={tocId} onClick={toggleToc} disabled={headings.length === 0}>
            {tocOpen ? <ChevronDown size={16} aria-hidden="true" /> : <ChevronRight size={16} aria-hidden="true" />} Contents
          </button>
          {!range && <span>No highlight is available for this source.</span>}
          <button type="button" className="markdown-source-jump is-quiet" aria-label="Jump to highlighted passage"
            title="Jump to highlighted passage" onClick={() => { userNavigated.current = true; jumpToPassage() }} disabled={!range}>
            <LocateFixed size={16} aria-hidden="true" /> Jump to passage
          </button>
        </div>
        <div className={`markdown-source-body${tocOpen && headings.length ? ' has-toc' : ''}`}>
          {tocOpen && headings.length > 0 && <nav id={tocId} className="markdown-source-toc" aria-label="Table of contents">
            <ol>{headings.map(heading => <li key={heading.id}>
              <a href={`#${heading.id}`} aria-current={activeHeading === heading.id ? 'location' : undefined}
                style={{ paddingInlineStart: `${12 + (heading.depth - minimumHeadingDepth) * 12}px` }}
                onClick={event => { event.preventDefault(); jumpToHeading(heading.id) }}>{heading.label}</a>
            </li>)}</ol>
          </nav>}
          <div className="markdown-source-viewport" ref={viewportRef} tabIndex={0} role="region" aria-label="Full source document">
            {renderedDocument}
          </div>
        </div>
      </section>
    </dialog>
  )
}
