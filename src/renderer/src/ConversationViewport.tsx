import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import type { ReactNode } from 'react'
import { createPortal } from 'react-dom'
import { ArrowDown, MessageSquarePlus } from 'lucide-react'
import type { ChatMessage } from '@shared/app-state'

interface SelectedText {
  text: string
  top: number
  left: number
}

function readChatSelection(viewport: HTMLElement | null): SelectedText | null {
  const selected = window.getSelection()
  const containerFor = (node: Node | null) => (node instanceof Element ? node : node?.parentElement)
    ?.closest('[data-chat-selectable]')
  const container = containerFor(selected?.anchorNode ?? null)
  if (!viewport || !selected || selected.isCollapsed || !selected.rangeCount || !container ||
      !viewport.contains(container) || container !== containerFor(selected.focusNode)) return null
  const text = selected.toString()
  if (!text.trim()) return null
  const rect = selected.getRangeAt(0).getBoundingClientRect()
  const bounds = viewport.getBoundingClientRect()
  if (rect.bottom < bounds.top || rect.top > bounds.bottom) return null
  return {
    text,
    left: Math.max(8, Math.min(rect.left, window.innerWidth - 168)),
    top: Math.max(bounds.top + 4, Math.min(rect.top - 42, bounds.bottom - 42))
  }
}

export function ConversationViewport({ messages, pending, children, onQuote, canQuote }: {
  messages: ChatMessage[]
  pending: boolean
  children: ReactNode
  onQuote: (text: string) => void
  canQuote: boolean
}) {
  const viewportRef = useRef<HTMLDivElement>(null)
  const navigatorRef = useRef<HTMLElement>(null)
  const questionElements = useRef(new Map<string, HTMLElement>())
  const atBottomRef = useRef(true)
  const previousLastId = useRef<string | undefined>(undefined)
  const [activeId, setActiveId] = useState<string>()
  const [atBottom, setAtBottom] = useState(true)
  const [selection, setSelection] = useState<SelectedText | null>(null)
  const [preview, setPreview] = useState<{ index: number; top: number; left: number } | null>(null)
  const questions = useMemo(() => messages.filter((message) => message.role === 'user'), [messages])

  function updateScrollPosition() {
    const viewport = viewportRef.current
    if (!viewport) return
    const bottom = viewport.scrollHeight - viewport.scrollTop - viewport.clientHeight < 48
    atBottomRef.current = bottom
    setAtBottom(bottom)
    const threshold = viewport.getBoundingClientRect().top + 64
    let current = questions[0]?.id
    for (const question of questions) {
      const element = questionElements.current.get(question.id)
      if (element && element.getBoundingClientRect().top <= threshold) current = question.id
      else break
    }
    setActiveId(bottom ? questions.at(-1)?.id : current)
  }

  useLayoutEffect(() => {
    const viewport = viewportRef.current
    if (!viewport) return
    questionElements.current = new Map(Array.from(viewport.querySelectorAll<HTMLElement>('[data-question-id]'))
      .map((element) => [element.dataset.questionId!, element]))
    const last = messages.at(-1)
    const newQuestion = last?.role === 'user' && last.id !== previousLastId.current
    if (atBottomRef.current || newQuestion) viewport.scrollTop = viewport.scrollHeight
    previousLastId.current = last?.id
    updateScrollPosition()
  }, [messages, pending])

  useEffect(() => {
    const viewport = viewportRef.current
    if (!viewport) return
    const observer = new ResizeObserver(() => {
      if (atBottomRef.current) viewport.scrollTop = viewport.scrollHeight
      updateScrollPosition()
    })
    observer.observe(viewport)
    if (viewport.firstElementChild) observer.observe(viewport.firstElementChild)
    return () => observer.disconnect()
  }, [questions])

  useEffect(() => {
    const navigator = navigatorRef.current
    const marker = navigator?.querySelector<HTMLElement>('[aria-current]')
    if (!navigator || !marker) return
    if (marker.offsetTop < navigator.scrollTop) navigator.scrollTop = marker.offsetTop
    if (marker.offsetTop + marker.offsetHeight > navigator.scrollTop + navigator.clientHeight) {
      navigator.scrollTop = marker.offsetTop + marker.offsetHeight - navigator.clientHeight
    }
  }, [activeId])

  useEffect(() => {
    const viewport = viewportRef.current
    if (!viewport) return
    let frame = 0
    let pointerDown = false
    const updateSelection = () => {
      cancelAnimationFrame(frame)
      frame = requestAnimationFrame(() => {
        setSelection(pointerDown ? null : readChatSelection(viewport))
      })
    }
    const dismiss = () => {
      cancelAnimationFrame(frame)
      setSelection(null)
    }
    const onPointerDown = (event: PointerEvent) => {
      if (event.target instanceof Element && event.target.closest('.selection-chat-action')) return
      pointerDown = true
      dismiss()
    }
    const onPointerUp = () => { pointerDown = false; updateSelection() }
    const onPointerCancel = () => { pointerDown = false; dismiss() }
    const onKeyDown = (event: KeyboardEvent) => { if (event.key === 'Escape') dismiss() }
    document.addEventListener('selectionchange', updateSelection)
    document.addEventListener('pointerdown', onPointerDown, true)
    document.addEventListener('pointerup', onPointerUp)
    document.addEventListener('pointercancel', onPointerCancel)
    document.addEventListener('keydown', onKeyDown)
    window.addEventListener('blur', onPointerCancel)
    viewport.addEventListener('scroll', dismiss)
    window.addEventListener('resize', dismiss)
    return () => {
      dismiss()
      document.removeEventListener('selectionchange', updateSelection)
      document.removeEventListener('pointerdown', onPointerDown, true)
      document.removeEventListener('pointerup', onPointerUp)
      document.removeEventListener('pointercancel', onPointerCancel)
      document.removeEventListener('keydown', onKeyDown)
      window.removeEventListener('blur', onPointerCancel)
      viewport.removeEventListener('scroll', dismiss)
      window.removeEventListener('resize', dismiss)
    }
  }, [])

  return (
    <div className="conversation-viewport">
      <div className="message-list" ref={viewportRef} onScroll={updateScrollPosition}>
        <div className="conversation-content">{children}</div>
      </div>
      {questions.length > 1 && (
        <nav className="question-navigator" ref={navigatorRef} aria-label="Questions in this chat" onScroll={() => setPreview(null)}>
          {questions.map((question, index) => (
            <button
              className="question-marker"
              key={question.id}
              type="button"
              aria-label={`Question ${index + 1}: ${question.text}`}
              aria-current={activeId === question.id ? 'location' : undefined}
              onMouseEnter={(event) => {
                const rect = event.currentTarget.getBoundingClientRect()
                setPreview({ index, top: rect.top, left: rect.right + 8 })
              }}
              onMouseLeave={() => setPreview(null)}
              onFocus={(event) => {
                const rect = event.currentTarget.getBoundingClientRect()
                setPreview({ index, top: rect.top, left: rect.right + 8 })
              }}
              onBlur={() => setPreview(null)}
              onClick={() => {
                const element = questionElements.current.get(question.id)
                const viewport = viewportRef.current
                if (!element || !viewport) return
                viewport.scrollTo({
                  top: viewport.scrollTop + element.getBoundingClientRect().top - viewport.getBoundingClientRect().top - 20,
                  behavior: window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth'
                })
                element.focus({ preventScroll: true })
              }}
            >
              <span className="question-marker-tick" />
            </button>
          ))}
        </nav>
      )}
      {preview && questions[preview.index] && createPortal(
        <div className="question-marker-preview" aria-hidden="true" style={{
          top: Math.max(8, Math.min(preview.top, window.innerHeight - 150)),
          left: Math.max(8, Math.min(preview.left, window.innerWidth - Math.min(300, window.innerWidth - 70) - 8))
        }}>
          <strong>Question {preview.index + 1}</strong><span>{questions[preview.index].text}</span>
        </div>, document.body
      )}
      {!atBottom && questions.length > 0 && (
        <button className="chat-jump-latest" type="button" aria-label="Go to latest message" title="Go to latest message"
          onClick={() => viewportRef.current?.scrollTo({
            top: viewportRef.current.scrollHeight,
            behavior: window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth'
          })}>
          <ArrowDown size={18} aria-hidden="true" />
        </button>
      )}
      {selection && canQuote && createPortal(
        <button className="selection-chat-action" type="button" style={{ top: selection.top, left: selection.left }}
          onPointerDown={(event) => event.preventDefault()}
          onClick={() => {
            // Selection-change rendering can lag behind the user's final range.
            const current = readChatSelection(viewportRef.current)
            if (current) onQuote(current.text)
            window.getSelection()?.removeAllRanges()
            setSelection(null)
          }}>
          <MessageSquarePlus size={16} aria-hidden="true" /> Add to chat
        </button>, document.body
      )}
    </div>
  )
}
