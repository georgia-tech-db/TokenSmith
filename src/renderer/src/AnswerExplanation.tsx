import { useEffect, useRef } from 'react'
import { BookOpen } from 'lucide-react'
import type { ChatMessage } from '@shared/app-state'
import { answerForDisplay } from '../../shared/study-chat-pipeline'
import { MessageText } from './MessageText'
import './answer-explanation.css'

export function AnswerExplanation({ message, pending, error, disabled, onSimplify, onSelectView }: {
  message: ChatMessage
  pending: boolean
  error?: string
  disabled: boolean
  onSimplify?: () => void
  onSelectView: (view: 'original' | 'simple') => void
}) {
  const simple = message.explanationView === 'simple' && Boolean(message.simplerExplanation)
  const simpleButton = useRef<HTMLButtonElement>(null)
  const retryButton = useRef<HTMLButtonElement>(null)
  const wasPending = useRef(pending)
  useEffect(() => {
    if (wasPending.current && !pending) {
      if (simple) simpleButton.current?.focus({ preventScroll: true })
      else if (error) retryButton.current?.focus({ preventScroll: true })
    }
    wasPending.current = pending
  }, [pending, simple, error])
  return <div className="answer-explanation">
    {message.simplerExplanation && <div className="answer-version-switch" role="group" aria-label="Answer version">
      <button type="button" aria-pressed={!simple} onClick={() => onSelectView('original')}>Original</button>
      <button ref={simpleButton} type="button" aria-pressed={simple} onClick={() => onSelectView('simple')}>Simpler explanation</button>
    </div>}
    <div data-chat-selectable data-chat-message-id={message.id}><MessageText text={answerForDisplay(message).text} /></div>
    {pending ? null : !message.simplerExplanation && onSimplify && <div className="answer-simplify-action">
      <button ref={retryButton} type="button" disabled={disabled} onClick={onSimplify}>
        <BookOpen size={16} aria-hidden="true" />
        <span>{error ? 'Try again' : 'Explain more simply'}</span>
      </button>
    </div>}
    {error && !pending && <p className="answer-simplify-error" role="alert">{error}</p>}
  </div>
}
