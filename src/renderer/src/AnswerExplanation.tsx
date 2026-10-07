import { useEffect, useRef } from 'react'
import { BookOpen } from 'lucide-react'
import type { AnswerView, ChatMessage } from '@shared/app-state'
import { answerForDisplay } from '../../shared/study-chat-pipeline'
import { MessageText } from './MessageText'
import './answer-explanation.css'

export function AnswerExplanation({ message, pending, error, errorView, disabled, onSimplify, onReasoning, onSelectView }: {
  message: ChatMessage
  pending: boolean
  error?: string
  errorView?: AnswerView
  disabled: boolean
  onSimplify?: () => void
  onReasoning?: () => void
  onSelectView: (view: AnswerView) => void
}) {
  const simple = message.explanationView === 'simple' && Boolean(message.simplerExplanation)
  const reasoning = message.explanationView === 'reasoning' && Boolean(message.reasoningAnswer)
  const simpleButton = useRef<HTMLButtonElement>(null)
  const reasoningButton = useRef<HTMLButtonElement>(null)
  const simplifyAction = useRef<HTMLButtonElement>(null)
  const wasPending = useRef(pending)
  useEffect(() => {
    if (wasPending.current && !pending) {
      const target = error ? (errorView === 'reasoning' ? reasoningButton : simplifyAction)
        : reasoning ? reasoningButton : simple ? simpleButton : undefined
      target?.current?.focus({ preventScroll: true })
    }
    wasPending.current = pending
  }, [pending, simple, reasoning, error, errorView])
  return <div className="answer-explanation">
    {(message.simplerExplanation || message.reasoningAnswer || onReasoning) && <div className="answer-version-switch" role="group" aria-label="Answer version">
      <button type="button" disabled={pending} aria-pressed={!simple && !reasoning} onClick={() => onSelectView('original')}>
        {message.reasoning?.used ? 'Reasoning' : 'Basic'}
      </button>
      {message.simplerExplanation && <button ref={simpleButton} type="button" disabled={pending} aria-pressed={simple} onClick={() => onSelectView('simple')}>Simpler explanation</button>}
      {(message.reasoningAnswer || onReasoning) && <button ref={reasoningButton} type="button"
        disabled={pending || (!message.reasoningAnswer && disabled)} aria-pressed={reasoning}
        title={message.reasoningAnswer ? 'Show the answer with reasoning' : 'Generate an answer with reasoning enabled. This may take longer.'}
        onClick={() => message.reasoningAnswer ? onSelectView('reasoning') : onReasoning?.()}>
        Reasoning
      </button>}
    </div>}
    <div data-chat-selectable data-chat-message-id={message.id}><MessageText text={answerForDisplay(message).text} /></div>
    {!pending && !message.simplerExplanation && onSimplify && <div className="answer-simplify-action">
      <button ref={simplifyAction} type="button" disabled={disabled} onClick={onSimplify}>
        <BookOpen size={16} aria-hidden="true" />
        <span>Explain more simply</span>
      </button>
    </div>}
    {error && !pending && <p className="answer-simplify-error" role="alert">{error}</p>}
  </div>
}
