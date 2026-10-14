import { useEffect, useRef } from 'react'
import { MessageSquare } from 'lucide-react'
import type { PracticeQuestion } from '@shared/practice'
import './answer-explanation.css'

export function PracticeComposer({ question, busy, isActive, onCheck }: {
  question: PracticeQuestion; busy: boolean; isActive: boolean
  onCheck: (answer: string) => void
}) {
  const choices = useRef<HTMLDivElement>(null)
  const attempt = question.attempts.at(-1)
  const binary = question.format === 'true_false'
  const answerReady = binary && typeof question.correctAnswer === 'boolean' && Boolean(question.referenceAnswer?.trim()) && question.sources.length > 0
  useEffect(() => {
    if (isActive && !attempt) choices.current?.querySelector<HTMLButtonElement>('button')?.focus()
  }, [question.id, isActive, attempt])
  return <div className="composer-area practice-composer" role="region" aria-label="Practice composer">
    {binary && !answerReady && <p className="practice-muted">This saved question has no usable answer guide. Prepare another question.</p>}
    <div className="composer">
      <div className="practice-choices" role="group" aria-label="Your answer" ref={choices}>
        {binary && ['True', 'False'].map(choice => <button key={choice} type="button"
          aria-pressed={(attempt?.answer ?? question.draft) === choice} disabled={busy || !answerReady || Boolean(attempt)}
          onClick={() => onCheck(choice)}>{choice}</button>)}
      </div>
    </div>
  </div>
}

export function PracticeActions({ onChat }: { onChat: () => void }) {
  return <button type="button" onClick={onChat}>
    <MessageSquare size={16} aria-hidden="true" />Chat about this question
  </button>
}
