import { useEffect, useId, useRef } from 'react'
import { ArrowRight, Pencil, SendHorizonal, X } from 'lucide-react'
import type { PracticeQuestion } from '@shared/practice'
import { MessageComposer } from './MessageComposer'
import { SelectedPassagePreview } from './SelectedPassagePreview'
import './answer-explanation.css'

interface QuestionControls {
  question: PracticeQuestion
  busy: boolean
  onChange: (update: (question: PracticeQuestion) => PracticeQuestion) => void
}

export function PracticeComposer({ question, busy, available, isActive, onChange, onCheck, onHelp }: QuestionControls & {
  available: boolean; isActive: boolean; onCheck: () => void; onHelp: () => void
}) {
  const input = useRef<HTMLTextAreaElement>(null)
  const choices = useRef<HTMLDivElement>(null)
  const choiceName = useId()
  const helping = question.composerMode === 'help'
  const binary = question.format === 'true_false'
  const attempt = question.attempts.at(-1)
  const value = helping ? question.helpDraft ?? '' : question.draft
  const answerReady = Boolean(question.rubric) && (!binary || typeof question.correctAnswer === 'boolean')
  const validValue = !helping && binary ? value === 'True' || value === 'False' : Boolean(value.trim())
  const submitDisabled = busy || ((!binary || helping) && !available) || !validValue || (!helping && !answerReady)
  const submitLabel = helping ? 'Send' : attempt ? 'Check revision' : 'Check answer'
  useEffect(() => {
    if (!isActive) return
    if (binary && !helping) {
      (choices.current?.querySelector<HTMLInputElement>('input:checked') ?? choices.current?.querySelector<HTMLInputElement>('input'))?.focus()
    } else input.current?.focus()
  }, [question.id, helping, isActive, binary])
  return <form className="composer-area practice-composer" aria-label="Practice composer" onSubmit={event => {
    event.preventDefault()
    if (submitDisabled) return
    if (helping) onHelp(); else onCheck()
  }}>
    <div className="answer-version-switch practice-input-mode" role="group" aria-label="Message intent">
      <button type="button" aria-pressed={!helping} disabled={busy || !answerReady}
        onClick={() => onChange(current => ({ ...current, composerMode: 'answer', reviewing: false }))}>Answer</button>
      <button type="button" aria-pressed={helping} disabled={busy}
        onClick={() => onChange(current => ({ ...current, composerMode: 'help' }))}>Ask for help</button>
    </div>
    {!answerReady && <p className="practice-muted">This saved question has no usable answer guide. Prepare another question.</p>}
    {binary && !helping ? <div className="composer">
      <div className="practice-choices" role="radiogroup" aria-label={attempt ? 'Revise your answer' : 'Your answer'} ref={choices}>
        {['True', 'False'].map(choice => <label key={choice}>
          <input type="radio" name={choiceName} value={choice} checked={value === choice} disabled={busy || !answerReady}
            onChange={() => onChange(current => ({ ...current, draft: choice, reviewing: false }))} />
          <span>{choice}</span>
        </label>)}
      </div>
      <button className="send-button" type="submit" disabled={submitDisabled} aria-label={submitLabel} title={submitLabel}>
        <SendHorizonal size={21} aria-hidden="true" />
      </button>
    </div> : <MessageComposer inputRef={input} isActive={isActive} value={value}
      onChange={text => onChange(current => ({ ...current, ...(helping ? { helpDraft: text } : { draft: text, reviewing: false }) }))}
      label={helping ? 'Question about this exercise' : attempt ? 'Revise your answer' : 'Your answer'}
      placeholder={helping ? 'Ask about this question...' : attempt ? 'Revise your answer...' : 'Write your answer...'}
      submitLabel={submitLabel}
      disabled={busy || (!helping && !answerReady)} submitDisabled={submitDisabled}>
      {helping && question.helpPassage && <SelectedPassagePreview passage={question.helpPassage} disabled={busy} label="From this practice"
        onRemove={() => onChange(current => ({ ...current, helpPassage: undefined }))} />}
    </MessageComposer>}
  </form>
}

export function PracticeActions({ question, busy, nextDisabled, onChange, onNext }: QuestionControls & {
  nextDisabled: boolean; onNext: () => void
}) {
  const attempt = question.attempts.at(-1)
  const revising = question.composerMode !== 'help' && !question.reviewing && Boolean(attempt)
  return <>
    {(attempt || question.answerRevealed) && question.rubric && (revising
      ? <button type="button" disabled={busy} onClick={() => onChange(current => ({ ...current, draft: current.attempts.at(-1)?.answer ?? current.draft, reviewing: true, composerMode: 'help' }))}><X size={16} aria-hidden="true" />Cancel revision</button>
      : <button type="button" disabled={busy} onClick={() => onChange(current => ({ ...current, composerMode: 'answer', reviewing: false }))}><Pencil size={16} aria-hidden="true" />Try again</button>)}
    <button type="button" disabled={nextDisabled} onClick={onNext}>Next question<ArrowRight size={16} aria-hidden="true" /></button>
  </>
}
