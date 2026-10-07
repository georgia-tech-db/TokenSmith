import { useEffect, useRef } from 'react'
import { ArrowLeft, ArrowRight, BookOpen, Check, Lightbulb, MessageSquare, Pencil } from 'lucide-react'
import type { ChatSource } from '@shared/app-state'
import type { PracticeQuestion, PracticeSession } from '@shared/practice'
import { MessageText } from './MessageText'
import { PracticeFeedbackView } from './PracticeFeedbackView'

export function PracticeQuestionView({ session, question, busy, available, onDraft, onCheck, onHint, onReveal,
  onRetry, onCancelRevision, onDiscuss, onNext, onPrevious, onFinish, onOpenSource }: {
  session: PracticeSession; question: PracticeQuestion; busy: boolean; available: boolean
  onDraft: (text: string) => void; onCheck: () => void; onHint: () => void; onReveal: () => void
  onRetry: () => void; onCancelRevision: () => void; onDiscuss: () => void; onNext: () => void; onPrevious: () => void; onFinish: () => void
  onOpenSource: (source: ChatSource, sources: ChatSource[]) => void
}) {
  const attempt = question.attempts.at(-1)
  const reviewing = question.reviewing && Boolean(attempt || question.answerRevealed)
  const correct = attempt && typeof attempt.feedback !== 'string' && attempt.feedback.verdict === 'correct'
  const needsReview = attempt && typeof attempt.feedback !== 'string' && attempt.feedback.verdict === 'needs_review'
  const input = useRef<HTMLTextAreaElement>(null)
  useEffect(() => { if (!reviewing && attempt) input.current?.focus() }, [reviewing, question.id])
  const nextDisabled = busy || (session.currentIndex + 1 >= session.questions.length && session.questions.length < session.totalQuestions && !available)
  return <>
    <div className="practice-row practice-between">
      <span>Question {session.currentIndex + 1} of {session.totalQuestions}</span>
      <button className="practice-link" type="button" onClick={onFinish} disabled={busy}>Finish session</button>
    </div>
    <div className="practice-progress" role="progressbar" aria-label="Questions reviewed" aria-valuemin={0} aria-valuemax={session.totalQuestions}
      aria-valuenow={session.questions.filter(item => item.attempts.length || item.answerRevealed).length}>
      {Array.from({ length: session.totalQuestions }, (_, index) => <span key={index} className={index === session.currentIndex ? 'is-current' : session.questions[index]?.attempts.length || session.questions[index]?.answerRevealed ? 'is-reviewed' : ''} />)}
    </div>
    <p className="practice-topic">{question.sources[0]?.sectionHeader || question.sources[0]?.documentTitle || session.title}</p>
    <div className="practice-question"><MessageText text={question.text} /></div>
    {!question.rubric && <p className="practice-muted" role="status">This older question has no saved grading guide. Its answers are preserved; start a new practice for updated feedback.</p>}
    {reviewing ? <section className="practice-submitted-answer" aria-label="Your submitted answer">
      <h3>Your answer</h3><MessageText text={attempt?.answer || question.draft || 'No answer submitted.'} />
    </section> : <form onSubmit={event => { event.preventDefault(); onCheck() }}>
      <label className="practice-answer-label" htmlFor={'practice-answer-' + question.id}>{attempt ? 'Revise your answer' : 'Your answer'}</label>
      <textarea ref={input} id={'practice-answer-' + question.id} value={question.draft} disabled={busy || !question.rubric}
        onChange={event => onDraft(event.target.value)} placeholder="Explain it in your own words..." rows={4} />
      <div className="practice-row practice-between practice-answer-actions">
        <div className="practice-row">
          <button type="button" className="practice-link" disabled={busy || !question.hint} onClick={onHint}><Lightbulb size={16} aria-hidden="true" />Hint</button>
          {attempt && <button type="button" className="practice-link" disabled={busy} onClick={onCancelRevision}>Cancel revision</button>}
        </div>
        <button className="practice-button is-primary" type="submit" disabled={busy || !available || !question.rubric || !question.draft.trim()}><Check size={16} aria-hidden="true" />{attempt ? 'Check revision' : 'Check answer'}</button>
      </div>
    </form>}
    {question.hintUsed && question.hint && !reviewing && <section className="practice-hint"><h3>Hint</h3><MessageText text={question.hint} /></section>}
    {attempt && <section className="practice-feedback" aria-label="Practice feedback">
      <div className="practice-row practice-between"><h3>{reviewing ? 'Feedback' : 'Feedback on your last answer'}</h3>
        <span className="practice-muted">Attempt {question.attempts.length}{attempt.assisted ? ' · With help or revision' : ''}</span></div>
      <PracticeFeedbackView feedback={attempt.feedback} />
      {question.attempts.length > 1 && <details><summary>Earlier attempts</summary>{question.attempts.slice(0, -1).map((item, index) => <section key={item.id} className="practice-past-attempt">
        <h3>Attempt {index + 1}</h3><h4>Your answer</h4><MessageText text={item.answer} /><PracticeFeedbackView feedback={item.feedback} />
      </section>)}</details>}
    </section>}
    {question.answerRevealed && question.referenceAnswer && <section className="practice-explanation" aria-label="Worked explanation">
      <h3>Explanation</h3><span className="practice-muted">Viewed before further attempts</span><MessageText text={question.referenceAnswer} />
    </section>}
    {(attempt || question.answerRevealed) && <details className="practice-passages"><summary><BookOpen size={16} aria-hidden="true" />Supporting passages ({question.sources.length})</summary>
      {question.sources.map((source, index) => <button className="practice-source" type="button" key={index} onClick={() => onOpenSource(source, question.sources)}>
        <span>{source.documentTitle || source.title}</span><span className="practice-muted">{source.sectionHeader || source.locator}</span><ArrowRight size={16} aria-hidden="true" />
      </button>)}
    </details>}
    <div className="practice-footer">
      <div className="practice-row">
        {reviewing && question.rubric && <button className={'practice-button' + (!correct && !needsReview ? ' is-primary' : '')} type="button" disabled={busy} onClick={onRetry}><Pencil size={16} aria-hidden="true" />Revise answer</button>}
        <button className="practice-button" type="button" disabled={busy || !question.referenceAnswer || question.answerRevealed} onClick={onReveal}><BookOpen size={16} aria-hidden="true" />{question.answerRevealed ? 'Explanation shown' : 'Show explanation'}</button>
        {(attempt || question.answerRevealed) && <button className="practice-button" type="button" disabled={busy} onClick={onDiscuss}><MessageSquare size={16} aria-hidden="true" />Discuss this</button>}
      </div>
      <button className={'practice-button' + (reviewing && (correct || needsReview) ? ' is-primary' : '')} type="button" disabled={nextDisabled} onClick={onNext}>{session.currentIndex + 1 === session.totalQuestions ? 'Finish practice' : 'Next question'}<ArrowRight size={16} aria-hidden="true" /></button>
    </div>
    {session.currentIndex > 0 && <button className="practice-link practice-previous" type="button" disabled={busy} onClick={onPrevious}><ArrowLeft size={16} aria-hidden="true" />Previous question</button>}
  </>
}
