import type { ReactNode } from 'react'
import { Database, PanelRightOpen } from 'lucide-react'
import { practiceQuestionText, type PracticeTopic, type PracticeQuestion } from '@shared/practice'
import { MessageText } from './MessageText'
import { PracticeFeedbackView } from './PracticeFeedbackView'
import { MessageAvatar } from './MessageAvatar'

export function PracticeQuestionView({ question, index, active, busy, onShowSources, sourcesOpen, actions, choices }: {
  question: PracticeQuestion; index: number; active: boolean; busy: boolean
  onShowSources?: () => void
  sourcesOpen: boolean
  actions?: ReactNode
  choices?: ReactNode
}) {
  return <section className={`practice-thread-question ${active ? 'is-active' : ''}`} data-question-id={question.id} tabIndex={-1} aria-label={`Practice question ${index + 1}`}>
    <div className="practice-question-messages">
      <div className="practice-question-heading"><span>Question {index + 1}</span>{question.concept && <small>{question.concept.title}</small>}</div>
      <div data-chat-selectable data-chat-message-id={question.id}><MessageText text={practiceQuestionText(question)} /></div>
      {choices}
      {question.attempts.map(attempt => <section key={attempt.id} className="practice-feedback practice-feedback-card" aria-label="Practice feedback">
        <p className="practice-submitted-answer">Your answer: <strong>{attempt.answer}</strong></p>
        {attempt.assisted && <p className="practice-muted">Answered with help · mastery unchanged</p>}
        <PracticeFeedbackView feedback={attempt.feedback} />
      </section>)}
    </div>
    <div className="practice-question-footer">
      {actions && <div className="answer-simplify-action practice-study-actions" role="group" aria-label="Practice actions">
        {actions}
      </div>}
      {onShowSources && <div className="practice-source-actions">
        <button className="source-chip" type="button" disabled={busy} aria-expanded={sourcesOpen} onClick={onShowSources}>
          <Database size={16} aria-hidden="true" /><span>{question.sources.length} {question.sources.length === 1 ? 'Source' : 'Sources'}</span><PanelRightOpen size={14} aria-hidden="true" />
        </button>
      </div>}
    </div>
  </section>
}

export function PracticeTopicView({ topic }: { topic: PracticeTopic }) {
  return <details className="practice-topic-reference" open><summary>From Chat</summary>
    {[{ role: 'user', text: topic.question }, { role: 'assistant', text: topic.answer }].map(({ role, text }) =>
      <div className="message-row" key={role}><MessageAvatar role={role === 'user' ? 'user' : 'assistant'} />
        <div><h3>{role === 'user' ? 'You' : 'TokenSmith'}</h3><MessageText text={text ?? ''} /></div></div>)}
  </details>
}
