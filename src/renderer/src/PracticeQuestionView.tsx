import type { ReactNode } from 'react'
import { Database, PanelRightOpen } from 'lucide-react'
import { practiceQuestionText, type PracticeTopic, type PracticeQuestion } from '@shared/practice'
import { MessageText } from './MessageText'
import { PracticeFeedbackView } from './PracticeFeedbackView'
import { MessageAvatar } from './MessageAvatar'

function PracticeMessage({ id, user = false, label, children }: { id: string; user?: boolean; label?: string; children: ReactNode }) {
  return <div className="message-row practice-message">
    <MessageAvatar role={user ? 'user' : 'assistant'} />
    <div><h3>{label ?? (user ? 'You' : 'TokenSmith')}</h3><div data-chat-selectable data-chat-message-id={id}>{children}</div></div>
  </div>
}

export function PracticeQuestionView({ question, index, active, busy, onShowSources, sourcesOpen, actions }: {
  question: PracticeQuestion; index: number; active: boolean; busy: boolean
  onShowSources?: () => void
  sourcesOpen: boolean
  actions?: ReactNode
}) {
  return <section className={`practice-thread-question ${active ? 'is-active' : ''}`} data-question-id={question.id} tabIndex={-1} aria-label={`Practice question ${index + 1}`}>
    <div className="practice-question-messages">
      <PracticeMessage id={question.id} label={`Question ${index + 1}`}>
        <MessageText text={practiceQuestionText(question)} />
      </PracticeMessage>
      {question.attempts.map(attempt => <div key={attempt.id}>
        <PracticeMessage id={attempt.id} user>
          <section aria-label="Your submitted answer"><MessageText text={attempt.answer} /></section>
        </PracticeMessage>
        <PracticeMessage id={`${attempt.id}:feedback`}>
          <section className="practice-feedback" aria-label="Practice feedback">
            {attempt.assisted && <p className="practice-muted" title="Chat help was shown before you answered. Reading source passages does not count as help.">Answered with help · mastery unchanged</p>}
            <PracticeFeedbackView feedback={attempt.feedback} />
          </section>
        </PracticeMessage>
      </div>)}
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
