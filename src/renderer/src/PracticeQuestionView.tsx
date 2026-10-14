import type { ReactNode } from 'react'
import { Database, PanelRightOpen } from 'lucide-react'
import { practiceQuestionText, practiceTimeline, type PracticeTopic, type PracticeQuestion } from '@shared/practice'
import { MessageText } from './MessageText'
import { PracticeFeedbackView } from './PracticeFeedbackView'
import { SelectedPassagePreview } from './SelectedPassagePreview'
import { MessageAvatar } from './MessageAvatar'

function PracticeMessage({ id, user = false, children }: { id: string; user?: boolean; children: ReactNode }) {
  return <div className="message-row practice-message">
    <MessageAvatar role={user ? 'user' : 'assistant'} />
    <div><h3>{user ? 'You' : 'TokenSmith'}</h3><div data-chat-selectable data-chat-message-id={id}>{children}</div></div>
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
      <PracticeMessage id={question.id}>
        <MessageText text={practiceQuestionText(question)} />
        {question.format !== 'true_false' && <p className="practice-muted">Saved short-answer exercise. Choose Next question for True/False practice.</p>}
      </PracticeMessage>
      {practiceTimeline(question).map((activity, eventIndex) => {
        if (activity.kind === 'attempt') {
          const attempt = question.attempts.find(item => item.id === activity.id)
          if (!attempt) return null
          return <div key={eventIndex}>
            <PracticeMessage id={attempt.id} user>
              <section aria-label="Your submitted answer"><MessageText text={attempt.answer} /></section>
            </PracticeMessage>
            <PracticeMessage id={`${attempt.id}:feedback`}>
              <section className="practice-feedback" aria-label="Practice feedback">
                {attempt.assisted && <p className="practice-muted">With help or revision</p>}
                <PracticeFeedbackView feedback={attempt.feedback} />
              </section>
            </PracticeMessage>
          </div>
        }
        if (activity.kind === 'discussion') {
          const discussion = question.discussions?.find(item => item.id === activity.id)
          if (!discussion) return null
          return <div key={eventIndex}>
            <PracticeMessage id={discussion.id} user>
              {discussion.selectedPassage && <SelectedPassagePreview passage={discussion.selectedPassage} label="From this practice" />}
              <section aria-label="Help question"><MessageText text={discussion.question} /></section>
            </PracticeMessage>
            <PracticeMessage id={`${discussion.id}:reply`}>
              <section aria-label="Help explanation"><MessageText text={discussion.answer} /></section>
            </PracticeMessage>
          </div>
        }
        return null
      })}
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
