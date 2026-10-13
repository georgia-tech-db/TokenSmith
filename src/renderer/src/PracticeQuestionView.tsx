import { useId, useState, type ReactNode } from 'react'
import { ArrowRight, BookOpen, ChevronDown, Database, Lightbulb, PanelRightOpen } from 'lucide-react'
import { practiceQuestionText, practiceTimeline, type PracticeQuestion } from '@shared/practice'
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

export function PracticeQuestionView({ question, index, active, busy, onContinue, onShowSources, sourcesOpen, onHint, onReveal, actions }: {
  question: PracticeQuestion; index: number; active: boolean; busy: boolean; onContinue: () => void
  onShowSources?: () => void
  sourcesOpen: boolean
  onHint: () => void
  onReveal: () => void
  actions?: ReactNode
}) {
  const [expandedHelp, setExpandedHelp] = useState<'hint' | 'explanation' | null>(null)
  const helpId = useId()
  function toggleHelp(kind: 'hint' | 'explanation') {
    const opening = expandedHelp !== kind
    if (opening) {
      if (kind === 'hint') onHint(); else onReveal()
    }
    setExpandedHelp(opening ? kind : null)
  }
  return <section className={`practice-thread-question ${active ? 'is-active' : ''}`} data-question-id={question.id} tabIndex={-1} aria-label={`Practice question ${index + 1}`}>
    <div className="practice-question-messages">
      <PracticeMessage id={question.id}>
        <MessageText text={practiceQuestionText(question)} />
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
      <div className="practice-study-actions" role="group" aria-label="Practice actions">
        {question.hint && <button type="button"
          aria-expanded={expandedHelp === 'hint'} aria-controls={`${helpId}-hint`} onClick={() => toggleHelp('hint')}>
          <Lightbulb size={16} aria-hidden="true" />Hint<ChevronDown size={14} aria-hidden="true" />
        </button>}
        {question.referenceAnswer && <button type="button"
          aria-expanded={expandedHelp === 'explanation'} aria-controls={`${helpId}-explanation`} onClick={() => toggleHelp('explanation')}>
          <BookOpen size={16} aria-hidden="true" />Explanation<ChevronDown size={14} aria-hidden="true" />
        </button>}
        {actions}
      </div>
      {(['hint', 'explanation'] as const).map(kind => <section key={kind} className="practice-inline-help" id={`${helpId}-${kind}`}
        hidden={expandedHelp !== kind} aria-label={kind === 'hint' ? 'Hint' : 'Worked explanation'}
        data-chat-selectable data-chat-message-id={`${question.id}:${kind}`}>
        {expandedHelp === kind && <MessageText text={(kind === 'hint' ? question.hint : question.referenceAnswer) ?? ''} />}
      </section>)}
      {(onShowSources || !active) && <div className="practice-source-actions">
        {onShowSources && <button className="source-chip" type="button" disabled={busy} aria-expanded={sourcesOpen} onClick={onShowSources}>
          <Database size={16} aria-hidden="true" /><span>{question.sources.length} {question.sources.length === 1 ? 'Source' : 'Sources'}</span><PanelRightOpen size={14} aria-hidden="true" />
        </button>}
        {!active && <button className="practice-link" type="button" disabled={busy} onClick={onContinue} aria-label={`Continue question ${index + 1}`}>Continue here<ArrowRight size={16} aria-hidden="true" /></button>}
      </div>}
    </div>
  </section>
}
