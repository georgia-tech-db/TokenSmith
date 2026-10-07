import { practiceVerdictLabels, type PracticeFeedback } from '@shared/practice'
import { MessageText } from './MessageText'

export function PracticeFeedbackView({ feedback }: { feedback: PracticeFeedback | string }) {
  if (typeof feedback === 'string') return <MessageText text={feedback} />
  const groups = [
    { status: 'met', title: 'What you got right' },
    { status: 'missing', title: 'What to explain' },
    { status: 'incorrect', title: 'What to correct' }
  ]
  return <div className="practice-feedback-details">
    <p className="practice-verdict">{practiceVerdictLabels[feedback.verdict]}</p>
    {feedback.questionIssue ? <MessageText text={feedback.questionIssue} /> : groups.map(group => {
      const checks = feedback.checks.filter(check => check.status === group.status)
      return checks.length > 0 && <section key={group.status}><h3>{group.title}</h3>
        {checks.length === 1 ? <MessageText text={checks[0].feedback} /> : <ul>{checks.map(check => <li key={check.criterionId}><MessageText text={check.feedback} /></li>)}</ul>}
      </section>
    })}
    {feedback.improvement && <section><h3>Since your last answer</h3><MessageText text={feedback.improvement} /></section>}
    {feedback.nextStep && <section><h3>For your revision</h3><MessageText text={feedback.nextStep} /></section>}
  </div>
}
