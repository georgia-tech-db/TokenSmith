import type { TopicMastery } from '@shared/practice-mastery'

export function MasteryMeter({ score, label = 'Estimated mastery', empty = 'Not practiced', compact = false }: {
  score: number | null; label?: string; empty?: string; compact?: boolean
}) {
  const percent = Math.round((score ?? 0) * 100)
  return <span className={`mastery-meter${compact ? ' mastery-meter-compact' : ''}`}>
    <span className="mastery-meter-label"><span>{label}</span><strong>{score === null ? empty : `${percent}%`}</strong></span>
    <span className="mastery-meter-track" role="meter" aria-label={label} aria-valuemin={0} aria-valuemax={100}
      aria-valuenow={percent} aria-valuetext={score === null ? empty : `${percent}%`}>
      <span style={{ width: `${percent}%` }} />
    </span>
  </span>
}

export function TopicMasteryDetails({ result }: { result?: TopicMastery }) {
  return <details className="mastery-details"><summary>About this estimate</summary>
    <p>This is an early estimate of your understanding, not a grade. It accounts for guessing on True/False questions and gives more weight to recent answers.</p>
    <p>{result?.answers ?? 0} independent {result?.answers === 1 ? 'answer' : 'answers'} · {result?.correct ?? 0} correct.
      {' '}Excluded: {result?.withHelp ?? 0} with help, {result?.repeated ?? 0} repeated.</p>
    <p>Only first answers without help count. Chat help before answering leaves mastery unchanged. Reading source passages, including Read while waiting, does not count as help. Revisions do not count again.</p>
    <p>Try different questions without help to build evidence. An empty bar means we have not assessed this topic yet. Clearing practice history keeps this estimate.</p>
    {result?.lastAssessedAt && <p>Last assessed <time dateTime={result.lastAssessedAt}>{new Date(result.lastAssessedAt).toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' })}</time>.</p>}
  </details>
}
