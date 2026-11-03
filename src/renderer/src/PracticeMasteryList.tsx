import { practiceMasteryLabel, practiceMasteryPercent, type TopicMastery } from '@shared/practice'

export function PracticeMasteryList({ title, entries, disabled, onChoose }: {
  title: string; entries: TopicMastery[]; disabled?: boolean; onChoose?: (topic: string) => void
}) {
  if (!entries.length) return null
  return <section className="practice-mastery" aria-label={title}>
    <h3>{title}</h3>
    <ul>{entries.map(entry => <li key={entry.topic}>
      {onChoose
        ? <button type="button" className="practice-link practice-mastery-topic" aria-label={`Revisit ${entry.topic}`} disabled={disabled} onClick={() => onChoose(entry.topic)}>{entry.topic}</button>
        : <span className="practice-mastery-topic">{entry.topic}</span>}
      <span className="practice-muted">{practiceMasteryLabel(entry)} · {entry.attempts} attempt{entry.attempts === 1 ? '' : 's'}</span>
      <span className="practice-mastery-bar" role="meter" aria-label={`Mastery of ${entry.topic}`} aria-valuemin={0} aria-valuemax={100}
        aria-valuenow={practiceMasteryPercent(entry)}><span style={{ width: `${practiceMasteryPercent(entry)}%` }} /></span>
    </li>)}</ul>
  </section>
}
