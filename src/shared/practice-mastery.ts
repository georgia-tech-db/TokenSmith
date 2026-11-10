import type { PracticeConcept, PracticeMasteryRecord, PracticeQuestion, PracticeSession, PracticeState, PracticeTopic } from './practice'
import { practiceConceptKey, scoredPracticeResponses, type ScoreCorrection } from './practice-study'
import { topicHash } from './topic-inventory'

// Provisional, versioned assumptions. This is an estimate, not a calibrated probability.
export const masteryModel = { version: 1, prior: 0.2, guess: 0.5, slip: 0.1, retention: 0.9 } as const
const priorOdds = Math.log(masteryModel.prior / (1 - masteryModel.prior))

export interface TopicMastery {
  concept: PracticeConcept
  score: number | null
  evidence: number
  answers: number
  correct: number
  withHelp: number
  repeated: number
  lastAssessedAt?: string
  recentMistake: boolean
}

export function updateMastery(evidence: number, correct: boolean) {
  const likelihood = correct ? (1 - masteryModel.slip) / masteryModel.guess : masteryModel.slip / (1 - masteryModel.guess)
  const next = masteryModel.retention * evidence + Math.log(likelihood)
  return { evidence: next, score: 1 / (1 + Math.exp(-(priorOdds + next))) }
}

const normalizeClaim = (text: string) => text.normalize('NFKC').toLowerCase().replace(/[^\p{L}\p{N}]+/gu, ' ').trim()

// Reusing the same supporting claim is not another independent mastery observation.
export function masteryEvidenceKey(question: Pick<PracticeQuestion, 'rubric'>): string | undefined {
  const quotes = question.rubric?.criteria.flatMap(c => c.evidence.map(e => normalizeClaim(e.quote))).filter(Boolean)
  return quotes?.length ? JSON.stringify([...new Set(quotes)].sort()) : undefined
}

function learningRecords(sessions: PracticeSession[], corrections: ScoreCorrection[] = []): PracticeMasteryRecord[] {
  const records: PracticeMasteryRecord[] = []
  for (const { session, question, attempt, correct } of scoredPracticeResponses(sessions, corrections)) {
    if (question.format !== 'true_false' || !question.concept) continue
    const key = practiceConceptKey(question.concept)
    if (question.masteryCheck?.version !== masteryModel.version || question.masteryCheck.conceptKey !== key) continue
    const claim = masteryEvidenceKey(question)
    if (!claim) continue
    records.push({ version: masteryModel.version, id: JSON.stringify([session.id, question.id]),
      concept: question.concept, attemptId: attempt.id, submittedAt: attempt.submittedAt!, correct,
      assisted: Boolean(attempt.assisted), statementFingerprint: topicHash(normalizeClaim(question.text)), evidenceFingerprint: topicHash(claim) })
  }
  return records
}

function mergeRecords(saved: PracticeMasteryRecord[], current: PracticeMasteryRecord[]) {
  return [...new Map([...saved, ...current].map(record => [record.id, record])).values()]
    .sort((a, b) => Date.parse(a.submittedAt) - Date.parse(b.submittedAt) || a.attemptId.localeCompare(b.attemptId))
}

export function removePracticeHistory(state: PracticeState, sessionId?: string): PracticeState {
  const removed = state.sessions.filter(s => sessionId === undefined || s.id === sessionId)
  if (!removed.length) return state
  return { ...state,
    masteryRecords: mergeRecords(state.masteryRecords ?? [], learningRecords(removed)),
    sessions: state.sessions.filter(s => !removed.includes(s)),
    activeSessionId: removed.some(s => s.id === state.activeSessionId) ? undefined : state.activeSessionId,
    ...(sessionId === undefined ? { draftTopic: undefined } : {}) }
}

export function practiceMastery(sessions: PracticeSession[], corrections: ScoreCorrection[] = [], saved: PracticeMasteryRecord[] = []): Map<string, TopicMastery> {
  const groups = new Map<string, TopicMastery>()
  const statements = new Set<string>(), evidence = new Set<string>()
  for (const record of mergeRecords(saved, learningRecords(sessions, corrections))) {
    if (record.version !== masteryModel.version) continue
    const key = practiceConceptKey(record.concept)
    const statementKey = JSON.stringify([key, record.statementFingerprint])
    const claimKey = JSON.stringify([key, record.evidenceFingerprint])
    const repeated = statements.has(statementKey) || evidence.has(claimKey)
    statements.add(statementKey); evidence.add(claimKey)
    const row = groups.get(key) ?? { concept: record.concept, score: null, evidence: 0,
      answers: 0, correct: 0, withHelp: 0, repeated: 0, recentMistake: false }
    groups.set(key, row)
    if (record.assisted) { row.withHelp++; continue }
    if (repeated) { row.repeated++; continue }
    Object.assign(row, updateMastery(row.evidence, record.correct))
    row.answers++; row.correct += Number(record.correct)
    row.lastAssessedAt = record.submittedAt
    row.recentMistake = !record.correct
  }
  return groups
}

export function chapterMastery(topics: PracticeTopic[], progress: Map<string, TopicMastery>) {
  const keys = [...new Set(topics.flatMap(t => t.concept ? [practiceConceptKey(t.concept)] : []))]
  const rows = keys.map(key => progress.get(key))
  return { total: keys.length, assessed: rows.filter(r => r?.score != null).length,
    // This parent bar measures coverage-weighted progress, not a probability of knowing the chapter.
    score: keys.length ? rows.reduce((sum, r) => sum + (r?.score ?? 0), 0) / keys.length : 0 }
}
