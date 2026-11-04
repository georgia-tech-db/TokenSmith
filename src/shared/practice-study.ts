import { isTrueFalseFeedback, type PracticeConcept, type PracticeQuestion, type PracticeSession } from './practice'
import type { ModelRuntimeSettings } from './app-state'
import { trueFalseStatementIssue } from './true-false-statement'

export interface AssessmentPack {
  version: 1
  id: string
  revision: string
  courseId: string
  title: string
  items: { id: string; revision: string; concept: { id: string; title: string }; question: string;
    format: 'true_false' | 'short_answer'; correctAnswer?: boolean; criteria: string[]; explanation: string; references?: string[] }[]
}

// Import creates new objects with a fixed allowlist; arbitrary pack fields never enter app state.
export function parseAssessmentPack(input: unknown): AssessmentPack {
  const object = (x: unknown): Record<string, unknown> => {
    if (!x || typeof x !== 'object' || Array.isArray(x)) throw new Error('Expected an assessment object.')
    return x as Record<string, unknown>
  }
  const str = (x: unknown, max = 20000): string => {
    if (typeof x !== 'string' || !x.trim() || x.length > max) throw new Error('Assessment fields must be nonempty text within the size limit.')
    return x.trim()
  }
  const p = object(input)
  if (p.version !== 1 || !Array.isArray(p.items) || !p.items.length || p.items.length > 200) throw new Error('Expected assessment version 1 with 1–200 items.')
  const seen = new Set<string>()
  const items: AssessmentPack['items'] = p.items.map(value => {
    const i = object(value), c = object(i.concept), id = str(i.id, 200)
    if (seen.has(id)) throw new Error(`Duplicate item: ${id}`)
    seen.add(id)
    if (i.format !== 'true_false' && i.format !== 'short_answer') throw new Error('Unsupported assessment item format.')
    if (i.format === 'true_false' && typeof i.correctAnswer !== 'boolean') throw new Error('True/False items require a boolean answer key.')
    if (i.format === 'true_false' && trueFalseStatementIssue(str(i.question))) throw new Error('True/False items must be declarative statements.')
    if (i.references !== undefined && (!Array.isArray(i.references) || i.references.length > 20)) throw new Error('References must be a list of at most 20 source locators.')
    if (!Array.isArray(i.criteria) || !i.criteria.length || i.criteria.length > 20) throw new Error('Each item needs reviewed scoring criteria.')
    return { id, revision: str(i.revision, 200), concept: { id: str(c.id, 200), title: str(c.title, 500) },
      question: str(i.question), format: i.format, correctAnswer: i.format === 'true_false' ? i.correctAnswer as boolean : undefined,
      criteria: i.criteria.map(x => str(x)), explanation: str(i.explanation), references: (i.references as unknown[] | undefined)?.map(x => str(x)) }
  })
  return { version: 1, id: str(p.id, 200), revision: str(p.revision, 200), courseId: str(p.courseId, 200), title: str(p.title, 500), items }
}

export function assessmentSession(pack: AssessmentPack, id: string, modelSettings: ModelRuntimeSettings, now: string): PracticeSession {
  return { id, title: pack.title, createdAt: now, documents: [], modelId: '', modelSettings, currentIndex: 0,
    sourcesExhausted: true, assessment: { packId: pack.id, version: pack.revision }, observations: [],
    questions: pack.items.map(i => ({ id: `${id}:${i.id}`, text: i.question, format: i.format, correctAnswer: i.correctAnswer,
      concept: { ...i.concept, courseId: pack.courseId }, item: { id: i.id, version: i.revision, keyVersion: pack.revision, origin: 'reviewed', references: i.references },
      referenceAnswer: i.explanation, rubric: { objective: i.concept.title, assumptions: [],
        criteria: i.criteria.map((description, n) => ({ id: `c${n + 1}`, description, evidence: [] })) },
      sources: [], draft: '', attempts: [], reviewing: false, timeline: [] })) }
}

export const assessmentLocked = (session: PracticeSession) => Boolean(session.assessment && !session.assessment.completedAt)

export function assessmentDisplayQuestion(q: PracticeQuestion): PracticeQuestion {
  return { id: q.id, text: q.text, format: q.format, sources: [], draft: q.draft, reviewing: false,
    attempts: q.attempts.map(a => ({ ...a, feedback: 'Response saved. Feedback is available after completing the assessment.' })), timeline: q.timeline }
}

export interface ScoreCorrection { sessionId: string; questionId: string; keyVersion: string; correct: boolean; reason: string }
export type PerformanceMethod = 'first_response' | 'recency' | 'assistance'
export function practicePerformance(sessions: PracticeSession[], method: PerformanceMethod = 'first_response', corrections: ScoreCorrection[] = []) {
  const groups = new Map<string, { concept: PracticeConcept; observations: number; assisted: number; correct: number; weight: number }>()
  const rows = sessions.flatMap(s => s.questions.map(q => ({ s, q, a: q.attempts[0] })))
    .filter(r => r.a?.submittedAt && r.a.observationVersion === 1 && r.q.concept && !assessmentLocked(r.s))
    .sort((a, b) => a.a.submittedAt!.localeCompare(b.a.submittedAt!) || a.a.id.localeCompare(b.a.id))
  for (const { s, q, a } of rows) {
    const correction = corrections.find(c => c.sessionId === s.id && c.questionId === q.id && c.keyVersion && c.reason)
    const correct = correction?.correct ?? (isTrueFalseFeedback(a.feedback) ? a.feedback.correct : undefined)
    if (correct === undefined) continue
    const key = JSON.stringify([q.concept!.courseId, q.concept!.id])
    const g = groups.get(key) ?? { concept: q.concept!, observations: 0, assisted: 0, correct: 0, weight: 0 }
    const decay = method === 'first_response' ? 1 : 0.9
    const weight = method === 'assistance' && a.assisted ? 0.5 : 1
    g.correct = g.correct * decay + Number(correct) * weight
    g.weight = g.weight * decay + weight
    g.observations++; g.assisted += Number(a.assisted)
    groups.set(key, g)
  }
  return [...groups.values()].map(g => ({ concept: g.concept, observations: g.observations, assisted: g.assisted,
    performance: g.correct / g.weight, method, estimatorVersion: 1 }))
}

// Explicit allowlist excludes provider credentials and unrelated conversations/settings.
export function exportPractice(sessions: PracticeSession[], participantId = '') {
  return { version: 1, participantId, sessions: sessions.map(s => ({
    id: s.id, title: s.title, createdAt: s.createdAt, assessment: s.assessment, observations: s.observations ?? [],
    modelId: s.modelId, modelSettings: { contextLength: s.modelSettings.contextLength, contextLengthMode: s.modelSettings.contextLengthMode,
      maxLength: s.modelSettings.maxLength, temperature: s.modelSettings.temperature, reasoningMode: s.modelSettings.reasoningMode,
      topP: s.modelSettings.topP, topK: s.modelSettings.topK, minP: s.modelSettings.minP, repeatPenalty: s.modelSettings.repeatPenalty },
    documents: s.documents.map(d => ({ materialId: d.materialId, documentId: d.documentId, title: d.title })),
    questions: s.questions.map(q => ({ id: q.id, text: q.text, format: q.format, concept: q.concept, item: q.item,
      correctAnswer: assessmentLocked(s) ? undefined : q.correctAnswer, referenceAnswer: assessmentLocked(s) ? undefined : q.referenceAnswer,
      rubric: assessmentLocked(s) ? undefined : q.rubric, generationMs: q.generationMs,
      sources: assessmentLocked(s) ? [] : q.sources.map(x => ({ materialId: x.materialId, documentId: x.documentId, chunkId: x.chunkId,
        sourceUnitId: x.sourceUnitId, excerpt: x.excerpt, context: x.context })), attempts: assessmentLocked(s) ? assessmentDisplayQuestion(q).attempts : q.attempts })) })) }
}
