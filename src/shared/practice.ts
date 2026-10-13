import type { ChatSource, ModelRuntimeSettings } from './app-state'
import type { StudyDocument, StudyDocumentRef } from './study-scope'

export type StudyMode = 'chat' | 'practice'

export interface PracticeRubric {
  objective: string
  assumptions: string[]
  criteria: { id: string; description: string; evidence: { sourceKey: string; quote: string }[] }[]
}

export interface PracticeFeedback {
  verdict: 'correct' | 'partial' | 'needs_work' | 'needs_review'
  checks: { criterionId: string; status: 'met' | 'missing' | 'incorrect'; feedback: string }[]
  improvement: string
  nextStep: string
  questionIssue: string
}

export interface PracticeAttempt {
  id: string
  answer: string
  feedback: PracticeFeedback | string
  assisted: boolean
  durationMs: number
}

export interface PracticeQuestion {
  id: string
  text: string
  sources: ChatSource[]
  draft: string
  attempts: PracticeAttempt[]
  reviewing: boolean
  rubric?: PracticeRubric
  hint?: string
  referenceAnswer?: string
  hintUsed?: boolean
  answerRevealed?: boolean
}

export interface PracticeSession {
  id: string
  title: string
  createdAt: string
  documents: StudyDocument[]
  modelId: string
  modelSettings: ModelRuntimeSettings
  totalQuestions: number
  questions: PracticeQuestion[]
  currentIndex: number
  completed: boolean
}

export interface PracticeState {
  mode: StudyMode
  sessions: PracticeSession[]
  activeSessionId?: string
  draftScope?: StudyDocumentRef[]
}

export interface PracticeReference {
  sessionId: string
  questionId: string
  question: string
  answer: string
  feedback: string
  sources: ChatSource[]
  documents: StudyDocumentRef[]
}

export const emptyPracticeState = (): PracticeState => ({ mode: 'chat', sessions: [] })

export function practiceSourceKey(source: ChatSource): string {
  return [source.materialId, source.documentId, source.sourceUnitId || source.chunkId || source.chunkRowid || source.excerpt].join(':')
}

export function practiceAssisted(question: PracticeQuestion): boolean {
  return Boolean(question.hintUsed || question.answerRevealed || question.attempts.length)
}

export const practiceVerdictLabels = { correct: 'Correct', partial: 'Partly there', needs_work: 'Needs another look', needs_review: 'Question needs review' }

export function practiceFeedbackText(feedback: PracticeFeedback | string): string {
  if (typeof feedback === 'string') return feedback
  return [practiceVerdictLabels[feedback.verdict], feedback.questionIssue,
    ...feedback.checks.map(check => check.feedback), feedback.improvement, feedback.nextStep].filter(Boolean).join('\n\n')
}

export function practiceReviewLabel(question: PracticeQuestion): string {
  const attempt = question.attempts.at(-1)
  if (!attempt) return question.answerRevealed ? 'Explanation viewed' : 'Not answered'
  if (typeof attempt.feedback === 'string') return 'Feedback saved'
  if (attempt.feedback.verdict === 'correct') return attempt.assisted ? 'Correct with help or revision' : 'Correct independently'
  return attempt.feedback.verdict === 'needs_review' ? 'Question needs review' : 'Revisit this concept'
}

export function practiceReference(session: PracticeSession, question: PracticeQuestion): PracticeReference {
  const attempt = question.attempts.at(-1)
  return { sessionId: session.id, questionId: question.id, question: question.text,
    answer: attempt?.answer ?? question.draft,
    feedback: attempt ? practiceFeedbackText(attempt.feedback) : question.answerRevealed ? question.referenceAnswer ?? '' : question.hintUsed ? question.hint ?? '' : '', sources: question.sources,
    documents: session.documents.map(({ materialId, documentId }) => ({ materialId, documentId })) }
}

export function practiceDiscussionPassage(reference: PracticeReference) {
  return {
    messageId: reference.questionId,
    role: 'assistant' as const,
    question: `Practice question: ${reference.question}${reference.answer ? `\n\nMy answer: ${reference.answer}` : ''}`,
    text: reference.feedback || reference.question
  }
}

export function normalizePracticeState(value?: PracticeState): PracticeState {
  if (!value || !Array.isArray(value.sessions)) return emptyPracticeState()
  const sessions = value.sessions.filter(session => session && typeof session.id === 'string' &&
    Array.isArray(session.documents) && Array.isArray(session.questions) && session.modelSettings)
  return { mode: value.mode === 'practice' ? 'practice' : 'chat', sessions, ...(value.draftScope ? { draftScope: value.draftScope } : {}),
    activeSessionId: sessions.some(session => session.id === value.activeSessionId) ? value.activeSessionId : undefined }
}
