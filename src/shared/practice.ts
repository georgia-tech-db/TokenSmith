import type { ChatMessage, ChatSelectedPassage, ChatSource, ModelRuntimeSettings } from './app-state'
import type { StudyDocument, StudyDocumentRef } from './study-scope'

export type StudyMode = 'chat' | 'practice'
export type PracticeQuestionFormat = 'short_answer' | 'true_false'

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

export interface PracticeTrueFalseFeedback {
  kind: 'true_false'
  correct: boolean
  explanation: string
}

export type PracticeAnswerFeedback = PracticeFeedback | PracticeTrueFalseFeedback | string

export interface PracticeAttempt {
  id: string
  answer: string
  feedback: PracticeAnswerFeedback
  assisted: boolean
  durationMs: number
}

export interface PracticeQuestion {
  id: string
  text: string
  format?: PracticeQuestionFormat
  correctAnswer?: boolean
  sources: ChatSource[]
  draft: string
  attempts: PracticeAttempt[]
  reviewing: boolean
  rubric?: PracticeRubric
  hint?: string
  referenceAnswer?: string
  hintUsed?: boolean
  answerRevealed?: boolean
  composerMode?: 'answer' | 'help'
  helpDraft?: string
  helpPassage?: ChatSelectedPassage
  discussions?: PracticeDiscussion[]
  timeline?: PracticeActivity[]
}

export interface PracticeDiscussion {
  id: string
  question: string
  answer: string
  sources: ChatSource[]
  durationMs: number
  selectedPassage?: ChatSelectedPassage
}

export type PracticeActivity = { kind: 'attempt' | 'discussion'; id: string } | { kind: 'hint' | 'explanation' }

export interface PracticeSession {
  id: string
  title: string
  createdAt: string
  documents: StudyDocument[]
  modelId: string
  modelSettings: ModelRuntimeSettings
  questions: PracticeQuestion[]
  currentIndex: number
  sourcesExhausted?: boolean
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
  return Boolean(question.hintUsed || question.answerRevealed || question.attempts.length || question.discussions?.length)
}

// Older saved questions have no timeline. Preserve their attempts when adding new activity.
export function practiceTimeline(question: PracticeQuestion): PracticeActivity[] {
  return question.timeline ?? [
    ...(question.hintUsed ? [{ kind: 'hint' as const }] : []),
    ...question.attempts.map(attempt => ({ kind: 'attempt' as const, id: attempt.id })),
    ...(question.answerRevealed ? [{ kind: 'explanation' as const }] : []),
    ...(question.discussions ?? []).map(discussion => ({ kind: 'discussion' as const, id: discussion.id }))
  ]
}

export function practiceMessages(question: PracticeQuestion): ChatMessage[] {
  return [{ id: question.id, role: 'assistant', text: practiceQuestionText(question) }, ...practiceTimeline(question).flatMap((activity): ChatMessage[] => {
    if (activity.kind === 'attempt') {
      const attempt = question.attempts.find(item => item.id === activity.id)
      return attempt ? [{ id: attempt.id, role: 'user', text: attempt.answer },
        { id: `${attempt.id}:feedback`, role: 'assistant', text: practiceFeedbackText(attempt.feedback) }] : []
    }
    if (activity.kind === 'discussion') {
      const discussion = question.discussions?.find(item => item.id === activity.id)
      return discussion ? [{ id: discussion.id, role: 'user', text: discussion.question, selectedPassage: discussion.selectedPassage },
        { id: `${discussion.id}:reply`, role: 'assistant', text: discussion.answer }] : []
    }
    return [{ id: `${question.id}:${activity.kind}`, role: 'assistant',
      text: (activity.kind === 'hint' ? question.hint : question.referenceAnswer) ?? '' }]
  })]
}

export function practiceQuestionText(question: PracticeQuestion): string {
  return question.format === 'true_false' ? `True/False: ${question.text}` : question.text
}

export function isTrueFalseFeedback(feedback: PracticeAnswerFeedback): feedback is PracticeTrueFalseFeedback {
  return typeof feedback !== 'string' && 'kind' in feedback && feedback.kind === 'true_false'
}

export function practiceFeedbackText(feedback: PracticeAnswerFeedback): string {
  if (typeof feedback === 'string') return feedback
  if (isTrueFalseFeedback(feedback)) return `${feedback.correct ? 'Correct' : 'Incorrect'}\n\n${feedback.explanation}`
  return [feedback.questionIssue,
    ...feedback.checks.map(check => check.feedback), feedback.improvement, feedback.nextStep].filter(Boolean).join('\n\n')
}

export function practiceReviewLabel(question: PracticeQuestion): string {
  const attempt = question.attempts.at(-1)
  if (!attempt) return question.answerRevealed ? 'Explanation viewed' : 'Not answered'
  if (typeof attempt.feedback === 'string') return 'Feedback saved'
  if (isTrueFalseFeedback(attempt.feedback)) return attempt.feedback.correct ? 'Correct' : 'Incorrect'
  if (attempt.feedback.verdict === 'needs_review') return 'Question needs review'
  if (attempt.feedback.verdict === 'correct') return 'Key points covered'
  const corrections = attempt.feedback.checks.filter(check => check.status === 'incorrect').length
  const missing = attempt.feedback.checks.filter(check => check.status === 'missing').length
  return [corrections && `${corrections} correction${corrections === 1 ? '' : 's'}`,
    missing && `${missing} point${missing === 1 ? '' : 's'} to explain`].filter(Boolean).join(', ')
}

export function practiceReference(session: PracticeSession, question: PracticeQuestion): PracticeReference {
  const attempt = question.attempts.at(-1)
  return { sessionId: session.id, questionId: question.id, question: practiceQuestionText(question),
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
