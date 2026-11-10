import type { ChatMessage, ChatSource, Conversation, ModelRuntimeSettings } from './app-state'
import type { StudyDocument, StudyDocumentRef } from './study-scope'
import { migratedContextMode } from './model-context'

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
  submittedAt?: string
  visibleMs?: number
  assistanceIds?: string[]
  observationVersion?: 1
  timingComplete?: boolean
}

export interface PracticeConcept { courseId: string; id: string; title: string }
export interface PracticeObservation {
  id: string
  questionId: string
  kind: 'presented' | 'paused' | 'submitted' | 'help_shown' | 'feedback_shown'
  at: string
  visibleMs?: number
  content?: string
  referenceId?: string
}

export interface PracticeQuestion {
  id: string
  text: string
  format?: PracticeQuestionFormat
  correctAnswer?: boolean
  requestedAnswer?: boolean
  sources: ChatSource[]
  draft: string
  attempts: PracticeAttempt[]
  rubric?: PracticeRubric
  referenceAnswer?: string
  concept?: PracticeConcept
  item?: { id: string; version: string; keyVersion: string; origin: 'generated' | 'reviewed'; references?: string[] }
  generationMs?: number
  masteryCheck?: { version: 1; conceptKey: string; verification?: 'none' }
}

export interface PracticeTopic {
  id?: string
  title: string
  objectives?: string[]
  concept?: PracticeConcept
  sources?: ChatSource[]
  prepared?: boolean
  conversationId?: string
  answerId?: string
  question?: string
  answer?: string
}

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
  questionSourceCursor?: number
  questionTargetAnswer?: boolean
  topic?: PracticeTopic
  observations?: PracticeObservation[]
  assessment?: { packId: string; version: string; completedAt?: string }
}

export interface PracticeState {
  mode: StudyMode
  sessions: PracticeSession[]
  masteryRecords?: PracticeMasteryRecord[]
  activeSessionId?: string
  draftScope?: StudyDocumentRef[]
  draftTopic?: PracticeTopic
  topicSuggestions?: { key: string; topics: PracticeTopic[] }
  draftModelId?: string
}

// Learning evidence survives deleting conversations; no answers, passages or chat text.
export interface PracticeMasteryRecord {
  version: 1
  id: string
  concept: PracticeConcept
  attemptId: string
  submittedAt: string
  correct: boolean
  assisted: boolean
  statementFingerprint: string
  evidenceFingerprint: string
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

export function practiceMessages(question: PracticeQuestion): ChatMessage[] {
  return [{ id: question.id, role: 'assistant', text: practiceQuestionText(question) }, ...question.attempts.flatMap((attempt): ChatMessage[] => [
    { id: attempt.id, role: 'user', text: attempt.answer },
    { id: `${attempt.id}:feedback`, role: 'assistant', text: practiceFeedbackText(attempt.feedback) }
  ])]
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
  if (!attempt) return 'Not answered'
  if (typeof attempt.feedback === 'string') return attempt.feedback === 'Awaiting independent review.' ? 'Awaiting review' : 'Feedback saved'
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
    feedback: attempt ? practiceFeedbackText(attempt.feedback) : '', sources: question.sources,
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

export function practiceChatConversation(id: string, session: PracticeSession, question: PracticeQuestion): Conversation {
  const reference = practiceReference(session, question)
  const messages: ChatMessage[] = [{ id: `${id}:question`, role: 'user',
    text: `${reference.question}${reference.answer ? `\n\n${question.attempts.length ? 'My answer' : 'My selected answer'}: ${reference.answer}` : ''}` }]
  // Copy only feedback the student has already seen, never the hidden answer guide.
  if (reference.feedback) messages.push({ id: `${id}:feedback`, role: 'assistant',
    text: reference.feedback, sources: reference.sources })
  return { id, modelId: session.modelId, title: `Practice: ${question.text.slice(0, 65)}`, period: 'Today', messages,
    practiceReference: reference, documentScope: reference.documents }
}

export function normalizePracticeState(value?: PracticeState): PracticeState {
  if (!value || !Array.isArray(value.sessions)) return emptyPracticeState()
  const sessions = value.sessions.filter(session => session && typeof session.id === 'string' &&
    Array.isArray(session.documents) && Array.isArray(session.questions) && session.modelSettings)
    .map(session => {
      // Repair source-view flags written by the earlier preview implementation.
      // Keep genuine help and student responses; simply opening course text is not assistance.
      const sourceRead = (id: string) => id.startsWith('source-read:')
      const questions = session.questions.map(question => ({ ...question, attempts: question.attempts.map(attempt => {
        if (!attempt.assistanceIds?.some(sourceRead)) return attempt
        const assistanceIds = attempt.assistanceIds.filter(id => !sourceRead(id))
        // Do not grant independent credit to an older answer that also used real help.
        const previous = question as PracticeQuestion & { hintUsed?: boolean; answerRevealed?: boolean; discussions?: unknown[] }
        return { ...attempt, assistanceIds, assisted: Boolean(assistanceIds.length || previous.hintUsed || previous.answerRevealed || previous.discussions?.length) }
      }) }))
      return { ...session, questions, ...(session.observations ? { observations: session.observations.filter(e => !sourceRead(e.id)) } : {}),
        modelSettings: { ...session.modelSettings, contextLengthMode: migratedContextMode(session.modelSettings) } }
    })
  return { mode: value.mode === 'practice' ? 'practice' : 'chat', sessions, draftModelId: value.draftModelId, ...(value.draftScope ? { draftScope: value.draftScope } : {}),
    ...(Array.isArray(value.masteryRecords) ? { masteryRecords: value.masteryRecords } : {}),
    ...(value.draftTopic?.title?.trim() ? { draftTopic: value.draftTopic } : {}),
    ...(value.topicSuggestions ? { topicSuggestions: value.topicSuggestions } : {}),
    activeSessionId: sessions.some(session => session.id === value.activeSessionId) ? value.activeSessionId : undefined }
}
