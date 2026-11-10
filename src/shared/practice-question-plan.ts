import { practiceSourceKey, type PracticeSession } from './practice'
import { sourcesInScope } from './study-scope'

export const practiceQuestionsPerPassage = 2

// Count saved questions, not requests: a failed or cancelled call uses no slot.
export function practiceQuestionPlan(session: PracticeSession, cursor = session.questionSourceCursor ?? session.questions.length) {
  const sources = [...new Map(sourcesInScope(session.topic?.sources ?? [], session.documents)
    .filter(source => (source.context || source.excerpt)?.trim())
    .map(source => [practiceSourceKey(source), source])).values()]
  const totalQuestions = sources.length * practiceQuestionsPerPassage
  const used = new Map<string, number>()
  for (const question of session.questions) {
    for (const key of new Set(question.sources.map(practiceSourceKey))) used.set(key, (used.get(key) ?? 0) + 1)
  }
  const start = Number.isSafeInteger(cursor) && cursor >= 0 ? cursor : 0
  if (session.questions.length < totalQuestions) {
    for (let offset = 0; offset < sources.length; offset++) {
      const source = sources[(start + offset) % sources.length]
      if ((used.get(practiceSourceKey(source)) ?? 0) < practiceQuestionsPerPassage) {
        return { passageCount: sources.length, totalQuestions, source, nextCursor: start + offset + 1 }
      }
    }
  }
  return { passageCount: sources.length, totalQuestions, source: undefined, nextCursor: start }
}
