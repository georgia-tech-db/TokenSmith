import { practiceSourceKey, type PracticeSession } from './practice'
import type { ChatSource } from './app-state'
import { sourcesInScope } from './study-scope'

export const practiceQuestionsPerPassage = 2

const exercise = (source: ChatSource) =>
  (source.chunkKind || source.tokensmithChunkKind) === 'exercise'

// The topic mapping supplies relevance. Do not infer topics or passage types from keywords.
function practicePassages(session: PracticeSession): ChatSource[] {
  return [...new Map(sourcesInScope(session.topic?.sources ?? [], session.documents)
    .filter(s => (s.context || s.excerpt)?.trim())
    .map(s => [practiceSourceKey(s), s])).values()]
    .sort((a, b) => Number(exercise(a)) - Number(exercise(b)))
}

export function practiceEvidence(session: PracticeSession, focus: ChatSource): ChatSource[] {
  const support = practicePassages(session).filter(s => practiceSourceKey(s) !== practiceSourceKey(focus))
  return [focus, ...support.slice(0, 3)]
}

// Count saved questions, not requests: a failed or cancelled call uses no slot.
export function practiceQuestionPlan(session: PracticeSession, cursor = session.questionSourceCursor ?? session.questions.length) {
  const sources = practicePassages(session)
  const totalQuestions = sources.length * practiceQuestionsPerPassage
  const used = new Map<string, number>()
  for (const question of session.questions) {
    const key = question.focusSourceKey || (question.sources[0] && practiceSourceKey(question.sources[0]))
    if (key) used.set(key, (used.get(key) ?? 0) + 1)
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
