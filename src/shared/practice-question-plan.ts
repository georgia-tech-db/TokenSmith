import { practiceSourceKey, type PracticeSession, type PracticeTopic, type PracticeQuestion } from './practice'
import { practiceConceptKey } from './practice-study'
import type { TopicMastery } from './practice-mastery'
import type { ChatSource } from './app-state'
import { sourcesInScope } from './study-scope'

export const practiceQuestionsPerPassage = 2

export const practiceTopics = (session: PracticeSession): PracticeTopic[] => session.topics ?? (session.topic ? [session.topic] : [])
export const practiceTopicKey = (topic: PracticeTopic) => topic.concept ? practiceConceptKey(topic.concept) : topic.id || topic.title
export const questionsForTopic = (session: PracticeSession, topic: PracticeTopic): PracticeQuestion[] =>
  session.questions.filter(q => q.concept ? practiceConceptKey(q.concept) === practiceTopicKey(topic) : !session.topics && session.topic && practiceTopicKey(session.topic) === practiceTopicKey(topic))

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
export function practiceQuestionPlan(session: PracticeSession, cursor = session.questionSourceCursor ?? session.questions.length,
  mastery: Map<string, TopicMastery> = new Map()) {
  if (session.topics) {
    const plans = session.topics.map(topic => {
      const key = practiceTopicKey(topic)
      const questions = questionsForTopic(session, topic)
      const plan = passagePlan({ ...session, topics: undefined, topic, questions }, session.topicSourceCursors?.[key] ?? questions.length)
      return { ...plan, topic, count: questions.length, key }
    })
    const last = session.questions.at(-1)?.concept
    let eligible = plans.filter(p => p.source)
    if (eligible.some(p => p.key !== (last && practiceConceptKey(last)))) eligible = eligible.filter(p => p.key !== (last && practiceConceptKey(last)))
    eligible.sort((a, b) => a.count - b.count || (mastery.get(a.key)?.score ?? 0) - (mastery.get(b.key)?.score ?? 0) ||
      (mastery.get(a.key)?.lastAssessedAt ?? '').localeCompare(mastery.get(b.key)?.lastAssessedAt ?? ''))
    const next = eligible[0]
    return { topic: next?.topic, source: next?.source, nextCursor: next?.nextCursor ?? 0,
      passageCount: plans.reduce((n, p) => n + p.passageCount, 0), totalQuestions: plans.reduce((n, p) => n + p.totalQuestions, 0) }
  }
  return { ...passagePlan(session, cursor), topic: session.topic }
}

function passagePlan(session: PracticeSession, cursor: number) {
  const sources = practicePassages(session)
  const totalQuestions = sources.length * practiceQuestionsPerPassage
  const used = new Map<string, number>()
  for (const question of session.questions) {
    const key = question.focusSourceKey || (question.sources[0] && practiceSourceKey(question.sources[0]))
    if (key) used.set(key, (used.get(key) ?? 0) + 1)
  }
  const start = Number.isSafeInteger(cursor) && cursor >= 0 ? cursor : 0
  if (session.questions.length < totalQuestions) {
    const leastUsed = Math.min(...sources.map(s => used.get(practiceSourceKey(s)) ?? 0))
    for (let offset = 0; offset < sources.length; offset++) {
      const source = sources[(start + offset) % sources.length]
      if ((used.get(practiceSourceKey(source)) ?? 0) === leastUsed && leastUsed < practiceQuestionsPerPassage) {
        return { passageCount: sources.length, totalQuestions, source, nextCursor: start + offset + 1 }
      }
    }
  }
  return { passageCount: sources.length, totalQuestions, source: undefined, nextCursor: start }
}
