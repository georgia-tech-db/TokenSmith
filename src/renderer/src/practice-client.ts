import type { ChatSource, CourseMaterial, LocalModel, TokenSmithSettings } from '@shared/app-state'
import type { TokenSmithBridge } from '@shared/bridge'
import type { PracticeQuestion, PracticeSession, PracticeTopic, PracticeMasteryRecord } from '@shared/practice'
import { practiceSourceKey, practiceQuestionText } from '../../shared/practice'
import { parsePracticeQuestion, quizQuestionPrompt } from '../../shared/quiz'
import { sourcesInScope } from '../../shared/study-scope'
import { practiceConceptKey } from '../../shared/practice-study'
import { loadSavedTopics } from '../../shared/saved-topics'
import { practiceMastery } from '../../shared/practice-mastery'
import { practiceEvidence, practiceQuestionPlan, practiceTopics, practiceTopicKey, questionsForTopic } from '../../shared/practice-question-plan'

export function createPracticeClient(bridge: TokenSmithBridge, model: LocalModel, settings: TokenSmithSettings,
  retrieval?: { materials: CourseMaterial[]; embeddingModels: LocalModel[] }, history: PracticeSession[] = [], random: () => number = Math.random, masteryRecords: PracticeMasteryRecord[] = []) {
  async function send(session: PracticeSession, requestId: string, prompt: string, sources: ChatSource[]) {
    if (model.id !== session.modelId || model.status !== 'ready') throw new Error('The model used for this session is unavailable. Reconnect it before continuing.')
    return bridge.sendChatMessage({
      requestId, practiceTask: 'true_false_question', prompt, messages: [], materials: [], model, settings,
      retrievedSources: sources, conversationContextMode: 'standalone', reasoning: false,
      applicationSettings: { ...settings.application, suggestionMode: 'off', explanationDepthEnabled: false },
      modelSettings: { ...session.modelSettings, systemMessage: '', reasoningMode: 'off', thinking: false,
        maxLength: Math.min(session.modelSettings.maxLength, 2048) }
    })
  }
  return {
    async prepareTopic(session: PracticeSession, signal: AbortSignal, requestId?: string): Promise<PracticeSession> {
      signal.throwIfAborted()
      if (!practiceTopics(session).length) return session
      if (session.topics) {
        const saved = await loadSavedTopics(bridge, session.documents, retrieval?.materials ?? [])
        signal.throwIfAborted()
        const topics = session.topics.map(topic => {
          const mapped = saved.find(t => practiceTopicKey(t) === practiceTopicKey(topic))
          if (!mapped) throw new Error('A selected topic is unavailable. Choose topics from the current book preparation.')
          return { ...mapped, prepared: true }
        })
        return { ...session, topics }
      }
      let topic = session.topic!
      // Inventory topics own their evidence. Reload that mapping so an older
      // session's search-expanded pool cannot override the saved topic.
      if (topic.id) {
        const saved = await loadSavedTopics(bridge, session.documents, retrieval?.materials ?? [])
        const mapped = saved.find(t => t.id === topic.id && t.concept?.courseId === topic.concept?.courseId)
        if (!mapped) throw new Error('This saved topic is unavailable. Choose a topic from the current book preparation.')
        topic = mapped
      }
      let sources = sourcesInScope(topic.sources ?? [], session.documents)
      if (!topic.id && !sources.length) {
        if (!retrieval) throw new Error('Select documents to practice this topic.')
        const ids = new Set(session.documents.map(document => document.materialId))
        const query = topic.question
          ? `${topic.question.slice(0, 1000)}\n${topic.answer?.slice(0, 1000) ?? ''}` : topic.title
        const result = await bridge.searchLibrary(query,
          retrieval.materials.filter(material => ids.has(material.id)).map(material => ({ ...material, isActive: true })),
          8, retrieval.embeddingModels, settings.application.searchMode,
          { embeddingGpuEnabled: settings.application.embeddingGpuEnabled, documents: session.documents, requestId })
        sources = sourcesInScope(result.sources, session.documents)
      }
      signal.throwIfAborted()
      sources = [...new Map(sources.filter(source => (source.context || source.excerpt)?.trim())
        .map(source => [practiceSourceKey(source), source])).values()]
      if (!sources.length) throw new Error('No supporting passages were found for this topic in the selected documents. Choose another topic or change the selected documents.')
      return { ...session, topic: { ...topic, sources, prepared: true } }
    },
    async question(session: PracticeSession, requestId: string, signal: AbortSignal,
      onSelection?: (nextCursor: number, sources: ChatSource[], targetAnswer: boolean, topic: PracticeTopic) => void): Promise<PracticeQuestion | null> {
      signal.throwIfAborted()
      const mastery = practiceMastery([...history.filter(s => s.id !== session.id), session], [], masteryRecords)
      // Each question owns its topic, even when its session spans several chapters.
      const selected = practiceQuestionPlan(session, undefined, mastery).topic
      if (!selected) return null
      const concept = selected.concept
      const conceptKey = concept && practiceConceptKey(concept)
      const related = history.filter(s => s.id !== session.id)
      const previous = [...related.sort((a, b) => a.createdAt.localeCompare(b.createdAt))
        .flatMap(s => questionsForTopic(s, selected)), ...questionsForTopic(session, selected)]
      const cursors = [...related, session].flatMap(s => {
        const value = s.topics ? s.topicSourceCursors?.[practiceTopicKey(selected)]
          : s.topic && practiceTopicKey(s.topic) === practiceTopicKey(selected) ? s.questionSourceCursor ?? s.questions.length : undefined
        return Number.isSafeInteger(value) && value! >= 0 ? [value!] : []
      })
      const cursor = Math.max(0, ...cursors)
      const focused = { ...session, topics: undefined, topic: selected, questions: questionsForTopic(session, selected) }
      const plan = practiceQuestionPlan(focused, cursor)
      if (!plan.source) return null
      const sources = practiceEvidence(focused, plan.source)
      const targetAnswer = session.questionTargetAnswer ?? random() < 0.5
      onSelection?.(plan.nextCursor, sources, targetAnswer, selected)
      signal.throwIfAborted()
      const sourceKey = practiceSourceKey(plan.source)
      const passageQuestions = [...history.filter(s => s.id !== session.id)
        .sort((a, b) => a.createdAt.localeCompare(b.createdAt)).flatMap(s => s.questions), ...session.questions]
        .filter(q => (q.focusSourceKey || (q.sources[0] && practiceSourceKey(q.sources[0]))) === sourceKey ||
          (q.sources.some(s => practiceSourceKey(s) === sourceKey) &&
            q.rubric?.criteria.some(c => c.evidence.some(e => e.sourceKey === sourceKey)))).slice(-12)
        .map(q => ({ question: practiceQuestionText(q), correctAnswer: q.correctAnswer,
          testedIdeas: q.rubric?.criteria.map(c => c.description) ?? [],
          supportingQuotes: [...new Set(q.rubric?.criteria.flatMap(c => c.evidence.map(e => e.quote)) ?? [])] }))
      const reply = await send(session, requestId, quizQuestionPrompt({ questionNumber: session.questions.length + 1,
        previousQuestions: previous.slice(-12).map(practiceQuestionText), previousQuestionsFromPassage: passageQuestions,
        previousTestedIdeas: [...new Set(previous.flatMap(q => [q.rubric?.objective, ...(q.rubric?.criteria.map(c => c.description) ?? [])]).filter((idea): idea is string => Boolean(idea)))].slice(-40),
        topic: selected, targetAnswer, focusSourceKey: sourceKey }), sources)
      signal.throwIfAborted()
      // Single generation call: parsing is structural only. Do not reject or repair
      // claims based on wording, citations, novelty, topic judgment or target parity.
      const prepared = parsePracticeQuestion(reply.text, reply.sources)
      return { id: crypto.randomUUID(), ...prepared, focusSourceKey: sourceKey, requestedAnswer: targetAnswer,
        ...(concept && conceptKey ? { concept, masteryCheck: { version: 1 as const, conceptKey, verification: 'none' as const } } : {}),
        draft: '', attempts: [] }
    }
  }
}
