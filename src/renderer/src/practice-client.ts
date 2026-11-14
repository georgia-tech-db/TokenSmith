import type { ChatSource, CourseMaterial, LocalModel, TokenSmithSettings } from '@shared/app-state'
import type { TokenSmithBridge } from '@shared/bridge'
import type { PracticeQuestion, PracticeSession } from '@shared/practice'
import { practiceSourceKey, practiceQuestionText } from '../../shared/practice'
import { parsePracticeQuestion, quizQuestionPrompt } from '../../shared/quiz'
import { sourcesInScope } from '../../shared/study-scope'
import { practiceConceptKey } from '../../shared/practice-study'
import { loadSavedTopics } from '../../shared/saved-topics'
import { practiceEvidence, practiceQuestionPlan } from '../../shared/practice-question-plan'

export function createPracticeClient(bridge: TokenSmithBridge, model: LocalModel, settings: TokenSmithSettings,
  retrieval?: { materials: CourseMaterial[]; embeddingModels: LocalModel[] }, history: PracticeSession[] = [], random: () => number = Math.random) {
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
      if (!session.topic) return session
      let topic = session.topic
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
      onSelection?: (nextCursor: number, sources: ChatSource[], targetAnswer: boolean) => void): Promise<PracticeQuestion | null> {
      signal.throwIfAborted()
      const concept = session.topic?.concept
      const conceptKey = concept && practiceConceptKey(concept)
      const related = history.filter(s => s.id !== session.id && conceptKey && s.topic?.concept && practiceConceptKey(s.topic.concept) === conceptKey)
      // Continue the topic's round-robin position across sessions and restarts.
      const cursor = Math.max(0, ...[...related, session].map(s =>
        Number.isSafeInteger(s.questionSourceCursor) && s.questionSourceCursor! >= 0 ? s.questionSourceCursor! : s.questions.length))
      const plan = practiceQuestionPlan(session, cursor)
      if (!plan.source) return null
      const sources = practiceEvidence(session, plan.source)
      const targetAnswer = session.questionTargetAnswer ?? random() < 0.5
      onSelection?.(plan.nextCursor, sources, targetAnswer)
      signal.throwIfAborted()
      const previous = [...related.sort((a, b) => a.createdAt.localeCompare(b.createdAt)).flatMap(s => s.questions), ...session.questions]
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
        topic: session.topic, targetAnswer, focusSourceKey: sourceKey }), sources)
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
