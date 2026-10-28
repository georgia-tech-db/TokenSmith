import type { ChatSource, CourseMaterial, LocalModel, ModelRuntimeSettings, TokenSmithSettings } from '@shared/app-state'
import type { TokenSmithBridge } from '@shared/bridge'
import type { PracticeQuestion, PracticeSession } from '@shared/practice'
import { practiceSourceKey, practiceReference, practiceDiscussionPassage, practiceQuestionText } from '../../shared/practice'
import { trimReferenceExchange } from '../../shared/study-chat-pipeline'
import { parsePracticeQuestion, parsePracticeTopics, quizQuestionPrompt } from '../../shared/quiz'
import { sourcesInScope, type StudyDocumentRef } from '../../shared/study-scope'

export function createPracticeClient(bridge: TokenSmithBridge, model: LocalModel, settings: TokenSmithSettings,
  retrieval?: { materials: CourseMaterial[]; embeddingModels: LocalModel[] }) {
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
    async topics(documents: StudyDocumentRef[], modelSettings: ModelRuntimeSettings, requestId: string, signal: AbortSignal) {
      const sources = sourcesInScope(await bridge.starterSources(retrieval?.materials ?? [], 24, documents), documents)
      signal.throwIfAborted()
      if (!sources.length) return []
      // Short temporary IDs are easier for small models to copy than stored chunk hashes.
      const evidence = sources.map((source, index) => ({ ...source, sourceUnitId: `topic-${index + 1}`,
        context: undefined, excerpt: (source.context || source.excerpt).slice(0, 500) }))
      const reply = await bridge.sendChatMessage({ requestId, practiceTask: 'topics', model, settings, materials: [], messages: [],
        prompt: 'Select a few named concepts or methods to practice. Name one concept per topic. Do not summarize the whole document or merge distinct concepts into broad categories.',
        retrievedSources: evidence,
        applicationSettings: { ...settings.application, suggestionMode: 'off', explanationDepthEnabled: false }, reasoning: false,
        modelSettings: { ...modelSettings, systemMessage: '', reasoningMode: 'off', thinking: false, maxLength: Math.min(modelSettings.maxLength, 768) } })
      signal.throwIfAborted()
      const included = new Set(reply.sources.map(practiceSourceKey))
      const originals = new Map(evidence.map((source, index) => [practiceSourceKey(source), sources[index]]))
      return parsePracticeTopics(reply.text, evidence.filter(source => included.has(practiceSourceKey(source))))
        .map(topic => ({ ...topic, sources: topic.sources!.map(source => originals.get(practiceSourceKey(source))!) }))
    },
    async prepareTopic(session: PracticeSession, signal: AbortSignal): Promise<PracticeSession> {
      if (!session.topic) return session
      let sources = sourcesInScope(session.topic.sources ?? [], session.documents)
      if (!session.topic.prepared) {
        if (!retrieval) throw new Error('Select documents in Library to practice this topic.')
        const ids = new Set(session.documents.map(document => document.materialId))
        const query = session.topic.question
          ? `${session.topic.question.slice(0, 1000)}\n${session.topic.answer?.slice(0, 1000) ?? ''}` : session.topic.title
        try {
          const result = await bridge.searchLibrary(query,
            retrieval.materials.filter(material => ids.has(material.id)).map(material => ({ ...material, isActive: true })),
            8, retrieval.embeddingModels, settings.application.searchMode,
            { embeddingGpuEnabled: settings.application.embeddingGpuEnabled, documents: session.documents })
          sources = [...sourcesInScope(result.sources, session.documents), ...sources]
        } catch (reason) { if (!sources.length) throw reason }
      }
      signal.throwIfAborted()
      sources = [...new Map(sources.filter(source => (source.context || source.excerpt)?.trim())
        .map(source => [practiceSourceKey(source), source])).values()].slice(0, 8)
      if (!sources.length) throw new Error('No supporting passages were found for this topic in the selected documents. Choose another topic or change the selection in Library.')
      return { ...session, topic: { ...session.topic, sources, prepared: true } }
    },
    async question(session: PracticeSession, requestId: string, signal: AbortSignal): Promise<PracticeQuestion | null> {
      const pool = session.topic ? sourcesInScope(session.topic.sources ?? [], session.documents) : []
      const offset = pool.length ? session.questions.length % pool.length : 0
      const sources = session.topic ? [...pool.slice(offset), ...pool.slice(0, offset)].slice(0, 4)
        : await bridge.practiceSources(session.documents,
          session.questions.flatMap(question => question.sources.map(practiceSourceKey)), session.questions.length)
      signal.throwIfAborted()
      if (!sources.length) return null
      const reply = await send(session, requestId,
        quizQuestionPrompt({ questionNumber: session.questions.length + 1,
          previousQuestions: session.questions.slice(-12).map(practiceQuestionText), topic: session.topic }), sources)
      signal.throwIfAborted()
      const prepared = parsePracticeQuestion(reply.text, reply.sources, 'true_false')
      if (session.topic && session.questions.some(question => question.text.trim().toLocaleLowerCase() === prepared.text.trim().toLocaleLowerCase())) {
        throw new Error('The model repeated an earlier question. Use the arrow to try another, or choose Practice another topic.')
      }
      return { id: crypto.randomUUID(), ...prepared,
        draft: '', attempts: [], reviewing: false, timeline: [] }
    },
    async discuss(session: PracticeSession, question: PracticeQuestion, prompt: string, requestId: string) {
      if (model.id !== session.modelId || model.status !== 'ready') throw new Error('The model used for this session is unavailable. Reconnect it before continuing.')
      const reference = practiceDiscussionPassage(practiceReference(session, question))
      const previous = question.discussions?.at(-1)
      // Only this exercise and the latest help exchange enter the prompt, not the practice transcript.
      const context = trimReferenceExchange({ question: reference.question,
        answer: [reference.text, previous && `Latest help question: ${previous.question}\nLatest explanation: ${previous.answer}`].filter(Boolean).join('\n\n') }, 4000)
      const selectedPassage = question.helpPassage
        ? { ...question.helpPassage, question: `${context.question}\n\n${context.answer}` }
        : { ...reference, question: context.question, text: context.answer }
      const reply = await bridge.sendChatMessage({ requestId, prompt, messages: [], materials: [], model, settings,
        selectedPassage, retrievedSources: question.sources, conversationContextMode: 'contextual',
        applicationSettings: { ...settings.application, suggestionMode: 'off' },
        modelSettings: { ...session.modelSettings, maxLength: Math.min(session.modelSettings.maxLength, 1536),
          systemMessage: [
            'This is a help discussion during practice. The original exercise and earlier feedback are reference context, not the task to answer now.',
            'Answer only the student\'s current help question. Explain the requested term, step, or example without completing the exercise unless they explicitly ask for its solution. Never grade this message or start another exercise.',
            'A True/False exercise is a claim to evaluate, not an established fact. If the learner challenges the claim or the saved explanation, check it against the evidence rather than assuming the answer key is right.',
            'Keep help focused: a few short paragraphs or a small worked example, not a full lesson. Preserve the exercise\'s qualifications and distinguish possible or worst-case outcomes from guarantees. Do not expand into unrelated background or code unless requested.'
          ].join('\n') }
      })
      if (!reply.text.trim()) throw new Error('No explanation was returned. Your question is saved; try again.')
      return reply
    }
  }
}
