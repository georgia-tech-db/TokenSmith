import type { ChatSource, CourseMaterial, LocalModel, TokenSmithSettings } from '@shared/app-state'
import type { TokenSmithBridge } from '@shared/bridge'
import type { PracticeQuestion, PracticeSession } from '@shared/practice'
import { practiceSourceKey, practiceReference, practiceDiscussionPassage, practiceQuestionText } from '../../shared/practice'
import { trimReferenceExchange } from '../../shared/study-chat-pipeline'
import { parsePracticeQuestion, checkPracticeQuestionAnswer, PracticeQuestionValidationError, quizQuestionPrompt } from '../../shared/quiz'
import { sourcesInScope } from '../../shared/study-scope'
import { practiceConceptKey } from '../../shared/practice-study'
import { masteryEvidenceKey } from '../../shared/practice-mastery'

export function createPracticeClient(bridge: TokenSmithBridge, model: LocalModel, settings: TokenSmithSettings,
  retrieval?: { materials: CourseMaterial[]; embeddingModels: LocalModel[] }, history: PracticeSession[] = []) {
  async function send(session: PracticeSession, requestId: string, prompt: string, sources: ChatSource[], practiceTask: 'true_false_question' | 'verify_question' = 'true_false_question') {
    if (model.id !== session.modelId || model.status !== 'ready') throw new Error('The model used for this session is unavailable. Reconnect it before continuing.')
    return bridge.sendChatMessage({
      requestId, practiceTask, prompt, messages: [], materials: [], model, settings,
      retrievedSources: sources, conversationContextMode: 'standalone', reasoning: false,
      applicationSettings: { ...settings.application, suggestionMode: 'off', explanationDepthEnabled: false },
      modelSettings: { ...session.modelSettings, systemMessage: '', reasoningMode: 'off', thinking: false,
        maxLength: Math.min(session.modelSettings.maxLength, practiceTask === 'verify_question' ? 512 : 2048) }
    })
  }
  return {
    async prepareTopic(session: PracticeSession, signal: AbortSignal): Promise<PracticeSession> {
      if (!session.topic) return session
      let sources = sourcesInScope(session.topic.sources ?? [], session.documents)
      if (!session.topic.prepared) {
        if (!retrieval) throw new Error('Select documents to practice this topic.')
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
      if (!sources.length) throw new Error('No supporting passages were found for this topic in the selected documents. Choose another topic or change the selected documents.')
      return { ...session, topic: { ...session.topic, sources, prepared: true } }
    },
    async question(session: PracticeSession, requestId: string, signal: AbortSignal,
      onAttempt?: (nextCursor: number, attempt: number, total: number) => void, onStage?: (label: string) => void): Promise<PracticeQuestion | null> {
      signal.throwIfAborted()
      const candidates = session.topic ? sourcesInScope(session.topic.sources ?? [], session.documents)
        : await bridge.practiceSources(session.documents,
          session.questions.flatMap(question => question.sources.map(practiceSourceKey)), session.questions.length)
      signal.throwIfAborted()
      // Saved topic citations and retrieval can refer to the same chunk with different unit IDs.
      const pool = [...new Map(candidates.map(source => [
        [source.materialId, source.documentId, source.chunkId || source.chunkRowid || practiceSourceKey(source)].join(':'), source
      ])).values()]
      if (!pool.length) return null
      const cursor = Number.isSafeInteger(session.questionSourceCursor) && session.questionSourceCursor! >= 0
        ? session.questionSourceCursor! : session.questions.length
      // One self-contained passage per attempt avoids stitching unrelated claims.
      // Visit alternatives before reusing evidence; a lone passage gets one repair.
      const total = Math.min(4, Math.max(2, pool.length))
      const concept = session.topic?.concept
      const conceptKey = concept && practiceConceptKey(concept)
      const previous = [...history.filter(s => s.id !== session.id).sort((a, b) => a.createdAt.localeCompare(b.createdAt)).flatMap(s => s.questions)
        .filter(q => conceptKey && q.concept && practiceConceptKey(q.concept) === conceptKey), ...session.questions]
      const prompt = { questionNumber: session.questions.length + 1,
        previousQuestions: previous.slice(-12).map(practiceQuestionText), topic: session.topic }
      let lastError: PracticeQuestionValidationError | undefined
      for (let attempt = 0; attempt < total; attempt++) {
        signal.throwIfAborted()
        const sources = [pool[(cursor + attempt) % pool.length]]
        onAttempt?.(cursor + attempt + 1, attempt + 1, total)
        signal.throwIfAborted()
        const correction = lastError && `The previous package failed validation: ${lastError.reason} Prepare a fresh, declarative True/False statement with a period, not a question or instruction. Use only the passage supplied NOW. Copy one short exact quote and its Evidence ID from the SAME supplied passage; never stitch table rows or cite a table header. Recheck the answer key against that evidence.`
        try {
          const reply = await send(session, requestId, quizQuestionPrompt({ ...prompt, correction }), sources)
          signal.throwIfAborted()
          const prepared = parsePracticeQuestion(reply.text, reply.sources, 'true_false')
          if (previous.some(question => question.text.trim().toLocaleLowerCase() === prepared.text.trim().toLocaleLowerCase()))
            throw new PracticeQuestionValidationError('The model repeated an earlier question. Choose a different claim on this topic.')
          if (conceptKey && previous.some(question => masteryEvidenceKey(question) === masteryEvidenceKey(prepared)))
            throw new PracticeQuestionValidationError('The question reuses an earlier supporting claim. Test another aspect of this topic using fresh evidence.')
          onStage?.('Checking answer')
          const checked = await send(session, requestId, JSON.stringify({ statement: prepared.text,
            topic: session.topic?.title, learningObjectives: session.topic?.objectives, previousQuestions: prompt.previousQuestions }), prepared.sources, 'verify_question')
          signal.throwIfAborted()
          checkPracticeQuestionAnswer(checked.text, prepared.correctAnswer!)
          return { id: crypto.randomUUID(), ...prepared,
            ...(concept && conceptKey ? { concept, masteryCheck: { version: 1 as const, conceptKey } } : {}),
            draft: '', attempts: [], reviewing: false, timeline: [] }
        } catch (error) {
          signal.throwIfAborted()
          if (error instanceof PracticeQuestionValidationError) lastError = error
          else if (error instanceof Error && /structured model response exceeded its output limit|Ollama returned an empty response/.test(error.message))
            lastError = new PracticeQuestionValidationError('The response was incomplete. Keep the question package concise.')
          else throw error // Connection/configuration failures need attention, not more model calls.
        }
      }
      throw new PracticeQuestionValidationError(lastError!.reason, total)
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
