import type { ChatSource, LocalModel, TokenSmithSettings } from '@shared/app-state'
import type { TokenSmithBridge } from '@shared/bridge'
import type { PracticeQuestion, PracticeSession } from '@shared/practice'
import { practiceSourceKey } from '../../shared/practice'
import { parsePracticeFeedback, parsePracticeQuestion, quizFeedbackPrompt, quizQuestionPrompt, type PracticeTask } from '../../shared/quiz'

export function createPracticeClient(bridge: TokenSmithBridge, model: LocalModel, settings: TokenSmithSettings) {
  async function send(session: PracticeSession, requestId: string, practiceTask: PracticeTask, prompt: string, sources: ChatSource[]) {
    if (model.id !== session.modelId || model.status !== 'ready') throw new Error('The model used for this session is unavailable. Reconnect it before continuing.')
    return bridge.sendChatMessage({
      requestId, practiceTask, prompt, messages: [], materials: [], model, settings,
      retrievedSources: sources, conversationContextMode: 'standalone', reasoning: false,
      applicationSettings: { ...settings.application, suggestionMode: 'off', explanationDepthEnabled: false },
      modelSettings: { ...session.modelSettings, systemMessage: '', reasoningMode: 'off', thinking: false,
        maxLength: Math.min(session.modelSettings.maxLength, practiceTask === 'question' ? 2048 : 1024) }
    })
  }
  return {
    async question(session: PracticeSession, requestId: string, signal: AbortSignal): Promise<PracticeQuestion> {
      const sources = await bridge.practiceSources(session.documents,
        session.questions.flatMap(question => question.sources.map(practiceSourceKey)), session.questions.length)
      signal.throwIfAborted()
      if (!sources.length) throw new Error('No unused passages remain in these documents. Finish this session or start with more documents.')
      const reply = await send(session, requestId, 'question', quizQuestionPrompt({ questionNumber: session.questions.length + 1,
        totalQuestions: session.totalQuestions, previousQuestions: session.questions.map(question => question.text) }), sources)
      signal.throwIfAborted()
      const prepared = parsePracticeQuestion(reply.text, reply.sources)
      return { id: crypto.randomUUID(), ...prepared,
        draft: '', attempts: [], reviewing: false }
    },
    async feedback(session: PracticeSession, question: PracticeQuestion, answer: string, requestId: string) {
      if (!question.rubric || !question.referenceAnswer) throw new Error('This older question has no saved grading guide. Start a new practice for updated feedback.')
      const previousAttempt = question.attempts.at(-1)
      const reply = await send(session, requestId, 'feedback', quizFeedbackPrompt({ answer, question: question.text,
        rubric: question.rubric, explanation: question.referenceAnswer, previousAttempt }), question.sources)
      return parsePracticeFeedback(reply.text, question.rubric, Boolean(previousAttempt))
    }
  }
}
