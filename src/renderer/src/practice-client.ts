import type { ChatSource, LocalModel, TokenSmithSettings } from '@shared/app-state'
import type { TokenSmithBridge } from '@shared/bridge'
import type { PracticeQuestion, PracticeSession } from '@shared/practice'
import { practiceSourceKey, practiceReference, practiceDiscussionPassage, practiceQuestionText } from '../../shared/practice'
import { trimReferenceExchange } from '../../shared/study-chat-pipeline'
import { parsePracticeFeedback, parsePracticeQuestion, quizFeedbackPrompt, quizQuestionPrompt, type PracticeTask } from '../../shared/quiz'

export function createPracticeClient(bridge: TokenSmithBridge, model: LocalModel, settings: TokenSmithSettings) {
  async function send(session: PracticeSession, requestId: string, practiceTask: PracticeTask, prompt: string, sources: ChatSource[]) {
    if (model.id !== session.modelId || model.status !== 'ready') throw new Error('The model used for this session is unavailable. Reconnect it before continuing.')
    return bridge.sendChatMessage({
      requestId, practiceTask, prompt, messages: [], materials: [], model, settings,
      retrievedSources: sources, conversationContextMode: 'standalone', reasoning: false,
      applicationSettings: { ...settings.application, suggestionMode: 'off', explanationDepthEnabled: false },
      modelSettings: { ...session.modelSettings, systemMessage: '', reasoningMode: 'off', thinking: false,
        maxLength: Math.min(session.modelSettings.maxLength, practiceTask === 'feedback' ? 1024 : 2048) }
    })
  }
  return {
    async question(session: PracticeSession, requestId: string, signal: AbortSignal): Promise<PracticeQuestion | null> {
      const sources = await bridge.practiceSources(session.documents,
        session.questions.flatMap(question => question.sources.map(practiceSourceKey)), session.questions.length)
      signal.throwIfAborted()
      if (!sources.length) return null
      const format = (session.questions.length + 1) % 3 === 0 ? 'true_false' : 'short_answer'
      const reply = await send(session, requestId, format === 'true_false' ? 'true_false_question' : 'question',
        quizQuestionPrompt({ questionNumber: session.questions.length + 1,
          previousQuestions: session.questions.slice(-12).map(practiceQuestionText) }), sources)
      signal.throwIfAborted()
      const prepared = parsePracticeQuestion(reply.text, reply.sources, format)
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
    },
    async feedback(session: PracticeSession, question: PracticeQuestion, answer: string, requestId: string) {
      if (question.format === 'true_false') throw new Error('True/False choices must be checked against their saved answer key, not sent for model grading.')
      if (!question.rubric || !question.referenceAnswer) throw new Error('This older question has no saved grading guide. Start a new practice for updated feedback.')
      const previousAttempt = question.attempts.at(-1)
      const reply = await send(session, requestId, 'feedback', quizFeedbackPrompt({ answer, question: question.text,
        rubric: question.rubric, explanation: question.referenceAnswer, previousAttempt }), question.sources)
      return parsePracticeFeedback(reply.text, question.rubric, Boolean(previousAttempt))
    }
  }
}
