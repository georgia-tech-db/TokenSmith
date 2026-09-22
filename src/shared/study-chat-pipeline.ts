import type { ChatMessage, ChatSource } from './app-state'
import { questionWithSelection } from './chat-selection'
import type {
  ChatReferenceExchange, EngineChatRequest, EngineQuestionRewriteRequest, QuestionRewrite
} from './engine'

export function lastChatExchange(messages: ChatMessage[]): ChatReferenceExchange | undefined {
  // Do not skip an unanswered student turn or replay an older, unrelated answer.
  const answer = messages.at(-1)
  if (answer?.role !== 'assistant') return undefined
  const question = messages.at(-2)
  if (question?.role !== 'user' || !question.text.trim() || !answer.text.trim()) return undefined
  return { question: questionWithSelection(question.text, question.selectedPassage), answer: answer.text }
}

export function trimReferenceExchange(exchange: ChatReferenceExchange, maxChars: number): ChatReferenceExchange {
  const clip = (text: string, limit: number) => text.length <= limit
    ? text
    : `${text.slice(0, Math.max(0, limit - 16))}\n[truncated]`
  const question = clip(exchange.question, Math.floor(maxChars / 3))
  return { question, answer: clip(exchange.answer, maxChars - question.length) }
}

interface RewritePipelineDependencies {
  resolve: (request: EngineQuestionRewriteRequest) => Promise<QuestionRewrite>
  search: (query: string) => Promise<ChatSource[]>
}

export async function prepareRewrittenStudyChat(
  request: EngineChatRequest,
  dependencies: RewritePipelineDependencies
): Promise<{
  resolution: QuestionRewrite
  request?: EngineChatRequest
  rewriteMs: number
  searchMs: number
}> {
  const previous = lastChatExchange(request.messages)
  const needsRewrite = Boolean(previous || request.selectedPassage)
  const rewriteStart = performance.now()
  const resolution: QuestionRewrite = needsRewrite
    ? await dependencies.resolve(request)
    : { mode: 'standalone', query: request.prompt, clarification: '' }
  const rewriteMs = needsRewrite ? performance.now() - rewriteStart : 0
  if (resolution.mode === 'clarify') {
    return { resolution, rewriteMs, searchMs: 0 }
  }

  // A selected passage is explicit context, even if the model labels its query standalone.
  const mode = request.selectedPassage ? 'contextual' : resolution.mode
  const query = mode === 'standalone' ? request.prompt : resolution.query
  if (!query.trim()) throw new Error('The question rewriter returned an empty search query.')
  const searchStart = performance.now()
  const retrievedSources = await dependencies.search(query)
  return {
    resolution: { ...resolution, mode, query },
    rewriteMs,
    searchMs: performance.now() - searchStart,
    request: {
      ...request,
      // A rewrite is a retrieval aid, never the student's replacement task.
      answerPrompt: request.prompt,
      retrievalQuery: query,
      conversationContextMode: mode,
      referenceExchange: mode === 'contextual' && !request.selectedPassage ? previous : undefined,
      retrievedSources
    }
  }
}
