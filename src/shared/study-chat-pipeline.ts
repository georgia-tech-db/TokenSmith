import type { ChatMessage } from './app-state'
import { questionWithSelection } from './chat-selection'
import { reasoningMode } from './reasoning'
import type {
  ChatReferenceExchange, EngineChatRequest, EngineQuestionRewriteRequest, LibrarySearchResult, QuestionRewrite
} from './engine'
import { startLatencySpan, type LatencySpan } from './latency-trace'

export function answerForDisplay(message: ChatMessage) {
  if (message.explanationView === 'reasoning' && message.reasoningAnswer) return message.reasoningAnswer
  return message.explanationView === 'simple' && message.simplerExplanation
    ? message.simplerExplanation
    : message
}

export function lastChatExchange(messages: ChatMessage[]): ChatReferenceExchange | undefined {
  // Do not skip an unanswered student turn or replay an older, unrelated answer.
  const answer = messages.at(-1)
  if (answer?.role !== 'assistant') return undefined
  const question = messages.at(-2)
  if (question?.role !== 'user' || !question.text.trim() || !answer.text.trim()) return undefined
  return { question: questionWithSelection(question.text, question.selectedPassage), answer: answerForDisplay(answer).text }
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
  search: (query: string) => Promise<LibrarySearchResult>
}

export async function prepareRewrittenStudyChat(
  request: EngineChatRequest,
  dependencies: RewritePipelineDependencies
): Promise<{
  resolution: QuestionRewrite
  request?: EngineChatRequest
  rewriteMs: number
  searchMs: number
  latencySpans: LatencySpan[]
}> {
  const previous = lastChatExchange(request.messages)
  const needsRewrite = Boolean(previous || request.selectedPassage ||
    (request.model?.engine === 'ollama' && reasoningMode(request.modelSettings) === 'auto'))
  const rewriteTimer = startLatencySpan('Rewriting')
  const resolution: QuestionRewrite = needsRewrite
    ? await dependencies.resolve(request)
    : { mode: 'standalone', query: request.prompt, clarification: '', reasoning: false }
  const rewriteSpan = rewriteTimer.finish({
    status: needsRewrite ? 'ok' : 'skipped',
    inCount: previous ? 1 : 0,
    details: {
      history_turns: previous ? 1 : 0,
      selected_passage: Boolean(request.selectedPassage),
      mode: resolution.mode
    }
  })
  const rewriteMs = needsRewrite ? rewriteSpan.durationMs : 0
  if (resolution.mode === 'clarify') {
    return { resolution, rewriteMs, searchMs: 0, latencySpans: [rewriteSpan] }
  }

  // A selected passage is explicit context, even if the model labels its query standalone.
  const mode = request.selectedPassage ? 'contextual' : resolution.mode
  const query = mode === 'standalone' ? request.prompt : resolution.query
  if (!query.trim()) throw new Error('The question rewriter returned an empty search query.')
  const searchTimer = startLatencySpan('Retrieval')
  const searchResult = await dependencies.search(query)
  const retrievedSources = searchResult.sources
  const retrievalSpan = searchTimer.finish({
    outCount: retrievedSources.length,
    details: {
      mode: request.applicationSettings?.searchMode ?? 'hybrid',
      reason: searchResult.reason ?? null
    },
    children: searchResult.retrievalChildren
  })
  return {
    resolution: { ...resolution, mode, query },
    rewriteMs,
    searchMs: retrievalSpan.durationMs,
    latencySpans: [rewriteSpan, retrievalSpan],
    request: {
      ...request,
      // A rewrite is a retrieval aid, never the student's replacement task.
      answerPrompt: request.prompt,
      retrievalQuery: query,
      conversationContextMode: mode,
      reasoning: resolution.reasoning,
      referenceExchange: mode === 'contextual' && !request.selectedPassage ? previous : undefined,
      retrievedSources
    }
  }
}
