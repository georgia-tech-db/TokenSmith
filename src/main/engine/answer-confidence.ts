import type { ChatSource } from '../../shared/app-state'
import type { AnswerConfidence } from '../../shared/confidence'
import { abstentionAnswer, scoreAnswerConfidence } from '../../shared/confidence'
import type { EngineChatRequest, EngineChatResponse } from '../../shared/engine'

function normalizeForAbstentionMatch(text: string): string {
  return text.toLowerCase().replace(/[^\p{L}\p{N}]+/gu, ' ').replace(/\s+/g, ' ').trim()
}

const abstentionMarker = normalizeForAbstentionMatch(abstentionAnswer)

/**
 * True when the model took the abstention instruction itself, rather than the
 * scorer having to force it.
 *
 * Only an answer that is essentially just that sentence counts. Models often
 * write a real answer and then tack the sentence on as a hedge; treating that
 * as an abstention would label a substantive answer "abstained" while still
 * showing it. Those trailing hedges go through `stripAbstentionHedge` instead.
 */
export function isAbstention(text: string): boolean {
  const normalized = normalizeForAbstentionMatch(text)

  return normalized.includes(abstentionMarker) && normalized.length <= abstentionMarker.length + 24
}

/**
 * Drops a trailing abstention sentence from an otherwise substantive answer, so
 * the answer is graded on what it actually claims.
 */
export function stripAbstentionHedge(text: string): string {
  const kept = text
    .split(/(?<=[.!?])\s+/)
    .filter((sentence) => !normalizeForAbstentionMatch(sentence).includes(abstentionMarker))
    .join(' ')
    .trim()

  return kept || text
}

export function scoredSources(request: EngineChatRequest, response: EngineChatResponse): ChatSource[] {
  return response.sources.length > 0 ? response.sources : request.retrievedSources ?? []
}

/**
 * Grades an answer and abstains in its place when the material does not support
 * it. The prompt already asks the model to abstain on its own; this is the
 * backstop for when it answers anyway.
 */
export function withAnswerConfidence(
  request: EngineChatRequest,
  response: EngineChatResponse
): EngineChatResponse {
  const sources = scoredSources(request, response)

  if (isAbstention(response.text)) {
    const scored = scoreAnswerConfidence(request.prompt, response.text, sources)
    const confidence: AnswerConfidence = {
      ...scored,
      status: 'unsupported',
      abstained: true
    }

    return { ...response, text: abstentionAnswer, confidence, followUpSuggestions: [] }
  }

  const text = stripAbstentionHedge(response.text)
  const scored = scoreAnswerConfidence(request.prompt, text, sources)

  if (scored.status === 'unsupported') {
    const confidence: AnswerConfidence = { ...scored, abstained: true }

    return {
      ...response,
      text: abstentionAnswer,
      confidence,
      followUpSuggestions: []
    }
  }

  return { ...response, text, confidence: scored }
}
