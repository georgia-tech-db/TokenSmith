import type { ChatSource } from '../../shared/app-state'
import type { AnswerConfidence, AnswerVerification } from '../../shared/confidence'
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

/** Runs the NLI check of `answer`'s claims against the passages. */
export type AnswerVerifier = (question: string, answer: string, passages: string[]) => Promise<AnswerVerification>

/** The question the answer responds to, as the answer prompt posed it. */
function answeredQuestion(request: EngineChatRequest): string {
  return (request.referenceExchange ? request.prompt : request.answerPrompt ?? request.prompt).trim()
}

/** The passage text the answer was written from, as the prompt gave it. */
function passageText(source: ChatSource): string {
  return (source.context || source.excerpt || '').trim()
}

async function verifySafely(
  verify: AnswerVerifier,
  question: string,
  answer: string,
  sources: ChatSource[]
): Promise<AnswerVerification> {
  if (sources.length === 0) {
    return { available: false, claims: [], error: 'No passages to check the answer against.' }
  }

  try {
    return await verify(question, answer, sources.map(passageText))
  } catch (error) {
    return { available: false, claims: [], error: error instanceof Error ? error.message : String(error) }
  }
}

/**
 * Settles what the answer is before it is shown: the model's own abstention
 * becomes the standard sentence, and a trailing hedge is dropped. The answer
 * is not checked here; `scoreAnswer` does that after it is on screen.
 */
export function prepareAnswer(request: EngineChatRequest, response: EngineChatResponse): EngineChatResponse {
  if (isAbstention(response.text)) {
    const scored = scoreAnswerConfidence(answeredQuestion(request), scoredSources(request, response), {
      available: false,
      claims: []
    })
    const confidence: AnswerConfidence = { ...scored, status: 'unsupported', abstained: true }

    return { ...response, text: abstentionAnswer, confidence, followUpSuggestions: [] }
  }

  return { ...response, text: stripAbstentionHedge(response.text) }
}

/**
 * Checks a prepared answer against the passages it was written from. If the
 * check cannot run, the answer is marked unverified.
 */
export async function scoreAnswer(
  request: EngineChatRequest,
  response: EngineChatResponse,
  verify: AnswerVerifier
): Promise<AnswerConfidence> {
  const sources = scoredSources(request, response)
  const question = answeredQuestion(request)
  const verification = await verifySafely(verify, question, response.text, sources)

  return scoreAnswerConfidence(question, sources, verification)
}

/** Prepares and scores an answer in one step. */
export async function withAnswerConfidence(
  request: EngineChatRequest,
  response: EngineChatResponse,
  verify: AnswerVerifier
): Promise<EngineChatResponse> {
  const prepared = prepareAnswer(request, response)
  if (prepared.confidence) {
    return prepared
  }

  return { ...prepared, confidence: await scoreAnswer(request, prepared, verify) }
}
