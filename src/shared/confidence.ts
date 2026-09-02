import type { ChatSource } from './app-state'

export type AnswerStatus = 'supported' | 'partial' | 'unsupported'

export interface AnswerConfidence {
  /** Combined grounding confidence in [0, 1]. */
  score: number
  status: AnswerStatus
  /** How well the retrieved passages match the question, in [0, 1]. */
  retrievalScore: number
  /** How much of the answer is carried by those passages, in [0, 1]. */
  alignmentScore: number
  /** True when the answer text was replaced by the abstention message. */
  abstained: boolean
}

export const abstentionAnswer =
  "I don't have enough information in the indexed material to answer that confidently."

/**
 * Chunk vectors are L2-normalized and searched with a FAISS inner-product index,
 * so a source score is a cosine similarity.
 *
 * These bounds are calibrated against nomic-embed-text, the embedder the app
 * recommends. That model has a high similarity floor: passages from an unrelated
 * subject still score around 0.35-0.48, while on-topic passages run 0.50-0.70.
 * Raw cosine alone therefore separates the two poorly, which is why retrieval
 * confidence blends it with the lexical overlap below. Retune `floor` and
 * `ceiling` for a different embedding model.
 */
export const retrievalCalibration = {
  floor: 0.45,
  ceiling: 0.7,
  topWeight: 0.6
}

export const confidenceThresholds = {
  /** At or above this combined score the answer reads as supported. */
  supported: 0.62,
  /** Below this combined score the answer is treated as unsupported. */
  partial: 0.35,
  /** Support also requires the passages themselves to be a real match. */
  minRetrievalForSupport: 0.45,
  /** Retrieval this weak means nothing relevant was found, whatever the model said. */
  maxRetrievalForAbstention: 0.25,
  /**
   * An answer whose wording is this closely carried by the passages came out of
   * the material, so it is never abstained over a weak retrieval score. This is
   * what saves a well-grounded answer to a question phrased in words the
   * material never uses.
   */
  strongAlignment: 0.8
}

const stopWords = new Set([
  'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'any', 'can', 'has', 'had', 'have',
  'was', 'were', 'this', 'that', 'these', 'those', 'with', 'from', 'they', 'them', 'their',
  'there', 'then', 'than', 'when', 'what', 'which', 'while', 'will', 'would', 'could', 'should',
  'into', 'over', 'such', 'some', 'more', 'most', 'also', 'been', 'being', 'does', 'did',
  'its', 'his', 'her', 'our', 'your', 'about', 'above', 'after', 'before', 'because',
  'how', 'why', 'who', 'whom', 'whose', 'where', 'each', 'other', 'only', 'same', 'both',
  'use', 'used', 'using', 'may', 'might', 'must', 'one', 'two', 'get', 'gets', 'let'
])

function clamp01(value: number): number {
  if (!Number.isFinite(value)) {
    return 0
  }

  return value < 0 ? 0 : value > 1 ? 1 : value
}

function rescale(value: number, floor: number, ceiling: number): number {
  if (ceiling <= floor) {
    return 0
  }

  return clamp01((value - floor) / (ceiling - floor))
}

/**
 * Folds the common English inflections onto one key so an answer saying
 * "transactions preserve" still matches a passage saying "transaction preserves".
 * Deliberately crude: this only has to make overlap counting fair.
 */
function wordKey(word: string): string {
  let key = word

  if (key.endsWith('ies') && key.length > 4) {
    key = `${key.slice(0, -3)}y`
  } else if (key.endsWith('es') && key.length > 4) {
    key = key.slice(0, -2)
  } else if (key.endsWith('s') && !key.endsWith('ss') && key.length > 3) {
    key = key.slice(0, -1)
  } else if (key.endsWith('ing') && key.length > 5) {
    key = key.slice(0, -3)
  } else if (key.endsWith('ed') && key.length > 4) {
    key = key.slice(0, -2)
  }

  return key.endsWith('e') ? key.slice(0, -1) : key
}

function contentWords(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s]+/gu, ' ')
    .split(/\s+/)
    .filter((word) => word.length > 2 && !stopWords.has(word))
    .map(wordKey)
}

/**
 * Scores the retrieved passages on embedding similarity alone, weighted toward
 * the single best passage.
 */
export function embeddingMatchScore(sources: ChatSource[]): number {
  const scores = sources
    .map((source) => source.score)
    .filter((score): score is number => typeof score === 'number' && Number.isFinite(score))

  if (scores.length === 0) {
    return 0
  }

  const ordered = [...scores].sort((left, right) => right - left)
  const top = ordered[0]
  const considered = ordered.slice(0, 3)
  const mean = considered.reduce((total, score) => total + score, 0) / considered.length
  const { floor, ceiling, topWeight } = retrievalCalibration

  return clamp01(rescale(top, floor, ceiling) * topWeight + rescale(mean, floor, ceiling) * (1 - topWeight))
}

/**
 * Scores how much of the question the material actually talks about. This is
 * what catches a question the library cannot answer at all: ask a scheduling
 * textbook about the capital of France and none of the question's content words
 * appear anywhere in what came back.
 */
export function questionContextOverlap(question: string, sources: ChatSource[]): number {
  const questionWords = new Set(contentWords(question))
  if (questionWords.size === 0 || sources.length === 0) {
    return 0
  }

  const contextWords = new Set(contentWords(sources.map((source) => source.excerpt ?? '').join(' ')))
  if (contextWords.size === 0) {
    return 0
  }

  let matched = 0
  for (const word of questionWords) {
    if (contextWords.has(word)) {
      matched += 1
    }
  }

  return clamp01(matched / questionWords.size)
}

/**
 * Scores how well the retrieved passages match the question. Needs no model and
 * no extra dependency: it blends embedding similarity with lexical overlap,
 * because neither signal is reliable enough alone.
 */
export function retrievalConfidence(question: string, sources: ChatSource[]): number {
  return clamp01(embeddingMatchScore(sources) * 0.5 + questionContextOverlap(question, sources) * 0.5)
}

/**
 * Scores how much of the answer is actually carried by the retrieved passages,
 * as the share of the answer's content words that appear in them. This is the
 * lexical stand-in for NLI entailment: it catches an answer the model produced
 * from its own weights rather than from the material.
 */
export function answerContextAlignment(answer: string, sources: ChatSource[]): number {
  const answerWords = new Set(contentWords(answer))
  if (answerWords.size === 0 || sources.length === 0) {
    return 0
  }

  const contextWords = new Set(contentWords(sources.map((source) => source.excerpt ?? '').join(' ')))
  if (contextWords.size === 0) {
    return 0
  }

  let matched = 0
  for (const word of answerWords) {
    if (contextWords.has(word)) {
      matched += 1
    }
  }

  return clamp01(matched / answerWords.size)
}

export function classifyAnswerStatus(
  score: number,
  retrievalScore: number,
  alignmentScore: number
): AnswerStatus {
  const carriedByContext = alignmentScore >= confidenceThresholds.strongAlignment
  const tooWeakToStand =
    score < confidenceThresholds.partial || retrievalScore < confidenceThresholds.maxRetrievalForAbstention

  if (!carriedByContext && tooWeakToStand) {
    return 'unsupported'
  }

  if (score >= confidenceThresholds.supported && retrievalScore >= confidenceThresholds.minRetrievalForSupport) {
    return 'supported'
  }

  return 'partial'
}

/**
 * Grades one answer against the passages it was given. Combines the two halves
 * evenly: a confident answer needs both a real retrieval match and text that
 * stays inside the material.
 */
export function scoreAnswerConfidence(
  question: string,
  answer: string,
  sources: ChatSource[]
): AnswerConfidence {
  const retrievalScore = retrievalConfidence(question, sources)
  const alignmentScore = answerContextAlignment(answer, sources)
  const score = clamp01(retrievalScore * 0.5 + alignmentScore * 0.5)
  const status = classifyAnswerStatus(score, retrievalScore, alignmentScore)

  return {
    score,
    status,
    retrievalScore,
    alignmentScore,
    abstained: false
  }
}

export function answerStatusLabel(status: AnswerStatus): string {
  if (status === 'supported') {
    return 'Supported'
  }

  return status === 'partial' ? 'Partially supported' : 'Not supported'
}
