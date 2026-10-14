import type { ChatSource } from './app-state'

export type AnswerStatus = 'supported' | 'partial' | 'unsupported' | 'unverified'

/** One claim graded against one passage by the NLI model, each in [0, 1]. */
export interface PassageEntailment {
  index: number
  entailment: number
  contradiction: number
}

/** The NLI model's reading of one sentence-level claim in the answer. */
export interface ClaimCheck {
  text: string
  /** The passage span that entails the claim best. */
  evidence?: string | null
  /** Names or numbers the claim uses that no passage contains. */
  missingTerms?: string[]
  passages: PassageEntailment[]
}

/** What the engine's `verify_answer` command returns. */
export interface AnswerVerification {
  /** False when the NLI model could not run; nothing was checked. */
  available: boolean
  claims: ClaimCheck[]
  error?: string
}

export type ClaimVerdict = 'supported' | 'likely' | 'unsupported' | 'contradicted'

export interface ClaimConfidence {
  text: string
  verdict: ClaimVerdict
  /** Strongest entailment any passage gives this claim, in [0, 1]. */
  entailment: number
  /** Strongest contradiction any passage gives this claim, in [0, 1]. */
  contradiction: number
  evidence?: string | null
  missingTerms?: string[]
  /** The source that entails this claim best, when one entails it at all. */
  sourceIndex?: number
}

/**
 * One retrieved passage's own grades. The combined `AnswerConfidence` says
 * whether to trust the answer; this says which passage earned that trust, so a
 * student can go read the one that actually carried it.
 */
export interface SourceConfidence {
  /** Index of the passage in the answer's source list. */
  index: number
  /** How well this passage alone matches the question, in [0, 1]. */
  retrievalScore: number
  /** Share of the answer's claims this passage entails, in [0, 1]. */
  alignmentScore: number
  /**
   * This passage's share of the grounded answer, in [0, 1]. Shares across the
   * source list sum to 1 whenever any passage carries the answer at all, so
   * they read directly as "how much of this answer came from here".
   */
  contribution: number
  /** True for the single largest contributor, when one carries the answer. */
  leading: boolean
}

/**
 * One source document's share, summed over every passage retrieved from it.
 * Retrieval works in passages, but a student asking "where did this come from"
 * means the document, so the passage-level split is rolled up before display.
 */
export interface DocumentConfidence {
  /** Document title, as the passages report it. */
  title: string
  /** This document's share of the grounded answer, in [0, 1]. */
  contribution: number
  /** Best question match among this document's passages, in [0, 1]. */
  retrievalScore: number
  /** How many retrieved passages came from this document. */
  passages: number
  /** True for the single largest contributing document. */
  leading: boolean
}

export interface AnswerConfidence {
  /** Grounding confidence in [0, 1]: how strongly the passages entail the answer's claims. */
  score: number
  status: AnswerStatus
  /** How well the retrieved passages match the question, in [0, 1]. Topical only. */
  retrievalScore: number
  /** Mean entailment of the answer's claims by the passages, in [0, 1]. */
  alignmentScore: number
  /** Per-claim NLI verdicts. Absent when verification did not run. */
  claims?: ClaimConfidence[]
  /** True when the model itself declined to answer. */
  abstained: boolean
  /** Per-passage breakdown, in the order the passages were retrieved. */
  sources?: SourceConfidence[]
  /** Per-document rollup of that breakdown, largest share first. */
  documents?: DocumentConfidence[]
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
  /**
   * Content words a question needs before lexical overlap carries its full
   * weight in the retrieval score. "Tell me abt hwk" has three, so a single
   * unmatched word would move the overlap by a third.
   */
  overlapEvidenceWords: 6
}

/**
 * Cut-offs on the NLI model's probabilities. Swept over the 35 questions (210
 * answers) in tests/benchmarks/buzzdb_nli_cases.json with
 * tests/benchmarks/sweep_nli_thresholds.py: picked on the "dev" questions,
 * then checked on the "heldout" ones. Run `npm run test:nli-benchmark` before
 * and after changing them.
 */
export const verificationThresholds = {
  /** A claim reads as supported only at or above this entailment. */
  supported: 0.85,
  /**
   * Below this entailment a claim counts as not carried by the passages.
   * Negated answers routinely score 0.6-0.8 because they reuse the passage's
   * words, so a lower bar lets them through.
   */
  entailed: 0.8,
  /**
   * A claim contradicted at least this strongly is contradicted, unless a
   * passage entails it more strongly still.
   */
  contradicted: 0.5,
  /**
   * Below this share of entailed claims the answer reads as not supported.
   * With the stricter `entailed` bar, half is enough: one sound claim out of
   * two no longer sinks a correct two-sentence paraphrase.
   */
  minEntailedShare: 0.5
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
 * How far to trust lexical overlap for this question, in [0, 0.5].
 *
 * Overlap is a share of the question's content words, so a short question
 * measures it from almost no evidence: "tell me abt hwk" yields three words,
 * none of which a textbook contains, and the resulting zero says more about the
 * student's shorthand than about the material. Scaling the weight with the
 * number of words keeps a confident signal confident and lets a thin one fall
 * back on embedding similarity, which does not care how the question was typed.
 */
export function overlapWeight(question: string): number {
  const words = contentWords(question).length
  if (words === 0) {
    return 0
  }

  return clamp01(words / confidenceThresholds.overlapEvidenceWords) * 0.5
}

/**
 * Scores how well the retrieved passages match the question. Needs no model and
 * no extra dependency: it blends embedding similarity with lexical overlap,
 * because neither signal is reliable enough alone.
 */
export function retrievalConfidence(question: string, sources: ChatSource[]): number {
  const weight = overlapWeight(question)

  return clamp01(
    embeddingMatchScore(sources) * (1 - weight) + questionContextOverlap(question, sources) * weight
  )
}

function maxOver(passages: PassageEntailment[], key: 'entailment' | 'contradiction'): number {
  return passages.reduce((best, passage) => Math.max(best, clamp01(passage[key])), 0)
}

export function claimVerdict(entailment: number, contradiction: number): ClaimVerdict {
  const { supported, entailed, contradicted } = verificationThresholds

  // When one passage contradicts the claim at least as strongly as another
  // entails it, the passages disagree with each other about it and the
  // contradiction wins: "neighboring keys are stored next to each other"
  // scores 0.83 entailment and 0.85 contradiction.
  if (contradiction >= contradicted && contradiction >= entailment) {
    return 'contradicted'
  }
  if (entailment >= supported) {
    return 'supported'
  }
  if (entailment >= entailed) {
    return 'likely'
  }

  return contradiction >= contradicted ? 'contradicted' : 'unsupported'
}

export function gradeClaims(claims: ClaimCheck[]): ClaimConfidence[] {
  return claims.map((claim) => {
    const entailment = maxOver(claim.passages, 'entailment')
    const contradiction = maxOver(claim.passages, 'contradiction')
    const verdict = claimVerdict(entailment, contradiction)
    const best = claim.passages.find((passage) => passage.entailment === entailment)

    return {
      text: claim.text,
      verdict,
      entailment,
      contradiction,
      evidence: claim.evidence ?? null,
      ...(claim.missingTerms?.length ? { missingTerms: claim.missingTerms } : {}),
      ...((verdict === 'supported' || verdict === 'likely') && best ? { sourceIndex: best.index } : {})
    }
  })
}

/**
 * Splits credit for the answer across the passages that entail it.
 *
 * Each entailed claim is worth the same, and its credit is divided evenly
 * among every passage that entails it. Shares sum to 1 across the list, so a
 * passage's number reads as its share of the grounded answer.
 */
export function sourceContributions(claims: ClaimCheck[], sourceCount: number): number[] {
  const credit = Array.from({ length: sourceCount }, () => 0)
  let total = 0

  for (const claim of claims) {
    const holders = claim.passages
      .filter((passage) => passage.index >= 0 && passage.index < sourceCount)
      .filter((passage) => passage.entailment >= verificationThresholds.entailed)
      .map((passage) => passage.index)

    if (holders.length === 0) {
      continue
    }

    for (const index of holders) {
      credit[index] += 1 / holders.length
    }
    total += 1
  }

  return total === 0 ? credit : credit.map((value) => clamp01(value / total))
}

/** Share of the claims that passage `index` entails, in [0, 1]. */
function passageAlignment(claims: ClaimCheck[], index: number): number {
  if (claims.length === 0) {
    return 0
  }

  const entailed = claims.filter((claim) =>
    claim.passages.some((passage) => passage.index === index && passage.entailment >= verificationThresholds.entailed)
  )

  return entailed.length / claims.length
}

/**
 * Decides the answer's status from its claims alone. Retrieval similarity is
 * deliberately left out: it measures topic, and an answer that contradicts a
 * passage is as on-topic as one that repeats it.
 */
export function classifyAnswerStatus(claims: ClaimConfidence[]): AnswerStatus {
  if (claims.length === 0) {
    return 'unverified'
  }

  const entailed = claims.filter((claim) => claim.verdict === 'supported' || claim.verdict === 'likely')
  const contradicted = claims.some((claim) => claim.verdict === 'contradicted')

  if (contradicted || entailed.length / claims.length < verificationThresholds.minEntailedShare) {
    return 'unsupported'
  }

  return claims.every((claim) => claim.verdict === 'supported') ? 'supported' : 'partial'
}

/**
 * Grades each retrieved passage on its own, then splits credit for the answer
 * between them.
 */
export function scoreSourceConfidences(
  question: string,
  claims: ClaimCheck[],
  sources: ChatSource[]
): SourceConfidence[] {
  const contributions = sourceContributions(claims, sources.length)
  let leadingIndex = -1

  contributions.forEach((contribution, index) => {
    if (contribution > 0 && (leadingIndex === -1 || contribution > contributions[leadingIndex])) {
      leadingIndex = index
    }
  })

  return sources.map((source, index) => ({
    index,
    retrievalScore: retrievalConfidence(question, [source]),
    alignmentScore: passageAlignment(claims, index),
    contribution: contributions[index] ?? 0,
    leading: index === leadingIndex
  }))
}

/**
 * The title a passage belongs under. `documentTitle` is the whole document;
 * `title` is the per-passage label, which for a PDF is the same document name,
 * so it is the right fallback.
 */
export function sourceDocumentKey(source: ChatSource): string {
  const title = (source.documentTitle ?? source.title ?? '').trim()

  return title || 'Untitled source'
}

/**
 * Rolls the passage-level split up to the documents the passages came from.
 * Shares add, because a document's contribution is everything its passages
 * carried; question match takes the best passage, because one strong passage
 * is what makes a document worth opening.
 */
export function scoreDocumentConfidences(sources: ChatSource[], scored: SourceConfidence[]): DocumentConfidence[] {
  const byTitle = new Map<string, DocumentConfidence>()

  sources.forEach((source, index) => {
    const title = sourceDocumentKey(source)
    const passage = scored[index]
    if (!passage) {
      return
    }

    const current = byTitle.get(title)
    if (current) {
      current.contribution = clamp01(current.contribution + passage.contribution)
      current.retrievalScore = Math.max(current.retrievalScore, passage.retrievalScore)
      current.passages += 1
      return
    }

    byTitle.set(title, {
      title,
      contribution: passage.contribution,
      retrievalScore: passage.retrievalScore,
      passages: 1,
      leading: false
    })
  })

  const documents = [...byTitle.values()].sort((left, right) => right.contribution - left.contribution)

  if (documents.length > 0 && documents[0].contribution > 0) {
    documents[0].leading = true
  }

  return documents
}

/**
 * Grades one answer from the NLI model's verdicts on its claims. When the
 * verification did not run, the answer is marked unverified rather than
 * guessed at: no label claims support that was never checked.
 */
export function scoreAnswerConfidence(
  question: string,
  sources: ChatSource[],
  verification: AnswerVerification
): AnswerConfidence {
  const retrievalScore = retrievalConfidence(question, sources)
  const checked = verification.available ? verification.claims : []
  const claims = gradeClaims(checked)
  const alignmentScore = claims.length
    ? claims.reduce((total, claim) => total + claim.entailment, 0) / claims.length
    : 0
  const scoredSources = scoreSourceConfidences(question, checked, sources)

  return {
    score: clamp01(alignmentScore),
    status: classifyAnswerStatus(claims),
    retrievalScore,
    alignmentScore,
    abstained: false,
    ...(verification.available ? { claims } : {}),
    sources: scoredSources,
    documents: scoreDocumentConfidences(sources, scoredSources)
  }
}

export function answerStatusLabel(status: AnswerStatus): string {
  if (status === 'supported') {
    return 'Supported'
  }
  if (status === 'unverified') {
    return 'Not verified'
  }

  return status === 'partial' ? 'Partially supported' : 'Not supported'
}
