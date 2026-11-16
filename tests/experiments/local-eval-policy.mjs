import { createHash } from 'node:crypto'

export function indexFingerprint(pdf, modelDigest, baseUrl, engineFiles) {
  if (!modelDigest) throw new Error('Missing embedding model digest')
  const hash = createHash('sha256').update(pdf).update(modelDigest).update(baseUrl)
  for (const [name, data] of engineFiles) hash.update(name).update(data)
  return hash.digest('hex')
}

export function localChatOptions(contextLength, maxTokens, temperature, seed) {
  if (!Number.isInteger(contextLength) || contextLength < 1024 || !Number.isInteger(maxTokens) || maxTokens < 1 || maxTokens >= contextLength) throw new Error('Invalid context or output token budget')
  return { num_ctx: contextLength, num_predict: maxTokens, temperature, seed }
}

export function gradeDecision(result, threshold) {
  const valid = result && typeof result.score === 'number' && Number.isFinite(result.score) && result.score >= 0 && result.score <= 1 &&
    ['correct', 'incorrect', 'manual_review'].includes(result.verdict) && typeof result.core_claim_satisfied === 'boolean' &&
    Array.isArray(result.contradictions) && result.contradictions.every(s => typeof s === 'string') && typeof result.rationale === 'string'
  if (!valid) return { passed: false, score: 0, verdict: 'manual_review' }
  const passed = result.verdict === 'correct' && result.score >= threshold && result.core_claim_satisfied && result.contradictions.length === 0
  return { passed, score: result.score, verdict: passed ? 'correct' : result.verdict === 'manual_review' ? 'manual_review' : 'incorrect' }
}
