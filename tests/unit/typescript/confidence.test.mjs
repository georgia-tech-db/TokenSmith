import assert from 'node:assert/strict'
import test from 'node:test'
import { requireTranspiledTs } from './ts-module-loader.mjs'

const {
  abstentionAnswer,
  answerContextAlignment,
  answerStatusLabel,
  classifyAnswerStatus,
  embeddingMatchScore,
  questionContextOverlap,
  retrievalConfidence,
  scoreAnswerConfidence
} = requireTranspiledTs('src/shared/confidence.ts')

const { isAbstention, stripAbstentionHedge, withAnswerConfidence } = requireTranspiledTs(
  'src/main/engine/answer-confidence.ts'
)

const strongSource = {
  title: 'Database Systems.pdf',
  locator: 'Page 4',
  excerpt: 'A transaction preserves atomicity and durability across a database crash.',
  score: 0.7
}
const weakSource = {
  title: 'Database Systems.pdf',
  locator: 'Page 9',
  excerpt: 'Logging records allow recovery after crashes.',
  score: 0.39
}

test('embeddingMatchScore is zero without sources or scores', () => {
  assert.equal(embeddingMatchScore([]), 0)
  assert.equal(embeddingMatchScore([{ title: 'a', locator: 'b', excerpt: 'c' }]), 0)
})

test('embeddingMatchScore separates an on-topic passage from an off-topic one', () => {
  const strong = embeddingMatchScore([strongSource])
  const weak = embeddingMatchScore([weakSource])

  assert.ok(strong > 0.9, `expected an on-topic match, got ${strong}`)
  assert.equal(weak, 0, `expected an off-topic match to floor out, got ${weak}`)
})

test('questionContextOverlap detects a question the material does not cover', () => {
  const covered = questionContextOverlap('What does a transaction preserve?', [strongSource])
  const uncovered = questionContextOverlap('What is the capital of France?', [strongSource])

  assert.ok(covered > 0.5, `expected a covered question, got ${covered}`)
  assert.equal(uncovered, 0, `expected an uncovered question, got ${uncovered}`)
})

test('questionContextOverlap is zero with nothing to compare against', () => {
  assert.equal(questionContextOverlap('Anything at all?', []), 0)
  assert.equal(questionContextOverlap('', [strongSource]), 0)
})

test('retrievalConfidence blends similarity with question overlap', () => {
  const strong = retrievalConfidence('What does a transaction preserve?', [strongSource])
  const weak = retrievalConfidence('What is the capital of France?', [weakSource])

  assert.ok(strong > 0.7, `expected a strong retrieval, got ${strong}`)
  assert.ok(weak < 0.2, `expected a weak retrieval, got ${weak}`)
})

test('answerContextAlignment rewards answers drawn from the passages', () => {
  const grounded = answerContextAlignment('Transactions preserve atomicity and durability.', [strongSource])
  const invented = answerContextAlignment('The capital of France is Paris.', [strongSource])

  assert.ok(grounded > 0.8, `expected a grounded answer, got ${grounded}`)
  assert.ok(invented < 0.2, `expected an ungrounded answer, got ${invented}`)
})

test('answerContextAlignment is zero when there is nothing to align against', () => {
  assert.equal(answerContextAlignment('Anything at all.', []), 0)
  assert.equal(answerContextAlignment('', [strongSource]), 0)
})

test('classifyAnswerStatus covers the three bands', () => {
  assert.equal(classifyAnswerStatus(0.9, 0.9, 0.9), 'supported')
  assert.equal(classifyAnswerStatus(0.5, 0.5, 0.5), 'partial')
  assert.equal(classifyAnswerStatus(0.2, 0.2, 0.2), 'unsupported')
})

test('classifyAnswerStatus withholds support when retrieval alone is weak', () => {
  assert.equal(classifyAnswerStatus(0.7, 0.3, 0.5), 'partial')
})

test('classifyAnswerStatus spares an answer the passages plainly carry', () => {
  // Weak retrieval, but the wording came straight out of the material: this is a
  // question phrased in words the material never uses, not an ungrounded answer.
  assert.equal(classifyAnswerStatus(0.6, 0.2, 1), 'partial')
  assert.equal(classifyAnswerStatus(0.6, 0.2, 0.3), 'unsupported')
})

test('scoreAnswerConfidence grades a grounded answer as supported', () => {
  const confidence = scoreAnswerConfidence(
    'What does a transaction preserve?',
    'Transactions preserve atomicity and durability.',
    [strongSource]
  )

  assert.equal(confidence.status, 'supported')
  assert.equal(confidence.abstained, false)
  assert.ok(confidence.score > 0.62)
})

test('scoreAnswerConfidence grades an off-material answer as unsupported', () => {
  const confidence = scoreAnswerConfidence(
    'What is the capital of France?',
    'The capital of France is Paris.',
    [weakSource]
  )

  assert.equal(confidence.status, 'unsupported')
})

test('isAbstention recognises the abstention sentence regardless of casing', () => {
  assert.equal(isAbstention(abstentionAnswer), true)
  assert.equal(isAbstention(`  ${abstentionAnswer.toUpperCase()}  `), true)
  assert.equal(isAbstention('Transactions preserve atomicity.'), false)
})

test('isAbstention ignores a trailing hedge on a substantive answer', () => {
  const hedged = `Transactions preserve atomicity and durability. However, ${abstentionAnswer}`

  assert.equal(isAbstention(hedged), false)
})

test('stripAbstentionHedge keeps the answer and drops the hedge', () => {
  const hedged = `Transactions preserve atomicity and durability. However, ${abstentionAnswer}`

  assert.equal(stripAbstentionHedge(hedged), 'Transactions preserve atomicity and durability.')
})

test('stripAbstentionHedge leaves a bare abstention alone', () => {
  assert.equal(stripAbstentionHedge(abstentionAnswer), abstentionAnswer)
})

test('withAnswerConfidence grades a hedged answer on what it actually claims', () => {
  const request = { prompt: 'What does a transaction preserve?', retrievedSources: [strongSource] }
  const response = {
    engineId: 'tokensmith',
    modelName: 'llama3.2',
    text: `Transactions preserve atomicity and durability. However, ${abstentionAnswer}`,
    sources: [strongSource]
  }

  const graded = withAnswerConfidence(request, response)

  assert.equal(graded.text, 'Transactions preserve atomicity and durability.')
  assert.equal(graded.confidence.status, 'supported')
  assert.equal(graded.confidence.abstained, false)
})

test('withAnswerConfidence replaces an unsupported answer with the abstention', () => {
  const request = { prompt: 'What is the capital of France?', retrievedSources: [weakSource] }
  const response = {
    engineId: 'tokensmith',
    modelName: 'llama3.2',
    text: 'The capital of France is Paris.',
    sources: [weakSource],
    followUpSuggestions: ['What else?']
  }

  const graded = withAnswerConfidence(request, response)

  assert.equal(graded.text, abstentionAnswer)
  assert.equal(graded.confidence.status, 'unsupported')
  assert.equal(graded.confidence.abstained, true)
  assert.deepEqual(graded.followUpSuggestions, [])
})

test('withAnswerConfidence leaves a supported answer untouched', () => {
  const request = { prompt: 'What does a transaction preserve?', retrievedSources: [strongSource] }
  const response = {
    engineId: 'tokensmith',
    modelName: 'llama3.2',
    text: 'Transactions preserve atomicity and durability.',
    sources: [strongSource],
    followUpSuggestions: ['What is durability?']
  }

  const graded = withAnswerConfidence(request, response)

  assert.equal(graded.text, response.text)
  assert.equal(graded.confidence.status, 'supported')
  assert.equal(graded.confidence.abstained, false)
  assert.deepEqual(graded.followUpSuggestions, ['What is durability?'])
})

test('withAnswerConfidence credits the model for abstaining on its own', () => {
  const request = { prompt: 'What does a transaction preserve?', retrievedSources: [strongSource] }
  const response = {
    engineId: 'tokensmith',
    modelName: 'llama3.2',
    text: abstentionAnswer,
    sources: [strongSource],
    followUpSuggestions: ['What else?']
  }

  const graded = withAnswerConfidence(request, response)

  assert.equal(graded.text, abstentionAnswer)
  assert.equal(graded.confidence.status, 'unsupported')
  assert.equal(graded.confidence.abstained, true)
})

test('withAnswerConfidence falls back to the requested sources when none come back', () => {
  const request = { prompt: 'What does a transaction preserve?', retrievedSources: [strongSource] }
  const response = {
    engineId: 'tokensmith',
    modelName: 'llama3.2',
    text: 'Transactions preserve atomicity and durability.',
    sources: []
  }

  assert.equal(withAnswerConfidence(request, response).confidence.status, 'supported')
})

test('answerStatusLabel reads plainly', () => {
  assert.equal(answerStatusLabel('supported'), 'Supported')
  assert.equal(answerStatusLabel('partial'), 'Partially supported')
  assert.equal(answerStatusLabel('unsupported'), 'Not supported')
})
