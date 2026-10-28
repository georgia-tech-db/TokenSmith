import assert from 'node:assert/strict'
import test from 'node:test'
import { requireTranspiledTs } from './ts-module-loader.mjs'

const {
  abstentionAnswer,
  answerStatusLabel,
  claimVerdict,
  classifyAnswerStatus,
  embeddingMatchScore,
  gradeClaims,
  overlapWeight,
  questionContextOverlap,
  retrievalConfidence,
  scoreAnswerConfidence,
  scoreDocumentConfidences,
  scoreSourceConfidences,
  sourceContributions
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

/** A claim as the engine's NLI check reports it: one [entailment, contradiction] per passage. */
function claim(text, perPassage) {
  return {
    text,
    evidence: null,
    passages: perPassage.map(([entailment, contradiction], index) => ({ index, entailment, contradiction }))
  }
}

function verified(...claims) {
  return { available: true, claims }
}

const unavailable = { available: false, claims: [], error: 'model missing' }

/** A stand-in for the engine that answers from a fixed table and records its calls. */
function fakeVerifier(verification) {
  const calls = []
  const verify = async (question, answer, passages) => {
    calls.push({ question, answer, passages })
    return verification
  }
  return { verify, calls }
}

function chatResponse(text, sources, extra = {}) {
  return { engineId: 'tokensmith', modelName: 'llama3.2', text, sources, ...extra }
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

test('claimVerdict separates supported, likely, contradicted and unsupported', () => {
  assert.equal(claimVerdict(0.95, 0.01), 'supported')
  assert.equal(claimVerdict(0.82, 0.1), 'likely')
  assert.equal(claimVerdict(0.1, 0.9), 'contradicted')
  assert.equal(claimVerdict(0.1, 0.2), 'unsupported')
  // Negations that reuse the passage's words land around 0.6-0.8 entailment.
  assert.equal(claimVerdict(0.7, 0.1), 'unsupported')
  // Entailment from one passage outweighs a weaker contradiction in another...
  assert.equal(claimVerdict(0.9, 0.6), 'supported')
  // ...but when the contradiction is at least as strong, it wins.
  assert.equal(claimVerdict(0.83, 0.85), 'contradicted')
})

test('gradeClaims takes the strongest reading across passages', () => {
  const [graded] = gradeClaims([claim('Pinned pages are never evicted.', [[0.1, 0.0], [0.92, 0.2]])])

  assert.equal(graded.verdict, 'supported')
  assert.equal(graded.entailment, 0.92)
  assert.equal(graded.contradiction, 0.2)
})

test('classifyAnswerStatus needs every claim entailed to read as supported', () => {
  const supported = gradeClaims([claim('a b c d', [[0.9, 0]]), claim('e f g h', [[0.85, 0]])])
  const oneLikely = gradeClaims([claim('a b c d', [[0.9, 0]]), claim('e f g h', [[0.65, 0]])])

  assert.equal(classifyAnswerStatus(supported), 'supported')
  assert.equal(classifyAnswerStatus(oneLikely), 'partial')
})

test('classifyAnswerStatus withholds an answer with a contradicted claim', () => {
  const graded = gradeClaims([
    claim('Pinning makes a page ineligible for eviction.', [[0.95, 0]]),
    claim('Pinned pages are written to disk immediately.', [[0.05, 0.8]]),
    claim('Unpinning makes a page a candidate for eviction.', [[0.9, 0]])
  ])

  assert.equal(classifyAnswerStatus(graded), 'unsupported')
})

test('classifyAnswerStatus withholds an answer the passages mostly do not carry', () => {
  const graded = gradeClaims([
    claim('Pinning makes a page ineligible for eviction.', [[0.95, 0]]),
    claim('BuzzDB tracks pins in the page header.', [[0.1, 0.1]]),
    claim('Pins are stored as atomic counters.', [[0.2, 0.1]])
  ])

  assert.equal(classifyAnswerStatus(graded), 'unsupported')
})

test('classifyAnswerStatus leaves an unchecked answer unverified', () => {
  assert.equal(classifyAnswerStatus([]), 'unverified')
})

test('the reviewer\'s negation no longer passes on shared words', () => {
  // "does" and "not" used to be dropped as stop words, so this matched the
  // passage word for word. The NLI check reads it as a contradiction.
  const negated = scoreAnswerConfidence('What does a transaction preserve?', [strongSource], verified(
    claim('A transaction does not preserve atomicity and durability.', [[0.0, 1.0]])
  ))

  assert.equal(negated.status, 'unsupported')
  assert.equal(negated.claims[0].verdict, 'contradicted')
})

test('a paraphrase sharing no words with the passage can still be supported', () => {
  const paraphrase = scoreAnswerConfidence('What does a transaction preserve?', [strongSource], verified(
    claim('Committed work is all-or-nothing and survives a crash.', [[0.91, 0.02]])
  ))

  assert.equal(paraphrase.status, 'supported')
})

test('scoreAnswerConfidence reports claims and mean entailment', () => {
  const confidence = scoreAnswerConfidence('What does a transaction preserve?', [strongSource], verified(
    claim('Transactions preserve atomicity.', [[0.9, 0]]),
    claim('Transactions preserve durability.', [[0.7, 0]])
  ))

  assert.equal(confidence.claims.length, 2)
  assert.ok(Math.abs(confidence.alignmentScore - 0.8) < 1e-9)
  assert.equal(confidence.score, confidence.alignmentScore)
  assert.equal(confidence.abstained, false)
})

test('scoreAnswerConfidence never claims support it could not check', () => {
  const confidence = scoreAnswerConfidence('What does a transaction preserve?', [strongSource], unavailable)

  assert.equal(confidence.status, 'unverified')
  assert.equal(confidence.claims, undefined)
  assert.equal(confidence.score, 0)
  // Retrieval is still reported; it just does not decide the status.
  assert.ok(confidence.retrievalScore > 0.7)
})

test('strong retrieval does not rescue a contradicted answer', () => {
  const confidence = scoreAnswerConfidence('What does a transaction preserve?', [strongSource], verified(
    claim('Transactions give up durability after a crash.', [[0.02, 0.97]])
  ))

  assert.ok(confidence.retrievalScore > 0.7)
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

test('withAnswerConfidence checks a hedged answer on what it actually claims', async () => {
  const request = { prompt: 'What does a transaction preserve?', retrievedSources: [strongSource] }
  const response = chatResponse(`Transactions preserve atomicity and durability. However, ${abstentionAnswer}`, [strongSource])
  const { verify, calls } = fakeVerifier(verified(claim('Transactions preserve atomicity and durability.', [[0.95, 0]])))

  const graded = await withAnswerConfidence(request, response, verify)

  assert.equal(calls[0].answer, 'Transactions preserve atomicity and durability.')
  assert.equal(graded.text, 'Transactions preserve atomicity and durability.')
  assert.equal(graded.confidence.status, 'supported')
  assert.equal(graded.confidence.abstained, false)
})

test('withAnswerConfidence checks against the text and question the model was given', async () => {
  const withContext = { ...strongSource, context: 'Full passage: a transaction preserves atomicity and durability.' }
  const request = {
    prompt: 'and durability?',
    answerPrompt: 'Does a transaction preserve durability?',
    retrievedSources: [withContext, weakSource]
  }
  const { verify, calls } = fakeVerifier(verified(claim('Yes, it preserves durability.', [[0.9, 0], [0, 0]])))

  await withAnswerConfidence(request, chatResponse('Yes, it preserves durability.', []), verify)

  assert.equal(calls[0].question, 'Does a transaction preserve durability?')
  assert.deepEqual(calls[0].passages, [withContext.context, weakSource.excerpt])
})

test('withAnswerConfidence shows an unsupported answer with low confidence', async () => {
  const request = { prompt: 'What is the capital of France?', retrievedSources: [weakSource] }
  const response = chatResponse('The capital of France is Paris.', [weakSource], { followUpSuggestions: ['What else?'] })
  const { verify } = fakeVerifier(verified(claim('The capital of France is Paris.', [[0.01, 0.1]])))

  const graded = await withAnswerConfidence(request, response, verify)

  assert.equal(graded.text, 'The capital of France is Paris.')
  assert.equal(graded.confidence.status, 'unsupported')
  assert.equal(graded.confidence.abstained, false)
  assert.ok(graded.confidence.score < 0.1)
  assert.deepEqual(graded.followUpSuggestions, ['What else?'])
})

test('withAnswerConfidence leaves a supported answer untouched', async () => {
  const request = { prompt: 'What does a transaction preserve?', retrievedSources: [strongSource] }
  const response = chatResponse('Transactions preserve atomicity and durability.', [strongSource], {
    followUpSuggestions: ['What is durability?']
  })
  const { verify } = fakeVerifier(verified(claim(response.text, [[0.97, 0]])))

  const graded = await withAnswerConfidence(request, response, verify)

  assert.equal(graded.text, response.text)
  assert.equal(graded.confidence.status, 'supported')
  assert.deepEqual(graded.followUpSuggestions, ['What is durability?'])
})

test('withAnswerConfidence shows the answer unverified when the check cannot run', async () => {
  const request = { prompt: 'What does a transaction preserve?', retrievedSources: [strongSource] }
  const response = chatResponse('Transactions preserve atomicity and durability.', [strongSource])
  const failing = async () => {
    throw new Error('worker crashed')
  }

  for (const verify of [failing, fakeVerifier(unavailable).verify]) {
    const graded = await withAnswerConfidence(request, response, verify)

    assert.equal(graded.text, response.text)
    assert.equal(graded.confidence.status, 'unverified')
    assert.equal(graded.confidence.abstained, false)
  }
})

test('withAnswerConfidence credits the model for abstaining on its own', async () => {
  const request = { prompt: 'What does a transaction preserve?', retrievedSources: [strongSource] }
  const response = chatResponse(abstentionAnswer, [strongSource], { followUpSuggestions: ['What else?'] })
  const { verify, calls } = fakeVerifier(verified())

  const graded = await withAnswerConfidence(request, response, verify)

  assert.equal(calls.length, 0, 'there is nothing to verify in an abstention')
  assert.equal(graded.text, abstentionAnswer)
  assert.equal(graded.confidence.status, 'unsupported')
  assert.equal(graded.confidence.abstained, true)
})

test('withAnswerConfidence falls back to the requested sources when none come back', async () => {
  const request = { prompt: 'What does a transaction preserve?', retrievedSources: [strongSource] }
  const { verify, calls } = fakeVerifier(verified(claim('Transactions preserve atomicity.', [[0.9, 0]])))

  const graded = await withAnswerConfidence(request, chatResponse('Transactions preserve atomicity.', []), verify)

  assert.deepEqual(calls[0].passages, [strongSource.excerpt])
  assert.equal(graded.confidence.status, 'supported')
})

test('withAnswerConfidence does not verify without passages', async () => {
  const { verify, calls } = fakeVerifier(verified())
  const graded = await withAnswerConfidence({ prompt: 'Anything?' }, chatResponse('Some answer here.', []), verify)

  assert.equal(calls.length, 0)
  assert.equal(graded.confidence.status, 'unverified')
})

test('answerStatusLabel reads plainly', () => {
  assert.equal(answerStatusLabel('supported'), 'Supported')
  assert.equal(answerStatusLabel('partial'), 'Partially supported')
  assert.equal(answerStatusLabel('unsupported'), 'Not supported')
  assert.equal(answerStatusLabel('unverified'), 'Not verified')
})

test('sourceContributions credits the passage that entails each claim', () => {
  const shares = sourceContributions(
    [
      claim('Transactions preserve atomicity.', [[0.9, 0], [0.1, 0]]),
      claim('Transactions preserve durability.', [[0.9, 0], [0.85, 0]])
    ],
    2
  )

  assert.deepEqual(shares, [0.75, 0.25])
})

test('sourceContributions is all zero when no passage entails the answer', () => {
  assert.deepEqual(sourceContributions([claim('Paris is in France.', [[0.1, 0], [0.2, 0]])], 2), [0, 0])
  assert.deepEqual(sourceContributions([], 0), [])
})

test('scoreSourceConfidences marks a single leading passage', () => {
  const scored = scoreSourceConfidences(
    'What does a transaction preserve?',
    [claim('A transaction preserves atomicity and durability.', [[0.05, 0], [0.95, 0]])],
    [weakSource, strongSource]
  )

  assert.deepEqual(scored.map((source) => source.leading), [false, true])
  assert.deepEqual(scored.map((source) => source.alignmentScore), [0, 1])
  assert.ok(scored[1].retrievalScore > scored[0].retrievalScore)
})

test('scoreSourceConfidences leads nothing when no passage carries the answer', () => {
  const scored = scoreSourceConfidences(
    'What is the capital of France?',
    [claim('The capital of France is Paris.', [[0.01, 0], [0.02, 0]])],
    [strongSource, weakSource]
  )

  assert.ok(scored.every((source) => !source.leading && source.contribution === 0))
})

test('scoreDocumentConfidences sums passage shares onto their document', () => {
  const sources = [
    { ...strongSource, documentTitle: 'Database Systems.pdf', locator: 'Page 4' },
    { ...strongSource, documentTitle: 'Database Systems.pdf', locator: 'Page 5' },
    { ...weakSource, documentTitle: 'Lecture Notes.pdf', locator: 'Page 9' }
  ]
  const claims = [
    claim('Transactions preserve atomicity.', [[0.9, 0], [0.1, 0], [0.1, 0]]),
    claim('Transactions preserve durability.', [[0.1, 0], [0.9, 0], [0.1, 0]]),
    claim('Logs allow recovery.', [[0.1, 0], [0.1, 0], [0.9, 0]])
  ]
  const documents = scoreDocumentConfidences(sources, scoreSourceConfidences('What does a transaction preserve?', claims, sources))

  assert.deepEqual(documents.map((document) => [document.title, document.passages, document.leading]), [
    ['Database Systems.pdf', 2, true],
    ['Lecture Notes.pdf', 1, false]
  ])
  assert.ok(Math.abs(documents[0].contribution - 2 / 3) < 1e-9)
})

test('scoreDocumentConfidences falls back to the passage title and leads nothing when unused', () => {
  const sources = [strongSource, weakSource]
  const documents = scoreDocumentConfidences(
    sources,
    scoreSourceConfidences('What is the capital of France?', [claim('Paris.', [[0, 0], [0, 0]])], sources)
  )

  assert.equal(documents.length, 1)
  assert.equal(documents[0].title, 'Database Systems.pdf')
  assert.equal(documents[0].passages, 2)
  assert.equal(documents[0].leading, false)
})

test('overlapWeight scales with the evidence the question offers', () => {
  assert.equal(overlapWeight(''), 0)
  assert.ok(
    overlapWeight('tell me abt hwk') <
      overlapWeight('what homework assignments are due each friday this semester')
  )
  // Caps once the question carries enough content words to trust the overlap.
  assert.equal(overlapWeight('weekly homework assignments grading policy and submission deadlines'), 0.5)
})
