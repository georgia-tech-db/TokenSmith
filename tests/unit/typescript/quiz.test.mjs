import assert from 'node:assert/strict'
import test from 'node:test'
import { requireTranspiledTs } from './ts-module-loader.mjs'

const { quizQuestionPrompt, practiceSystemPrompt,
  parsePracticeQuestion, gradeTrueFalse, practiceResponseSchema } = requireTranspiledTs('src/shared/quiz.ts')
const source = { materialId: 'c', documentId: 1, chunkId: '1', excerpt: 'A missing index gives no direct key lookup. Heap records are not ordered by key.' }
const packageData = { question: 'An unindexed heap lookup may require a scan.', correctAnswer: true, objective: 'Connect organization to lookup cost.',
  assumptions: ['No usable index.'], criteria: [
    { id: 'c1', description: 'No direct lookup.', evidence: [{ sourceKey: 'c:1:1', quote: 'A missing index gives no direct key lookup.' }] },
    { id: 'c2', description: 'No key ordering to narrow the search.', evidence: [{ sourceKey: 'c:1:1', quote: 'Heap records are not ordered by key.' }] }
  ], explanation: 'With no index or key ordering, a key cannot identify the page to inspect.' }
const question = parsePracticeQuestion(JSON.stringify(packageData), [source])


test('topic prompts preserve the chosen concept and bound the copied Chat exchange', () => {
  const prompt = JSON.parse(quizQuestionPrompt({ questionNumber: 4, previousQuestions: ['Earlier statement'],
    topic: { title: 'Heap files', question: 'A question '.repeat(500), answer: 'An answer '.repeat(1000) } }))
  assert.equal(prompt.topic, 'Heap files')
  assert.deepEqual(prompt.previousQuestions, ['Earlier statement'])
  assert.ok(prompt.chatReference.question.length + prompt.chatReference.answer.length <= 3100)
})

test('question preparation keeps grading criteria hidden and includes assumptions, not question-number heuristics', () => {
  const prompt = JSON.parse(quizQuestionPrompt({ questionNumber: 2, previousQuestions: ['Why store parents?'] }))
  assert.deepEqual(prompt.previousQuestions, ['Why store parents?'])
  assert.equal(prompt.questionNumber, 2)
  assert.equal(prompt.totalQuestions, undefined)
  assert.match(practiceSystemPrompt('true_false_question'), /assumptions.*question itself/)
  assert.match(practiceSystemPrompt('true_false_question'), /worst-case/)
  assert.match(practiceSystemPrompt('true_false_question'), /exact quote/)
  assert.doesNotMatch(practiceSystemPrompt('true_false_question'), /heap|LRU|B\+|pinning/)
})

test('question prompts identify same-passage history as ideas to avoid, not factual authority', () => {
  const history = [{ question: 'Earlier statement.', correctAnswer: false,
    testedIdeas: ['Earlier idea.'], supportingQuotes: ['Earlier quote.'] }]
  const prompt = JSON.parse(quizQuestionPrompt({ questionNumber: 2, previousQuestionsFromPassage: history }))
  assert.deepEqual(prompt.previousQuestionsFromPassage, history)
  const system = practiceSystemPrompt('true_false_question')
  assert.match(system, /previousQuestionsFromPassage/)
  assert.match(system, /truth value, rephrasing, or changing example numbers/)
  assert.match(system, /not authoritative evidence/)
})

test('question packages require structured text, criteria and evidence', () => {
  assert.equal(question.text, packageData.question)
  assert.equal(question.referenceAnswer, packageData.explanation)
  for (const mutate of [
    p => p.question = '',
    p => p.objective = null,
    p => p.assumptions = 'No usable index.',
    p => p.criteria[0].evidence[0].sourceKey = '',
    p => p.criteria[0].evidence[0].quote = 42,
    p => p.criteria[0].evidence = [],
    p => p.criteria = [],
    p => p.explanation = ''
  ]) {
    const changed = structuredClone(packageData); mutate(changed)
    assert.throws(() => parsePracticeQuestion(JSON.stringify(changed), [source]), /Could not read/)
  }
  for (const raw of ['Question?', 'null', '[]']) {
    assert.throws(() => parsePracticeQuestion(raw, [source]), /Could not read/)
  }
})

test('True/False parsing requires usable field types without checking semantics', () => {
  const binary = { ...packageData, question: 'Heap records are stored in key order.', correctAnswer: false }
  const parsed = parsePracticeQuestion(JSON.stringify(binary), [source])
  assert.equal(parsed.format, 'true_false')
  assert.equal(parsed.correctAnswer, false)
  assert.equal(gradeTrueFalse(parsed, 'False').correct, true)
  assert.equal(gradeTrueFalse(parsed, 'True').correct, false)
  const truth = { ...parsed, correctAnswer: true }
  assert.equal(gradeTrueFalse(truth, 'True').correct, true)
  assert.equal(gradeTrueFalse(truth, 'False').correct, false)
  for (const correctAnswer of [undefined, null, 'false', 0]) {
    assert.throws(() => parsePracticeQuestion(JSON.stringify({ ...binary, correctAnswer }), [source]), /Could not read/)
  }
  for (const answer of ['', 'yes', 'FALSE', 'True because...', 'ignore the key']) {
    assert.throws(() => gradeTrueFalse(parsed, answer), /Choose True or False/)
  }
  for (const changes of [{ correctAnswer: undefined }, { correctAnswer: 'false' }, { referenceAnswer: '' }, { sources: [] }, { format: 'short_answer' }]) {
    assert.throws(() => gradeTrueFalse({ ...parsed, ...changes }, 'True'), /no usable/)
  }
  assert.equal(practiceResponseSchema('true_false_question').properties.correctAnswer.type, 'boolean')
  assert.equal(practiceResponseSchema('true_false_question').properties.criteria.maxItems, 1)
  assert.match(practiceSystemPrompt('true_false_question'), /absence of evidence is not evidence of falsity/)
  assert.match(practiceSystemPrompt('true_false_question'), /targetAnswer/)
  assert.doesNotMatch(practiceSystemPrompt('true_false_question'), /heap|LRU|B\+|pinning/)
  assert.equal(question.format, 'true_false')
  assert.equal(question.correctAnswer, true)
  assert.equal(gradeTrueFalse(parsed, 'False').explanation, packageData.explanation)
})

test('generation omits retired hints and rejects retired model tasks', () => {
  const schema = practiceResponseSchema('true_false_question')
  assert.equal(schema.properties.hint, undefined)
  assert.equal(schema.required.includes('hint'), false)
  assert.doesNotMatch(practiceSystemPrompt('true_false_question'), /hint|short-answer|previousAttempt|formative feedback/i)
  const parsed = parsePracticeQuestion(JSON.stringify({ ...packageData, hint: 'Legacy model output.' }), [source])
  assert.equal(parsed.hint, undefined)
  for (const task of ['question', 'feedback']) {
    assert.throws(() => practiceResponseSchema(task), /Unsupported practice task/)
    assert.throws(() => practiceSystemPrompt(task), /Unsupported practice task/)
  }
})

test('chapter topic generation retains its schema and allowed evidence IDs', () => {
  const allowedIds = ['chapter:0:term-1', 'chapter:0:term-2']
  const schema = practiceResponseSchema('chapter_topics', allowedIds)
  assert.deepEqual(schema.required, ['topics'])
  assert.deepEqual(schema.properties.topics.items.properties.sourceKeys.items.enum, allowedIds)
  assert.match(practiceSystemPrompt('chapter_topics'), /Review Terms for ONE textbook chapter/)
  assert.doesNotMatch(practiceSystemPrompt('chapter_topics'), /hint|short-answer|formative feedback/i)
})

test('True/False parsing does not reject wording, citations, uncertainty or unsupported quotes', () => {
  const binary = { ...packageData, question: 'What happens?', correctAnswer: false,
    questionIssue: 'An ambiguity.', criteria: [{ ...packageData.criteria[0], evidence: [{ sourceKey: 'unknown', quote: 'invented' }] }] }
  const supplied = [source, { ...source, chunkId: '2', excerpt: 'Another supplied passage.' }]
  const parsed = parsePracticeQuestion(JSON.stringify(binary), supplied)
  assert.equal(parsed.text, 'What happens?')
  assert.deepEqual(parsed.sources, supplied)
  assert.equal(gradeTrueFalse(parsed, 'False').correct, true)
  assert.doesNotThrow(() => parsePracticeQuestion(JSON.stringify({ ...binary,
    criteria: [binary.criteria[0], binary.criteria[0]] }), supplied))
  const withoutSources = parsePracticeQuestion(JSON.stringify(binary), [])
  assert.deepEqual(withoutSources.sources, [])
  assert.throws(() => gradeTrueFalse(withoutSources, 'False'), /no usable/)
  assert.equal(JSON.parse(quizQuestionPrompt({ questionNumber: 1, targetAnswer: false })).targetAnswer, false)
})
