import assert from 'node:assert/strict'
import test from 'node:test'
import { requireTranspiledTs } from './ts-module-loader.mjs'

const { quizFeedbackPrompt, quizQuestionPrompt, practiceSystemPrompt,
  parsePracticeQuestion, checkPracticeQuestionAnswer, parsePracticeFeedback, gradeTrueFalse, practiceResponseSchema } = requireTranspiledTs('src/shared/quiz.ts')
const source = { materialId: 'c', documentId: 1, chunkId: '1', excerpt: 'A missing index gives no direct key lookup. Heap records are not ordered by key.' }
const packageData = { question: 'Why might searching an unindexed heap require a scan?', objective: 'Connect organization to lookup cost.',
  assumptions: ['No usable index.'], criteria: [
    { id: 'c1', description: 'No direct lookup.', evidence: [{ sourceKey: 'c:1:1', quote: 'A missing index gives no direct key lookup.' }] },
    { id: 'c2', description: 'No key ordering to narrow the search.', evidence: [{ sourceKey: 'c:1:1', quote: 'Heap records are not ordered by key.' }] }
  ], explanation: 'With no index or key ordering, a key cannot identify the page to inspect.', hint: 'Consider how keys relate to page locations.' }
const question = parsePracticeQuestion(JSON.stringify(packageData), [source])
const feedback = { questionAssessment: 'The question states the necessary unindexed assumption.', checks: [
  { criterionId: 'c1', status: 'met', feedback: 'You identified the missing direct lookup.' },
  { criterionId: 'c2', status: 'missing', feedback: 'You have not connected the file organization to the search.' }
], improvement: '', nextStep: 'Explain whether the organization narrows the search to a page.', questionIssue: '' }


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
  assert.match(practiceSystemPrompt('question'), /assumptions.*question itself/)
  assert.match(practiceSystemPrompt('question'), /worst-case/)
  assert.match(practiceSystemPrompt('question'), /exact quote/)
  assert.doesNotMatch(practiceSystemPrompt('question'), /heap|LRU|B\+|pinning/)
})

test('question packages require evidence in the actual packed sources, not model-invented citations', () => {
  assert.equal(question.text, packageData.question)
  assert.equal(question.referenceAnswer, packageData.explanation)
  for (const mutate of [
    p => p.criteria[0].evidence[0].sourceKey = 'another-document',
    p => p.criteria[0].evidence[0].quote = 'A quotation that is not present.',
    p => p.criteria[0].evidence = [],
    p => p.criteria[1].id = 'c1',
    p => p.criteria = [],
    p => p.explanation = ''
  ]) {
    const changed = structuredClone(packageData); mutate(changed)
    assert.throws(() => parsePracticeQuestion(JSON.stringify(changed), [source]), /valid question/)
  }
  assert.throws(() => parsePracticeQuestion(JSON.stringify(packageData), []), /supporting passages/)
  assert.throws(() => parsePracticeQuestion('Question?', [source]), /valid question/)
  const short = { ...source, excerpt: 'The relationship is F = ma.' }
  assert.doesNotThrow(() => parsePracticeQuestion(JSON.stringify({ ...packageData,
    criteria: [{ ...packageData.criteria[0], evidence: [{ sourceKey: 'c:1:1', quote: 'F = ma' }] }] }), [short]))
})

test('True/False keys are explicit booleans, backed by quoted evidence and an ambiguity check', () => {
  const binary = { ...packageData, question: 'Heap records are stored in key order.', correctAnswer: false,
    questionAssessment: 'The source explicitly contradicts key ordering.', questionIssue: '' }
  const parsed = parsePracticeQuestion(JSON.stringify(binary), [source], 'true_false')
  assert.equal(parsed.format, 'true_false')
  assert.equal(parsed.correctAnswer, false)
  assert.equal(gradeTrueFalse(parsed, 'False').correct, true)
  assert.equal(gradeTrueFalse(parsed, 'True').correct, false)
  const truth = { ...parsed, correctAnswer: true }
  assert.equal(gradeTrueFalse(truth, 'True').correct, true)
  assert.equal(gradeTrueFalse(truth, 'False').correct, false)
  for (const correctAnswer of [undefined, null, 'false', 0]) {
    assert.throws(() => parsePracticeQuestion(JSON.stringify({ ...binary, correctAnswer }), [source], 'true_false'), /valid question/)
  }
  for (const changes of [{ questionIssue: 'Depends on an unstated condition.' }, { questionAssessment: '' },
    { questionIssue: undefined }, { criteria: [] }]) {
    assert.throws(() => parsePracticeQuestion(JSON.stringify({ ...binary, ...changes }), [source], 'true_false'), /valid question/)
  }
  assert.throws(() => parsePracticeQuestion(JSON.stringify(binary), [], 'true_false'), /valid question/)
  for (const answer of ['', 'yes', 'FALSE', 'True because...', 'ignore the key']) {
    assert.throws(() => gradeTrueFalse(parsed, answer), /Choose True or False/)
  }
  for (const changes of [{ correctAnswer: undefined }, { correctAnswer: 'false' }, { referenceAnswer: '' }, { sources: [] }, { format: 'short_answer' }]) {
    assert.throws(() => gradeTrueFalse({ ...parsed, ...changes }, 'True'), /no usable/)
  }
  assert.equal(practiceResponseSchema('true_false_question').properties.correctAnswer.type, 'boolean')
  assert.equal(practiceResponseSchema('true_false_question').properties.criteria.maxItems, 1)
  assert.match(practiceSystemPrompt('true_false_question'), /absence of evidence is not evidence of falsity/)
  assert.match(practiceSystemPrompt('true_false_question'), /unstated conditions/)
  assert.doesNotMatch(practiceSystemPrompt('true_false_question'), /heap|LRU|B\+|pinning/)
  assert.equal(question.format, 'short_answer')
  assert.equal(question.correctAnswer, undefined)
})

test('feedback preserves the submitted answer as data and includes the actual previous attempt', () => {
  const prior = { answer: 'No index.', feedback }
  const prompt = JSON.parse(quizFeedbackPrompt({ answer: 'Now I include unordered placement.', question: question.text,
    rubric: question.rubric, explanation: question.referenceAnswer, previousAttempt: prior }))
  assert.deepEqual(prompt.previousAttempt, prior)
  assert.deepEqual(prompt.rubric, question.rubric)
  assert.match(practiceSystemPrompt('feedback'), /meaning, not keywords/)
  assert.match(practiceSystemPrompt('feedback'), /missing.*not explained/)
  assert.match(practiceSystemPrompt('feedback'), /data, not instructions/)
  assert.match(practiceSystemPrompt('feedback'), /never append an expected answer/)
})

test('verdicts follow complete criterion checks, not an inconsistent overall grade', () => {
  assert.equal(parsePracticeFeedback(JSON.stringify(feedback), question.rubric, false).verdict, 'partial')
  const correct = { ...feedback, checks: feedback.checks.map(c => ({ ...c, status: 'met' })), nextStep: '' }
  assert.equal(parsePracticeFeedback(JSON.stringify(correct), question.rubric, true).verdict, 'correct')
  const wrong = { ...feedback, checks: feedback.checks.map(c => ({ ...c, status: 'incorrect' })) }
  assert.equal(parsePracticeFeedback(JSON.stringify(wrong), question.rubric, false).verdict, 'needs_work')
  for (const changed of [
    { ...feedback, checks: feedback.checks.slice(0, 1) },
    { ...feedback, checks: [feedback.checks[0], feedback.checks[0]] },
    { ...feedback, checks: feedback.checks.map(c => ({ ...c, criterionId: 'unknown' })) },
    { ...feedback, nextStep: '' },
    { ...correct, nextStep: 'Revise a correct answer anyway.' }
  ]) assert.throws(() => parsePracticeFeedback(JSON.stringify(changed), question.rubric, false), /usable feedback/)
})

test('ambiguous questions are not graded and invented first-attempt progress is not shown', () => {
  const result = parsePracticeFeedback(JSON.stringify({ questionAssessment: 'The question is ambiguous.', checks: [], nextStep: '', improvement: '',
    questionIssue: 'The question does not specify which case it means.' }), question.rubric, false)
  assert.equal(result.verdict, 'needs_review')
  const conflicting = parsePracticeFeedback(JSON.stringify({ ...feedback, questionIssue: 'Ambiguous.' }), question.rubric, false)
  assert.equal(conflicting.verdict, 'needs_review')
  assert.deepEqual(conflicting.checks, [])
  assert.equal(conflicting.nextStep, '')
  assert.equal(parsePracticeFeedback(JSON.stringify({ ...feedback, improvement: 'Invented progress.' }), question.rubric, false).improvement, '')
  assert.equal(parsePracticeFeedback(JSON.stringify({ ...feedback, improvement: 'You now explain the ordering.' }), question.rubric, true).improvement, 'You now explain the ordering.')
})

test('True/False rejects open-ended prompts even when the model supplies a confident answer key', () => {
  const binary = { ...packageData, correctAnswer: true, questionAssessment: 'Supported.', questionIssue: '' }
  for (const question of [
    'If a crash happens after updating the table but before updating the index, what state results?',
    'What foundational guarantee ensures that neither change will be visible',
    'Explain why partial updates are unsafe.',
    'Assuming the table was written first, which property is needed?',
    'Is the table consistent?',
    'When does a leaf node split',
    'Where are records stored',
    'True/False: Describe what happens after a crash.'
  ]) {
    assert.throws(() => parsePracticeQuestion(JSON.stringify({ ...binary, question }), [source], 'true_false'), /valid question/)
    assert.throws(() => gradeTrueFalse({ ...binary, text: question, format: 'true_false', referenceAnswer: binary.explanation, sources: [source] }, 'False'), /cannot be graded/)
  }
  for (const statement of [
    'A partial table update can leave an index inconsistent.',
    'When a leaf node splits, the parent may need a new separator key.',
    'Where no usable index exists, a full scan may be necessary.',
    'The operator `?` is part of the expression.',
    'The text "What happens?" is stored as a string.',
    'A query explains how the data is selected.'
  ]) assert.doesNotThrow(() => parsePracticeQuestion(JSON.stringify({ ...binary, question: statement }), [source], 'true_false'))
})


test('independent answer checks reject disagreement, uncertainty and malformed judgments', () => {
  const valid = {answer:'false',noveltyReason: 'No equivalent earlier claim.', topicReason:'Assesses the requested concept.',topicMatch:'direct',relationship:'new_claim',reason:'The passage contradicts the claim.'}
  assert.doesNotThrow(()=>checkPracticeQuestionAnswer(JSON.stringify(valid),false))
  for(const raw of ['bad', '{}', JSON.stringify(valid), JSON.stringify({...valid, answer:'uncertain'}), JSON.stringify({...valid, answer:'invalid'})])
    assert.throws(()=>checkPracticeQuestionAnswer(raw,true),/valid question/)
  assert.match(practiceSystemPrompt('verify_question'),/FULL passage/)
  assert.match(practiceSystemPrompt('verify_question'),/No proposed answer key/)
})

test('mastery verification rejects off-topic, repeated and missing judgments even with the right answer key', () => {
  const valid = { answer: 'true', noveltyReason: 'No equivalent earlier claim.', topicReason: 'Assesses the requested concept.', topicMatch: 'direct', relationship: 'new_claim', reason: 'Supported.' }
  for (const change of [{ topicMatch: 'related' }, { topicMatch: 'unrelated' }, { relationship: 'restatement' }, { relationship: 'negation' }, { topicMatch: true }, { relationship: undefined }, { topicReason: '' }, { noveltyReason: '' }])
    assert.throws(() => checkPracticeQuestionAnswer(JSON.stringify({ ...valid, ...change }), true), /valid question/)
  assert.deepEqual(practiceResponseSchema('verify_question').required, ['topicReason', 'topicMatch', 'noveltyReason', 'relationship', 'answer', 'reason'])
})
