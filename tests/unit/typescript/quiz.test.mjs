import assert from 'node:assert/strict'
import test from 'node:test'
import { requireTranspiledTs } from './ts-module-loader.mjs'

const { quizFeedbackPrompt, quizQuestionPrompt, practiceSystemPrompt,
  parsePracticeQuestion, parsePracticeTopics, parsePracticeFeedback, gradeTrueFalse, practiceResponseSchema } = requireTranspiledTs('src/shared/quiz.ts')
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

test('topic suggestions require evidence from the packed documents and short concept names', () => {
  const raw = topics => JSON.stringify({ topics })
  const topic = { title: 'Heap files', sourceKeys: ['c:1:1'] }
  assert.deepEqual(parsePracticeTopics(raw([topic, { ...topic, title: 'heap files' }]), [source]), [{ title: 'heap files', sources: [source] }])
  for (const invalid of [{ ...topic, sourceKeys: ['other:1:1'] }, { ...topic, sourceKeys: [] }, { ...topic, title: '' }, { ...topic, title: 'A question?' }]) {
    assert.throws(() => parsePracticeTopics(raw([invalid]), [source]), /usable topic suggestions/)
  }
  assert.match(practiceSystemPrompt('topics'), /B\+ trees or Volcano model/)
  assert.match(practiceSystemPrompt('topics'), /do not suggest them unless the documents cover them/)
})

test('True/False packages ask for a statement field and say why a previous package was rejected', () => {
  const schema = practiceResponseSchema('true_false_question')
  assert.equal(schema.properties.question, undefined)
  assert.match(schema.properties.statement.description, /declarative sentence.*Never a question/)
  assert.equal(schema.required[0], 'statement')
  assert.match(practiceSystemPrompt('true_false_question'), /statement in statement.*never a question/)
  assert.equal(practiceResponseSchema('question').properties.question.type, 'string')
  const first = JSON.parse(quizQuestionPrompt({ questionNumber: 1 }))
  assert.equal(first.task, 'Prepare a source-backed True/False statement package')
  assert.equal(first.previousPackageRejected, undefined)
  assert.equal(JSON.parse(quizQuestionPrompt({ questionNumber: 1, format: 'short_answer' })).task, 'Prepare a source-backed question package')
  const retry = JSON.parse(quizQuestionPrompt({ questionNumber: 1, rejection: 'The True/False item is a question, not a statement: Why?' }))
  assert.equal(retry.previousPackageRejected, 'The True/False item is a question, not a statement: Why?. Return a new package that fixes this problem.')
  assert.ok(JSON.parse(quizQuestionPrompt({ questionNumber: 1, rejection: 'x'.repeat(2000) })).previousPackageRejected.length < 460)
})

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
    p => p.criteria[0].evidence[0].quote = 'A quotation that is not present.',
    p => { p.criteria[0].evidence[0].sourceKey = 'another-document'; p.criteria[0].evidence[0].quote = 'A quotation that is not present.' },
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

test('question evidence quoted from a different supplied passage is credited to that passage', () => {
  const other = { materialId: 'c', documentId: 1, chunkId: '2', excerpt: 'Buffer pools cache disk pages in memory.' }
  const changed = structuredClone(packageData)
  changed.criteria[1].evidence[0] = { sourceKey: 'c:1:1', quote: 'Buffer pools cache disk pages in memory.' }
  const parsed = parsePracticeQuestion(JSON.stringify(changed), [source, other])
  assert.equal(parsed.rubric.criteria[1].evidence[0].sourceKey, 'c:1:2')
  assert.deepEqual(parsed.sources, [source, other])
  changed.criteria[1].evidence[0].sourceKey = 'invented-key'
  assert.equal(parsePracticeQuestion(JSON.stringify(changed), [source, other]).rubric.criteria[1].evidence[0].sourceKey, 'c:1:2')
})

test('question evidence ignores stray spaces that PDF extraction leaves around punctuation', () => {
  const pdf = { ...source, excerpt: 'Let V2 and V3 denote the version vectors at N2 and N3 . Then V2 [2] = 2 ( for N2 ) .' }
  const changed = structuredClone(packageData)
  changed.criteria = [{ ...packageData.criteria[0], evidence: [{ sourceKey: 'c:1:1', quote: 'Let V2 and V3 denote the version vectors at N2 and N3.' }] },
    { ...packageData.criteria[1], evidence: [{ sourceKey: 'c:1:1', quote: 'Then V2 [2] = 2 (for N2).' }] }]
  assert.doesNotThrow(() => parsePracticeQuestion(JSON.stringify(changed), [pdf]))
  changed.criteria[1].evidence[0].quote = 'Then V2[2] = 2'
  assert.throws(() => parsePracticeQuestion(JSON.stringify(changed), [pdf]), /valid question/)
})

test('rejected question packages keep the specific reason for logging', () => {
  const changed = structuredClone(packageData)
  changed.criteria[0].evidence[0].quote = 'A quotation that is not present.'
  assert.throws(() => parsePracticeQuestion(JSON.stringify(changed), [source]),
    error => /valid question/.test(error.message) && /c1 was not found.*A quotation that is not present/.test(error.cause))
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
  for (const question of ['The Volcano exchange-operator model covers which specific aspects?', 'Which aspects does the Volcano model cover.',
    'How does insertion propagate a split', '**Is a heap ordered by key?**', 'Heap records are ordered by key?"']) {
    assert.throws(() => parsePracticeQuestion(JSON.stringify({ ...binary, question }), [source], 'true_false'),
      error => /valid question/.test(error.message) && /a question, not a statement/.test(error.cause), question)
  }
  for (const question of ['When a heap has no index, a lookup may scan every page.', 'Where no index exists, records are scanned.',
    'However, heap records are not ordered by key.', 'Whatever the key, heap records are not ordered by it.']) {
    assert.doesNotThrow(() => parsePracticeQuestion(JSON.stringify({ ...binary, question }), [source], 'true_false'), question)
  }
  assert.doesNotThrow(() => parsePracticeQuestion(JSON.stringify(packageData), [source]))
  const { question: statement, ...rest } = binary
  const fromStatement = parsePracticeQuestion(JSON.stringify({ ...rest, statement }), [source], 'true_false')
  assert.equal(fromStatement.text, statement)
  assert.throws(() => parsePracticeQuestion(JSON.stringify({ ...rest, statement: 'Which order do heap records use?' }), [source], 'true_false'), /valid question/)
  for (const answer of ['', 'yes', 'FALSE', 'True because...', 'ignore the key']) {
    assert.throws(() => gradeTrueFalse(parsed, answer), /Choose True or False/)
  }
  for (const changes of [{ correctAnswer: undefined }, { correctAnswer: 'false' }, { referenceAnswer: '' }, { sources: [] }, { format: 'short_answer' }]) {
    assert.throws(() => gradeTrueFalse({ ...parsed, ...changes }, 'True'), /no usable/)
  }
  assert.equal(practiceResponseSchema('true_false_question').properties.correctAnswer.type, 'boolean')
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
