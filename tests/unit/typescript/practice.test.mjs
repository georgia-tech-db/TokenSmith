import assert from 'node:assert/strict'
import test from 'node:test'
import { requireTranspiledTs } from './ts-module-loader.mjs'

const { studyDocumentKey } = requireTranspiledTs('src/shared/study-scope.ts')
const { emptyPracticeState, normalizePracticeState, practiceSourceKey, practiceReference,
  practiceDiscussionPassage, practiceChatConversation, practiceReviewLabel, practiceMessages, practiceFeedbackText } = requireTranspiledTs('src/shared/practice.ts')
const { parsePracticeQuestion, gradeTrueFalse } = requireTranspiledTs('src/shared/quiz.ts')
const { createPracticeClient } = requireTranspiledTs('src/renderer/src/practice-client.ts')
const { practiceEvidence, practiceQuestionPlan } = requireTranspiledTs('src/shared/practice-question-plan.ts')
const { prepareStudyChatMessages } = requireTranspiledTs('src/main/engine/study-chat-format.ts')

const source = { materialId: 'c1', documentId: 8, chunkId: 'ch.1', sourceUnitId: 'unit.1', excerpt: 'Ancestor paths let insertion propagate splits to parents.' }
const packageData = { question: 'Why keep a path?', objective: 'Explain split propagation.', assumptions: [], criteria: [
  { id: 'c1', description: 'Propagation to parents.', evidence: [{ sourceKey: 'c1:8:unit.1', quote: source.excerpt }] }
], explanation: 'The path lets insertion revisit parents after a split.' }
const trueFalseData = { ...packageData, question: 'Paths let insertion propagate splits to parents.', correctAnswer: true,
  questionAssessment: 'The passage supports the statement.', questionIssue: '' }
const question = { id: 'q1', text: packageData.question, format: 'short_answer', sources: [source],
  rubric: { objective: packageData.objective, assumptions: packageData.assumptions, criteria: packageData.criteria },
  referenceAnswer: packageData.explanation, draft: 'To revisit ancestors.', attempts: [] }
const feedback = { questionAssessment: 'The question is supported and answerable.', checks: [{ criterionId: 'c1', status: 'met', feedback: 'You identified where updates propagate.' }], improvement: '', nextStep: '', questionIssue: '' }
const model = { id: 'model1', status: 'ready' }
const settings = { application: { suggestionMode: 'on', explanationDepthEnabled: true } }
const session = { id: 's1', documents: [{ materialId: 'c1', documentId: 8 }], modelId: model.id,
  modelSettings: { contextLengthMode: 'auto', contextLength: 8192, maxLength: 4096 }, questions: [], currentIndex: 0, topic: { title: 'Splits', sources: [source] } }

function harness(overrides = {}) {
  const calls = []
  const bridge = {
    practiceSources: async (...args) => { calls.push(['sources', ...args]); return [source] },
    sendChatMessage: async request => { calls.push(['model', request]); return {
      text: JSON.stringify(trueFalseData), sources: [source] } },
    ...overrides
  }
  return { calls, client: createPracticeClient(bridge, model, settings, undefined, [], () => 0) }
}

test('topic retrieval is scoped once, reused on Next, and never falls back to random passages', async () => {
  const calls = []
  const outside = { ...source, materialId: 'other' }
  const bridge = {
    searchLibrary: async (...args) => { calls.push(['search', ...args]); return { sources: [outside, source, source] } },
    practiceSources: async () => { throw new Error('Unrelated whole-document sampling') },
    sendChatMessage: async request => { calls.push(['model', request]); return { text: JSON.stringify(trueFalseData), sources: request.retrievedSources } }
  }
  const client = createPracticeClient(bridge, model, settings, { materials: [{ id: 'c1' }, { id: 'other' }], embeddingModels: [{ id: 'nomic' }] })
  const signal = new AbortController().signal
  const prepared = await client.prepareTopic({ ...session, topic: { title: 'B+ trees', question: 'How do splits propagate?', answer: 'An earlier explanation.' } }, signal)
  assert.deepEqual(calls[0][2], [{ id: 'c1', isActive: true }])
  assert.deepEqual(calls[0].at(-1).documents, session.documents)
  assert.deepEqual(prepared.topic.sources, [source])
  assert.equal(prepared.topic.prepared, true)
  const generated = await client.question(prepared, 'first', signal)
  await client.prepareTopic(prepared, signal)
  assert.equal(calls.filter(call => call[0] === 'search').length, 1)
  assert.equal(JSON.parse(calls[1][1].prompt).topic, 'B+ trees')
  assert.equal((await client.question({ ...prepared, questions: [generated] }, 'duplicate', signal)).text, generated.text)
})

test('unsupported and cancelled topic retrieval cannot generate a whole-document question', async () => {
  const bridge = { searchLibrary: async () => ({ sources: [] }), sendChatMessage: async () => { throw new Error('Unexpected model call') } }
  const client = createPracticeClient(bridge, model, settings, { materials: [{ id: 'c1' }], embeddingModels: [] })
  await assert.rejects(client.prepareTopic({ ...session, topic: { title: 'Unknown topic' } }, new AbortController().signal), /No supporting passages/)
  const controller = new AbortController(); controller.abort()
  await assert.rejects(client.prepareTopic({ ...session, topic: { title: 'B+ trees', sources: [source] } }, controller.signal), /aborted/)
})


test('Chat copies the selected exercise and visible feedback without revealing an unanswered key', () => {
  const binary = { ...question, ...parsePracticeQuestion(JSON.stringify(trueFalseData), [source]), draft: 'False' }
  const unanswered = practiceChatConversation('chat-1', session, binary)
  assert.equal(unanswered.modelId, session.modelId)
  assert.equal(unanswered.messages.length, 1)
  assert.match(unanswered.messages[0].text, /My selected answer: False/)
  assert.equal(unanswered.messages.some(message => message.text.includes(binary.referenceAnswer)), false)
  const checked = { ...binary, attempts: [{ id: 'a1', answer: 'False', feedback: gradeTrueFalse(binary, 'False'), assisted: false, durationMs: 0 }] }
  const before = structuredClone(checked)
  const copied = practiceChatConversation('chat-2', session, checked)
  assert.equal(copied.modelId, session.modelId)
  assert.deepEqual(copied.messages.map(message => message.role), ['user', 'assistant'])
  assert.match(copied.messages[0].text, /My answer: False/)
  assert.match(copied.messages[1].text, /^Incorrect\n\n/)
  assert.deepEqual(copied.messages[1].sources, [source])
  assert.deepEqual(copied.documentScope, [{ materialId: 'c1', documentId: 8 }])
  assert.deepEqual(checked, before)
})

test('practice state preserves saved attempts, scopes, and old transcripts across reloads', () => {
  assert.deepEqual(normalizePracticeState(), emptyPracticeState())
  assert.deepEqual(normalizePracticeState({ sessions: null }), emptyPracticeState())
  const saved = { mode: 'practice', activeSessionId: session.id, draftModelId: model.id, sessions: [session] }
  assert.deepEqual(normalizePracticeState(JSON.parse(JSON.stringify(saved))), saved)
  assert.equal(normalizePracticeState({ ...saved, activeSessionId: 'missing' }).activeSessionId, undefined)
})

test('saved practice context defaults migrate to Auto without changing custom sizes or student work', () => {
  for (const [contextLength, contextLengthMode, expected] of [
    [2048, undefined, 'auto'], [8192, undefined, 'auto'],
    [16384, undefined, 'manual'], [2048, 'manual', 'manual'], [16384, 'auto', 'auto']
  ]) {
    const old = { ...session, questions: [question], modelSettings: { ...session.modelSettings, contextLength, contextLengthMode } }
    const saved = { mode: 'practice', activeSessionId: old.id, sessions: [old] }
    const restored = normalizePracticeState(JSON.parse(JSON.stringify(saved)))
    assert.equal(restored.sessions[0].modelSettings.contextLengthMode, expected)
    assert.equal(restored.sessions[0].modelSettings.contextLength, contextLength)
    assert.deepEqual(restored.sessions[0].questions, old.questions)
    assert.deepEqual(restored.sessions[0].documents, old.documents)
    assert.equal(restored.activeSessionId, old.id)
    assert.equal(old.modelSettings.contextLengthMode, contextLengthMode)
    assert.deepEqual(normalizePracticeState(restored), restored)
  }
})

test('resuming a legacy 2K practice session leaves room for True/False questions', async () => {
  const old = { ...session, questions: [question], modelSettings: { ...session.modelSettings,
    contextLength: 2048, contextLengthMode: undefined } }
  const restored = normalizePracticeState({ mode: 'practice', sessions: [old] }).sessions[0]
  const requests = []
  const { client } = harness({ sendChatMessage: async request => {
    requests.push(request)
    const prepared = prepareStudyChatMessages({ ...request,
      model: { ...model, engine: 'ollama', contextLength: 262144 } })
    assert.equal(prepared.budget.modelContextTokens, 262144)
    assert.ok(prepared.budget.estimatedPromptTokens + prepared.budget.answerReserveTokens + prepared.budget.safetyMarginTokens <= 262144)
    assert.deepEqual(prepared.sources, [source])
    return { text: request.practiceTask === 'true_false_question' ? JSON.stringify(trueFalseData) : 'An explanation.', sources: [source] }
  } })
  await client.question(restored, 'next', new AbortController().signal)
  assert.equal(requests.length, 1)
  assert.ok(requests.every(request => request.modelSettings.contextLengthMode === 'auto'))
})

test('document and evidence identity includes collection and document', () => {
  assert.notEqual(studyDocumentKey(source), studyDocumentKey({ ...source, materialId: 'c2' }))
  assert.notEqual(studyDocumentKey(source), studyDocumentKey({ ...source, documentId: 9 }))
  assert.equal(practiceSourceKey(source), 'c1:8:unit.1')
})

test('review labels describe saved responses', () => {
  assert.equal(practiceReviewLabel(question), 'Not answered')
  assert.equal(practiceReviewLabel({ ...question, attempts: [{ feedback: 'Awaiting independent review.' }] }), 'Awaiting review')
})

test('Chat handoff uses the submitted answer and feedback, never the unrevealed explanation', () => {
  assert.equal(practiceReference(session, question).feedback, '')
  const answered = { ...question, draft: 'Unsubmitted revision', attempts: [{ answer: 'Submitted answer', feedback: 'Actual feedback' }] }
  const reference = practiceReference(session, answered)
  assert.equal(reference.answer, 'Submitted answer')
  assert.deepEqual(reference.sources, [source])
  assert.deepEqual(reference.documents, session.documents)
  assert.deepEqual(practiceDiscussionPassage(reference), { messageId: 'q1', role: 'assistant',
    question: 'Practice question: Why keep a path?\n\nMy answer: Submitted answer', text: 'Actual feedback' })
})

test('one question call prepares the question, saved criteria and explanation', async () => {
  const { calls, client } = harness()
  const result = await client.question({ ...session, questions: [question] }, 'r1', new AbortController().signal)
  const request = calls[0][1]
  assert.equal(request.practiceTask, 'true_false_question')
  assert.deepEqual(request.messages, [])
  assert.deepEqual(request.materials, [])
  assert.deepEqual(request.retrievedSources, [source])
  assert.equal(request.applicationSettings.suggestionMode, 'off')
  assert.equal(request.modelSettings.thinking, false)
  assert.equal(request.modelSettings.maxLength, 2048)
  assert.equal(result.text, trueFalseData.question)
  assert.deepEqual(result.rubric, question.rubric)
  assert.equal(result.referenceAnswer, packageData.explanation)
  assert.equal('hint' in result, false)
  assert.deepEqual(result.attempts, [])
  assert.equal(calls.length, 1)
  assert.equal(client.help, undefined)
})

test('practice generates only source-backed True/False statements from the first question onward', async () => {
  const { calls, client } = harness({ sendChatMessage: async request => {
    calls.push(['model', request])
    return { text: JSON.stringify({ ...trueFalseData, question: `In example ${JSON.parse(request.prompt).questionNumber}, paths let insertion propagate splits to parents.` }), sources: request.retrievedSources }
  } })
  const topic = { ...session.topic, sources: Array.from({ length: 3 }, (_, i) => ({ ...source, sourceUnitId: `unit.${i}` })) }
  const questions = []
  for (let index = 0; index < 6; index++) {
    questions.push(await client.question({ ...session, topic, questions }, `r${index}`, new AbortController().signal))
  }
  assert.ok(questions.every(question => question.format === 'true_false'))
  const generated = calls.filter(call => call[0] === 'model')
  assert.equal(generated.length, 6)
  assert.ok(generated.every(call => call[1].practiceTask === 'true_false_question'))
  assert.equal(generated[2][1].modelSettings.maxLength, 2048)
  const binary = questions[2]
  assert.equal(binary.correctAnswer, true)
  assert.equal(gradeTrueFalse(binary, 'False').correct, false)
  assert.equal(gradeTrueFalse(binary, 'True').correct, true)
  assert.equal(calls.filter(call => call[0] === 'model').length, 6, 'Choice checking has no model call')
  assert.equal(client.feedback, undefined, 'Practice has no model grading operation')
  const saved = { ...session, questions }
  assert.deepEqual(normalizePracticeState(JSON.parse(JSON.stringify({ mode: 'practice', sessions: [saved] }))).sessions[0].questions, questions)
  assert.match(practiceMessages(binary)[0].text, /^True\/False:/)
  assert.match(practiceReference(session, binary).question, /^True\/False:/)
  assert.equal(practiceReference(session, binary).feedback, '', 'The hidden key/explanation must not leak into help')
})

test('written feedback and history omit overall verdicts while binary feedback is explicit', () => {
  for (const verdict of ['partial', 'needs_work', 'correct']) {
    const parsed = { ...feedback, verdict }
    assert.equal(practiceFeedbackText(parsed), feedback.checks[0].feedback)
    assert.doesNotMatch(practiceMessages({ ...question, attempts: [{ id: 'attempt', answer: 'My answer', feedback: parsed }] }).at(-1).text,
      /Partly there|Needs another look|^Correct/)
  }
  const binary = { kind: 'true_false', correct: false, explanation: 'The path enables parent updates.' }
  assert.equal(practiceFeedbackText(binary), 'Incorrect\n\nThe path enables parent updates.')
  assert.equal(practiceReviewLabel({ ...question, attempts: [{ feedback: binary }] }), 'Incorrect')
})

test('saved short-answer questions remain readable but cannot be graded', () => {
  const { calls, client } = harness()
  const saved = { ...question, attempts: [{ id: 'old', answer: 'My previous answer.', feedback: 'Saved feedback.' }] }
  assert.equal(practiceMessages(saved).at(-1).text, 'Saved feedback.')
  assert.equal(client.feedback, undefined)
  assert.throws(() => gradeTrueFalse(saved, 'True'), /answer key/)
  assert.equal(calls.length, 0)
})

test('invalid model output fails once and an empty topic cannot sample the whole book', async () => {
  const empty = harness()
  assert.equal(await empty.client.question({ ...session, topic: undefined }, 'r', new AbortController().signal), null)
  assert.equal(empty.calls.length, 0)
  for (const reply of [{ text: '', sources: [source] }, { text: 'Question?', sources: [] },
    { text: JSON.stringify(packageData), sources: [source] }]) {
    let count = 0
    const { client } = harness({ sendChatMessage: async () => { count++; return reply } })
    await assert.rejects(client.question(session, 'r', new AbortController().signal), /Could not read/)
    assert.equal(count, 1)
  }
})

test('topic and same-passage history remain bounded across earlier sessions', async () => {
  const calls = []
  const concept = { courseId: 'book', id: 'splits', title: 'Splits' }
  const current = { ...session, topic: { ...session.topic, concept } }
  const questions = Array.from({ length: 80 }, (_, index) => ({ ...question, id: `q${index}`, text: `Question ${index}` }))
  const client = createPracticeClient({ sendChatMessage: async request => {
    calls.push(request); return { text: JSON.stringify(trueFalseData), sources: request.retrievedSources }
  } }, model, settings, undefined, [{ ...current, id: 'old', createdAt: '2026-01-01', questions }])
  await client.question(current, 'r', new AbortController().signal)
  const input = JSON.parse(calls[0].prompt)
  assert.equal(input.questionNumber, 1)
  assert.equal(input.totalQuestions, undefined)
  assert.equal(input.previousQuestions.length, 12)
  assert.equal(input.previousQuestions[0], 'Question 68')
  assert.equal(input.previousQuestionsFromPassage.length, 12)
  assert.equal(input.previousQuestionsFromPassage[0].question, 'Question 68')
})

test('practice messages use saved attempts directly with stable selectable IDs', () => {
  const value = { ...question, attempts: [{ id: 'a1', answer: 'First', feedback: 'Correct' }] }
  assert.deepEqual(practiceMessages(value).map(message => message.text), ['Why keep a path?', 'First', 'Correct'])
  assert.equal(new Set(practiceMessages(value).map(message => message.id)).size, 3)
})

test('quality gates are absent: repeated wording, invented quotes and requested-key mismatches remain observable', async () => {
  const output = { ...trueFalseData, question: 'What happens?', correctAnswer: false, questionIssue: 'Uncertain.',
    criteria: [{ ...trueFalseData.criteria[0], evidence: [{ sourceKey: 'invented', quote: 'Not in the passage.' }] }] }
  const { calls, client } = harness({ sendChatMessage: async request => {
    calls.push(['model', request]); return { text: JSON.stringify(output), sources: [source] }
  } })
  const result = await client.question({ ...session, questions: [{ ...question, text: output.question }] }, 'r', new AbortController().signal)
  assert.equal(calls.length, 1)
  assert.equal(result.requestedAnswer, true)
  assert.equal(result.correctAnswer, false, 'Do not silently change the returned key to the requested key')
  assert.equal(result.text, output.question)
  assert.equal(result.rubric.criteria[0].evidence[0].sourceKey, 'invented')
  assert.deepEqual(result.sources, [source], 'Show the actual input passage, regardless of generated citation text')
  assert.equal(gradeTrueFalse(result, 'False').correct, true)
})

test('random targets allow runs of the same answer and do not alternate by question number', async () => {
  const draws = [0, 0.49, 0.5, 0.99, 0.1, 0.9], targets = []
  const client = createPracticeClient({ sendChatMessage: async request => {
    targets.push(JSON.parse(request.prompt).targetAnswer)
    return { text: JSON.stringify({ ...trueFalseData, correctAnswer: targets.at(-1) }), sources: [source] }
  } }, model, settings, undefined, [], () => draws.shift())
  for (let i = 0; i < 6; i++) await client.question(session, String(i), new AbortController().signal)
  assert.deepEqual(targets, [true, true, false, false, true, false])
  assert.equal(draws.length, 0)
})

test('manual retry keeps the truth target and advances the passage across saved restarts', async () => {
  const pool = [source, { ...source, chunkId: 'ch.2', sourceUnitId: 'unit.2' }]
  let saved = { ...session, topic: { title: 'Splits', sources: pool } }, calls = 0, draws = 0
  const requests = []
  const client = createPracticeClient({ sendChatMessage: async request => {
    requests.push(request)
    if (++calls === 1) return { text: 'broken JSON', sources: request.retrievedSources }
    return { text: JSON.stringify(trueFalseData), sources: request.retrievedSources }
  } }, model, settings, undefined, [], () => { draws++; return 0.8 })
  const remember = (cursor, _sources, target) => { saved = { ...saved, questionSourceCursor: cursor, questionTargetAnswer: target } }
  await assert.rejects(client.question(saved, 'first', new AbortController().signal, remember), /Could not read/)
  saved = normalizePracticeState(JSON.parse(JSON.stringify({ sessions: [saved] }))).sessions[0]
  await client.question(saved, 'retry', new AbortController().signal, remember)
  assert.equal(draws, 1)
  assert.deepEqual(requests.map(r => r.retrievedSources[0].chunkId), ['ch.1', 'ch.2'])
  assert.deepEqual(requests.map(r => JSON.parse(r.prompt).targetAnswer), [false, false])
})

test('rotation stays within the topic, wraps, and continues across sessions without crossing collections', async () => {
  const concept = { courseId: 'book', id: 'splits', title: 'Splits' }
  const pool = [source, { ...source, chunkId: 'ch.2', sourceUnitId: 'unit.2' }, { ...source, chunkId: 'ch.3', sourceUnitId: 'unit.3' }]
  const old = { ...session, id: 'old', createdAt: '2026-01-01', questionSourceCursor: 2, topic: { ...session.topic, concept }, questions: [{ ...question, text: 'Old claim.' }] }
  const unrelated = { ...old, id: 'unrelated', questionSourceCursor: 99, topic: { ...old.topic, concept: { ...concept, courseId: 'other' } } }
  let saved = { ...session, topic: { title: 'Splits', concept, sources: pool } }
  const requests = []
  const client = createPracticeClient({ sendChatMessage: async request => {
    requests.push(request); return { text: JSON.stringify(trueFalseData), sources: request.retrievedSources }
  } }, model, settings, undefined, [unrelated, old], () => 0)
  for (let i = 0; i < 4; i++) {
    const result = await client.question(saved, String(i), new AbortController().signal, cursor => { saved = { ...saved, questionSourceCursor: cursor } })
    assert.deepEqual(result.masteryCheck, { version: 1, conceptKey: JSON.stringify(['book', 'splits']), verification: 'none' })
  }
  assert.deepEqual(requests.map(r => r.retrievedSources[0].chunkId), ['ch.3', 'ch.1', 'ch.2', 'ch.3'])
  assert.deepEqual(JSON.parse(requests[0].prompt).previousQuestions, ['Old claim.'])
})

test('a mapped topic is authoritative even when an older session contains broad search results', async () => {
  const mapped = Array.from({ length: 10 }, (_, i) => ({ ...source, materialId: 'book/1', chunkId: `ch.${i}`, sourceUnitId: `part.${i}` }))
  const topic = { id: 'saved', title: 'Splits', concept: { courseId: 'book', id: 'saved', title: 'Splits' }, sources: mapped }
  const inventory = { version: 3, complete: true, indexRevision: 'revision', topics: [topic] }
  const bridge = { loadTopicInventory: async () => inventory, searchLibrary: async () => { throw new Error('Must not search') } }
  const client = createPracticeClient(bridge, model, settings, { materials: [{ id: 'book/1', collectionRevision: 'revision' }], embeddingModels: [] })
  const prepared = await client.prepareTopic({ ...session, documents: [{ materialId: 'book/1', documentId: 8 }],
    topic: { ...topic, prepared: true, sources: [{ ...mapped[0], chunkId: 'unrelated' }] } }, new AbortController().signal)
  assert.deepEqual(prepared.topic.sources, mapped, 'Preserve all mapped passages, including those beyond eight')
  inventory.topics = []
  await assert.rejects(client.prepareTopic(prepared, new AbortController().signal), /saved topic is unavailable/)
})

test('cancellation and connection failure never trigger another model call', async () => {
  let count = 0
  const controller = new AbortController()
  const { client } = harness({ sendChatMessage: async () => { count++; throw new Error('Connection refused') } })
  await assert.rejects(client.question(session, 'cancel', controller.signal, () => controller.abort()), { name: 'AbortError' })
  assert.equal(count, 0)
  await assert.rejects(client.question(session, 'offline', new AbortController().signal), /Connection refused/)
  assert.equal(count, 1)
})

test('one passage yields two questions, ten yield twenty, and completion makes no extra model call', async () => {
  for (const count of [1, 10]) {
    const pool = Array.from({ length: count }, (_, i) => ({ ...source, chunkId: `ch.${i}`, sourceUnitId: `unit.${i}` }))
    let saved = { ...session, topic: { ...session.topic, sources: pool }, questions: [] }
    const requests = []
    const client = createPracticeClient({ sendChatMessage: async request => {
      requests.push(request); return { text: JSON.stringify(trueFalseData), sources: request.retrievedSources }
    } }, model, settings)
    assert.equal(practiceQuestionPlan(saved).totalQuestions, count * 2)
    for (let i = 0; i < count * 2; i++) {
      const next = await client.question(saved, String(i), new AbortController().signal, cursor => { saved.questionSourceCursor = cursor })
      saved = normalizePracticeState(JSON.parse(JSON.stringify({ sessions: [{ ...saved, questions: [...saved.questions, next] }] }))).sessions[0]
    }
    assert.equal(await client.question(saved, 'complete', new AbortController().signal), null)
    assert.equal(requests.length, count * 2)
    assert.deepEqual(requests.map(r => r.retrievedSources[0].chunkId), [...pool, ...pool].map(s => s.chunkId))
    assert.equal(practiceQuestionPlan(saved).source, undefined)
  }
})

test('question limits ignore duplicate, empty and out-of-scope mapped passages', () => {
  const mapped = [source, { ...source }, { ...source, materialId: 'another-book' },
    { ...source, sourceUnitId: 'empty', excerpt: '' }]
  assert.equal(practiceQuestionPlan({ ...session, topic: { ...session.topic, sources: mapped } }).totalQuestions, 2)
  assert.equal(practiceQuestionPlan({ ...session, topic: { ...session.topic, sources: [] } }).totalQuestions, 0)
})

test('evidence bundles stay in scope, deduplicate and prefer explanatory metadata without topic keywords', () => {
  const exercise = { ...source, chunkId: 'exercise', sourceUnitId: 'exercise', chunkKind: 'exercise' }
  const context = [1, 2, 3, 4].map(i => ({ ...source, chunkId: `context${i}`, sourceUnitId: `context${i}` }))
  const current = { ...session, topic: { ...session.topic, sources: [exercise, ...context, source, source,
    { ...source, materialId: 'other' }, { ...source, sourceUnitId: 'blank', excerpt: '' }] } }
  assert.deepEqual(practiceEvidence(current, source), [source, ...context.slice(0, 3)])
  assert.deepEqual(practiceEvidence(current, exercise), [exercise, ...context.slice(0, 3)])
  assert.equal(practiceQuestionPlan(current).source.chunkId, 'context1')
})

test('uncited supporting evidence does not consume another passage’s allowance or same-passage history', async () => {
  const second = { ...source, chunkId: 'second', sourceUnitId: 'second' }
  const requests = []
  const client = createPracticeClient({ sendChatMessage: async request => {
    requests.push(request); return { text: JSON.stringify(trueFalseData), sources: request.retrievedSources }
  } }, model, settings)
  let current = { ...session, topic: { ...session.topic, sources: [source, second] } }
  const first = await client.question(current, 'first', new AbortController().signal,
    cursor => { current.questionSourceCursor = cursor })
  assert.equal(first.focusSourceKey, practiceSourceKey(source))
  assert.deepEqual(first.sources, [source, second])
  current = { ...current, questions: [first] }
  const next = await client.question(current, 'next', new AbortController().signal)
  assert.equal(next.focusSourceKey, practiceSourceKey(second))
  assert.deepEqual(JSON.parse(requests[1].prompt).previousQuestionsFromPassage, [])
  assert.equal(JSON.parse(requests[1].prompt).focusSourceKey, practiceSourceKey(second))
  assert.equal(practiceQuestionPlan({ ...current, questions: [first, { ...first, id: 'another' }] }).source.chunkId, 'second')
  assert.equal(requests.length, 2)
})

test('a question citing a supporting passage is included when that passage becomes the focus', async () => {
  const second = { ...source, chunkId: 'second', sourceUnitId: 'second' }
  const earlier = { ...question, focusSourceKey: practiceSourceKey(source), sources: [source, second],
    rubric: { ...question.rubric, criteria: [{ id: 'c1', description: 'Already tested supporting idea',
      evidence: [{ sourceKey: practiceSourceKey(second), quote: second.excerpt }] }] } }
  const requests = []
  const client = createPracticeClient({ sendChatMessage: async request => {
    requests.push(request); return { text: JSON.stringify(trueFalseData), sources: request.retrievedSources }
  } }, model, settings)
  await client.question({ ...session, questionSourceCursor: 1, questions: [earlier],
    topic: { ...session.topic, sources: [source, second] } }, 'next', new AbortController().signal)
  const prompt = JSON.parse(requests[0].prompt)
  assert.equal(prompt.previousQuestionsFromPassage.length, 1)
  assert.deepEqual(prompt.previousQuestionsFromPassage[0].testedIdeas, ['Already tested supporting idea'])
})

test('question sources reflect what the generator actually received after context packing', async () => {
  const second = { ...source, chunkId: 'second', sourceUnitId: 'second' }
  const client = createPracticeClient({ sendChatMessage: async request => {
    assert.equal(request.retrievedSources.length, 2)
    return { text: JSON.stringify(trueFalseData), sources: [source] }
  } }, model, settings)
  const result = await client.question({ ...session, topic: { ...session.topic, sources: [source, second] } }, 'packed', new AbortController().signal)
  assert.deepEqual(result.sources, [source])
})

test('rotation skips passages with two saved questions and failed attempts do not consume slots', async () => {
  const second = { ...source, chunkId: 'ch.2', sourceUnitId: 'unit.2' }
  let saved = { ...session, topic: { ...session.topic, sources: [source, second] }, questionSourceCursor: 0,
    questions: [{ ...question, id: 'first' }, { ...question, id: 'second' }] }
  const requests = []
  const client = createPracticeClient({ sendChatMessage: async request => {
    requests.push(request)
    return { text: requests.length === 1 ? 'broken JSON' : JSON.stringify(trueFalseData), sources: request.retrievedSources }
  } }, model, settings)
  const remember = (cursor, _sources, target) => { saved = { ...saved, questionSourceCursor: cursor, questionTargetAnswer: target } }
  await assert.rejects(client.question(saved, 'failed', new AbortController().signal, remember), /Could not read/)
  for (let i = 0; i < 2; i++) {
    const next = await client.question(saved, String(i), new AbortController().signal, remember)
    saved = { ...saved, questions: [...saved.questions, next], questionTargetAnswer: undefined }
  }
  assert.equal(await client.question(saved, 'complete', new AbortController().signal), null)
  assert.deepEqual(requests.map(r => r.retrievedSources[0].chunkId), ['ch.2', 'ch.2', 'ch.2'])
  assert.equal(saved.questions.length, 4)
})

test('same-passage history survives the topic history cutoff and includes tested ideas, not student answers', async () => {
  const concept = { courseId: 'book', id: 'splits', title: 'Splits' }
  const earlier = { ...question, text: 'Earlier same-passage claim.', correctAnswer: true, draft: 'Private student answer' }
  const otherPassage = { ...source, chunkId: 'other', sourceUnitId: 'other' }
  const history = [{ ...session, id: 'old', createdAt: '2026-01-01', questionSourceCursor: 0,
    topic: { ...session.topic, concept }, questions: [earlier, ...Array.from({ length: 15 }, (_, i) =>
      ({ ...question, id: `q${i}`, text: `Other passage ${i}.`, sources: [otherPassage] }))] },
    { ...session, id: 'other-book', createdAt: '2026-01-02', questions: [{ ...earlier, sources: [{ ...source, materialId: 'another-book' }] }] }]
  const requests = []
  const client = createPracticeClient({ sendChatMessage: async request => {
    requests.push(request); return { text: JSON.stringify(trueFalseData), sources: request.retrievedSources }
  } }, model, settings, undefined, history)
  const current = { ...session, createdAt: '2026-01-03', topic: { ...session.topic, concept } }
  const next = await client.question(current, 'new-session', new AbortController().signal)
  const prompt = JSON.parse(requests[0].prompt)
  assert.equal(prompt.previousQuestions.includes(earlier.text), false, 'Old question fell outside recent topic history')
  assert.deepEqual(prompt.previousQuestionsFromPassage, [{ question: earlier.text, correctAnswer: true,
    testedIdeas: [packageData.criteria[0].description], supportingQuotes: [source.excerpt] }])
  assert.equal(requests[0].prompt.includes('Private student answer'), false)
  assert.equal(practiceQuestionPlan({ ...current, questions: [next] }).totalQuestions, 2, 'Earlier sessions do not consume this session’s allowance')
})
