import assert from 'node:assert/strict'
import test from 'node:test'
import { requireTranspiledTs } from './ts-module-loader.mjs'

const { studyDocumentKey } = requireTranspiledTs('src/shared/study-scope.ts')
const { emptyPracticeState, normalizePracticeState, practiceSourceKey, practiceAssisted, practiceReference,
  practiceDiscussionPassage, practiceChatConversation, practiceReviewLabel, practiceTimeline, practiceMessages, practiceFeedbackText } = requireTranspiledTs('src/shared/practice.ts')
const { parsePracticeQuestion, gradeTrueFalse } = requireTranspiledTs('src/shared/quiz.ts')
const { createPracticeClient } = requireTranspiledTs('src/renderer/src/practice-client.ts')
const { prepareStudyChatMessages } = requireTranspiledTs('src/main/engine/study-chat-format.ts')

const source = { materialId: 'c1', documentId: 8, chunkId: 'ch.1', sourceUnitId: 'unit.1', excerpt: 'Ancestor paths let insertion propagate splits to parents.' }
const packageData = { question: 'Why keep a path?', objective: 'Explain split propagation.', assumptions: [], criteria: [
  { id: 'c1', description: 'Propagation to parents.', evidence: [{ sourceKey: 'c1:8:unit.1', quote: source.excerpt }] }
], explanation: 'The path lets insertion revisit parents after a split.', hint: 'Consider which node changes next.' }
const trueFalseData = { ...packageData, question: 'Paths let insertion propagate splits to parents.', correctAnswer: true,
  questionAssessment: 'The passage supports the statement.', questionIssue: '' }
const question = { id: 'q1', ...parsePracticeQuestion(JSON.stringify(packageData), [source]), draft: 'To revisit ancestors.', attempts: [], reviewing: false }
const feedback = { questionAssessment: 'The question is supported and answerable.', checks: [{ criterionId: 'c1', status: 'met', feedback: 'You identified where updates propagate.' }], improvement: '', nextStep: '', questionIssue: '' }
const model = { id: 'model1', status: 'ready' }
const settings = { application: { suggestionMode: 'on', explanationDepthEnabled: true } }
const session = { id: 's1', documents: [{ materialId: 'c1', documentId: 8 }], modelId: model.id,
  modelSettings: { contextLengthMode: 'auto', contextLength: 8192, maxLength: 4096 }, questions: [], currentIndex: 0 }

function harness(overrides = {}) {
  const calls = []
  const bridge = {
    practiceSources: async (...args) => { calls.push(['sources', ...args]); return [source] },
    sendChatMessage: async request => { calls.push(['model', request]); return {
      text: JSON.stringify(trueFalseData), sources: [source] } },
    ...overrides
  }
  return { calls, client: createPracticeClient(bridge, model, settings) }
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
  await assert.rejects(client.question({ ...prepared, questions: [generated] }, 'duplicate', signal), /repeated an earlier question/)
})

test('unsupported and cancelled topic retrieval cannot generate a whole-document question', async () => {
  const bridge = { searchLibrary: async () => ({ sources: [] }), sendChatMessage: async () => { throw new Error('Unexpected model call') } }
  const client = createPracticeClient(bridge, model, settings, { materials: [{ id: 'c1' }], embeddingModels: [] })
  await assert.rejects(client.prepareTopic({ ...session, topic: { title: 'Unknown topic' } }, new AbortController().signal), /No supporting passages/)
  const controller = new AbortController(); controller.abort()
  await assert.rejects(client.prepareTopic({ ...session, topic: { title: 'B+ trees', sources: [source] } }, controller.signal), /aborted/)
})

test('topic discovery uses one model call and preserves full source text only for returned evidence', async () => {
  const calls = []
  const fullSource = { ...source, context: source.excerpt.repeat(100) }
  const bridge = {
    starterSources: async (...args) => { calls.push(['sample', ...args]); return [fullSource, { ...source, materialId: 'other' }] },
    sendChatMessage: async request => { calls.push(['model', request]); return {
      text: JSON.stringify({ topics: [{ title: 'B+ trees', sourceKeys: ['c1:8:topic-1'] }] }), sources: request.retrievedSources } }
  }
  const client = createPracticeClient(bridge, model, settings, { materials: [{ id: 'c1' }], embeddingModels: [] })
  const topics = await client.topics(session.documents, session.modelSettings, 'topics', new AbortController().signal)
  assert.equal(calls.length, 2)
  assert.deepEqual(calls[0][3], session.documents)
  assert.equal(calls[1][1].practiceTask, 'topics')
  assert.equal(calls[1][1].reasoning, false)
  assert.equal(calls[1][1].retrievedSources.length, 1)
  assert.ok(calls[1][1].retrievedSources[0].excerpt.length <= 500)
  assert.deepEqual(topics[0].sources, [fullSource])
  const cached = { mode: 'practice', sessions: [], topicSuggestions: { key: 'scope', topics } }
  assert.deepEqual(normalizePracticeState(JSON.parse(JSON.stringify(cached))).topicSuggestions, cached.topicSuggestions)
})

test('Chat copies the selected exercise and visible feedback without revealing an unanswered key', () => {
  const binary = { ...question, ...parsePracticeQuestion(JSON.stringify(trueFalseData), [source], 'true_false'), draft: 'False' }
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
  assert.equal(practiceAssisted({ ...binary, chatOpened: true }), true)
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

test('resuming a legacy 2K practice session leaves room for True/False questions and help', async () => {
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
  await client.discuss(restored, question, 'Explain this term.', 'help')
  assert.equal(requests.length, 2)
  assert.ok(requests.every(request => request.modelSettings.contextLengthMode === 'auto'))
})

test('document and evidence identity includes collection and document', () => {
  assert.notEqual(studyDocumentKey(source), studyDocumentKey({ ...source, materialId: 'c2' }))
  assert.notEqual(studyDocumentKey(source), studyDocumentKey({ ...source, documentId: 9 }))
  assert.equal(practiceSourceKey(source), 'c1:8:unit.1')
})

test('hints, explanations, and revisions are not mistaken for independent attempts', () => {
  assert.equal(practiceAssisted(question), false)
  for (const change of [{ hintUsed: true }, { answerRevealed: true }, { attempts: [{ answer: 'first' }] }, { discussions: [{ answer: 'Help' }] }]) {
    assert.equal(practiceAssisted({ ...question, ...change }), true)
  }
  assert.equal(practiceReviewLabel(question), 'Not answered')
  assert.equal(practiceReviewLabel({ ...question, answerRevealed: true }), 'Explanation viewed')
  assert.equal(practiceReviewLabel({ ...question, attempts: [{ feedback: { verdict: 'partial', checks: [{ status: 'missing' }] } }] }), '1 point to explain')
  assert.equal(practiceReviewLabel({ ...question, attempts: [{ feedback: { verdict: 'correct' }, assisted: true }] }), 'Key points covered')
})

test('discussion uses the submitted answer and feedback, never unrevealed explanations or hints', () => {
  assert.equal(practiceReference(session, question).feedback, '')
  const answered = { ...question, draft: 'Unsubmitted revision', attempts: [{ answer: 'Submitted answer', feedback: 'Actual feedback' }] }
  const reference = practiceReference(session, answered)
  assert.equal(reference.answer, 'Submitted answer')
  assert.deepEqual(reference.sources, [source])
  assert.deepEqual(reference.documents, session.documents)
  assert.deepEqual(practiceDiscussionPassage(reference), { messageId: 'q1', role: 'assistant',
    question: 'Practice question: Why keep a path?\n\nMy answer: Submitted answer', text: 'Actual feedback' })
})

test('one question call prepares the question, saved criteria, hint and explanation', async () => {
  const { calls, client } = harness()
  const result = await client.question({ ...session, questions: [question] }, 'r1', new AbortController().signal)
  assert.deepEqual(calls[0], ['sources', session.documents, ['c1:8:unit.1'], 1])
  const request = calls[1][1]
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
  assert.equal(result.hint, packageData.hint)
  assert.deepEqual(result.attempts, [])
  assert.equal(calls.length, 2)
  assert.equal(client.help, undefined)
})

test('practice generates only source-backed True/False statements from the first question onward', async () => {
  const { calls, client } = harness()
  const questions = []
  for (let index = 0; index < 6; index++) {
    questions.push(await client.question({ ...session, questions }, `r${index}`, new AbortController().signal))
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

test('cancellation during passage loading prevents model calls', async () => {
  let release
  const { client, calls } = harness({ practiceSources: () => new Promise(resolve => { release = resolve }) })
  const controller = new AbortController()
  const pending = client.question(session, 'cancel', controller.signal)
  controller.abort(); release([source])
  await assert.rejects(pending, { name: 'AbortError' })
  assert.equal(calls.length, 0)
})

test('invalid model output fails explicitly; older questions never get invented rubrics', async () => {
  const empty = harness({ practiceSources: async () => [] })
  assert.equal(await empty.client.question(session, 'r', new AbortController().signal), null)
  for (const reply of [{ text: '', sources: [source] }, { text: 'Question?', sources: [] },
    { text: JSON.stringify(packageData), sources: [source] }]) {
    const { client } = harness({ sendChatMessage: async () => reply })
    await assert.rejects(client.question(session, 'r', new AbortController().signal), /valid question/)
  }
})

test('open-ended practice bounds question history while retaining all used source identities', async () => {
  const { calls, client } = harness()
  const questions = Array.from({ length: 80 }, (_, index) => ({ ...question, id: `q${index}`, text: `Question ${index}` }))
  await client.question({ ...session, questions }, 'r', new AbortController().signal)
  assert.equal(calls[0][2].length, 80)
  const input = JSON.parse(calls[1][1].prompt)
  assert.equal(input.questionNumber, 81)
  assert.equal(input.totalQuestions, undefined)
  assert.equal(input.previousQuestions.length, 12)
  assert.equal(input.previousQuestions[0], 'Question 68')
})

test('inline help is not grading and does not reveal the hidden answer or unrelated exercises', async () => {
  const { calls, client } = harness({ sendChatMessage: async request => {
    calls.push(['help', request]); return { text: 'Explanation', sources: [source] }
  } })
  const withHistory = { ...question, discussions: Array.from({ length: 50 }, (_, index) => ({
    id: `help${index}`, question: `Help ${index}`, answer: index === 49 ? 'Last explanation' : 'OLD UNRELATED HELP'
  })) }
  await client.discuss({ ...session, questions: [question, { ...question, text: 'OTHER EXERCISE' }] }, withHistory, 'Why is that needed?', 'help')
  const request = calls[0][1]
  assert.equal(calls.length, 1)
  assert.equal(request.practiceTask, undefined)
  assert.equal(request.prompt, 'Why is that needed?')
  assert.equal(request.applicationSettings.suggestionMode, 'off')
  assert.deepEqual(request.messages, [])
  assert.deepEqual(request.retrievedSources, [source])
  const body = JSON.stringify(request)
  assert.match(body, /Why keep a path/)
  assert.match(body, /Help 49/)
  assert.match(body, /Last explanation/)
  assert.doesNotMatch(body, /OLD UNRELATED HELP|OTHER EXERCISE/)
  assert.ok(!body.includes(packageData.explanation))
  assert.ok(!body.includes(packageData.hint))
  assert.equal(request.rubric, undefined)
})

test('help context stays bounded and retains an explicit quotation', async () => {
  const { calls, client } = harness({ sendChatMessage: async request => {
    calls.push(request); return { text: 'Explanation', sources: [source] }
  } })
  await client.discuss(session, { ...question,
    helpPassage: { messageId: 'q1', role: 'assistant', text: 'Selected phrase' },
    discussions: [{ question: 'Explain', answer: 'Long context '.repeat(2000) }] }, 'Why?', 'help')
  assert.equal(calls[0].selectedPassage.text, 'Selected phrase')
  assert.ok(calls[0].selectedPassage.question.length < 4100)
  await assert.rejects(client.discuss({ ...session, modelId: 'missing' }, question, 'Why?', 'help'), /unavailable/)
})

test('timeline keeps feedback, help and later revisions in order with stable selectable messages', () => {
  const value = { ...question, attempts: [{ id: 'a1', answer: 'First', feedback: 'Partial' }, { id: 'a2', answer: 'Revision', feedback: 'Correct' }],
    discussions: [{ id: 'd1', question: 'Explain?', answer: 'Help' }], hint: 'Hint', hintUsed: true,
    timeline: [{ kind: 'attempt', id: 'a1' }, { kind: 'discussion', id: 'd1' }, { kind: 'hint' }, { kind: 'attempt', id: 'a2' }] }
  assert.deepEqual(practiceTimeline(value), value.timeline)
  assert.deepEqual(practiceMessages(value).map(message => message.text), ['Why keep a path?', 'First', 'Partial', 'Explain?', 'Help', 'Hint', 'Revision', 'Correct'])
  const messages = practiceMessages(value)
  assert.equal(new Set(messages.map(message => message.id)).size, messages.length)
  assert.equal(practiceTimeline({ ...value, timeline: undefined }).length, 4)
  assert.deepEqual(normalizePracticeState({ mode: 'practice', sessions: [{ ...session, questions: [value] }] }).sessions[0].questions[0], value)
})
