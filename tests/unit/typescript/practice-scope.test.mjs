import assert from 'node:assert/strict'
import test from 'node:test'
import { requireTranspiledTs } from './ts-module-loader.mjs'
const { createPracticeClient } = requireTranspiledTs('src/renderer/src/practice-client.ts')
const { practiceQuestionPlan, practiceTopicKey } = requireTranspiledTs('src/shared/practice-question-plan.ts')
const { practiceMastery, removePracticeHistory } = requireTranspiledTs('src/shared/practice-mastery.ts')
const { normalizePracticeState, practiceSourceKey } = requireTranspiledTs('src/shared/practice.ts')
const { gradeTrueFalse } = requireTranspiledTs('src/shared/quiz.ts')
const model = { id: 'gemma', status: 'ready' }
const settings = { application: {} }
const topics = ['Alpha', 'Beta', 'Gamma'].map((title, i) => ({ id: title, title, concept: { courseId: 'book', id: title, title },
  sources: Array.from({ length: i + 1 }, (_, n) => ({ materialId: 'book/1', documentId: 1, sourceUnitId: `${title}-${n}`, excerpt: `${title} supported rule ${n}.` })) }))
const session = () => ({ id: 'new', title: 'Chapter', createdAt: '2026-11-13', documents: [{ materialId: 'book/1', documentId: 1 }],
  modelId: model.id, modelSettings: { maxLength: 2048 }, topics, questions: [], currentIndex: 0 })
const signal = () => new AbortController().signal
const inventory = { version: 3, complete: true, indexRevision: 'r1', topics }
function harness(history = []) {
  const requests = []
  const bridge = { loadTopicInventory: async () => inventory, searchLibrary: () => { throw Error('Unexpected retrieval call') },
    sendChatMessage: async request => {
      requests.push(request)
      const source = request.retrievedSources[0]
      return { sources: request.retrievedSources, text: JSON.stringify({ objective: source.excerpt, question: source.excerpt,
        assumptions: [], criteria: [{ id: 'c1', description: source.excerpt, evidence: [{ sourceKey: practiceSourceKey(source), quote: source.excerpt }] }],
        explanation: source.excerpt, correctAnswer: JSON.parse(request.prompt).targetAnswer }) }
    } }
  return { requests, bridge, client: createPracticeClient(bridge, model, settings, { materials: [{ id: 'book/1', collectionRevision: 'r1' }], embeddingModels: [] }, history) }
}
const remember = s => (cursor, _sources, target, topic) => { s.topicSourceCursors = { ...s.topicSourceCursors, [practiceTopicKey(topic)]: cursor }; s.questionTargetAnswer = target }
function answer(s, q, correct) {
  const response = (q.correctAnswer === correct) ? 'True' : 'False'
  q.attempts.push({ id: `a-${q.id}`, answer: response, feedback: gradeTrueFalse(q, response), assisted: false, observationVersion: 1, durationMs: 1000, submittedAt: new Date().toISOString() })
  s.questions.push(q); s.questionTargetAnswer = undefined
}

test('mixed scope covers every topic, honors unequal limits, resumes and uses one call per question', async () => {
  const { client, requests } = harness()
  let s = await client.prepareTopic(session(), signal())
  assert.equal(practiceQuestionPlan(s).totalQuestions, 12)
  for (let i = 0; i < 12; i++) {
    const q = await client.question(s, String(i), signal(), remember(s))
    assert(q)
    assert.equal(q.concept.id, JSON.parse(requests[i].prompt).topic)
    assert(requests[i].retrievedSources.every(source => source.excerpt.startsWith(q.concept.title)))
    answer(s, q, i % 2 === 0)
    s = normalizePracticeState(JSON.parse(JSON.stringify({ sessions: [s] }))).sessions[0]
  }
  assert.equal(new Set(s.questions.slice(0, 3).map(q => q.concept.id)).size, 3)
  assert.deepEqual(topics.map(t => s.questions.filter(q => q.concept.id === t.id).length), [2, 4, 6])
  assert.equal(await client.question(s, 'finished', signal()), null)
  assert.equal(requests.length, 12)
})

test('correct and incorrect answers change only their assigned topic; deletion preserves all topic mastery', async () => {
  const { client } = harness()
  const s = session()
  for (const correct of [true, false, true]) {
    const before = practiceMastery([s])
    const q = await client.question(s, crypto.randomUUID(), signal(), remember(s))
    answer(s, q, correct)
    const after = practiceMastery([s])
    const changed = [...after].filter(([key, value]) => JSON.stringify(value) !== JSON.stringify(before.get(key))).map(([key]) => key)
    assert.deepEqual(changed, [JSON.stringify([q.concept.courseId, q.concept.id])])
    assert.equal(after.get(changed[0]).correct, Number(correct))
  }
  const before = practiceMastery([s])
  const cleared = removePracticeHistory({ mode: 'practice', sessions: [s], activeSessionId: s.id })
  assert.deepEqual(practiceMastery(cleared.sessions, [], cleared.masteryRecords), before)
})

test('selection prioritizes coverage, then lower mastery; other collections cannot receive credit', () => {
  const s = session()
  const progress = new Map(topics.map((t, i) => [practiceTopicKey(t), { score: i === 1 ? .1 : .9 }]))
  assert.equal(practiceQuestionPlan(s, undefined, progress).topic.id, 'Beta')
  s.questions.push({ id: 'beta', concept: topics[1].concept, sources: topics[1].sources, focusSourceKey: practiceSourceKey(topics[1].sources[0]), attempts: [] })
  assert.notEqual(practiceQuestionPlan(s, undefined, progress).topic.id, 'Beta')
  s.topics = [{ ...topics[0], concept: { ...topics[0].concept, courseId: 'different-book' } }]
  assert.equal(practiceQuestionPlan(s).totalQuestions, 2)
  assert.equal(practiceQuestionPlan(s).topic.concept.courseId, 'different-book')
})

test('mixed-session history follows question topic and shared cited passage without leaking answers', async () => {
  const old = session(); old.id = 'old'; old.createdAt = '2026-11-12'
  const shared = topics[0].sources[0]
  old.questions.push({ id: 'old-q', concept: topics[1].concept, text: 'Prior related question', sources: [shared], focusSourceKey: practiceSourceKey(shared), attempts: [],
    draft: 'Private answer', rubric: { objective: 'Earlier objective', criteria: [{ description: 'Earlier idea', evidence: [{ sourceKey: practiceSourceKey(shared), quote: shared.excerpt }] }] } })
  const { client, requests } = harness([old])
  const s = session()
  await client.question(s, 'shared', signal())
  let prompt = JSON.parse(requests[0].prompt)
  assert.equal(prompt.previousQuestionsFromPassage[0].question, 'Prior related question')
  assert(!requests[0].prompt.includes('Private answer'))
  s.topics = [topics[1]]
  await client.question(s, 'topic', signal())
  prompt = JSON.parse(requests[1].prompt)
  assert.deepEqual(prompt.previousTestedIdeas, ['Earlier objective', 'Earlier idea'])
  assert.deepEqual(prompt.previousQuestions, ['Prior related question'])
})

test('unavailable scope, cancellation and failed generation never make extra calls or consume a slot', async () => {
  const { client, bridge, requests } = harness()
  await assert.rejects(client.prepareTopic({ ...session(), topics: [{ ...topics[0], concept: { ...topics[0].concept, id: 'removed' } }] }, signal()), /unavailable/)
  const s = session(), before = practiceQuestionPlan(s).totalQuestions
  const controller = new AbortController(); controller.abort()
  await assert.rejects(client.question(s, 'cancel', controller.signal), { name: 'AbortError' })
  assert.equal(requests.length, 0)
  let calls = 0
  bridge.sendChatMessage = async () => { calls++; throw Error('offline') }
  await assert.rejects(client.question(s, 'failed', signal(), remember(s)), /offline/)
  assert.equal(calls, 1); assert.equal(s.questions.length, 0)
  assert.equal(practiceQuestionPlan(s).totalQuestions, before)
  assert.equal(practiceQuestionPlan({ ...s, documents: [] }).source, undefined)
})
