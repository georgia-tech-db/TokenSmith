import assert from 'node:assert/strict'
import test from 'node:test'
import { requireTranspiledTs } from './ts-module-loader.mjs'
const { createAnswerSourcePreview, questionForSourceAnswer } = requireTranspiledTs('src/renderer/src/answer-source-preview.ts')
function deferred() { let resolve; const promise = new Promise(r => {resolve=r}); return {promise,resolve} }

test('opens the first source once and closes when the answer arrives', async () => {
  const changes=[]; const preview=createAnswerSourcePreview(value=>changes.push(value))
  preview.begin('answer')
  let loads=0
  await preview.open('answer',async()=>{loads++;return 'first source'})
  await preview.open('answer',async()=>{loads++;return 'second source'})
  assert.equal(changes.at(-1),'first source')
  assert.equal(loads,1)
  preview.finish('answer')
  assert.equal(changes.at(-1),null)
  await preview.open('answer',async()=>{loads++;return 'too late'})
  assert.equal(loads,1)
})
test('a source that finishes loading after the answer cannot reopen the preview', async () => {
  const changes=[]; const preview=createAnswerSourcePreview(value=>changes.push(value)),load=deferred()
  preview.begin('answer')
  const pending=preview.open('answer',()=>load.promise)
  preview.finish('answer')
  load.resolve('late PDF')
  await pending
  assert.ok(changes.every(value=>value===null))
})
test('dismissal, navigation and Stop invalidate a pending preview without reopening it', async () => {
  const changes=[]; const preview=createAnswerSourcePreview(value=>changes.push(value)),load=deferred()
  preview.begin('answer')
  const pending=preview.open('answer',()=>load.promise)
  preview.dismiss()
  load.resolve('late Markdown')
  await pending
  await preview.open('answer',async()=> 'reopened')
  assert.ok(changes.every(value=>value===null))
})
test('old answer completion and out-of-order source loads cannot close a newer preview', async () => {
  const changes=[];const preview=createAnswerSourcePreview(value=>changes.push(value)),old=deferred()
  preview.begin('old')
  const pending=preview.open('old',()=>old.promise)
  preview.begin('new')
  await preview.open('new',async()=> 'new source')
  preview.finish('old')
  old.resolve('old source')
  await pending
  assert.equal(changes.at(-1),'new source')
})
test('answer completion leaves a manually chosen source alone', async () => {
  let displayed;const preview=createAnswerSourcePreview(value=>{displayed=value})
  preview.begin('answer')
  await preview.open('answer',async()=> 'automatic')
  preview.dismiss()
  displayed='manual source'
  preview.finish('answer')
  assert.equal(displayed,'manual source')
})
test('source load failure is optional and does not interrupt generation', async () => {
  const changes=[];const preview=createAnswerSourcePreview(value=>changes.push(value))
  preview.begin('answer')
  await preview.open('answer',async()=>{throw new Error('Missing source')})
  assert.equal(changes.at(-1),null)
  preview.finish('answer')
})


test('keeping a visible passage open preserves it after completion until explicit dismissal', async () => {
  const changes = []; const preview = createAnswerSourcePreview(value => changes.push(value))
  preview.begin('answer')
  await preview.open('answer', async () => 'passage')
  assert.equal(preview.keepOpen('answer'), true)
  const updates = changes.length
  preview.finish('answer')
  assert.equal(changes.length, updates)
  assert.equal(changes.at(-1), 'passage')
  await preview.open('answer', async () => 'late source')
  assert.equal(changes.at(-1), 'passage')
  preview.dismiss()
  assert.equal(changes.at(-1), null)
})
test('a keep-open request cannot retain a hidden or unrelated source', async () => {
  const changes = []; const preview = createAnswerSourcePreview(value => changes.push(value)), load = deferred()
  preview.begin('answer')
  const pending = preview.open('answer', () => load.promise)
  assert.equal(preview.keepOpen('answer'), false)
  preview.finish('answer')
  load.resolve('too late')
  await pending
  assert.ok(changes.every(value => value === null))
  preview.begin('new')
  await preview.open('new', async () => 'new passage')
  assert.equal(preview.keepOpen('answer'), false)
  preview.finish('new')
  assert.equal(changes.at(-1), null)
})
test('starting another question resets keep-open without letting the old completion close its preview', async () => {
  let displayed; const preview = createAnswerSourcePreview(value => { displayed = value })
  preview.begin('old')
  await preview.open('old', async () => 'old passage')
  preview.keepOpen('old')
  preview.begin('new')
  assert.equal(displayed, null)
  await preview.open('new', async () => 'new passage')
  preview.finish('old')
  assert.equal(displayed, 'new passage')
  preview.finish('new')
  assert.equal(displayed, null)
})

test('a historical source uses its own preceding question, not the latest question', () => {
  const firstAnswer = { id: 'a1', role: 'assistant', text: 'First answer' }
  const messages = [
    { id: 'q1', role: 'user', text: 'Why flush clean pages?' }, firstAnswer,
    { id: 'q2', role: 'user', text: 'What is implicit pinning?' },
    { id: 'a2', role: 'assistant', text: 'Second answer' }
  ]
  assert.equal(questionForSourceAnswer(messages, firstAnswer), 'Why flush clean pages?')
  assert.equal(questionForSourceAnswer(messages, messages[3]), 'What is implicit pinning?')
})
test('saved answer context takes precedence, and a missing source owner does not borrow a question', () => {
  const question = { id: 'q1', role: 'user', text: 'Displayed question' }
  const answer = { id: 'a1', role: 'assistant', text: 'Answer', answerContext: { prompt: 'Original request' } }
  assert.equal(questionForSourceAnswer([question, answer], answer), 'Original request')
  assert.equal(questionForSourceAnswer([question]), undefined)
  assert.equal(questionForSourceAnswer([question], { id: 'missing', role: 'assistant', text: '' }), undefined)
})
