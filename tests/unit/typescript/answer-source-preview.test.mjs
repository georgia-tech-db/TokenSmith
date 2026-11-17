import assert from 'node:assert/strict'
import test from 'node:test'
import { requireTranspiledTs } from './ts-module-loader.mjs'
const { createSourcePane, questionForSourceAnswer } = requireTranspiledTs('src/renderer/src/answer-source-preview.ts')
const sources = [
  { path: '/book.md', title: 'First', excerpt: 'Atomic transactions', lineFrom: 3 },
  { path: '/slides.pdf', title: 'Second', excerpt: 'Durability', pageStart: 2 }
]
function setup() {
  const changes = []
  return { changes, pane: createSourcePane(value => changes.push(value)) }
}
test('completion leaves the passage mounted and rejects late previews', () => {
  const { pane, changes } = setup()
  pane.begin('answer')
  assert.equal(pane.open('answer', sources, 'What is atomicity?'), true)
  const displayed = changes.at(-1), count = changes.length
  pane.finish('answer')
  assert.equal(pane.open('answer', sources), false)
  assert.equal(changes.length, count, 'No publish on completion: preserve DOM and scroll position')
  assert.equal(changes.at(-1), displayed)
  assert.equal(displayed.question, 'What is atomicity?')
})
test('closing prevents pending results from reopening the pane', () => {
  const { pane, changes } = setup()
  pane.begin('answer'); pane.dismiss()
  assert.equal(pane.open('answer', sources), false)
  assert.equal(changes.at(-1), null)
})
test('old completion cannot interfere with a newer question', () => {
  const { pane, changes } = setup()
  pane.begin('old'); pane.open('old', sources)
  pane.begin('new'); pane.finish('old')
  assert.equal(pane.open('old', sources), false)
  assert.equal(pane.open('new', [sources[1]], 'New question'), true)
  assert.equal(changes.at(-1).question, 'New question')
})
test('manual selection and navigation survive completion and new automatic requests', () => {
  const { pane, changes } = setup()
  pane.begin('answer'); pane.open('answer', sources)
  pane.move(1)
  assert.equal(changes.at(-1).index, 1)
  pane.finish('answer'); pane.begin('next')
  assert.equal(pane.open('next', sources), false)
  pane.dismissAutomatic()
  assert.equal(changes.at(-1).index, 1)
  pane.move(1)
  assert.equal(changes.at(-1).index, 1, 'No out of range navigation')
  pane.select(sources[0], sources, 'Historical question')
  assert.equal(changes.at(-1).question, 'Historical question')
  pane.dismiss(); pane.begin('third')
  assert.equal(pane.open('third', sources), true)
})
test('disabling automatic reading dismisses only automatic selection', () => {
  const { pane, changes } = setup()
  pane.begin('answer'); pane.open('answer', sources)
  pane.dismissAutomatic()
  assert.equal(changes.at(-1), null)
  assert.equal(pane.open('answer', sources), false)
  pane.select(sources[1], sources)
  pane.dismissAutomatic()
  assert.equal(changes.at(-1).index, 1)
})
test('empty or unsupported sources never produce an invalid pane', () => {
  const { pane, changes } = setup()
  pane.begin('answer')
  assert.equal(pane.open('answer', []), false)
  assert.equal(pane.open('answer', [{ path: '/video.mp4', excerpt: '' }]), false)
  assert.equal(changes.at(-1), null)
})
test('sources retain the question that produced their answer, including historical turns', () => {
  const messages = [{ id: 'u1', role: 'user', text: 'Old question' }, { id: 'a1', role: 'assistant', text: 'Old answer' },
    { id: 'u2', role: 'user', text: 'New question' }, { id: 'a2', role: 'assistant', text: 'New answer', answerContext: { prompt: 'Exact submitted prompt' } }]
  assert.equal(questionForSourceAnswer(messages, messages[1]), 'Old question')
  assert.equal(questionForSourceAnswer(messages, messages[3]), 'Exact submitted prompt')
  assert.equal(questionForSourceAnswer(messages), undefined)
  assert.equal(questionForSourceAnswer(messages, { id: 'missing' }), undefined)
})
