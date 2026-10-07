import assert from 'node:assert/strict'
import test from 'node:test'
import { requireTranspiledTs } from './ts-module-loader.mjs'

const { javascriptRouteForQuestion } = requireTranspiledTs('src/main/engine/javascript-routing.ts')

test('clear source lookups, counts, and one operation answer directly', () => {
  assert.equal(javascriptRouteForQuestion('hi', false), 'direct')
  assert.equal(javascriptRouteForQuestion('Hello there!', false), 'direct')
  assert.equal(javascriptRouteForQuestion('Thanks.', false), 'direct')
  assert.equal(javascriptRouteForQuestion('Which editions were published after 2000, and how many are there?', true), 'direct')
  assert.equal(javascriptRouteForQuestion('According to the textbook, how many editions are listed?', true), 'direct')
  assert.equal(javascriptRouteForQuestion('How many entries are in this list: red, blue, green?', false), 'direct')
  assert.equal(javascriptRouteForQuestion('What is 6 times 7?', false), 'direct')
  assert.equal(javascriptRouteForQuestion('120 ÷ 3', false), 'direct')
  assert.equal(javascriptRouteForQuestion('Explain the formula for compound interest.', true), 'direct')
  assert.equal(javascriptRouteForQuestion('According to the notes, what is the formula for average speed?', true), 'direct')
})

test('derived values and repeated calculations use the computation tool', () => {
  assert.equal(javascriptRouteForQuestion('Hi, calculate the average of 3, 5, and 7.', false), 'tool')
  assert.equal(javascriptRouteForQuestion('Calculate the average of these values and show each difference.', false), 'tool')
  assert.equal(javascriptRouteForQuestion('List consecutive percentage changes and their arithmetic mean.', true), 'tool')
  assert.equal(javascriptRouteForQuestion('Compute the weighted contributions and overall score.', true), 'tool')
  assert.equal(javascriptRouteForQuestion('Convert three temperatures and calculate their average.', false), 'tool')
})

test('unclear intent is left to the model', () => {
  assert.equal(javascriptRouteForQuestion('How has enrollment changed?', true), 'model')
  assert.equal(javascriptRouteForQuestion('Compare these options.', false), 'model')
})
