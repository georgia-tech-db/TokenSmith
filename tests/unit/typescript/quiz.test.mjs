import assert from 'node:assert/strict'
import test from 'node:test'
import { requireTranspiledTs } from './ts-module-loader.mjs'

const {
  quizFeedbackPrompt,
  quizQuestionPrompt,
  quizSourcePoolSize,
  quizSourceLimit,
  quizTotalQuestions
} = requireTranspiledTs('src/shared/quiz.ts')

test('quizQuestionPrompt asks for one source-backed short-answer question only', () => {
  const prompt = quizQuestionPrompt({
    questionNumber: 2,
    previousQuestions: ['What is atomicity?'],
    totalQuestions: 5
  })

  assert.match(prompt, /Create quiz question 2 of 5/)
  assert.match(prompt, /important properties, theorems, conditions, or rules/)
  assert.match(prompt, /short-answer question/)
  assert.match(prompt, /provided PDF excerpts/)
  assert.match(prompt, /Previous quiz questions to avoid repeating/)
  assert.match(prompt, /What is atomicity\?/)
  assert.match(prompt, /Do not repeat/)
  assert.match(prompt, /What is this book on/)
  assert.match(prompt, /document metadata/)
  assert.match(prompt, /tables of contents/)
  assert.match(prompt, /Do not answer/)
  assert.match(prompt, /Do not mention page numbers/)
  assert.match(prompt, /where the answer appears/)
  assert.match(prompt, /depends on knowing a page number/)
  assert.match(prompt, /Return only the question/)
})

test('quizFeedbackPrompt grades without asking the next question', () => {
  const prompt = quizFeedbackPrompt({
    answer: 'It guarantees all-or-nothing updates.',
    question: 'What does atomicity guarantee?',
    questionNumber: 1,
    totalQuestions: 5
  })

  assert.match(prompt, /Grade the student answer/)
  assert.match(prompt, /Address the learner directly/)
  assert.match(prompt, /Do not say "the student"/)
  assert.match(prompt, /Do not start with "Feedback:"/)
  assert.match(prompt, /Good\/Partial\/Needs work/)
  assert.match(prompt, /Expected answer:/)
  assert.match(prompt, /- One concise answer\./)
  assert.match(prompt, /Do not ask a new quiz question/)
  assert.match(prompt, /Do not mention page numbers/)
  assert.match(prompt, /as stated on page 36/)
  assert.match(prompt, /Quiz question 1 of 5: What does atomicity guarantee\?/)
  assert.match(prompt, /Student answer: It guarantees all-or-nothing updates\./)
})

test('quiz constants keep the first version small', () => {
  assert.equal(quizTotalQuestions, 5)
  assert.equal(quizSourceLimit, 4)
  assert.equal(quizSourcePoolSize, 32)
})
