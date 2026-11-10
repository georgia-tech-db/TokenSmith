import assert from 'node:assert/strict'
import { choosePracticeTopic, idlePractice } from './practice-topic-helpers.mjs'
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright')
const browser = await chromium.launch({ headless: true, executablePath: process.env.CHROME_PATH })
const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } })
page.setDefaultTimeout(10000)
const base = process.env.PRACTICE_UI_URL || 'http://127.0.0.1:5197/tests/ui/practice.html'
const practice = page.locator('.practice-frame')
const errors = []; page.on('pageerror', error => errors.push(error.message))
try {
  await page.goto(`${base}?run=${crypto.randomUUID()}&practice&open-question-first`)
  await choosePracticeTopic(page)
  await practice.getByLabel('Your answer', { exact: true }).waitFor()
  await idlePractice(page)
  assert.equal(await page.evaluate(() => window.practiceTest.calls.filter(c => c.kind === 'true_false_question').length), 2)
  assert.equal(await practice.getByText('If a crash interrupts the table update, what state results?', { exact: true }).count(), 0)
  assert.equal(await practice.getByRole('button', { name: 'True', exact: true }).isEnabled(), true)

  // Persist a previously graded malformed question and reload the real App.
  await page.evaluate(async () => {
    const state = structuredClone(window.practiceTest.state())
    const question = state.practice.sessions[0].questions[0]
    question.text = 'If a crash interrupts the table update, what state results?'
    question.attempts = [{ id: 'old-bad-attempt', answer: 'False', assisted: false, durationMs: 0,
      feedback: { kind: 'true_false', correct: false, explanation: 'An invalid answer guide.' } }]
    question.timeline = [{ kind: 'attempt', id: 'old-bad-attempt' }]
    await window.tokensmith.saveAppState(state)
  })
  await page.reload()
  await practice.getByText(/It cannot be graded/).waitFor()
  assert.equal(await practice.getByText('Incorrect', { exact: true }).count(), 0)
  assert.equal(await practice.getByRole('button', { name: 'False', exact: true }).count(), 0)
  await practice.getByRole('button', { name: 'Prepare another question', exact: true }).waitFor()
  await practice.getByRole('button', { name: 'Chat about this question', exact: true }).click()
  const chat = page.locator('.chat-frame')
  await chat.getByText(/It cannot be graded/).waitFor()
  assert.equal(await chat.getByText('Incorrect', { exact: true }).count(), 0)
  await chat.getByRole('region', { name: 'Suggested practice follow-up questions' }).getByRole('button').first().waitFor()

  await page.goto(`${base}?run=${crypto.randomUUID()}&practice&open-question-always`)
  await choosePracticeTopic(page)
  await practice.getByRole('alert').waitFor()
  await idlePractice(page)
  assert.equal(await page.evaluate(() => window.practiceTest.calls.filter(c => c.kind === 'true_false_question').length), 2)
  assert.equal(await practice.getByRole('button', { name: 'True', exact: true }).count(), 0)
  assert.equal(await page.evaluate(() => window.practiceTest.state().practice.sessions[0].questions.length), 0)
  assert.deepEqual(errors, [])
  console.log('Practice statements passed: one repair attempt, malformed text never exposed as True/False, saved bad grades suppressed, Chat transfer preserves warning and suggestions, repeated invalid output stops safely.')
} finally { await browser.close() }
