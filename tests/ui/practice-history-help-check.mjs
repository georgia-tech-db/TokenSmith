import { choosePracticeTopic, idlePractice, reviewPracticeQuestion, answerThenReview, advancePractice } from './practice-topic-helpers.mjs'
import assert from 'node:assert/strict'
import { mkdir } from 'node:fs/promises'

// Real UI and persistence with deterministic replies; no user data or model calls.
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright')
const browser = await chromium.launch({ headless: true, executablePath: process.env.CHROME_PATH })
const base = process.env.PRACTICE_UI_URL || 'http://127.0.0.1:5188/tests/ui/practice.html'
const output = process.env.PRACTICE_HISTORY_OUTPUT || '/tmp/tokensmith-practice-history-ui'
await mkdir(output, { recursive: true })
const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } })
page.setDefaultTimeout(10000)
const errors = []
page.on('pageerror', error => errors.push(error.message))
const practice = page.locator('.practice-frame')
const questions = practice.locator('.practice-thread-question')
const first = questions.first()
const control = name => first.getByRole('button', { name, exact: true })
const session = () => page.evaluate(() => window.practiceTest.state().practice.sessions[0])
const idle = () => idlePractice(page)
async function open(suffix = '') {
  await page.goto(`${base}?run=${crypto.randomUUID()}&practice&show-sources${suffix}`)
  await choosePracticeTopic(page)
  await practice.getByLabel('Your answer', { exact: true }).waitFor()
  await idle()
}
async function next(count) { await advancePractice(page, count) }
async function checkLayout() {
  await first.locator('.practice-question-footer').scrollIntoViewIfNeeded()
  const message = await first.locator('.practice-message').last().boundingBox()
  const actions = await first.locator('.practice-source-actions').boundingBox()
  assert.ok(actions.y - message.y - message.height >= 10 && actions.y - message.y - message.height <= 14,
    'Source controls should sit close to the last message')
  assert.equal(await practice.getByRole('button', { name: /^Continue question/ }).count(), 0)
  assert.equal(await practice.locator('.message-list').evaluate(el => el.scrollWidth <= el.clientWidth), true)
}
try {
  for (const [name, width, height] of [['desktop', 1440, 1000], ['mobile', 390, 844]]) {
    await page.setViewportSize({ width, height })
    await open()
    await answerThenReview(page, 'True')
    await first.getByRole('region', { name: 'Practice feedback', exact: true }).waitFor()
    await idle()
    assert.equal((await session()).questions[0].attempts[0].assisted, false)
    await next(2)
    await answerThenReview(page, 'False')
    await checkLayout()
    await page.screenshot({ path: `${output}/${name}-completed-question.png` })

    assert.equal(await practice.getByRole('button', { name: /^(Hint|Explanation|Try again)$/ }).count(), 0)
    await page.reload()
    await practice.getByLabel('Your answer', { exact: true }).waitFor()
    assert.equal((await session()).currentIndex, 1)
    assert.equal(await practice.getByRole('button', { name: 'False', exact: true }).evaluate(el => el.getAttribute('aria-pressed') === 'true'), true)
    await practice.getByRole('navigation', { name: 'Questions in this practice' }).getByRole('button', { name: /^Question 1:/ }).click()
    const chat = page.locator('.chat-frame')
    await control('Chat about this question').click()
    await chat.waitFor({ state: 'visible' })
    assert.match(await chat.locator('.message-list').innerText(), /My answer: True/)
    await chat.getByRole('button', { name: 'Practice', exact: true }).click()
    const saved = await session()
    assert.equal(saved.currentIndex, 0)
    assert.equal(saved.questions[0].attempts.length, 1)
    assert.equal(saved.questions[0].attempts[0].assisted, false)
    assert.equal(saved.questions[0].discussions?.length ?? 0, 0)
    assert.equal(saved.questions[1].draft, 'False')
    await page.screenshot({ path: `${output}/${name}-earlier-help.png` })
    await reviewPracticeQuestion(page, 2)
    assert.equal((await session()).currentIndex, 1)
    assert.equal(await practice.getByRole('button', { name: 'False', exact: true }).evaluate(el => el.getAttribute('aria-pressed') === 'true'), true)
  }

  // Removed options must also stay absent while another question is loading.
  await page.setViewportSize({ width: 1440, height: 1000 })
  await open('&slow-model')
  await next(2)
  await practice.getByRole('button', { name: 'True', exact: true }).click()
  await page.waitForFunction(() => document.querySelector('.practice-frame').getAttribute('aria-busy') === 'true')
  assert.equal(await practice.getByRole('button', { name: /^(Hint|Explanation|Try again)$/ }).count(), 0)
  assert.equal(await practice.locator('.practice-pending').count(), 0)
  assert.equal(await page.locator('.generation-status-label').innerText(), 'Preparing question')
  await page.waitForFunction(() => window.practiceTest.state().practice.sessions[0].questions.length === 3)
  await idle()
  const saved = await session()
  assert.equal(saved.currentIndex, 2)
  assert.equal(Boolean(saved.questions[0].hintUsed), false)
  assert.equal(Boolean(saved.questions[0].answerRevealed), false)
  assert.equal(saved.questions[0].attempts.length, 1)
  assert.ok(saved.questions.slice(1).every(question => !question.hintUsed && !question.answerRevealed))
  assert.equal(await page.evaluate(() => window.practiceTest.calls.filter(call => call.kind === 'true_false_question').length), 3)
  assert.deepEqual(errors, [])
  console.log('Practice history passed: Chat copies of completed questions, removed controls, unchanged drafts and prior answers, reload, pending generation, desktop/mobile layout.')
  console.log(`Screenshots: ${output}`)
} catch (error) {
  await page.screenshot({ path: `${output}/failure.png` })
  throw error
} finally { await browser.close() }
