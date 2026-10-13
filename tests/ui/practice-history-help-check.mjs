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
const idle = () => practice.locator('.practice-pending').waitFor({ state: 'hidden' })
const calls = () => page.evaluate(() => window.practiceTest.calls.length)
async function open(suffix = '') {
  await page.goto(`${base}?run=${crypto.randomUUID()}&practice&show-sources${suffix}`)
  await practice.getByLabel('Your answer', { exact: true }).waitFor()
  await idle()
}
async function next(count) {
  await practice.getByRole('button', { name: 'Next question', exact: true }).click()
  await page.waitForFunction(count => window.practiceTest.state().practice.sessions[0].questions.length === count, count)
  await idle()
}
async function checkLayout() {
  await first.locator('.practice-question-footer').scrollIntoViewIfNeeded()
  const message = await first.locator('.practice-message').last().boundingBox()
  const actions = await first.locator('.practice-study-actions').boundingBox()
  assert.ok(actions.y - message.y - message.height >= 10 && actions.y - message.y - message.height <= 14,
    'Help controls should sit close to the last message')
  const sources = await control('1 Source').boundingBox()
  const resume = await first.getByRole('button', { name: 'Continue question 1', exact: true }).boundingBox()
  const separated = resume.x >= sources.x + sources.width + 15 || resume.y >= sources.y + sources.height + 7
  assert.ok(separated, 'Sources and Continue here need spacing, including when wrapped')
  assert.equal(await practice.locator('.message-list').evaluate(el => el.scrollWidth <= el.clientWidth), true)
}
try {
  for (const [name, width, height] of [['desktop', 1440, 1000], ['mobile', 390, 844]]) {
    await page.setViewportSize({ width, height })
    await open()
    await practice.getByLabel('Your answer', { exact: true }).fill('The path lets insertion update a parent after a split.')
    await practice.getByRole('button', { name: 'Check answer', exact: true }).click()
    await first.getByRole('region', { name: 'Practice feedback', exact: true }).waitFor()
    await idle()
    assert.equal((await session()).questions[0].attempts[0].assisted, false)
    await next(2)
    await practice.getByLabel('Your answer', { exact: true }).fill('Keep the current draft.')
    await checkLayout()
    await page.screenshot({ path: `${output}/${name}-completed-question.png` })

    const before = await calls()
    await control('Hint').click()
    assert.equal(await first.getByRole('region', { name: 'Hint', exact: true }).isVisible(), true)
    await control('Explanation').focus()
    await control('Explanation').press('Enter')
    assert.equal(await first.getByRole('region', { name: 'Worked explanation' }).isVisible(), true)
    assert.equal(await first.getByRole('region', { name: 'Hint', exact: true }).count(), 0)
    let saved = await session()
    assert.equal(saved.currentIndex, 1, 'Reading earlier help must not change the active question')
    assert.equal(saved.questions[0].hintUsed, true)
    assert.equal(saved.questions[0].answerRevealed, true)
    assert.equal(saved.questions[0].attempts[0].assisted, false, 'Do not retroactively mark a submitted answer assisted')
    assert.equal(Boolean(saved.questions[1].hintUsed || saved.questions[1].answerRevealed), false)
    assert.equal(saved.questions[1].draft, 'Keep the current draft.')
    assert.equal(await calls(), before, 'Reading saved help must not call the model')
    await page.screenshot({ path: `${output}/${name}-earlier-explanation.png` })
    await control('Explanation').click()
    await control('Explanation').click()
    assert.equal((await session()).questions[0].timeline.filter(event => event.kind === 'explanation').length, 1)

    await page.reload()
    await practice.getByLabel('Your answer', { exact: true }).waitFor()
    assert.equal(await control('Hint').isEnabled(), true)
    assert.equal(await control('Explanation').isEnabled(), true)
    assert.equal(await first.getByRole('region', { name: 'Worked explanation' }).count(), 0)
    await control('Explanation').click()
    assert.equal((await session()).currentIndex, 1)
    assert.equal(await practice.getByLabel('Your answer', { exact: true }).inputValue(), 'Keep the current draft.')
    await first.getByRole('button', { name: 'Continue question 1', exact: true }).click()
    await practice.getByRole('button', { name: 'Try again', exact: true }).click()
    await practice.getByLabel('Revise your answer').fill('The path propagates the split to a parent.')
    await practice.getByRole('button', { name: 'Check revision', exact: true }).click()
    await page.waitForFunction(() => window.practiceTest.state().practice.sessions[0].questions[0].attempts.length === 2)
    await idle()
    saved = await session()
    assert.equal(saved.questions[0].attempts[1].assisted, true)
    assert.equal(saved.questions[1].draft, 'Keep the current draft.')
  }

  // Skipped questions also keep help available while another question is loading.
  await page.setViewportSize({ width: 1440, height: 1000 })
  await open('&slow-model')
  await next(2)
  await practice.getByRole('button', { name: 'Next question', exact: true }).click()
  await practice.locator('.practice-pending').waitFor()
  assert.equal(await control('Hint').isEnabled(), true)
  assert.equal(await control('Explanation').isEnabled(), true)
  await control('Hint').click()
  await control('Explanation').click()
  assert.equal(await practice.locator('.practice-pending').isVisible(), true)
  await page.waitForFunction(() => window.practiceTest.state().practice.sessions[0].questions.length === 3)
  await idle()
  const saved = await session()
  assert.equal(saved.currentIndex, 2)
  assert.equal(saved.questions[0].hintUsed, true)
  assert.equal(saved.questions[0].answerRevealed, true)
  assert.equal(saved.questions[0].attempts.length, 0)
  assert.ok(saved.questions.slice(1).every(question => !question.hintUsed && !question.answerRevealed))
  assert.equal(await page.evaluate(() => window.practiceTest.calls.filter(call => call.kind === 'question' || call.kind === 'true_false_question').length), 3)
  assert.deepEqual(errors, [])
  console.log('Practice history passed: help on completed/skipped questions, correct ownership, unchanged drafts and prior grades, assisted revisions, reload, pending generation, desktop/mobile layout.')
  console.log(`Screenshots: ${output}`)
} catch (error) {
  await page.screenshot({ path: `${output}/failure.png` })
  throw error
} finally { await browser.close() }
