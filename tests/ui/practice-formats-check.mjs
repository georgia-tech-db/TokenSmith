import { choosePracticeTopic, idlePractice, reviewPracticeQuestion, answerThenReview, advancePractice } from './practice-topic-helpers.mjs'
import assert from 'node:assert/strict'
import { mkdir } from 'node:fs/promises'

// Real components and saved state, deterministic model replies, isolated from user data.
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright')
const browser = await chromium.launch({ headless: true, executablePath: process.env.CHROME_PATH })
const base = process.env.PRACTICE_UI_URL || 'http://127.0.0.1:5188/tests/ui/practice.html'
const output = '/tmp/tokensmith-practice-formats-ui'
await mkdir(output, { recursive: true })
const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } })
page.setDefaultTimeout(10000)
const errors = []
page.on('pageerror', error => errors.push(error.message))
const practice = page.locator('.practice-frame')
const active = practice.locator('.practice-thread-question.is-active')
const composer = practice.locator('.practice-composer')
const state = () => page.evaluate(() => window.practiceTest.state().practice.sessions[0])
const calls = () => page.evaluate(() => window.practiceTest.calls.filter(call => ['question', 'true_false_question', 'feedback', 'discussion'].includes(call.kind)).length)
const idle = () => idlePractice(page)
async function open(suffix = '') {
  await page.goto(`${base}?run=${crypto.randomUUID()}&practice&show-sources${suffix}`)
  await choosePracticeTopic(page)
  await practice.getByLabel('Your answer', { exact: true }).waitFor()
  await idle()
}
async function next(count) { await advancePractice(page, count) }
try {
  for (const [name, width, height] of [['desktop', 1440, 1000], ['mobile', 390, 844]]) {
    await page.setViewportSize({ width, height })
    await open()
    assert.equal((await state()).questions[0].format, 'true_false')
    assert.equal(await practice.getByRole('button', { name: /^(Hint|Explanation|Try again)$/ }).count(), 0)
    assert.match(await active.locator('.practice-question-messages .message-text').first().innerText(), /^True\/False: /)
    assert.equal(await active.getByText('True or false?', { exact: true }).count(), 0)
    assert.equal(await composer.locator('textarea').count(), 0)
    assert.equal(await composer.getByRole('button', { pressed: true }).count(), 0, 'No default answer')
    assert.equal(await active.locator('.source-chip').count(), 0, 'Sources do not reveal the answer before an attempt')
    assert.equal(await active.getByText(/Record each internal node/).count(), 0)

    const before = await calls()
    await answerThenReview(page, 'False')
    await active.getByText('Incorrect', { exact: true }).waitFor()
    assert.equal(await calls(), before + 1, 'Only preparing the next question calls the model')
    assert.equal(await active.getByText(/Record each internal node/).count(), 1)
    assert.equal((await state()).questions[0].attempts[0].assisted, false)
    await page.screenshot({ path: `${output}/${name}-true-false-feedback.png` })
    assert.equal(await practice.getByRole('button', { name: /^(Hint|Explanation|Try again|Check revision|Answer)$/ }).count(), 0)
    assert.equal(await composer.getByRole('button', { name: 'True', exact: true }).isDisabled(), true)
    assert.equal(await calls(), before + 1)
    assert.deepEqual((await state()).questions[0].attempts.map(attempt => [attempt.answer, attempt.feedback.correct, attempt.assisted]),
      [['False', false, false]])
    await page.reload()
    await practice.getByRole('button', { name: 'Chat about this question', exact: true }).waitFor()
    assert.equal(await active.getByText('Incorrect', { exact: true }).count(), 1)
    assert.equal(await active.getByText('Correct', { exact: true }).count(), 0)
    await active.getByRole('button', { name: '1 Source', exact: true }).click()
    await practice.locator('.source-tray-card').click()
    await page.getByRole('dialog', { name: 'Source Markdown viewer' }).waitFor()
    await page.getByRole('button', { name: 'Close Markdown viewer' }).click()
    await practice.getByRole('button', { name: 'Close Library', exact: true }).click()
    await next(2)
    await answerThenReview(page, 'False')
    await next(3)
    await practice.getByRole('navigation', { name: 'Questions in this practice' }).getByRole('button', { name: /^Question 2:/ }).click()
    assert.equal(await composer.getByRole('button', { name: 'False', exact: true }).evaluate(el => el.getAttribute('aria-pressed') === 'true'), true)
    await reviewPracticeQuestion(page, 3)
    await next(4)
    assert.ok((await state()).questions.every(question => question.format === 'true_false'))
    assert.equal(await composer.locator('textarea').count(), 0)
    assert.equal(await composer.getByRole('button', { name: /^(True|False)$/ }).count(), 2)
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true)
    if (name === 'desktop') {
      await next(5)
      await next(6)
      assert.equal((await state()).questions[5].correctAnswer, false, 'False must be a valid saved answer key')
      await page.evaluate(async () => {
        const saved = window.practiceTest.state()
        saved.models[0].status = 'missing'
        await window.tokensmith.saveAppState(saved)
      })
      await page.reload()
      await composer.getByRole('button', { name: 'False', exact: true }).waitFor()
      await answerThenReview(page, 'False')
      const before = await calls()
      await active.getByText('Correct', { exact: true }).waitFor()
      assert.equal(await calls(), before, 'Saved choices can be checked even when the model is unavailable')
    }
  }

  // Old questions and attempts stay readable, without any grading controls.
  await page.setViewportSize({ width: 1440, height: 1000 })
  await open()
  await page.evaluate(async () => {
    const saved = window.practiceTest.state()
    const session = saved.practice.sessions[0]
    const binary = session.questions[0]
    const old = { ...binary, id: 'legacy', format: 'short_answer', text: 'Explain ancestor paths.',
      draft: 'My saved written draft.', composerMode: 'answer', timeline: undefined,
      attempts: [{ id: 'old-attempt', answer: 'My earlier answer.', feedback: 'My saved feedback.', assisted: false, durationMs: 100 }] }
    session.questions = [old, { ...old, id: 'legacy-unanswered', format: undefined, attempts: [] }, binary]
    session.currentIndex = 0
    await window.tokensmith.saveAppState(saved)
  })
  await page.reload()
  await practice.getByRole('button', { name: 'Chat about this question', exact: true }).waitFor()
  assert.equal(await composer.getByRole('button', { name: 'Answer', exact: true }).count(), 0)
  assert.equal(await practice.getByRole('button', { name: /Check answer|Check revision|Try again/ }).count(), 0)
  assert.equal(await active.getByText('My earlier answer.', { exact: true }).isVisible(), true)
  assert.equal(await active.getByText('My saved feedback.', { exact: true }).isVisible(), true)
  assert.equal((await state()).questions[0].draft, 'My saved written draft.')
  const beforeNext = await calls()
  await reviewPracticeQuestion(page, 3)
  assert.equal((await state()).currentIndex, 2, 'Next skips archived short-answer exercises')
  assert.equal(await calls(), beforeNext, 'An existing T/F question needs no new generation')

  // Archived written answers remain readable; another topic starts current T/F practice.
  const legacyId = (await state()).id
  await practice.getByRole('button', { name: 'Practice another topic', exact: true }).click()
  await choosePracticeTopic(page)
  await practice.getByLabel('Your answer', { exact: true }).waitFor()
  await idle()
  assert.equal((await state()).questions[0].format, 'true_false')
  assert.equal(await page.evaluate(id => window.practiceTest.state().practice.sessions.find(s => s.id === id).questions[0].attempts[0].answer, legacyId), 'My earlier answer.')

  // A bad answer key or ambiguity is rejected before showing a gradable statement.
  for (const flag of ['ambiguous-true-false', 'bad-true-false-key']) {
    await page.goto(`${base}?run=${crypto.randomUUID()}&practice&${flag}`)
    await choosePracticeTopic(page)
    await practice.getByRole('alert').waitFor()
    assert.equal((await state()).questions.length, 0)
    assert.equal(await composer.getByRole('radio').count(), 0)
    assert.equal(await practice.getByRole('button', { name: 'Retry question', exact: true }).isVisible(), true)
  }
  assert.deepEqual(errors, [])
  console.log('True/False-only Practice UI passed: legacy history without grading, T/F choices, local correct/incorrect checks, single attempts, separate drafts, reload, sources, navigation, invalid-key/ambiguity rejection, desktop/mobile.')
  console.log(`Screenshots: ${output}`)
} catch (error) {
  await page.screenshot({ path: `${output}/failure.png` })
  throw error
} finally { await browser.close() }
