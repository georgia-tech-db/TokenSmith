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
const idle = () => practice.locator('.practice-pending').waitFor({ state: 'hidden' })
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
try {
  for (const [name, width, height] of [['desktop', 1440, 1000], ['mobile', 390, 844]]) {
    await page.setViewportSize({ width, height })
    await open()
    await practice.getByLabel('Your answer', { exact: true }).fill('The path lets insertion backtrack.')
    await practice.getByRole('button', { name: 'Check answer', exact: true }).click()
    await active.getByRole('region', { name: 'Practice feedback' }).waitFor()
    await idle()
    assert.equal(await active.locator('.practice-verdict').count(), 0, 'No overall verdict on a written answer')
    assert.equal(await active.getByText('What you got right', { exact: true }).isVisible(), true)
    assert.equal(await active.getByText('What to explain', { exact: true }).isVisible(), true)
    assert.doesNotMatch(await practice.innerText(), /Partly there|Needs another look/)
    await page.screenshot({ path: `${output}/${name}-written-feedback.png` })
    await next(2)
    await practice.getByLabel('Your answer', { exact: true }).fill('Keep this written draft.')
    await next(3)
    assert.equal((await state()).questions[2].format, 'true_false')
    assert.match(await active.locator('.practice-question-messages .message-text').first().innerText(), /^True\/False: /)
    assert.equal(await active.getByText('True or false?', { exact: true }).count(), 0)
    assert.equal(await composer.locator('textarea').count(), 0)
    assert.equal(await composer.getByRole('radio', { checked: true }).count(), 0, 'No default answer')
    assert.equal(await composer.getByRole('button', { name: 'Check answer' }).isDisabled(), true)
    assert.equal(await active.locator('.source-chip').count(), 0, 'Sources do not reveal the answer before an attempt')
    assert.equal(await active.getByText(/Record each internal node/).count(), 0)

    await composer.getByRole('radio', { name: 'True', exact: true }).focus()
    await composer.getByRole('radio', { name: 'True', exact: true }).press('ArrowRight')
    assert.equal(await composer.getByRole('radio', { name: 'False', exact: true }).isChecked(), true)
    await page.screenshot({ path: `${output}/${name}-true-false-choice.png` })
    await composer.getByRole('button', { name: 'Ask for help', exact: true }).click()
    await practice.getByLabel('Question about this exercise').fill('Keep this help draft.')
    await composer.getByRole('button', { name: 'Answer', exact: true }).click()
    assert.equal(await composer.getByRole('radio', { name: 'False', exact: true }).isChecked(), true)
    await page.reload()
    await composer.getByRole('radio', { name: 'False', exact: true }).waitFor()
    assert.equal(await composer.getByRole('radio', { name: 'False', exact: true }).isChecked(), true)
    assert.equal((await state()).questions[2].correctAnswer, true)
    const before = await calls()
    await composer.getByRole('button', { name: 'Check answer', exact: true }).click()
    await active.getByText('Incorrect', { exact: true }).waitFor()
    assert.equal(await calls(), before, 'Binary checking must not call the model')
    assert.equal(await active.getByText(/Record each internal node/).count(), 1)
    assert.equal(await practice.getByLabel('Question about this exercise').inputValue(), 'Keep this help draft.')
    assert.equal((await state()).questions[2].attempts[0].assisted, false)
    await page.screenshot({ path: `${output}/${name}-true-false-feedback.png` })
    await active.getByRole('button', { name: 'Try again', exact: true }).click()
    await composer.getByRole('radio', { name: 'True', exact: true }).check()
    await composer.getByRole('button', { name: 'Check revision', exact: true }).click()
    await active.getByText('Correct', { exact: true }).waitFor()
    assert.equal(await calls(), before)
    assert.deepEqual((await state()).questions[2].attempts.map(attempt => [attempt.answer, attempt.feedback.correct, attempt.assisted]),
      [['False', false, false], ['True', true, true]])
    await page.reload()
    await practice.getByLabel('Question about this exercise').waitFor()
    assert.equal(await active.getByText('Incorrect', { exact: true }).count(), 1)
    assert.equal(await active.getByText('Correct', { exact: true }).count(), 1)
    await active.getByRole('button', { name: '1 Source', exact: true }).click()
    await practice.locator('.source-tray-card').click()
    await page.getByRole('dialog', { name: 'Source Markdown viewer' }).waitFor()
    await page.getByRole('button', { name: 'Close Markdown viewer' }).click()
    await practice.getByRole('button', { name: 'Close materials', exact: true }).click()
    await practice.locator('.practice-thread-question').nth(1).getByRole('button', { name: 'Continue question 2', exact: true }).click()
    assert.equal(await practice.getByLabel('Your answer', { exact: true }).inputValue(), 'Keep this written draft.')
    await practice.getByRole('button', { name: 'Next question', exact: true }).click()
    await next(4)
    assert.equal((await state()).questions[3].format, 'short_answer')
    assert.equal(await composer.locator('textarea').count(), 1)
    assert.equal(await composer.getByRole('radio').count(), 0)
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
      await composer.getByRole('radio', { name: 'False', exact: true }).waitFor()
      await composer.getByRole('radio', { name: 'False', exact: true }).check()
      const before = await calls()
      await composer.getByRole('button', { name: 'Check answer', exact: true }).click()
      await active.getByText('Correct', { exact: true }).waitFor()
      assert.equal(await calls(), before, 'Saved choices can be checked even when the model is unavailable')
    }
  }

  // A bad answer key or ambiguity is rejected before showing a gradable statement.
  for (const flag of ['ambiguous-true-false', 'bad-true-false-key']) {
    await open(`&${flag}`)
    await next(2)
    await practice.getByRole('button', { name: 'Next question', exact: true }).click()
    await practice.getByRole('alert').waitFor()
    assert.equal((await state()).questions.length, 2)
    assert.equal(await composer.getByRole('radio').count(), 0)
    assert.equal((await state()).questions[1].attempts.length, 0)
  }
  assert.deepEqual(errors, [])
  console.log('Mixed Practice UI passed: specific written feedback, T/F choices, local correct/incorrect checks, revisions, separate drafts, reload, sources, navigation, invalid-key/ambiguity rejection, desktop/mobile.')
  console.log(`Screenshots: ${output}`)
} catch (error) {
  await page.screenshot({ path: `${output}/failure.png` })
  throw error
} finally { await browser.close() }
