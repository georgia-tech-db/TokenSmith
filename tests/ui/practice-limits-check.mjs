import assert from 'node:assert/strict'
import { mkdir } from 'node:fs/promises'
import { choosePracticeTopic, idlePractice, reviewPracticeQuestion } from './practice-topic-helpers.mjs'

const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright')
const browser = await chromium.launch({ headless: true, executablePath: process.env.CHROME_PATH })
const page = await browser.newPage()
page.setDefaultTimeout(10000)
const base = process.env.PRACTICE_UI_URL || 'http://127.0.0.1:5198/tests/ui/practice.html'
const output = '/tmp/tokensmith-practice-limits'
await mkdir(output, { recursive: true })
const errors = []; page.on('pageerror', error => errors.push(error.message))
const practice = page.locator('.practice-frame')
const session = () => page.evaluate(() => window.practiceTest.state().practice.sessions[0])
const modelCalls = () => page.evaluate(() => window.practiceTest.calls.filter(c => c.kind === 'true_false_question'))
try {
  for (const [name, width, height, passages] of [['desktop', 1440, 1000, 1], ['mobile', 390, 844, 1], ['larger-topic', 1440, 1000, 6]]) {
    await page.setViewportSize({ width, height })
    await page.goto(`${base}?run=${crypto.randomUUID()}&practice${passages > 1 ? '&many-sources' : ''}`)
    await choosePracticeTopic(page); await idlePractice(page)
    const limit = passages * 2
    for (let i = 0; i < limit; i++) {
      await page.waitForFunction(count => window.practiceTest.state().practice.sessions[0].questions.length === count, i + 1)
      await idlePractice(page)
      assert.equal(await practice.getByText(/Practice complete\./).count(), 0, 'Completion waits for the last answer')
      await practice.getByLabel('Your answer', { exact: true }).getByRole('button', { name: 'True', exact: true }).click()
    }
    await practice.getByText(`Practice complete. You've finished this topic's ${limit}-question session. Choose Practice another topic to continue.`, { exact: true }).waitFor()
    await idlePractice(page)
    assert.equal((await session()).questions.length, limit)
    assert.equal((await modelCalls()).length, limit, 'No generation call after the last answer')
    assert.equal(await practice.getByRole('button', { name: 'Retry question', exact: true }).count(), 0)
    assert.equal(await practice.getByRole('button', { name: 'Practice another topic', exact: true }).isEnabled(), true)
    const samePassage = JSON.parse((await modelCalls())[passages].request.prompt).previousQuestionsFromPassage
    assert.equal(samePassage.length, 1)
    assert.ok(samePassage[0].testedIdeas.length)
    await reviewPracticeQuestion(page, 1)
    assert.equal((await modelCalls()).length, limit, 'Reviewing old feedback does not generate questions')
    await page.reload(); await idlePractice(page)
    await practice.getByText(/Practice complete\./).waitFor()
    assert.equal((await session()).questions.length, limit)
    assert.equal((await modelCalls()).length, 0, 'Reloading a completed session makes no question call')
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true)
    await page.screenshot({ path: `${output}/${name}.png` })
    await practice.getByRole('button', { name: 'Practice another topic', exact: true }).click()
    await practice.getByLabel('Your topic', { exact: true }).waitFor()
  }
  assert.deepEqual(errors, [])
  console.log('Practice limits passed: 1/6 passages produce 2/12 questions, no completion call, passage history, reload, review, topic switching, desktop/mobile.')
} catch (error) {
  await page.screenshot({ path: `${output}/failure.png` })
  console.error(await page.locator('body').innerText()); throw error
} finally { await browser.close() }
