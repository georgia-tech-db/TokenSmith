// Regression: optional delayed advance, cancellation, and manual quiz navigation.
import assert from 'node:assert/strict'
import { choosePracticeTopic, idlePractice } from './practice-topic-helpers.mjs'
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright')
const browser = await chromium.launch({ headless: true, executablePath: process.env.CHROME_PATH })
const page = await browser.newPage()
const base = process.env.PRACTICE_UI_URL || 'http://127.0.0.1:5197/tests/ui/practice.html'
const practice = page.locator('.practice-frame')
const button = name => practice.getByRole('button', { name, exact: true })
const session = () => page.evaluate(() => window.practiceTest.state().practice.sessions[0])
const calls = () => page.evaluate(() => window.practiceTest.calls.filter(c => c.kind === 'true_false_question').length)
try {
  for (const viewport of [{ width: 1440, height: 1000 }, { width: 390, height: 844 }]) {
    await page.setViewportSize(viewport)
    await page.goto(`${base}?run=${crypto.randomUUID()}&practice`)
    await choosePracticeTopic(page); await idlePractice(page)
    assert.equal(await button('Next question').isDisabled(), true)
    const first = (await session()).questions[0]
    await button(first.correctAnswer ? 'False' : 'True').click(); await idlePractice(page)
    assert.equal(await calls(), 1)
    assert.equal((await session()).currentIndex, 0)
    await practice.getByText('Incorrect', { exact: true }).waitFor()
    assert.equal(await button('True').isDisabled(), true)
    assert.equal(await practice.locator('.practice-thread-question').count(), 1)
    assert.equal(await practice.locator('.message-avatar').count(), 0)
    await button('Next question').click(); await idlePractice(page)
    assert.equal(await calls(), 2)
    assert.equal((await session()).currentIndex, 1)
    const second = (await session()).questions[1]
    await button(second.correctAnswer ? 'True' : 'False').click(); await idlePractice(page)
    await button('View results').waitFor()
    assert.equal(await calls(), 2)
    await button('Previous').click()
    await practice.getByText('Incorrect', { exact: true }).waitFor()
    await button('Next question').click()
    assert.equal(await calls(), 2)
    await page.reload(); await idlePractice(page)
    await button('View results').click()
    await practice.getByRole('heading', { name: 'Your practice progress' }).waitFor()
    assert.equal(await calls(), 0, 'Reload and review do not generate questions')
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true)
  }
  for (const mode of ['advance', 'stay', 'settings', 'manual']) {
    await page.goto(`${base}?run=${crypto.randomUUID()}&practice&auto-advance`)
    await choosePracticeTopic(page); await idlePractice(page)
    await button('True').click(); await idlePractice(page)
    await button('Stay here').waitFor()
    assert.equal((await session()).currentIndex, 0, 'Feedback remains visible during delay')
    if (mode === 'stay') await button('Stay here').click()
    if (mode === 'settings') {
      await page.getByRole('button', { name: 'Settings', exact: true }).click()
      await page.getByLabel('Automatically advance in Practice', { exact: true }).uncheck()
      await page.getByRole('button', { name: 'Chat', exact: true }).click()
    }
    if (mode === 'manual') await button('Next question').click()
    // Deliberately exceed the delay to catch a canceled timer firing later.
    await page.waitForTimeout(5500)
    await idlePractice(page)
    const advanced = mode === 'advance' || mode === 'manual'
    assert.equal(await calls(), advanced ? 2 : 1)
    assert.equal((await session()).currentIndex, advanced ? 1 : 0)
    if (!advanced) {
      await button('Next question').click(); await idlePractice(page)
    }
    await button('False').click(); await idlePractice(page)
    await button('View results').waitFor()
    assert.equal(await button('Stay here').count(), 0, 'Final answer does not start a timer')
    await page.waitForTimeout(5500)
    assert.equal(await calls(), 2, 'No generation beyond the final question')
  }
  console.log('Quiz navigation: delayed advance, Stay here, settings opt-out, manual Next, completion, reload, desktop/mobile.')
} finally { await browser.close() }
