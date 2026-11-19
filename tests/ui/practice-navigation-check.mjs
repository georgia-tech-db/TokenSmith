import assert from 'node:assert/strict'
import { choosePracticeTopic, idlePractice } from './practice-topic-helpers.mjs'
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright')
const browser = await chromium.launch({ headless: true, executablePath: process.env.CHROME_PATH })
try {
  for (const width of [1440, 390]) {
    const page = await browser.newPage({ viewport: { width, height: 1000 } })
    const errors = []
    page.on('pageerror', error => errors.push(error.message))
    await page.goto(`${process.env.PRACTICE_UI_URL || 'http://127.0.0.1:5188/tests/ui/practice.html'}?run=${crypto.randomUUID()}`)
    const nav = page.getByRole('navigation', { name: 'Primary navigation' })
    await page.locator('.chat-frame').waitFor()
    assert.equal(await page.getByRole('group', { name: 'Study mode' }).count(), 0)
    await nav.getByRole('button', { name: 'Practice', exact: true }).click()
    await page.locator('.practice-frame').waitFor()
    assert.equal(await page.locator('.chat-frame').count(), 0)
    assert.equal(await nav.getByRole('button', { name: 'Practice', exact: true }).getAttribute('aria-current'), 'page')
    await choosePracticeTopic(page)
    await idlePractice(page)
    await page.getByRole('button', { name: 'Chat about this question', exact: true }).click()
    await page.locator('.chat-frame').waitFor()
    assert.equal(await page.locator('.practice-frame').count(), 0)
    await page.getByLabel('Message', { exact: true }).fill('Explain the ancestor path')
    await page.getByRole('button', { name: 'Send message', exact: true }).click()
    await page.getByRole('button', { name: 'Practice this topic', exact: true }).first().click()
    await page.locator('.practice-frame').waitFor()
    await idlePractice(page)
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true)
    assert.deepEqual(errors, [])
    await page.close()
  }
  console.log('Practice navigation passed at desktop and mobile widths')
} finally { await browser.close() }
