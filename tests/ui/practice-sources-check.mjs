import { choosePracticeTopic, idlePractice, reviewPracticeQuestion, answerThenReview, advancePractice } from './practice-topic-helpers.mjs'
import assert from 'node:assert/strict'
import { mkdir } from 'node:fs/promises'

// Real UI and reader, deterministic sources and replies. No user data or model calls.
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright')
const browser = await chromium.launch({ headless: true, executablePath: process.env.CHROME_PATH })
const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } })
page.setDefaultTimeout(10000)
const base = process.env.PRACTICE_UI_URL || 'http://127.0.0.1:5188/tests/ui/practice.html'
const output = '/tmp/tokensmith-practice-sources-ui'
await mkdir(output, { recursive: true })
const errors = []
page.on('pageerror', error => errors.push(error.message))
const practice = page.locator('.practice-frame')
const materials = practice.getByRole('complementary', { name: 'Library' })
const cards = materials.locator('.source-tray-card')
const button = name => practice.getByRole('button', { name, exact: true })
const reader = page.getByRole('dialog', { name: 'Source Markdown viewer' })
const state = () => page.evaluate(() => window.practiceTest.state().practice)
const current = async () => { const value = await state(); const session = value.sessions.find(s => s.id === value.activeSessionId); return session.questions[session.currentIndex] }
const open = async (flags = '&show-sources') => {
  await page.goto(`${base}?run=${crypto.randomUUID()}&practice${flags}`)
  await choosePracticeTopic(page)
  await practice.getByLabel('Your answer', { exact: true }).waitFor()
}
const reveal = async () => {
  await answerThenReview(page, 'True')
  await practice.getByRole('region', { name: 'Practice feedback' }).last().waitFor()
}
const next = async () => {
  const value = await state()
  const session = value.sessions.find(s => s.id === value.activeSessionId)
  await advancePractice(page, session.currentIndex + 2)
}
try {
  // Sources appear after a submitted choice, without a reveal button.
  await open()
  assert.equal(await cards.count(), 0)
  assert.equal(await practice.locator('.source-chip').count(), 0)
  assert.equal(await practice.getByRole('button', { name: /^(Hint|Explanation|Try again)$/ }).count(), 0)
  await reveal()
  await cards.first().waitFor()
  assert.equal(await cards.count(), 1)
  assert.match(await cards.innerText(), /Trees[\s\S]*Lines 1-4/)
  await cards.click()
  await reader.waitFor()
  assert.equal(await page.evaluate(() => window.practiceTest.calls.at(-1).source.documentId), 1)
  await page.getByRole('button', { name: 'Close Markdown viewer' }).click()
  await page.screenshot({ path: `${output}/sources-desktop.png` })

  // Moving forward hides the old evidence; reopening an earlier question restores its sources.
  await next()
  assert.equal(await cards.count(), 0)
  await reveal()
  await cards.first().waitFor()
  assert.match(await cards.innerText(), /Trees/)
  await cards.click()
  await reader.waitFor()
  assert.equal(await page.evaluate(() => window.practiceTest.calls.at(-1).source.documentId), 1)
  await page.getByRole('button', { name: 'Close Markdown viewer' }).click()
  await practice.locator('.practice-thread-question').first().getByRole('button', { name: '1 Source', exact: true }).click()
  assert.equal((await state()).sessions[0].currentIndex, 0)
  assert.match(await cards.innerText(), /Trees/)
  await page.reload()
  await cards.first().waitFor()
  assert.match(await cards.innerText(), /Trees/)

  // Reader errors remain in the source panel, retry works, and navigation clears stale errors.
  await open('&show-sources&source-error')
  await reveal()
  await cards.click()
  await materials.getByRole('alert').waitFor()
  assert.match(await materials.getByRole('alert').innerText(), /Test source read failed/)
  assert.equal((await current()).attempts.length, 1)
  await next()
  await reveal()
  assert.equal(await materials.getByRole('alert').count(), 0)
  await cards.click()
  await reader.waitFor()
  await page.getByRole('button', { name: 'Close Markdown viewer' }).click()

  // The reader navigates every passage in the topic question.
  await open('&show-sources&many-sources')
  await reveal()
  assert.equal(await cards.count(), 4)
  await button('4 Sources').click()
  await cards.first().click()
  await reader.waitFor()
  for (let i = 2; i <= 4; i++) {
    await page.getByRole('button', { name: 'Next source', exact: true }).click()
    await reader.getByText(`Source ${i} of 4`, { exact: true }).waitFor()
  }
  assert.equal(await page.getByRole('button', { name: 'Next source', exact: true }).isDisabled(), true)
  await page.getByRole('button', { name: 'Close Markdown viewer' }).click()

  // A late reader response must not open the previous question's document.
  await open('&show-sources&slow-reader')
  await reveal()
  await cards.click()
  await next()
  await page.waitForTimeout(1800)
  assert.equal(await reader.count(), 0)

  // Both modes respect Show sources; disabling it leaves no source buttons or cards.
  await open('')
  await reveal()
  assert.equal(await cards.count(), 0)
  assert.equal(await practice.locator('.source-chip').count(), 0)

  // Chat uses the same cards and reader; changing modes must keep their evidence separate.
  await open()
  await reveal()
  await button('Chat').click()
  const chat = page.locator('.chat-frame')
  await chat.locator('.composer textarea').fill('Why record ancestors during insertion?')
  await chat.locator('.send-button').click()
  await chat.locator('.source-tray-card').first().waitFor()
  await chat.locator('.source-tray-card').first().click()
  await reader.waitFor()
  assert.match(await reader.innerText(), /ancestor path/)
  await page.getByRole('button', { name: 'Close Markdown viewer' }).click()
  await chat.getByRole('button', { name: 'Practice', exact: true }).click()
  await cards.waitFor()
  assert.equal(await cards.count(), 1)
  assert.match(await cards.innerText(), /Trees/)

  // Mobile source button opens Library without changing the checked answer.
  await page.setViewportSize({ width: 390, height: 844 })
  await open()
  await reveal()
  await button('1 Source').click()
  await materials.waitFor({ state: 'visible' })
  await cards.click()
  await reader.waitFor()
  await page.screenshot({ path: `${output}/reader-mobile.png` })
  await page.getByRole('button', { name: 'Close Markdown viewer' }).click()
  await page.screenshot({ path: `${output}/sources-mobile.png` })
  await materials.getByRole('button', { name: 'Close Library' }).click()
  assert.equal((await current()).attempts[0].answer, 'True')
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true)
  assert.deepEqual(errors, [])
  console.log('Practice sources passed: shared cards, MD/TXT reader, source gating, question navigation, reload, errors, full source navigation, stale-response cancellation, settings and mobile.')
} catch (error) {
  await page.screenshot({ path: `${output}/failure.png` })
  console.error(await page.locator('body').innerText())
  throw error
} finally { await browser.close() }
