import assert from 'node:assert/strict'
import { mkdir } from 'node:fs/promises'
import { choosePracticeTopic, idlePractice, reviewPracticeQuestion } from './practice-topic-helpers.mjs'

// Production App and reader; controlled replies and documents, no user data or model calls.
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright')
const browser = await chromium.launch({ headless: true, executablePath: process.env.CHROME_PATH })
const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } })
page.setDefaultTimeout(12000)
const base = process.env.PRACTICE_UI_URL || 'http://127.0.0.1:5198/tests/ui/practice.html'
const output = '/tmp/tokensmith-practice-source-preview'
await mkdir(output, { recursive: true })
const errors = []; page.on('pageerror', error => errors.push(error.message))
const practice = page.locator('.practice-frame')
const reader = page.getByRole('dialog', { name: 'Source Markdown viewer' })
const close = () => page.getByRole('button', { name: 'Close Markdown viewer' }).click()
const session = () => page.evaluate(() => window.practiceTest.state().practice.sessions[0])
const open = async (flags = '&show-sources&preview-sources&slow-model') => {
  await page.goto(`${base}?run=${crypto.randomUUID()}&practice${flags}`)
  await choosePracticeTopic(page)
}
try {
  await open()
  await reader.waitFor()
  assert.equal(await practice.getAttribute('aria-busy'), 'true')
  assert.equal(await reader.getAttribute('aria-modal'), 'false')
  const evidence = await page.evaluate(() => {
    const calls = window.practiceTest.calls
    return [calls.find(c => c.kind === 'read-source').source.path,
      calls.find(c => c.kind === 'true_false_question').request.retrievedSources[0].path]
  })
  assert.equal(evidence[0], evidence[1])
  await page.screenshot({ path: `${output}/desktop.png` })
  await idlePractice(page)
  await reader.waitFor({ state: 'hidden' })
  assert.equal((await session()).questions.length, 1)
  assert.equal((await session()).observations.some(e => e.kind === 'help_shown'), false)
  await practice.getByRole('region', { name: 'Sources', exact: true }).waitFor()
  assert.equal(await practice.locator('.source-tray-card').count(), 1, 'Sources are shown before answering, without clicking Source')
  await practice.getByRole('button', { name: 'True', exact: true }).click()
  await page.waitForFunction(() => window.practiceTest.state().practice.sessions[0].questions[0].attempts.length === 1)
  assert.equal((await session()).questions[0].attempts[0].assisted, false)
  assert.equal(await practice.getByText('Answered with help · mastery unchanged', { exact: true }).count(), 0)
  assert.equal(await practice.getByText('With help or revision', { exact: true }).count(), 0)
  await page.reload()
  assert.equal((await session()).questions[0].attempts[0].assisted, false)
  console.log('PASS source matches generation, closes on completion, source preview does not count as help after reload')

  await open()
  await reader.waitFor(); await close()
  await idlePractice(page)
  assert.equal(await reader.count(), 0)
  assert.equal(await page.evaluate(() => window.practiceTest.calls.filter(c => c.kind === 'read-source').length), 1)
  console.log('PASS dismissal stays closed until generation finishes')

  await open('&show-sources')
  await idlePractice(page)
  assert.equal(await page.evaluate(() => window.practiceTest.calls.filter(c => c.kind === 'read-source').length), 0)
  await practice.locator('.source-tray-card').first().click()
  await reader.waitFor(); await close()
  await practice.getByRole('button', { name: 'True', exact: true }).click()
  await idlePractice(page)
  assert.equal((await session()).questions[0].attempts[0].assisted, false)
  await reviewPracticeQuestion(page, 1)
  await practice.locator('.source-tray-card').first().click()
  await reader.waitFor(); await close()
  console.log('PASS sources visible before answering; manual and automatic reading preserve independent answers')

  for (const receiveHelp of [false, true]) {
    await open('&show-sources')
    await idlePractice(page)
    await practice.getByRole('button', { name: 'Chat about this question', exact: true }).click()
    const chat = page.locator('.chat-frame')
    await chat.waitFor({ state: 'visible' })
    if (receiveHelp) {
      await chat.getByLabel('Message', { exact: true }).fill('Why keep the ancestor path?')
      await chat.getByLabel('Message', { exact: true }).press('Enter')
      await page.waitForFunction(() => window.practiceTest.state().practice.sessions[0].observations.some(e => e.kind === 'help_shown'))
    }
    await chat.getByRole('button', { name: 'Practice', exact: true }).click()
    await practice.getByRole('button', { name: 'True', exact: true }).click()
    await idlePractice(page)
    assert.equal((await session()).questions[0].attempts[0].assisted, receiveHelp,
      'Only a received Chat explanation counts as help; opening Chat alone does not')
  }
  console.log('PASS actual Chat help counts, opening Chat alone does not')

  await open('&show-sources&preview-sources&slow-reader')
  await idlePractice(page)
  await page.waitForTimeout(1700)
  assert.equal(await reader.count(), 0)
  assert.equal((await session()).observations.some(e => e.kind === 'help_shown'), false)
  console.log('PASS late document does not open or count as help')

  await open('&show-sources&preview-sources&slow-reader&slow-model')
  await page.waitForFunction(() => window.practiceTest.calls.some(c => c.kind === 'read-source'))
  await page.getByRole('button', { name: 'Stop', exact: true }).click()
  await page.waitForTimeout(2200)
  assert.equal(await reader.count(), 0)
  assert.equal((await session()).questions.length, 0)
  console.log('PASS Stop cancels preview and ignores late completion')

  await open('&show-sources&preview-sources&slow-reader&slow-model')
  await page.waitForFunction(() => window.practiceTest.calls.some(c => c.kind === 'read-source'))
  await page.getByRole('button', { name: 'Settings', exact: true }).click()
  await idlePractice(page)
  assert.equal(await reader.count(), 0)
  assert.equal((await session()).observations.some(e => e.kind === 'help_shown'), false)
  await page.getByRole('checkbox', { name: 'Read while waiting', exact: true }).uncheck()
  await page.reload()
  await page.getByRole('button', { name: 'Settings', exact: true }).click()
  await page.getByRole('checkbox', { name: 'Read while waiting', exact: true }).waitFor()
  assert.equal(await page.getByRole('checkbox', { name: 'Read while waiting', exact: true }).isChecked(), false)
  console.log('PASS navigation prevents hidden preview and preference persists')

  await open('&show-sources&preview-sources&source-error')
  await idlePractice(page)
  assert.equal((await session()).questions.length, 1)
  assert.equal((await session()).observations.some(e => e.kind === 'help_shown'), false)
  assert.equal(await reader.count(), 0)
  await open('&show-sources&preview-sources&slow-model&error-first')
  await reader.waitFor(); await idlePractice(page)
  assert.equal(await reader.count(), 0)
  assert.equal((await session()).questions.length, 0)
  console.log('PASS missing source is optional and failed generation closes reader')

  await open('&show-sources&preview-sources&slow-model&many-sources')
  await reader.waitFor()
  await reader.getByRole('button', { name: 'Next source', exact: true }).click()
  await reader.getByText('Source 2 of 6', { exact: true }).waitFor()
  await idlePractice(page)
  assert.equal(await reader.isVisible(), true)
  await close()
  console.log('PASS manually selected source stays open when question arrives')

  await page.setViewportSize({ width: 390, height: 844 })
  await open()
  await reader.waitFor()
  assert.equal(await page.getByRole('button', { name: 'Stop', exact: true }).isVisible(), true)
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth + 1), false)
  await page.screenshot({ path: `${output}/mobile.png` })
  await close(); await idlePractice(page)
  await practice.getByRole('button', { name: 'True', exact: true }).waitFor()
  assert.deepEqual(errors, [])
  console.log('PASS mobile layout and no browser errors')
} catch (error) {
  await page.screenshot({ path: `${output}/failure.png` })
  console.error(await page.locator('body').innerText())
  throw error
} finally { await browser.close() }
