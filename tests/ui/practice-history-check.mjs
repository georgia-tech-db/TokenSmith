import assert from 'node:assert/strict'
import { mkdir, readFile } from 'node:fs/promises'
import { choosePracticeTopic, idlePractice, answerThenReview } from './practice-topic-helpers.mjs'

const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright')
const browser = await chromium.launch({ headless: true, executablePath: process.env.CHROME_PATH })
const page = await browser.newPage({ viewport: { width: 1440, height: 1000 }, acceptDownloads: true })
page.setDefaultTimeout(10000)
const base = process.env.PRACTICE_UI_URL || 'http://127.0.0.1:5198/tests/ui/practice.html'
const output = '/tmp/tokensmith-practice-history'
await mkdir(output, { recursive: true })
const errors = []; page.on('pageerror', error => errors.push(error.message))
const practice = page.locator('.practice-frame')
const button = name => practice.getByRole('button', { name, exact: true })
try {
  for (const [name, width, height] of [['desktop', 1440, 1000], ['mobile', 390, 844]]) {
    await page.setViewportSize({ width, height })
    await page.goto(`${base}?run=${crypto.randomUUID()}&practice`)
    await choosePracticeTopic(page); await idlePractice(page)
    const answer = await page.evaluate(() => window.practiceTest.state().practice.sessions[0].questions[0].correctAnswer ? 'True' : 'False')
    await answerThenReview(page, answer)
    await practice.locator('.practice-live-mastery').getByText('31%', { exact: true }).waitFor()
    const actions = practice.getByRole('group', { name: 'Practice session actions' })
    assert.equal(await practice.locator('.practice-topic-bar').count(), 1, 'Normal practice has no second Export row')
    assert.deepEqual(await actions.getByRole('button').allTextContents(), ['Practice another topic', 'Export practice record', 'View mastery'])
    const topicButton = button('Practice another topic'), exportButton = button('Export practice record')
    const style = locator => locator.evaluate(el => {
      const s = getComputedStyle(el); return [s.backgroundColor, s.borderStyle, s.borderRadius, s.fontSize, s.padding]
    })
    assert.deepEqual(await style(topicButton), await style(exportButton))
    assert.deepEqual(await style(button('View mastery')), await style(exportButton))
    if (name === 'desktop') assert.ok(Math.abs((await topicButton.boundingBox()).y - (await exportButton.boundingBox()).y) < 2)
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true)
    await page.screenshot({ path: `${output}/${name}.png` })
    await button('View mastery').click()
    await practice.getByRole('heading', { name: 'Your practice progress', exact: true }).waitFor()
    await practice.locator('.practice-review-mastery').getByRole('meter').waitFor()
    await button('Return to practice').click()
    const downloadPromise = page.waitForEvent('download')
    page.once('dialog', dialog => dialog.accept(''))
    await exportButton.click()
    const download = await downloadPromise
    const exported = JSON.parse(await readFile(await download.path(), 'utf8'))
    assert.equal(exported.sessions.length, 1)
    assert.equal(exported.sessions[0].questions[0].attempts[0].feedback.correct, true)

    const originalChats = await page.evaluate(() => JSON.parse(JSON.stringify(window.practiceTest.state().conversations)))
    if (!await button('Clear Practice History').isVisible()) await button('Toggle history').click()
    let confirmation
    page.once('dialog', dialog => { confirmation = dialog.message(); void dialog.dismiss() })
    await button('Clear Practice History').click()
    assert.match(confirmation, /mastery.*kept/)
    assert.equal(await page.evaluate(() => window.practiceTest.state().practice.sessions.length), 1)
    page.once('dialog', dialog => dialog.accept())
    await button('Clear Practice History').click()
    await page.waitForFunction(() => window.practiceTest.state().practice.sessions.length === 0)
    const records = await page.evaluate(() => window.practiceTest.state().practice.masteryRecords)
    assert.equal(records.length, 1)
    assert.equal(records[0].correct, true)
    assert.equal(records[0].assisted, false)
    assert.equal('questions' in records[0], false)
    assert.equal('sources' in records[0], false)
    await page.reload()
    assert.deepEqual(await page.evaluate(() => window.practiceTest.state().practice.masteryRecords), records)
    assert.deepEqual(await page.evaluate(() => window.practiceTest.state().conversations), originalChats)
    assert.equal(await page.evaluate(() => window.practiceTest.state().practice.sessions.length), 0)
  }
  assert.deepEqual(errors, [])
  console.log('PASS: compact matching action buttons, working export, cancelled clear, mastery survives clear/reload, Chat preserved, desktop/mobile.')
} catch (error) {
  await page.screenshot({ path: `${output}/failure.png` })
  throw error
} finally { await browser.close() }
