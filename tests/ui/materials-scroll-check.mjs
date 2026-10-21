import { choosePracticeTopic, idlePractice, reviewPracticeQuestion, answerThenReview, advancePractice } from './practice-topic-helpers.mjs'
import assert from 'node:assert/strict'
import { mkdir } from 'node:fs/promises'

// Real shared panes with a long deterministic catalog; no model calls or user data.
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright')
const browser = await chromium.launch({ headless: true, executablePath: process.env.CHROME_PATH })
const base = process.env.PRACTICE_UI_URL || 'http://127.0.0.1:5188/tests/ui/practice.html'
const output = '/tmp/tokensmith-materials-scroll-ui'
await mkdir(output, { recursive: true })
const page = await browser.newPage()
page.setDefaultTimeout(10000)
const errors = []
page.on('pageerror', error => errors.push(error.message))
const chat = page.locator('.chat-frame')
const practice = page.locator('.practice-frame')
const pane = frame => frame.getByRole('complementary', { name: 'Library' })
const toggle = frame => frame.getByRole('button', { name: 'Toggle Library', exact: true })
const show = async frame => { if (!await pane(frame).isVisible()) await toggle(frame).click() }
const hide = async frame => { if (await pane(frame).isVisible()) await toggle(frame).click() }
const scrollTop = element => element.evaluate(el => el.scrollTop)
const metrics = element => element.evaluate(el => ({ height: el.clientHeight, total: el.scrollHeight, width: el.clientWidth, fullWidth: el.scrollWidth }))
const near = (actual, expected, label) => assert.ok(Math.abs(actual - expected) <= 1, `${label}: ${actual} vs ${expected}`)
async function checkScrolling(frame, label) {
  await show(frame)
  const panel = pane(frame)
  const documents = panel.getByRole('region', { name: 'Folders and documents', exact: true })
  const sources = panel.getByRole('region', { name: 'Source passages', exact: true })
  const heading = panel.getByRole('heading', { name: 'Sources', exact: true })
  assert.equal(await heading.count(), 1)
  assert.equal(await panel.getByText('Supporting passages', { exact: true }).count(), 0)
  assert.equal(await panel.getByText('Top sources', { exact: true }).count(), 0)
  assert.equal(await panel.getByRole('heading', { name: 'Library', exact: true }).count(), 1)
  const docsSize = await metrics(documents), sourceSize = await metrics(sources)
  assert.ok(docsSize.total > docsSize.height && docsSize.height > 90, 'Long document lists must scroll within a usable area')
  assert.ok(sourceSize.total > sourceSize.height && sourceSize.height > 90, 'Sources must have their own usable scrolling area')
  assert.ok(docsSize.fullWidth <= docsSize.width && sourceSize.fullWidth <= sourceSize.width, 'Neither list should overflow horizontally')
  const header = await panel.locator(':scope > header').boundingBox()
  const footer = await panel.locator(':scope > footer').boundingBox()
  const sourceHeading = await heading.boundingBox()
  const transcriptScroll = await scrollTop(frame.locator('.message-list'))
  await documents.hover()
  await page.mouse.wheel(0, 480)
  await page.waitForFunction(el => el.scrollTop > 0, await documents.elementHandle())
  near(await scrollTop(sources), 0, 'Scrolling documents must not move sources')
  const documentScroll = await scrollTop(documents)
  await sources.hover()
  await page.mouse.wheel(0, 260)
  await page.waitForFunction(el => el.scrollTop > 0, await sources.elementHandle())
  near(await scrollTop(documents), documentScroll, 'Scrolling sources must not move documents')
  near(await scrollTop(frame.locator('.message-list')), transcriptScroll, 'The transcript must not move')
  assert.deepEqual(await panel.locator(':scope > header').boundingBox(), header)
  assert.deepEqual(await panel.locator(':scope > footer').boundingBox(), footer)
  assert.deepEqual(await heading.boundingBox(), sourceHeading)
  assert.ok(footer.y + footer.height <= (await page.viewportSize()).height)

  // Keyboard scrolling reaches the final entry without moving either fixed heading.
  await documents.focus()
  await documents.press('End')
  await page.waitForFunction(el => el.scrollTop + el.clientHeight >= el.scrollHeight - 2, await documents.elementHandle())
  await sources.focus()
  await sources.press('End')
  await page.waitForFunction(el => el.scrollTop + el.clientHeight >= el.scrollHeight - 2, await sources.elementHandle())
  const cards = sources.locator('.source-tray-card')
  await cards.last().click()
  await page.getByRole('dialog', { name: 'Source Markdown viewer' }).waitFor()
  await page.getByRole('button', { name: 'Close Markdown viewer' }).click()
  await page.screenshot({ path: `${output}/${label}.png` })
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth && document.documentElement.scrollHeight <= innerHeight), true)
  return { documents: await documents.boundingBox(), sources: await sources.boundingBox(), header, footer }
}
try {
  for (const [name, viewport] of Object.entries({ desktop: { width: 1440, height: 900 }, compact: { width: 900, height: 640 }, mobile: { width: 390, height: 844 } })) {
    await page.setViewportSize(viewport)
    await page.goto(`${base}?run=${crypto.randomUUID()}&show-sources&many-sources&long-catalog`)
    await page.waitForFunction(() => window.practiceTest.state().studyScope?.length > 20)
    await show(chat)
    const body = pane(chat).locator('.study-materials-body')
    const docs = pane(chat).getByRole('region', { name: 'Folders and documents' })
    near((await docs.boundingBox()).height, (await body.boundingBox()).height, 'Documents use the full body when there are no sources')
    await hide(chat)
    await chat.getByLabel('Message', { exact: true }).fill('Why record ancestors during insertion?')
    await chat.getByRole('button', { name: 'Send message', exact: true }).click()
    await page.waitForFunction(() => window.practiceTest.state().conversations.some(c => c.messages.some(m => m.role === 'assistant')))
    const chatLayout = await checkScrolling(chat, `${name}-chat`)
    await hide(chat)
    await chat.getByRole('button', { name: 'Practice', exact: true }).click()
    await choosePracticeTopic(page)
    await practice.getByLabel('Your answer', { exact: true }).waitFor()
    await answerThenReview(page, 'True')
    await practice.getByRole('region', { name: 'Practice feedback' }).waitFor()
    const practiceLayout = await checkScrolling(practice, `${name}-practice`)
    for (const part of Object.keys(chatLayout)) {
      for (const dimension of ['x', 'y', 'width', 'height']) near(practiceLayout[part][dimension], chatLayout[part][dimension], `${name}: matching ${part} ${dimension}`)
    }

    // New questions replace the source list and reset it to the highest-ranked source.
    await hide(practice)
    await advancePractice(page, 2)
    await page.waitForFunction(() => window.practiceTest.state().practice.sessions[0].questions.length === 2)
    await answerThenReview(page, 'True')
    await show(practice)
    near(await scrollTop(pane(practice).getByRole('region', { name: 'Source passages' })), 0, 'New question sources start at the top')
    assert.match(await pane(practice).locator('.source-tray-card').first().innerText(), /Buffer pool/)
    const sources = pane(practice).getByRole('region', { name: 'Source passages' })
    await sources.focus()
    await sources.press('End')
    await page.waitForFunction(el => el.scrollTop > 0, await sources.elementHandle())
    await hide(practice)
    await practice.locator('.practice-thread-question').first().getByRole('button', { name: '4 Sources', exact: true }).click()
    near(await scrollTop(sources), 0, 'Selecting earlier evidence resets the existing source list')
    assert.match(await pane(practice).locator('.source-tray-card').first().innerText(), /Trees/)
  }
  assert.deepEqual(errors, [])
  console.log('Library scrolling passed: identical Chat/Practice layout, independent wheel and keyboard scrolling, fixed headings/footer, full-height empty-source state, reader access and source reset on desktop, compact and mobile.')
} catch (error) {
  await page.screenshot({ path: `${output}/failure.png` })
  console.error(await page.locator('body').innerText())
  throw error
} finally { await browser.close() }
