import assert from 'node:assert/strict'
import { mkdir } from 'node:fs/promises'

const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright')
const browser = await chromium.launch({ headless: true, executablePath: process.env.CHROME_PATH })
const base = process.env.TOPIC_PROGRESS_UI_URL || 'http://127.0.0.1:5197/tests/ui/topic-progress.html'
const output = process.env.TOPIC_PROGRESS_UI_OUTPUT || '/tmp/tokensmith-mastery-ui'
await mkdir(output, { recursive: true })
const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } })
page.setDefaultTimeout(8000)
const errors = []
page.on('pageerror', error => errors.push(error.message))
const button = name => page.getByRole('button', { name, exact: true })
const topicRow = title => page.locator('.practice-topic-entry').filter({ has: button(title) })
const score = async (container, expected) => {
  await container.locator('.mastery-meter-label strong').getByText(expected, { exact: true }).waitFor()
  assert.equal(await container.getByRole('meter', { name: 'Estimated mastery', exact: true }).getAttribute('aria-valuetext'), expected)
}
const openChapter = async (collection = 'Book A') => {
  await page.getByRole('button', { name: new RegExp(`Chapter 1 — Indexes.*${collection}`) }).click()
  await page.getByRole('heading', { name: 'Chapter 1 — Indexes' }).waitFor()
}
try {
  for (const [name, width, height] of [['desktop', 1440, 1000], ['mobile', 390, 844]]) {
    await page.setViewportSize({ width, height })
    await page.goto(`${base}?run=${crypto.randomUUID()}`)
    await page.getByText('Book A · 2 of 3 topics assessed').waitFor()
    assert.equal(await page.getByRole('button', { name: /Chapter 1 — Indexes.*Book A/ }).getByRole('meter').getAttribute('aria-valuenow'), '12')
    await page.screenshot({ path: `${output}/chapters-${name}.png`, fullPage: true })
    await button('Recent mistakes').click()
    assert.equal(await page.locator('.practice-chapter-button').count(), 1)
    await openChapter()
    assert.equal(await page.locator('.practice-topic-entry').count(), 1)
    await score(topicRow('Tree insertion'), '5%')
    await button('Details for Tree insertion').click()
    await topicRow('Tree insertion').getByText('About this estimate', { exact: true }).click()
    await topicRow('Tree insertion').getByText(/1 independent answer · 0 correct/).waitFor()
    assert.equal(await topicRow('Tree insertion').locator('time').getAttribute('datetime'), '2026-11-09T12:00:00Z')
    await topicRow('Tree insertion').getByText('About this estimate', { exact: true }).click()
    await button('Details for Tree insertion').click()
    await button('All').click()
    assert.equal(await page.locator('.practice-topic-entry').count(), 3)
    await button('Tree insertion').click()
    const selection = await page.evaluate(() => window.topicProgressTest.selected)
    assert.deepEqual(selection.concept, { courseId: 'a', id: 'trees', title: 'Tree insertion' })
    assert.equal(selection.sources[0].chunkId, 'trees-passage')
    assert.equal(selection.sources[0].materialId, 'a/1')
    await score(page.getByLabel('Active topic'), '5%')
    // Keyboard toggling, search + filter composition, and no overflow at either size.
    await button('Not practiced').focus()
    await page.keyboard.press('Space')
    assert.equal(await button('Not practiced').getAttribute('aria-pressed'), 'true')
    assert.equal(await page.locator('.practice-topic-entry').count(), 1)
    await score(topicRow('Range queries'), 'Not practiced')
    await page.getByLabel('Find a topic in this chapter').fill('missing')
    await page.getByText('No matching topics in this chapter.').waitFor()
    await page.getByLabel('Find a topic in this chapter').fill('')
    await button('All').click()
    await page.screenshot({ path: `${output}/topics-${name}.png`, fullPage: true })
    const rowHeight = (await topicRow('Tree insertion').boundingBox()).height
    assert.ok(rowHeight <= (name === 'desktop' ? 60 : 95), `Compact ${name} topic row: ${rowHeight}px`)
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true)
    assert.equal(await page.evaluate(() => getComputedStyle(document.querySelector('.practice-scroll')).overflowY), 'auto')

    // A new assisted success keeps the mistake filter; a new independent success clears it.
    await page.evaluate(() => window.topicProgressTest.answer(true, true))
    await score(topicRow('Tree insertion'), '5%')
    await button('Recent mistakes').click()
    await button('Tree insertion').waitFor()
    await page.evaluate(() => window.topicProgressTest.answer(true))
    await page.getByText('No matching topics in this chapter.').waitFor()
    await button('All').click()
    await score(topicRow('Tree insertion'), '10%')
    await score(page.getByLabel('Active topic'), '10%')
    assert.deepEqual(await page.evaluate(() => window.topicProgressTest.reads), ['a'], 'Progress updates only read existing session props')
    await page.reload()
    await openChapter()
    await score(topicRow('Tree insertion'), '10%')
    await page.evaluate(() => window.topicProgressTest.deleteSession('mistake'))
    await score(topicRow('Tree insertion'), '31%')
    await button('Practice next: Range queries').click()
    assert.equal(await page.evaluate(() => window.topicProgressTest.selected.concept.id), 'ranges')

    // Same name and local topic ID in another collection must have no progress.
    await button('Recent mistakes').click()
    await button('Book B').click()
    await openChapter('Book B')
    assert.equal(await button('All').getAttribute('aria-pressed'), 'true', 'Scope changes reset the filter')
    await score(topicRow('Tree insertion'), 'Not practiced')
    await button('Both books').click()
    await page.getByText('Book A · 2 of 3 topics assessed').waitFor()
    await page.getByText('Book B · 0 of 1 topics assessed').waitFor()
    await button('Not practiced').click()
    await openChapter('Book B')
    await button('Tree insertion').click()
    assert.equal(await page.evaluate(() => window.topicProgressTest.selected.concept.courseId), 'b')
    await button('All chapters').click()
    assert.equal(await button('Not practiced').getAttribute('aria-pressed'), 'true')
    await page.evaluate(() => window.topicProgressTest.clear())
    await page.getByText('Book A · 0 of 3 topics assessed').waitFor()
    assert.equal(await page.getByRole('button', { name: /Chapter 1 — Indexes.*Book A/ }).getByRole('meter').getAttribute('aria-valuenow'), '0')
  }
  assert.deepEqual(errors, [])
  console.log(`PASS: mastery meters, hierarchy, filters, collection identity, source preservation, help, live updates, deletion, reload, keyboard and desktop/mobile layouts. Screenshots: ${output}`)
} finally {
  await browser.close()
}
