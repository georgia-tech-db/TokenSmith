import assert from 'node:assert/strict'
import { mkdir } from 'node:fs/promises'

// Reuse the isolated study fixture; no saved user data, downloads, or model calls.
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright')
const browser = await chromium.launch({ headless: true, executablePath: process.env.CHROME_PATH })
const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } })
page.setDefaultTimeout(10000)
const output = '/tmp/tokensmith-source-preview-ui'
await mkdir(output, { recursive: true })
const base = process.env.PRACTICE_UI_URL || 'http://127.0.0.1:5193/tests/ui/practice.html'
const errors = []
page.on('pageerror', error => errors.push(error.message))

await page.addInitScript(() => {
  const params = new URLSearchParams(location.search)
  const text = ['# Buffer management', '', 'Introductory context.', '',
    '## Flushing Pages to Disk', '',
    'BuzzDB flushes every evicted page, whether it changed or not.', '',
    '| Action | BuzzDB | With a dirty bit |', '| --- | --- | --- |',
    '| Insert 1 | Flush | Mark dirty |', '| Insert 2 | Flush | Keep dirty |', '',
    '```cpp', 'storage_manager.flush(evictedPageId, pageMap[evictedPageId]->second);', '```', '',
    ...Array.from({ length: 30 }, (_, i) => `Paragraph ${i + 1}: surrounding source context remains available.\n`)].join('\n')
  function pdfData() {
    const stream = label => `BT /F1 18 Tf 40 700 Td (${label}) Tj ET`
    const one = stream('Source page one'), two = stream('Source page two')
    const objects = ['<< /Type /Catalog /Pages 2 0 R >>', '<< /Type /Pages /Kids [3 0 R 4 0 R] /Count 2 >>',
      '<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 5 0 R >> >> /Contents 6 0 R >>',
      '<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 5 0 R >> >> /Contents 7 0 R >>',
      '<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>',
      `<< /Length ${one.length} >>\nstream\n${one}\nendstream`, `<< /Length ${two.length} >>\nstream\n${two}\nendstream`]
    let pdf = '%PDF-1.4\n'; const offsets = [0]
    objects.forEach((object, i) => { offsets.push(pdf.length); pdf += `${i + 1} 0 obj\n${object}\nendobj\n` })
    const xref = pdf.length
    pdf += `xref\n0 ${objects.length + 1}\n0000000000 65535 f \n`
    pdf += offsets.slice(1).map(offset => `${String(offset).padStart(10, '0')} 00000 n \n`).join('')
    pdf += `trailer\n<< /Size ${objects.length + 1} /Root 1 0 R >>\nstartxref\n${xref}\n%%EOF`
    return `data:application/pdf;base64,${btoa(pdf)}`
  }
  const control = { reads: 0, answer: null, finish: null, request: null }
  window.previewCheck = control
  Object.defineProperty(window, 'tokensmith', { configurable: true, set(bridge) {
    const load = bridge.loadAppState
    bridge.loadAppState = async () => {
      const state = await load()
      state.settings.application.showSources = true
      state.settings.application.theme = params.has('alternate-theme') ? 'sarah-and-duck' : 'light'
      if (params.has('historical-answer')) {
        const source = {
          title: 'BuzzDB', documentTitle: 'BuzzDB', materialId: 'c1', documentId: 1,
          path: params.has('pdf') ? '/fixture/db/notes.pdf' : '/fixture/db/Trees.md',
          chunkId: 'original-source', locator: 'Chunk 100', sectionHeader: 'Flushing Pages to Disk',
          lineFrom: 7, lineTo: 12, ...(params.has('pdf') ? { pageStart: 1, pageEnd: 1 } : {}),
          context: 'BuzzDB flushes every evicted page, whether it changed or not.',
          excerpt: 'BuzzDB flushes every evicted page, whether it changed or not.'
        }
        state.activeConversationId = 'historical-chat'
        state.conversations = [{ id: 'historical-chat', title: 'Historical answer check', period: 'Today', messages: [
          { id: 'original-question', role: 'user', text: 'Why does BuzzDB flush an unchanged page during eviction?' },
          { id: 'original-answer', role: 'assistant', text: 'The flush call is unconditional.', sources: [source],
            conversationContextMode: 'standalone', explanationDepth: 'standard',
            reasoning: { modelId: 'test', supported: true, used: false },
            answerContext: { prompt: 'Why does BuzzDB flush an unchanged page during eviction?',
              modelSettings: state.settings.modelSettingsById.test } },
          { id: 'later-question', role: 'user', text: 'What is implicit pinning?' },
          { id: 'later-answer', role: 'assistant', text: 'This later answer describes implicit pinning.',
            sources: [{ ...source, chunkId: 'later-source' }], conversationContextMode: 'standalone',
            answerContext: { prompt: 'What is implicit pinning?' } }
        ] }]
      }
      return state
    }
    const search = bridge.searchLibrary
    bridge.searchLibrary = async (...args) => {
      const result = await search(...args)
      return { ...result, sources: result.sources.slice(0, 2).map((source, i) => ({
        ...source, title: 'BuzzDB', documentTitle: 'BuzzDB', locator: `Chunk ${100 + i}`,
        sectionHeader: i === 0 ? 'Flushing Pages to Disk' : undefined,
        context: 'BuzzDB flushes every evicted page, whether it changed or not.',
        excerpt: 'BuzzDB flushes every evicted page, whether it changed or not.',
        lineFrom: 7, lineTo: 12,
        ...(params.has('pdf') ? { path: '/fixture/db/notes.pdf', pageStart: i + 1, pageEnd: i + 1 } : {})
      })) }
    }
    bridge.getMarkdownForSource = async source => {
      control.reads++
      return { title: 'BuzzDB', path: source.path, text, chunkText: source.context,
        sectionHeader: source.sectionHeader, locator: source.locator, lineFrom: 7, lineTo: 12 }
    }
    bridge.getPdfForSource = async source => {
      control.reads++
      return { title: 'Lecture notes', path: source.path, page: source.pageStart, dataUrl: pdfData() }
    }
    bridge.sendChatMessage = (request, onAnswer) => new Promise(resolve => {
      control.request = request
      const reply = { text: 'An unchanged page is flushed because the call is unconditional.', sources: request.retrievedSources || [] }
      control.answer = () => onAnswer?.(reply, true)
      control.finish = () => resolve(reply)
    })
    Object.defineProperty(window, 'tokensmith', { configurable: true, writable: true, value: bridge })
  } })
})

const chat = page.locator('.chat-frame')
const preview = page.locator('.automatic-source-preview')
const question = 'Why does BuzzDB flush an unchanged page during eviction?'
async function open(suffix = '', prompt = question) {
  await page.goto(`${base}?run=${crypto.randomUUID()}&many-sources${suffix}`)
  await chat.getByLabel('Message', { exact: true }).fill(prompt)
  await chat.getByRole('button', { name: 'Send message', exact: true }).click()
  await preview.waitFor()
  await page.waitForFunction(() => typeof window.previewCheck.answer === 'function')
}
const finish = async () => { await page.evaluate(() => { window.previewCheck.answer(); window.previewCheck.finish() }) }
try {
  await open()
  assert.equal(await preview.locator('.source-viewer-heading strong').innerText(), 'Flushing Pages to Disk')
  assert.equal(await preview.locator('.source-viewer-question-preview').innerText(), question)
  assert.equal(await preview.locator('.source-viewer-details').evaluate(el => el.open), false)
  await preview.locator('.source-viewer-details summary').click()
  assert.match(await preview.locator('.source-viewer-details').innerText(), /Chunk 100.*Lines 7/)
  await preview.locator('.source-viewer-details summary').click()
  assert.equal(await preview.locator('table').count(), 1)
  assert.ok(await preview.locator('[data-source-selected]').count() > 0)
  await preview.getByRole('button', { name: 'Keep open', exact: true }).click()
  await page.evaluate(() => document.fonts.ready)
  const viewport = preview.locator('.markdown-source-viewport')
  const scroll = await viewport.evaluate(el => { el.dataset.retained = 'yes'; el.scrollTop = 250; return el.scrollTop })
  const reads = await page.evaluate(() => window.previewCheck.reads)
  await finish()
  await preview.getByRole('status').filter({ hasText: 'Answer ready' }).waitFor()
  assert.equal(await viewport.getAttribute('data-retained'), 'yes')
  assert.equal(await viewport.evaluate(el => el.scrollTop), scroll)
  assert.equal(await page.evaluate(() => window.previewCheck.reads), reads)
  await preview.getByRole('button', { name: 'Jump to highlighted passage' }).click()
  assert.notEqual(await viewport.evaluate(el => el.scrollTop), scroll)
  await page.screenshot({ path: `${output}/markdown-kept-open.png` })
  await preview.getByRole('button', { name: 'Return to chat' }).click()
  await preview.waitFor({ state: 'detached' })
  assert.match(await chat.innerText(), /An unchanged page is flushed/)
  await page.locator('.source-tray-card').first().click()
  const manualReader = page.getByRole('dialog', { name: 'Source Markdown viewer' })
  await manualReader.waitFor()
  assert.equal(await manualReader.locator('.source-viewer-question-preview').innerText(), question)
  await manualReader.getByRole('button', { name: 'Return to chat' }).click()
  await manualReader.waitFor({ state: 'detached' })

  await open()
  await finish()
  await preview.waitFor({ state: 'detached' })

  for (const format of ['markdown', 'pdf']) {
    await open(format === 'pdf' ? '&pdf' : '')
    await preview.getByRole('button', { name: 'Return to chat', exact: true }).click()
    await preview.waitFor({ state: 'detached' })
    const navigation = page.getByRole('complementary', { name: 'Primary navigation' })
    await navigation.getByRole('button', { name: 'Settings', exact: true }).click()
    await navigation.getByRole('button', { name: 'Chat', exact: true }).click()
    assert.equal(await preview.count(), 0, 'Returning to Chat must not reopen a dismissed source')
    await chat.locator('.source-tray-card').first().click()
    const reopenedReader = page.getByRole('dialog', { name: format === 'pdf' ? 'Source PDF viewer' : 'Source Markdown viewer' })
    await reopenedReader.waitFor()
    assert.equal(await reopenedReader.locator('.source-viewer-question-preview').innerText(), question)
    assert.equal(await reopenedReader.locator('.source-viewer-status').count(), 0)
    await page.evaluate(() => window.previewCheck.answer())
    await reopenedReader.getByRole('status').filter({ hasText: 'Answer ready' }).waitFor()
    assert.equal(await reopenedReader.locator('.source-viewer-question-preview').innerText(), question)
    await reopenedReader.getByRole('button', { name: 'Return to chat', exact: true }).click()
    await reopenedReader.waitFor({ state: 'detached' })
    await chat.locator('.source-tray-card').first().click()
    await reopenedReader.waitFor()
    assert.equal(await reopenedReader.locator('.source-viewer-question-preview').innerText(), question)
    await reopenedReader.getByRole('status').filter({ hasText: 'Answer ready' }).waitFor()
    await page.evaluate(() => window.previewCheck.finish())
    await reopenedReader.getByRole('button', { name: 'Return to chat', exact: true }).click()
    await reopenedReader.waitFor({ state: 'detached' })
  }

  for (const format of ['markdown', 'pdf']) {
    for (const action of ['Explain more simply', 'Reasoning']) {
      await page.goto(`${base}?run=${crypto.randomUUID()}&historical-answer${format === 'pdf' ? '&pdf' : ''}`)
      await chat.getByRole('button', { name: 'Historical answer check', exact: true }).click()
      await chat.getByRole('button', { name: action, exact: true }).first().click()
      await preview.waitFor()
      assert.equal(await preview.locator('.source-viewer-question-preview').innerText(), question)
      await page.waitForFunction(() => typeof window.previewCheck.answer === 'function')
      await preview.getByRole('button', { name: 'Return to chat', exact: true }).click()
      await preview.waitFor({ state: 'detached' })
      await chat.locator('.source-chip').last().click()
      await chat.locator('.source-tray-card').first().click()
      const reader = page.getByRole('dialog', { name: format === 'pdf' ? 'Source PDF viewer' : 'Source Markdown viewer' })
      await reader.waitFor()
      assert.equal(await reader.locator('.source-viewer-question-preview').innerText(), question,
        `${action} must retain its original question after selecting another answer's sources`)
      assert.equal(await page.evaluate(() => window.previewCheck.request.retrievedSources[0].chunkId), 'original-source')
      await finish()
      await reader.getByRole('status').filter({ hasText: 'Answer ready' }).waitFor()
      assert.equal(await reader.locator('.source-viewer-question-preview').innerText(), question)
      await reader.getByRole('button', { name: 'Return to chat', exact: true }).click()
      await reader.waitFor({ state: 'detached' })
    }
  }

  for (const action of ['Explain more simply', 'Reasoning']) {
    await page.goto(`${base}?run=${crypto.randomUUID()}&historical-answer&pdf`)
    await chat.getByRole('button', { name: 'Historical answer check', exact: true }).click()
    await chat.getByRole('button', { name: action, exact: true }).first().click()
    await preview.waitFor()
    await page.waitForFunction(() => typeof window.previewCheck.answer === 'function')
    await preview.getByRole('button', { name: 'Keep open', exact: true }).click()
    await page.evaluate(() => window.previewCheck.answer())
    await preview.getByRole('status').filter({ hasText: 'Answer ready' }).waitFor()
    await page.evaluate(() => window.previewCheck.finish())
    await chat.getByRole('button', { name: 'Stop', exact: true }).waitFor({ state: 'detached' })
    assert.equal(await preview.evaluate(element => element.contains(document.activeElement)), true,
      'A completed retry must not move keyboard focus behind a retained PDF')
    await page.keyboard.press('Escape')
    await preview.waitFor({ state: 'detached' })
  }

  await open()
  await preview.getByRole('button', { name: 'Next source' }).click()
  await preview.getByText('Source 2 of 2', { exact: true }).waitFor()
  assert.equal(await preview.locator('.source-viewer-heading strong').innerText(), 'BuzzDB')
  await finish()
  await preview.getByRole('status').filter({ hasText: 'Answer ready' }).waitFor()
  assert.equal(await preview.locator('.source-viewer-question-preview').innerText(), question)
  await preview.locator('.markdown-source-viewport').focus()
  await page.keyboard.press('Escape')
  await preview.waitFor({ state: 'detached' })

  await open()
  await preview.getByRole('button', { name: 'Keep open', exact: true }).click()
  await page.getByRole('button', { name: 'Stop', exact: true }).click()
  await preview.getByRole('status').filter({ hasText: 'Generation stopped' }).waitFor()
  await finish()
  assert.match(await preview.innerText(), /Generation stopped/)
  await preview.getByRole('button', { name: 'Return to chat' }).click()

  await page.setViewportSize({ width: 720, height: 850 })
  await open('&alternate-theme', `${question} Explain the code and the difference between clean and dirty pages. `.repeat(5))
  await preview.locator('.source-viewer-question summary').click()
  assert.equal(await preview.locator('.source-viewer-question').evaluate(el => el.open), true)
  assert.equal(await page.evaluate(() => document.documentElement.dataset.theme), 'sarah-and-duck')
  const heading = await preview.locator('.source-viewer-heading strong').boundingBox()
  assert.ok(heading.width > 180 && heading.height < 40, 'The section title must stay readable in a narrow window')
  assert.equal(await preview.evaluate(el => el.scrollWidth <= el.clientWidth), true)
  assert.ok((await preview.locator('.markdown-source-viewport').boundingBox()).height > 200)
  await page.screenshot({ path: `${output}/markdown-alternate-narrow.png` })
  await finish()
  await preview.waitFor({ state: 'detached' })

  await page.setViewportSize({ width: 1440, height: 1000 })
  await open('&pdf')
  assert.equal(await preview.getByRole('button', { name: 'Return to chat', exact: true })
    .evaluate(button => button === document.activeElement), true, 'An automatic PDF must receive keyboard focus')
  await page.keyboard.press('Escape')
  await preview.waitFor({ state: 'detached' })
  await finish()
  await chat.getByRole('button', { name: 'Stop', exact: true }).waitFor({ state: 'detached' })
  assert.equal(await preview.count(), 0, 'Completing an answer must not reopen a dismissed PDF')

  await open('&pdf')
  await preview.locator('.pdf-js-page img').waitFor()
  assert.equal(await preview.locator('.source-viewer-heading strong').innerText(), 'Flushing Pages to Disk')
  await preview.getByRole('button', { name: 'Next page', exact: true }).click()
  await preview.locator('.source-viewer-heading').getByText('Page 2 of 2', { exact: true }).waitFor()
  assert.equal(await preview.locator('.source-viewer-heading strong').innerText(), 'Lecture notes')
  await preview.getByRole('button', { name: 'Back to cited page' }).click()
  await preview.locator('.source-viewer-heading').getByText('Lecture notes · Page 1 of 2', { exact: true }).waitFor()
  await preview.getByRole('button', { name: 'Keep open', exact: true }).click()
  await preview.getByRole('textbox', { name: 'Search this PDF', exact: true }).focus()
  await finish()
  await preview.getByRole('status').filter({ hasText: 'Answer ready' }).waitFor()
  assert.equal(await preview.getByRole('textbox', { name: 'Search this PDF', exact: true })
    .evaluate(input => input === document.activeElement), true, 'Completion must preserve focus already inside the PDF')
  await page.screenshot({ path: `${output}/pdf-kept-open.png` })
  await preview.getByRole('button', { name: 'Return to chat' }).click()
  assert.deepEqual(errors, [])
  console.log('Source preview UI passed: question, metadata fallback, intact table, retained reader/scroll, default close, source navigation, PDF focus/Escape, reopened Markdown/PDF question and status after Settings, historical simplification/reasoning ownership, Stop, both app themes/narrow layout, and PDF cited-page controls.')
  console.log(`Screenshots: ${output}`)
} catch (error) {
  await page.screenshot({ path: `${output}/failure.png` })
  console.error(await page.locator('body').innerText())
  throw error
} finally { await browser.close() }
