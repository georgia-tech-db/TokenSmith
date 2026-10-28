import assert from 'node:assert/strict'
import { mkdir } from 'node:fs/promises'

const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright')
const browser = await chromium.launch({ headless: true, executablePath: process.env.CHROME_PATH })
const base = process.env.QUERY_PLAN_UI_URL || 'http://127.0.0.1:5196'
const output = process.env.QUERY_PLAN_OUTPUT || '/tmp/tokensmith-query-plan-ui'
await mkdir(output, { recursive: true })
const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } })
page.setDefaultTimeout(10000)
const errors = []
page.on('pageerror', error => errors.push(error.message))
const answer = page.locator('.answer-explanation').first()
const button = name => answer.getByRole('button', { name, exact: true })
const panel = () => answer.getByRole('region', { name: 'Query plan', exact: true })
const generation = { stage: 'Answer generation', startedAt: 1, durationMs: 300 }
const trace = { totalDurationMs: 300, spans: [generation] }
async function openSavedAnswer() {
  const chat = page.locator('.conversation-select').filter({ hasText: /^Reasoning retry test$/ })
  const history = page.getByRole('button', { name: 'Toggle history', exact: true })
  await history.waitFor()
  if (await history.getAttribute('aria-expanded') === 'false') await history.click()
  await chat.click()
  for (const name of ['Toggle history', 'Toggle Library']) {
    const toggle = page.getByRole('button', { name, exact: true })
    if (await toggle.getAttribute('aria-expanded') === 'true') await toggle.click()
  }
}
async function total(value) {
  await page.waitForFunction(expected => document.querySelector('.query-plan-totals dd')?.textContent === expected, value)
}
async function layout() {
  assert.equal(await answer.evaluate(el => el.scrollWidth <= el.clientWidth + 1), true, 'Answer must not overflow horizontally')
  assert.equal(await panel().evaluate(el => el.scrollWidth <= el.clientWidth + 1), true, 'Plan must wrap within the answer')
}
try {
  await page.goto(`${base}/tests/ui/query-plan.html`)
  await button('Explain more simply').waitFor()
  assert.equal(await button('Show query plan').count(), 0, 'Legacy answers have no empty action')
  await page.evaluate(value => window.queryPlanTest.setTrace(value), trace)
  await button('Show query plan').waitFor()
  assert.deepEqual(await answer.locator('.answer-simplify-action button').allTextContents(), ['Explain more simply', 'Show query plan'])
  await button('Show query plan').focus()
  await page.keyboard.press('Enter')
  await panel().waitFor()
  assert.equal(await button('Hide query plan').getAttribute('aria-expanded'), 'true')
  assert.equal(await button('Hide query plan').getAttribute('aria-controls'), await panel().getAttribute('id'))
  await total('300 ms')
  await page.evaluate(value => window.queryPlanTest.setTrace(value), { totalDurationMs: 700, spans: [generation, { stage: 'Follow-up suggestions', startedAt: 2, durationMs: 400 }] })
  await total('700 ms')
  await button('Hide query plan').focus()
  await page.keyboard.press('Space')
  await panel().waitFor({ state: 'hidden' })
  for (const bad of [null, { spans: null }, { spans: 'invalid' }, { spans: [null, { stage: 'bad', durationMs: -1 }] }]) {
    await page.evaluate(value => window.queryPlanTest.setTrace(value), bad)
    await button('Show query plan').waitFor({ state: 'hidden' })
    assert.match(await answer.innerText(), /An existing saved answer/)
  }
  await page.evaluate(value => window.queryPlanTest.setTrace(value), trace)
  await button('Show query plan').click()
  await total('300 ms')

  for (const [name, width, height] of [['desktop', 1440, 1000], ['mobile', 390, 844]]) {
    await page.setViewportSize({ width, height })
    await page.goto(`${base}/tests/ui/reasoning-retry.html?plan-test&run=${crypto.randomUUID()}`)
    await openSavedAnswer()
    await button('Show query plan').waitFor()
    const actions = await answer.locator('.answer-simplify-action button').allTextContents()
    assert.equal(actions[actions.indexOf('Explain more simply') + 1], 'Show query plan')
    await button('Show query plan').click()
    await total('5.42 s')
    assert.match(await panel().innerText(), /Answer ready in\s+5.40 s/)
    await panel().getByRole('button', { name: 'Explain all stages', exact: true }).click()
    assert.ok(await panel().locator('.query-plan-stage-description').count() >= 9)
    await panel().getByRole('button', { name: 'Hide stage guides', exact: true }).click()
    await layout()
    await button('Hide query plan').scrollIntoViewIfNeeded()
    await page.screenshot({ path: `${output}/${name}-original.png` })

    await button('Explain more simply').click()
    await page.waitForFunction(() => window.reasoningTest.state().conversations.find(chat => chat.id === 'retry-test').messages[1].simplerExplanation?.latencyTrace?.spans.length === 2)
    await button('Hide query plan').waitFor()
    await total('700 ms')
    assert.equal(await panel().locator('[data-stage="Library search"]').count(), 0, 'Simple answer must not borrow the original retrieval plan')
    await button('Basic').click()
    await total('5.42 s')
    await button('Reasoning').click()
    await page.waitForFunction(() => window.reasoningTest.state().conversations.find(chat => chat.id === 'retry-test').messages[1].reasoningAnswer?.latencyTrace?.spans.length === 2)
    await total('1.30 s')
    await button('Simpler explanation').click()
    await total('700 ms')
    await page.reload()
    await openSavedAnswer()
    await button('Show query plan').click()
    await total('700 ms')
    await layout()
    await button('Hide query plan').scrollIntoViewIfNeeded()
    await page.screenshot({ path: `${output}/${name}-saved-simple.png` })
  }
  assert.deepEqual(errors, [])
  console.log('Query plan passed: action order, keyboard access, late/partial/malformed traces, original/simple/reasoning plans, persisted reload, desktop/mobile layout.')
  console.log(`Screenshots: ${output}`)
} catch (error) {
  await page.screenshot({ path: `${output}/failure.png` })
  throw error
} finally {
  await browser.close()
}
