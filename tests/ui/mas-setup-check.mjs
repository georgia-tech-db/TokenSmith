import assert from 'node:assert/strict'
import { createServer } from 'vite'
import react from '@vitejs/plugin-react'
import { resolve } from 'node:path'
import { mkdir } from 'node:fs/promises'
import { chromium } from 'playwright'
const server = await createServer({ configFile: false, cacheDir: 'node_modules/.vite-mas-test',
  optimizeDeps: { entries: ['tests/ui/cloud-generators.html'] }, plugins: [react()],
  resolve: { alias: { '@shared': resolve('src/shared') } }, server: { host: '127.0.0.1', port: 0 } })
await server.listen()
const browser = await chromium.launch({ headless: true, executablePath: process.env.CHROME_PATH })
try {
  const context = await browser.newContext({ viewport: { width: 1440, height: 1100 } })
  await context.route('**/*', route => new URL(route.request().url()).hostname === '127.0.0.1' ? route.continue() : route.abort())
  const page = await context.newPage()
  const errors = []; page.on('pageerror', error => errors.push(error.message))
  const base = `http://127.0.0.1:${server.httpServer.address().port}/tests/ui/cloud-generators.html`
  await page.goto(`${base}?mas&persist`)
  const setup = page.getByRole('region', { name: 'Model setup' })
  await setup.getByText('Ollama is optional.', { exact: false }).waitFor()
  assert.equal(await page.getByRole('button', { name: /Install Ollama|Start Ollama|Download Nomic|Download Gemma/ }).count(), 0)
  for (const [role, modelName] of [['generator', 'gpt-4.1-mini'], ['embedder', 'text-embedding-test']]) {
    await setup.getByRole('button', { name: 'Connect a cloud model', exact: true }).click()
    const dialog = page.getByRole('dialog')
    await dialog.getByRole('button', { name: /^OpenAI Connect with/ }).click()
    await dialog.getByLabel('Model type', { exact: true }).selectOption(role)
    if (role === 'generator') await dialog.getByLabel('OpenAI API key', { exact: true }).fill('test-key')
    await dialog.getByRole('button', { name: 'Connect and find models', exact: true }).click()
    await dialog.getByRole('radio', { name: modelName, exact: true }).check()
    await dialog.getByRole('button', { name: 'Use this model', exact: true }).click()
    await dialog.waitFor({ state: 'hidden' })
  }
  await page.waitForFunction(() => window.cloudTest.state().models.filter(m => m.engine === 'remote' && m.status === 'ready').length === 2)
  assert.equal((await page.evaluate(() => window.cloudTest.state().models)).some(m => m.engine === 'ollama'), false)
  await page.reload()
  await setup.getByRole('button', { name: 'Add documents', exact: true }).waitFor()
  assert.equal((await page.evaluate(() => window.cloudTest.state().models)).filter(m => m.engine === 'remote').length, 2)
  await setup.scrollIntoViewIfNeeded()
  await mkdir('tmp/mas-ui', { recursive: true }); await page.screenshot({ path: 'tmp/mas-ui/cloud-only.png', fullPage: true })
  // A separate fresh session can connect a model that is already served by Ollama.
  await context.clearCookies(); await page.evaluate(() => sessionStorage.clear())
  await page.goto(`${base}?mas&ollama`)
  await setup.getByRole('button', { name: 'Connect Ollama models', exact: true }).click()
  await page.getByRole('button', { name: 'Use Chat Model', exact: true }).click()
  await page.waitForFunction(() => window.cloudTest.state().models.some(m => m.engine === 'ollama' && m.status === 'ready'))
  assert.equal(await page.getByRole('button', { name: /^Download$|^Delete|^Start Ollama|^Install Ollama/ }).count(), 0)
  await page.getByRole('button', { name: 'Disconnect', exact: true }).click()
  await page.waitForFunction(() => !window.cloudTest.state().models.some(m => m.engine === 'ollama'))
  assert.ok((await page.evaluate(() => window.cloudTest.calls)).some(c => c.kind === 'disconnect'))
  assert.deepEqual(errors, [])
  console.log('PASS MAS cloud-only onboarding, persisted cloud models, and optional Ollama connection/disconnection')
} finally { await browser.close(); await server.close() }
