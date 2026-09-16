import React, { useState } from 'react'
import { createRoot } from 'react-dom/client'
import { ConversationViewport } from '../../src/renderer/src/ConversationViewport'
import { addQuoteToDraft } from '../../src/renderer/src/chat-interactions'
import '../../src/renderer/src/chat-interactions.css'

const text = 'Compaction would invalidate all external pointers referencing those old locations.'
const quote = 'external pointers referencing those old locations.'
const paint = () => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))
function select(start, end) {
  const node = document.querySelector('#source').firstChild
  const range = document.createRange()
  range.setStart(node, start); range.setEnd(node, end)
  const selection = window.getSelection()
  selection.removeAllRanges(); selection.addRange(range)
  document.dispatchEvent(new Event('selectionchange'))
}
function Fixture() {
  const [draft, setDraft] = useState('Explain this.')
  const [result, setResult] = useState('Ready')
  async function run(kind) {
    setDraft('Explain this.')
    setResult('Running')
    const source = document.querySelector('#source')
    const start = text.indexOf('external')
    if (kind === 'drag') source.dispatchEvent(new PointerEvent('pointerdown', {bubbles: true, button: 0, buttons: 1}))
    select(start, start + 20)
    await paint()
    if (kind === 'drag' && document.querySelector('.selection-chat-action')) {
      document.dispatchEvent(new PointerEvent('pointercancel', {bubbles: true}))
      setResult('FAIL: quote action appeared during an unfinished drag'); return
    }
    select(start, text.length)
    if (kind === 'drag') {
      source.dispatchEvent(new PointerEvent('pointerup', {bubbles: true, button: 0}))
      await paint()
    }
    // Click before the queued selection-change frame, reproducing an outdated snapshot.
    document.querySelector('.selection-chat-action')?.click()
    await paint()
    const actual = document.querySelector('textarea').value
    setResult(actual === `> ${quote}\n\nExplain this.` ? `PASS: ${kind}` : `FAIL: ${kind}: ${actual}`)
  }
  return <main>
    <h1>Selection regression checks</h1>
    <button onClick={() => run('snapshot')}>Check final selection</button>
    <button onClick={() => run('drag')}>Check unfinished drag</button>
    <output aria-live="polite">{result}</output>
    <ConversationViewport messages={[]} pending={false} canQuote onQuote={value => setDraft(old => addQuoteToDraft(old, value))}>
      <div data-chat-selectable><p id="source">{text}</p><p>A separate paragraph with <strong>formatted text</strong> for manual selection.</p></div>
      <div data-chat-selectable><p>Another message must not be combined with the first one.</p></div>
    </ConversationViewport>
    <label>Draft<textarea value={draft} onChange={event => setDraft(event.target.value)} /></label>
  </main>
}
document.head.insertAdjacentHTML('beforeend', '<style>body{font:16px system-ui;padding:28px}main{max-width:760px}button{margin:8px;padding:8px}output{display:block;margin:12px}.conversation-viewport{position:relative}.message-list{height:240px;overflow:auto;border:1px solid #ccc;padding:20px}textarea{display:block;width:100%;height:150px}.selection-chat-action{margin:0;color:#111;background:#fff}</style>')
createRoot(document.getElementById('root')).render(<Fixture />)
