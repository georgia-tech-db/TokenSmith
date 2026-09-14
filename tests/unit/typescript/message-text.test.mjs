import assert from 'node:assert/strict'
import test from 'node:test'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { requireTranspiledTs } from './ts-module-loader.mjs'

const { MessageText } = requireTranspiledTs('src/renderer/src/MessageText.tsx')
const render = (text) => renderToStaticMarkup(createElement(MessageText, { text }))
const mathCount = (html) => (html.match(/class="katex"/g) ?? []).length

test('renders inline arithmetic and LaTeX with bare and explicit delimiters', () => {
  for (const input of ['( 20 + 50 = 70 )', String.raw`\(\frac{1}{2} \times 50 = 25\)`, String.raw`(\text{Total} = 20 \times 3)`]) {
    const html = render(input)
    assert.equal(mathCount(html), 1)
    assert.match(html, /message-math-inline/)
    assert.match(html, /<math/)
  }
})

test('renders display equations, nested parentheses, and literal currency within math', () => {
  for (const input of [String.raw`[ \text{Final Price} = $50 + (0.235 \times $50) ]`, String.raw`\[
\text{Final Price} = \$50 + (0.235 \times $50)
\]`, '[ 50 + (0.235 * 50) = 61.75 ]']) {
    const html = render(input)
    assert.equal(mathCount(html), 1)
    assert.match(html, /message-math-display/)
    assert.match(html, /katex-display/)
  }
})

test('preserves currency, ordinary parentheses, brackets, and existing Markdown', () => {
  assert.equal(render('**Price:** $50 and $61.75 (before tax) [estimated].\n\n- **First** item\n- Second item\n\n1. One\n2. Two'),
    '<div class="message-text"><p><strong>Price:</strong> $50 and $61.75 (before tax) [estimated].</p><ul><li><strong>First</strong> item</li><li>Second item</li></ul><ol><li>One</li><li>Two</li></ol></div>')
  assert.equal(mathCount(render('($50) [$61.75] ($50 and $61.75)')), 0)
})

test('mixes math with bold, lists, paragraphs, and multiline display content', () => {
  const html = render(String.raw`**Result (20 + 50 = 70)**

- Half: \(\frac{1}{2}\)
- Price: $50

1. Total: [
\text{Total} = 20
+ 50
]

Done.`)
  assert.equal(mathCount(html), 3)
  assert.match(html, /<strong>Result <span/)
  assert.match(html, /<ul><li>Half:/)
  assert.match(html, /<ol><li>Total:/)
  assert.match(html, /<p>Done\.<\/p>/)
})

test('malformed and unclosed math stays readable without throwing', () => {
  for (const input of [String.raw`\(\frac{}\)`, String.raw`[\frac{1}{]`, String.raw`\(\frac{1}{2}`, '[ 20 + 50', String.raw`\(\unknowncommand{x}\)`]) {
    assert.equal(render(input), `<div class="message-text"><p>${input}</p></div>`)
  }
  assert.equal(mathCount(render(String.raw`\(\frac{}\) then (1 + 2 = 3)`)), 1)
})

test('escapes raw HTML and disables trusted LaTeX HTML commands', () => {
  assert.match(render('<img src=x onerror=alert(1)>'), /&lt;img/)
  const html = render(String.raw`\(\href{javascript:alert(1)}{click}\)`)
  assert.doesNotMatch(html, /href="javascript:|<a /)
  assert.equal(mathCount(render('MATHPLACEHOLDER0END (1 + 2)')), 1)
  assert.match(render('MATHPLACEHOLDER0END (1 + 2)'), /<p>MATHPLACEHOLDER0END /)
})
