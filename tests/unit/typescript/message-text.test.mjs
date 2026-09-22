import assert from 'node:assert/strict'
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { pathToFileURL } from 'node:url'
import test from 'node:test'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import ts from 'typescript'

// Compile the actual renderer to ESM; its Markdown plugins are ESM-only.
const runtimePath = resolve('.coverage-ts-runtime/message-text.mjs')
mkdirSync(resolve('.coverage-ts-runtime'), { recursive: true })
for (const [sourceName, outputName] of [['MessageText.tsx', 'message-text.mjs'], ['remark-model-math.ts', 'remark-model-math.mjs']]) {
  const sourcePath = resolve('src/renderer/src', sourceName)
  const output = ts.transpileModule(readFileSync(sourcePath, 'utf8'), {
    fileName: sourcePath,
    compilerOptions: {
      module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022, jsx: ts.JsxEmit.ReactJSX,
      inlineSourceMap: true, inlineSources: true, sourceRoot: `${dirname(sourcePath)}/`
    }
  }).outputText.replace("from './remark-model-math'", "from './remark-model-math.mjs'")
  writeFileSync(resolve('.coverage-ts-runtime', outputName), output)
}
const { MessageText } = await import(pathToFileURL(runtimePath).href)
const render = (text) => renderToStaticMarkup(createElement(MessageText, { text }))
const renderSuggestion = (text) => renderToStaticMarkup(createElement('button', null,
  createElement(MessageText, { text, inline: true })))

test('suggested questions render code, emphasis, and math inside one button', () => {
  const html = renderSuggestion('How does `std::upper_bound` handle **equal** keys in $O(\\log N)$?')
  assert.match(html, /<code>std::upper_bound<\/code>/)
  assert.match(html, /<strong>equal<\/strong>/)
  assert.match(html, /class="katex"/)
  assert.doesNotMatch(html, /<p>|<div|\$|`|\*\*/)
  assert.equal((html.match(/<button/g) || []).length, 1)
})

test('suggested questions cannot add links, controls, images, or block layout to a button', () => {
  const html = renderSuggestion('# [Docs](https://example.com)\n\n- [ ] a task\n\n![pixel](https://example.com/pixel)\n\n```cpp\npath.push_back(node);\n```\n\n$$\nx^2\n$$')
  assert.doesNotMatch(html, /<(?:a|input|img|div|p|pre|h1|ul|li)(?:\s|>)|href=|src=/)
  assert.match(html, /Docs/)
  assert.match(html, /path.push_back/)
  assert.match(html, /class="katex"/)
  assert.equal((html.match(/<button/g) || []).length, 1)
})

test('renders three nested list levels without flattening them', () => {
  assert.equal(render('- First level\n  1. Second level\n     - Third level\n  2. Second item\n- Last item'),
    '<div class="message-text"><ul>\n<li>First level\n<ol>\n<li>Second level\n<ul>\n<li>Third level</li>\n</ul>\n</li>\n<li>Second item</li>\n</ol>\n</li>\n<li>Last item</li>\n</ul></div>')
})

test('keeps ordered numbering across blank lines and nested bullets', () => {
  const html = render('1. Timestamp 3:\n   - Outside the range.\n\n2. Timestamp 7:\n   - Inside the range.\n\n3. Timestamp 11:\n   - Inside the range.\n\n4. Timestamp 15:\n   - Outside the range.')
  assert.equal((html.match(/<ol>/g) || []).length, 1)
  assert.equal((html.match(/<ul>/g) || []).length, 4)
  assert.match(render('3. Continue\n4. Finish'), /<ol start="3">/)
})

test('keeps mixed cache explanation lists under their parent bullet', () => {
  const html = render(String.raw`- **Mechanism:** Repeated access earns promotion.

  1. **Hot Pages:** Enter the protected **LRU list**.
  2. **Scan Pages:** Enter the **FIFO queue**.

- **Consequence:** Scan pages are evicted first.

For example, index pages ` + '`A` and `B`' + String.raw` stay hot while $	ext{P}1$ through $	ext{P}5$ are scanned.`)
  assert.match(html, /<li>\s*<p><strong>Mechanism:<\/strong>[^]*?<ol>[^]*?<\/ol>\s*<\/li>\s*<li>\s*<p><strong>Consequence:/)
  assert.match(html, /<code>A<\/code> and <code>B<\/code>/)
  assert.equal((html.match(/class="katex"/g) || []).length, 2)
})

test('renders comparison math and preserves nested index bullets', () => {
  const html = render(String.raw`- **Equality Lookups:**
  - **Hash Index:** $O(1)$.
  - **B+ Tree:** $O(\log N)$.
- **Range Queries:**
  - **Hash Index:** $O(N)$.
  - **B+ Tree:** $O(\log N + k)$.`)
  assert.equal((html.match(/<ul>/g) || []).length, 3)
  assert.equal((html.match(/class="katex"/g) || []).length, 4)
  assert.doesNotMatch(html, /\$/)
})

test('renders SIMD math with adjacent prose, emphasis, and inline code', () => {
  const html = render(String.raw`If $\text{timestamp} \ge \text{startTimestamp}$, fill the lane with $1$s, otherwise $0$s.

The ` + '`in_range_mask`' + String.raw` combines $\ge \text{start}$ and $\le \text{end}$ using ` + '`vandq_u32`' + '. Only *both* passing conditions keep the lane set.')
  assert.equal((html.match(/class="katex"/g) || []).length, 5)
  assert.match(html, /<code>in_range_mask<\/code>/)
  assert.match(html, /<em>both<\/em>/)
  assert.match(html, /<\/span>s, otherwise/)
  assert.doesNotMatch(html, /\$/)
})

test('renders block math but leaves dollar signs and whitespace in code alone', () => {
  const html = render('$$\nx^2 + y^2 = z^2\n$$\n\n`$O(1)$`\n\n```cpp\nif (ok) {\n    // $1$ and **literal**\n}\n```')
  assert.match(html, /class="katex-display"/)
  assert.match(html, /<code>\$O\(1\)\$<\/code>/)
  assert.match(html, /<pre><code class="language-cpp">if \(ok\) \{\n    \/\/ \$1\$ and \*\*literal\*\*/)
  assert.match(render(String.raw`It costs \$5. An unfinished expression: $x +`), /It costs \$5\. An unfinished expression: \$x \+/)
})

test('malformed math stays readable without crashing the answer', () => {
  const html = render(String.raw`Before $\frac{1}{$ after.

Next paragraph.`)
  assert.match(html, /Before/)
  assert.match(html, /class="katex-error"/)
  assert.match(html, /Next paragraph/)
})

test('renders tables, headings, quotes and repeated items without merging paragraphs', () => {
  const html = render('### Details\n\nFirst paragraph.\n\nSecond paragraph.\n\n> A quote.\n\n| Key | Value |\n| --- | --- |\n| x | 1 |\n\n- Same\n- Same')
  assert.match(html, /<h3>Details<\/h3>/)
  assert.match(html, /<p>First paragraph\.<\/p>\n<p>Second paragraph\.<\/p>/)
  assert.match(html, /<blockquote>/)
  assert.match(html, /<div class="message-table"><table>/)
  assert.equal((html.match(/<li>Same<\/li>/g) || []).length, 2)
})

test('does not guess list structure from prose or flat bullets', () => {
  assert.match(render('A sentence. 1. Still prose. - Not a new list.'), /<p>A sentence\. 1\. Still prose\. - Not a new list\.<\/p>/)
  const html = render('- Heading\n- Detail')
  assert.equal((html.match(/<ul>/g) || []).length, 1)
})

test('model content cannot run HTML, open local files, or load images', () => {
  const html = render('<script>alert(1)</script>\n\n[bad](javascript:alert) [local](file:///tmp/test) [relative](./test)\n\n![remote](https://example.com/pixel) ![local](file:///tmp/test.png)\n\n$\\href{javascript:alert(1)}{x}$')
  assert.doesNotMatch(html, /<script|<img|<iframe|href=|src=/)
  const link = render('[Docs](https://example.com/docs)')
  assert.match(link, /href="https:\/\/example.com\/docs" target="_blank" rel="noopener noreferrer"/)
})

test('renders standard and bare parentheses/bracket equations', () => {
  for (const expression of [String.raw`\(\frac{1}{2} \times 50 = 25\)`, '( 20 + 50 = 70 )', '( 50 + (0.235 * 50) = 61.75 )']) {
    assert.match(render(expression), /class="katex"/)
    assert.doesNotMatch(render(expression), /katex-display|katex-error/)
  }
  for (const expression of [String.raw`[ \text{Final Price} = $50 + (0.235 \times $50) ]`, String.raw`\[
\frac{1}{2} + \text{Total}_{new} = 70
\]`]) {
    assert.match(render(expression), /class="katex-display"/)
    assert.doesNotMatch(render(expression), /katex-error|TOKSMATH/)
  }
})

test('keeps prices as text alongside existing dollar-delimited math', () => {
  const html = render(String.raw`It costs $50 or $61.75. **Total:** $100. Compare $O(\log N)$, $1$s, and $20 + 50 = 70$.`)
  assert.match(html, /It costs \$50 or \$61\.75\. <strong>Total:<\/strong> \$100\./)
  assert.equal((html.match(/class="katex"/g) || []).length, 3)
  assert.doesNotMatch(html, /katex-error/)
  assert.match(render('($50) [$61.75] (before tax) [estimated]'), /\(\$50\) \[\$61\.75\] \(before tax\) \[estimated\]/)
})

test('keeps new delimiters out of code, links, and reference definitions', () => {
  const html = render('`(1 + 2)` and `\\(x\\)`\n\n```text\n[1 + 2]\n$50 and $61.75\n```\n\n    (3 + 4)\n\n[1 + 2](https://example.com/calc(1+2))\n\n[ref][1 + 2]\n\n[1 + 2]: https://example.com')
  assert.doesNotMatch(html, /class="katex"|TOKSMATH/)
  assert.match(html, /<code>\(1 \+ 2\)<\/code>/)
  assert.match(html, /href="https:\/\/example.com\/calc\(1\+2\)"/)
  assert.match(html, /\$50 and \$61\.75/)
})

test('new math works inside emphasis, lists, tables, and inline suggestions', () => {
  const html = render(String.raw`**Result: (20 + 50 = 70)**

- First: \(x_1 + x_2\)
  - Second: [ \frac{1}{2} ]

| Value |
| --- |
| (1 + 2) |`)
  assert.equal((html.match(/class="katex"/g) || []).length, 4)
  assert.match(html, /<strong>Result:/)
  assert.match(html, /<table>/)
  assert.equal((html.match(/<ul>/g) || []).length, 2)
  const suggestion = renderSuggestion(String.raw`What is \[\frac{1}{2}\]?`)
  assert.match(suggestion, /class="katex-display"/)
  assert.doesNotMatch(suggestion, /<(?:div|p|pre)(?:\s|>)/)
})

test('incomplete and invalid new equations do not break the message', () => {
  for (const input of [String.raw`\(\frac{}\)`, String.raw`[\frac{1}{]`, String.raw`\(20 + 50`, '[20 + 50']) {
    const html = render('Before ' + input + '\n\nAfter.')
    assert.match(html, /Before/)
    assert.match(html, /After\./)
    assert.doesNotMatch(html, /TOKSMATH/)
  }
  const html = render(String.raw`\(\href{javascript:alert(1)}{x}\)`)
  assert.doesNotMatch(html, /href=|<script|<img/)
  assert.match(render('TOKSMATH0END (1 + 2)'), /TOKSMATH0END/)
})


test('currency cannot hide code spans or alter their literal contents', () => {
  const html = render('Pay $50 then `(1 + 2)` and `$61.75`.')
  assert.match(html, /Pay \$50 then <code>\(1 \+ 2\)<\/code> and <code>\$61\.75<\/code>/)
  assert.doesNotMatch(html, /TOKSMATH|class="katex"/)
})
