import assert from 'node:assert/strict'
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { pathToFileURL } from 'node:url'
import test from 'node:test'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import ts from 'typescript'

const runtime = resolve('.coverage-ts-runtime/markdown-source-viewer')
mkdirSync(runtime, { recursive: true })
for (const filename of ['markdown-source.ts', 'MarkdownSourceViewer.tsx']) {
  const sourcePath = resolve('src/renderer/src', filename)
  const output = ts.transpileModule(readFileSync(sourcePath, 'utf8'), {
    fileName: sourcePath,
    compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022, jsx: ts.JsxEmit.ReactJSX }
  }).outputText.replace("from './markdown-source'", "from './markdown-source.mjs'")
  writeFileSync(resolve(runtime, filename.replace(/\.tsx?$/, '.mjs')), output)
}
const { resolveSourceLineRange, sourceAnchorIndex } = await import(pathToFileURL(resolve(runtime, 'markdown-source.mjs')).href)
const { MarkdownSourceViewer } = await import(pathToFileURL(resolve(runtime, 'MarkdownSourceViewer.mjs')).href)
const render = (text, lineFrom, lineTo, chunkText = '') => renderToStaticMarkup(createElement(MarkdownSourceViewer, {
  viewer: { title: 'Source', path: '/study/source.md', text, lineFrom, lineTo, chunkText }, onClose() {}
}))

test('keeps the full document in one rendered pane and counts hidden comments in source offsets', () => {
  const html = render('# First section\n\nEarlier text.\n\n<!-- tokensmith-chunk id="42" -->\n\n## Selected section\n\n**Cited** text.\n\n## Later section\n\nLater text.', 7, 9)
  assert.equal((html.match(/aria-label="Full source document"/g) || []).length, 1)
  assert.match(html, /<h1 data-source-line-start="1" data-source-line-end="1">First section<\/h1>/)
  assert.match(html, /<h2 data-source-line-start="7" data-source-line-end="7" data-source-selected="true">Selected section<\/h2>/)
  assert.match(html, /<p data-source-line-start="9" data-source-line-end="9" data-source-selected="true"><strong>Cited<\/strong> text\.<\/p>/)
  assert.match(html, /<h2 data-source-line-start="11" data-source-line-end="11">Later section<\/h2>/)
  assert.doesNotMatch(html, /tokensmith-chunk|Full Chunk|Markdown File|<pre>/)
})

test('targets a line within fenced code without losing whitespace or code formatting', () => {
  const html = render('# Code\n\n```cpp\nint first = 1;\n  int second = 2;\nint third = 3;\n```\n\nAfter.', 5, 5)
  assert.match(html, /<code class="language-cpp">/)
  assert.match(html, /data-source-line-start="4" data-source-line-end="4">int first = 1;\n<\/span>/)
  assert.match(html, /data-source-line-start="5" data-source-line-end="5" data-source-selected="true">  int second = 2;\n<\/span>/)
  assert.match(html, /data-source-line-start="6" data-source-line-end="6">int third = 3;\n<\/span>/)
  assert.doesNotMatch(html, /class="markdown-source-code-line" data-source-line-start="7"/)
})

test('handles indented and nested fenced code offsets', () => {
  const indented = render('Before.\n\n    first\n    second\n\nAfter.', 4, 4)
  assert.match(indented, /data-source-line-start="4" data-source-line-end="4" data-source-selected="true">second\n<\/span>/)
  const nested = render('> ```js\n> first\n> second\n> ```', 3, 3)
  assert.match(nested, /data-source-line-start="3" data-source-line-end="3" data-source-selected="true">second\n<\/span>/)
})

test('keeps math anchors when KaTeX replaces the original block', () => {
  const html = render('Before.\n\n$$\nx^2 + y^2\n$$\n\nAfter.', 4, 4)
  assert.match(html, /class="markdown-source-math" data-source-line-start="3" data-source-line-end="5" data-source-selected="true"/)
  assert.match(html, /class="katex-display"/)
  assert.match(html, /data-source-line-start="7" data-source-line-end="7">After\./)
})

test('renders GFM tables and nested lists with usable block anchors', () => {
  const html = render('| Key | Value |\n| --- | --- |\n| a | 1 |\n| b | 2 |\n\n- Parent\n  - Child', 4, 7)
  assert.match(html, /<div class="message-table"><table data-source-line-start="1"/)
  assert.match(html, /<tr data-source-line-start="4" data-source-line-end="4" data-source-selected="true">/)
  assert.match(html, /<li data-source-line-start="7" data-source-line-end="7" data-source-selected="true">Child<\/li>/)
})

test('handles CRLF, missing offsets, invalid ranges, and unmatched chunks honestly', () => {
  const text = '# Title\r\n\r\nSelected line.\r\nSecond line.\r\n'
  assert.deepEqual(resolveSourceLineRange(text, 3, 99), { from: 3, to: 5 })
  assert.deepEqual(resolveSourceLineRange(text, 3, 1), { from: 3, to: 3 })
  assert.deepEqual(resolveSourceLineRange(text, undefined, undefined, 'Selected line.\nSecond line.'), { from: 3, to: 4 })
  assert.deepEqual(resolveSourceLineRange(text, -1, 100, 'Second line.'), { from: 4, to: 4 })
  assert.equal(resolveSourceLineRange(text, 500, 600, 'Unknown chunk'), undefined)
  const html = render(text, undefined, undefined, 'Unknown chunk')
  assert.match(html, /Chunk position unavailable/)
  assert.doesNotMatch(html, /data-source-selected/)
})

test('chooses the closest precise anchor and skips unrendered lines', () => {
  const ranges = [{ from: 1, to: 1 }, { from: 5, to: 20 }, { from: 6, to: 6 }, { from: 12, to: 15 }, { from: 25, to: 26 }]
  assert.equal(sourceAnchorIndex(ranges, 6), 2)
  assert.equal(sourceAnchorIndex(ranges, 14), 3)
  assert.equal(sourceAnchorIndex(ranges, 3), 1)
  assert.equal(sourceAnchorIndex(ranges, 23), 4)
  assert.equal(sourceAnchorIndex(ranges, 100), 4)
  assert.equal(sourceAnchorIndex([], 1), -1)
})

test('source content cannot run HTML, load remote images, or open local files', () => {
  const html = render('<script>alert(1)</script>\n\n![pixel](https://example.com/pixel)\n\n[local](file:///tmp/secret) [bad](javascript:alert) [Docs](https://example.com/docs)', 3, 5)
  assert.doesNotMatch(html, /<script|<img|<iframe|src=|href="file:|href="javascript:/)
  assert.match(html, /href="https:\/\/example.com\/docs" target="_blank" rel="noopener noreferrer"/)
})
