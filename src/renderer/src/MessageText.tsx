import type { ReactNode } from 'react'
import katex from 'katex'

// Do not use dollar delimiters: model answers often contain ordinary prices.
// Bare delimiters are accepted only for LaTeX commands or numeric arithmetic.
function looksLikeMath(value: string) {
  return /\\(?:frac|times|text|sqrt|cdot|div|boxed)\b/.test(value)
    || (/^[\d\s.,+\-*/=<>^%()$]+$/.test(value) && /\d/.test(value) && /[+*/=<>^]|\d\s*-\s*\d/.test(value))
}

function mathNode(source: string, expression: string, displayMode: boolean, key: string): ReactNode {
  try {
    // A dollar within an equation is a literal currency symbol, too.
    const html = katex.renderToString(expression.replace(/(^|[^\\])\$/g, '$1\\$'), {
      displayMode,
      throwOnError: true,
      trust: false,
      strict: 'ignore',
      maxSize: 10,
      maxExpand: 1000
    })
    return <span key={key} className={displayMode ? 'message-math-display' : 'message-math-inline'} dangerouslySetInnerHTML={{ __html: html }} />
  } catch {
    // React escapes the original text; incomplete streamed equations remain readable.
    return source
  }
}

function protectMath(text: string) {
  const math = new Map<string, ReactNode>()
  let prefix = 'MATHPLACEHOLDER'
  while (text.includes(prefix)) prefix += 'X'
  let protectedText = ''
  let cursor = 0
  const opening = /\\[([]|[([]/g
  let match: RegExpExecArray | null
  while ((match = opening.exec(text)) !== null) {
    const explicit = match[0].startsWith('\\')
    const bracket = match[0].endsWith('[')
    const close = bracket ? ']' : ')'
    const contentStart = opening.lastIndex
    let end = -1
    let closeLength = 1
    if (explicit) {
      end = text.indexOf('\\' + close, contentStart)
      closeLength = 2
      if (end < 0) break
    } else {
      let depth = 1
      for (let i = contentStart; i < text.length; i++) {
        if (text[i - 1] === '\\') continue
        if (text[i] === match[0]) depth++
        if (text[i] === close && --depth === 0) { end = i; break }
      }
      if (end < 0) continue
    }
    const expression = text.slice(contentStart, end)
    if (!explicit && !looksLikeMath(expression)) continue
    const token = `${prefix}${math.size}END`
    math.set(token, mathNode(text.slice(match.index, end + closeLength), expression, bracket, token))
    protectedText += text.slice(cursor, match.index) + token
    cursor = end + closeLength
    opening.lastIndex = cursor
  }
  return { protectedText: protectedText + text.slice(cursor), math }
}

function renderMathTokens(text: string, math: Map<string, ReactNode>): ReactNode[] {
  if (math.size === 0) return [text]
  const pattern = new RegExp(`(${[...math.keys()].join('|')})`, 'g')
  return text.split(pattern).map((part) => math.get(part) ?? part)
}

function normalizeMessageMarkdown(text: string) {
  return text
    .replace(/\r\n/g, '\n')
    .replace(/([.!?:;)])\s+([*-])\s+(?=\*\*|[A-Z0-9])/g, '$1\n$2 ')
    .replace(/(\*\*)\s+([*-])\s+(?=\*\*|[A-Z0-9])/g, '$1\n$2 ')
    .replace(/([.!?:;)])\s+(\d+\.)\s+(?=\*\*|[A-Z0-9])/g, '$1\n$2 ')
    .trim()
}

function renderInlineMarkdown(text: string, math: Map<string, ReactNode>): ReactNode[] {
  const parts: ReactNode[] = []
  const boldPattern = /\*\*([^*]+)\*\*/g
  let lastIndex = 0
  let match: RegExpExecArray | null

  while ((match = boldPattern.exec(text)) !== null) {
    if (match.index > lastIndex) {
      parts.push(...renderMathTokens(text.slice(lastIndex, match.index), math))
    }

    parts.push(<strong key={`${match.index}-${match[1]}`}>{renderMathTokens(match[1], math)}</strong>)
    lastIndex = match.index + match[0].length
  }

  if (lastIndex < text.length) {
    parts.push(...renderMathTokens(text.slice(lastIndex), math))
  }

  return parts
}

export function MessageText({ text }: { text: string }) {
  const { protectedText, math } = protectMath(text)
  const normalizedText = normalizeMessageMarkdown(protectedText)
  const blocks: ReactNode[] = []
  let paragraphLines: string[] = []
  let bulletItems: string[] = []
  let numberedItems: string[] = []

  function flushParagraph() {
    if (paragraphLines.length === 0) {
      return
    }

    const paragraph = paragraphLines.join(' ').replace(/\s+/g, ' ').trim()
    if (paragraph) {
      blocks.push(<p key={`p-${blocks.length}`}>{renderInlineMarkdown(paragraph, math)}</p>)
    }
    paragraphLines = []
  }

  function flushBullets() {
    if (bulletItems.length === 0) {
      return
    }

    blocks.push(
      <ul key={`ul-${blocks.length}`}>
        {bulletItems.map((item) => (
          <li key={item}>{renderInlineMarkdown(item, math)}</li>
        ))}
      </ul>
    )
    bulletItems = []
  }

  function flushNumbered() {
    if (numberedItems.length === 0) {
      return
    }

    blocks.push(
      <ol key={`ol-${blocks.length}`}>
        {numberedItems.map((item) => (
          <li key={item}>{renderInlineMarkdown(item, math)}</li>
        ))}
      </ol>
    )
    numberedItems = []
  }

  for (const rawLine of normalizedText.split('\n')) {
    const line = rawLine.trim()

    if (!line) {
      flushParagraph()
      flushBullets()
      flushNumbered()
      continue
    }

    const bulletMatch = line.match(/^[-*]\s+(.+)$/)
    if (bulletMatch) {
      flushParagraph()
      flushNumbered()
      bulletItems.push(bulletMatch[1])
      continue
    }

    const numberedMatch = line.match(/^\d+\.\s+(.+)$/)
    if (numberedMatch) {
      flushParagraph()
      flushBullets()
      numberedItems.push(numberedMatch[1])
      continue
    }

    flushBullets()
    flushNumbered()
    paragraphLines.push(line)
  }

  flushParagraph()
  flushBullets()
  flushNumbered()

  return (
    <div className="message-text">
      {blocks.length > 0 ? blocks : <p>{text}</p>}
    </div>
  )
}

