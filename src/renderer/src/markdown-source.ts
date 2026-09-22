import type { Element, Nodes, Properties, Root } from 'hast'

export interface SourceLineRange {
  from: number
  to: number
}

export function resolveSourceLineRange(text: string, lineFrom?: number, lineTo?: number, chunkText = ''): SourceLineRange | undefined {
  const normalized = text.replace(/\r\n?/g, '\n')
  const lineCount = normalized.split('\n').length
  if (lineFrom !== undefined && Number.isInteger(lineFrom) && lineFrom >= 1 && lineFrom <= lineCount) {
    return {
      from: lineFrom,
      to: lineTo !== undefined && Number.isInteger(lineTo) && lineTo >= lineFrom
        ? Math.min(lineTo, lineCount)
        : lineFrom
    }
  }

  const chunk = chunkText.replace(/\r\n?/g, '\n').trim()
  const offset = chunk ? normalized.indexOf(chunk) : -1
  if (offset < 0) return undefined
  const from = normalized.slice(0, offset).split('\n').length
  return { from, to: from + chunk.split('\n').length - 1 }
}

// Prefer the smallest rendered block containing the line. On a hidden comment
// or blank line, use the next visible block, or the last block at end of file.
export function sourceAnchorIndex(ranges: SourceLineRange[], line: number): number {
  let containing = -1
  let next = -1
  let previous = -1
  ranges.forEach((range, index) => {
    if (range.from <= line && range.to >= line &&
        (containing < 0 || range.to - range.from < ranges[containing].to - ranges[containing].from)) {
      containing = index
    }
    if (range.from > line && (next < 0 || range.from < ranges[next].from)) next = index
    if (range.to < line && (previous < 0 || range.to >= ranges[previous].to)) previous = index
  })
  return containing >= 0 ? containing : next >= 0 ? next : previous
}

const sourceBlocks = new Set(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'ul', 'ol', 'li', 'blockquote', 'pre', 'table', 'tr', 'hr'])

export function rehypeSourceLines({ range }: { range?: SourceLineRange }) {
  return (tree: Root, file: { value: unknown }) => {
    const source = String(file.value)
    const properties = (from: number, to: number): Properties => ({
      dataSourceLineStart: from,
      dataSourceLineEnd: to,
      ...(range && from <= range.to && to >= range.from ? { dataSourceSelected: 'true' } : {})
    })

    const visit = (node: Nodes) => {
      if (node.type === 'element' && node.position && sourceBlocks.has(node.tagName)) {
        const { start, end } = node.position
        const code = node.tagName === 'pre' && node.children[0]?.type === 'element' && node.children[0].tagName === 'code'
          ? node.children[0] : undefined
        const classes = code?.properties.className
        if (code && Array.isArray(classes) && classes.includes('language-math')) {
          // KaTeX replaces the pre node. Keep its original position on a wrapper.
          const math: Element = { ...node }
          node.tagName = 'div'
          node.properties = { className: ['markdown-source-math'], ...properties(start.line, end.line) }
          node.children = [math]
          return
        }

        node.properties = { ...node.properties, ...properties(start.line, end.line) }
        if (code && code.children.length === 1 && code.children[0].type === 'text') {
          const rawStart = source.slice(start.offset, (start.offset ?? 0) + 20)
          const firstLine = start.line + (/^\s{0,3}(?:`{3,}|~{3,})/.test(rawStart) ? 1 : 0)
          // Preserve code whitespace and provide exact anchors inside long fences.
          code.children = code.children[0].value.split(/(?<=\n)/).map((value, index) => ({
            type: 'element',
            tagName: 'span',
            properties: { className: ['markdown-source-code-line'], ...properties(firstLine + index, firstLine + index) },
            children: [{ type: 'text', value }]
          }))
        }
      }
      if ('children' in node) node.children.forEach(visit)
    }
    visit(tree)
  }
}
