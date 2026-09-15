import type { Root, RootContent, Text } from 'mdast'
import type { InlineMath } from 'mdast-util-math'
import type { Plugin } from 'unified'

function looksLikeMath(value: string) {
  return /\\[a-zA-Z]+\b/.test(value)
    || (/^[\d\s.,+\-*/=<>^%()$]+$/.test(value) && /\d/.test(value) && /[+*/=<>^]|\d\s*-\s*\d/.test(value))
}

// Use the existing Markdown parser to identify regions that must stay literal.
function literalRanges(tree: Root) {
  const ranges: Array<[number, number]> = []
  function visit(node: Root | RootContent) {
    if (['code', 'inlineCode', 'html', 'link', 'linkReference', 'image', 'imageReference', 'definition'].includes(node.type)) {
      const start = node.position?.start.offset
      const end = node.position?.end.offset
      if (start !== undefined && end !== undefined) ranges.push([start, end])
    } else if ('children' in node) {
      node.children.forEach(visit)
    }
  }
  visit(tree)
  return ranges
}

// Extend the existing remark/KaTeX pipeline without changing Markdown block parsing.
export const remarkModelMath: Plugin<[], Root> = function () {
  const parse = this.parser
  if (!parse) return
  this.parser = (source, file) => {
    // Mask dollars for this structural pass so two prices cannot hide a code
    // span inside a mistakenly parsed inline-math node. Offsets stay unchanged.
    const structuralSource = source.replace(/\$/g, 'S')
    const structure = parse(structuralSource, file) as Root
    const ranges = literalRanges(structure)
    const equations = new Map<string, InlineMath>()
    let prefix = 'TOKSMATH'
    while (source.includes(prefix)) prefix += 'X'
    let normalized = ''
    let rangeIndex = 0

    for (let i = 0; i < source.length;) {
      while (ranges[rangeIndex] && ranges[rangeIndex][1] <= i) rangeIndex++
      const range = ranges[rangeIndex]
      if (range && range[0] <= i) {
        normalized += source.slice(i, range[1])
        i = range[1]
        continue
      }

      // Keep existing $...$ and $$...$$ math. A price must not consume the
      // dollar sign of the next price (or a later equation) as its closing fence.
      if (source[i] === '$') {
        const fence = source[i + 1] === '$' ? '$$' : '$'
        let end = source.indexOf(fence, i + fence.length)
        while (end >= 0 && source[end - 1] === '\\') end = source.indexOf(fence, end + fence.length)
        const value = end < 0 ? '' : source.slice(i + fence.length, end)
        const price = fence === '$' && /^\d/.test(source.slice(i + 1))
          && (end < 0 || /^\d/.test(source.slice(end + 1))
            || (!looksLikeMath(value) && !/^\d[\d,.]*$/.test(value)))
        if (price) {
          normalized += '\\$'
          i++
          continue
        }
        if (end >= 0) {
          normalized += source.slice(i, end + fence.length)
          i = end + fence.length
          continue
        }
      }

      const explicit = source[i] === '\\' && /[([]/.test(source[i + 1] ?? '')
      const open = source[i + (explicit ? 1 : 0)]
      if (open === '(' || open === '[') {
        const start = i + (explicit ? 2 : 1)
        const close = open === '(' ? ')' : ']'
        let end = -1
        if (explicit) {
          end = source.indexOf('\\' + close, start)
        } else {
          let depth = 1
          for (let j = start; j < source.length; j++) {
            if (source[j] === '\\') { j++; continue }
            if (source[j] === open) depth++
            if (source[j] === close && --depth === 0) { end = j; break }
          }
        }
        const value = source.slice(start, end)
        // Never turn an expression spanning a code span or link into math.
        if (end >= 0 && (!range || end < range[0]) && (explicit || looksLikeMath(value))) {
          const token = `${prefix}${equations.size}END`
          const expression = value.replace(/(?<!\\)\$/g, '\\$')
          equations.set(token, {
            type: 'inlineMath',
            value: expression,
            data: {
              hName: 'code',
              hProperties: { className: ['language-math', open === '[' ? 'math-display' : 'math-inline'] },
              hChildren: [{ type: 'text', value: expression }]
            }
          })
          normalized += token
          i = end + (explicit ? 2 : 1)
          continue
        }
        if (explicit && end < 0) {
          normalized += source.slice(i)
          break
        }
      }
      // Preserve Markdown escapes, including escaped dollars and backslashes.
      const length = source[i] === '\\' && i + 1 < source.length ? 2 : 1
      normalized += source.slice(i, i + length)
      i += length
    }

    if (normalized === structuralSource) return structure
    const tree = parse(normalized, file) as Root
    const tokenPattern = new RegExp(`(${prefix}\\d+END)`, 'g')
    function restore(node: Root | RootContent) {
      if (!('children' in node)) return
      // Only text nodes receive equations; code, links, and existing math stay intact.
      const children: RootContent[] = []
      for (const child of node.children) {
        if (child.type === 'text') {
          for (const part of child.value.split(tokenPattern)) {
            if (part) children.push(equations.get(part) ?? { type: 'text', value: part } satisfies Text)
          }
        } else {
          restore(child)
          children.push(child)
        }
      }
      node.children = children as typeof node.children
    }
    restore(tree)
    return tree
  }
}
