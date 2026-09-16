import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'

const rootDir = resolve(new URL('../..', import.meta.url).pathname)

export const buzzdbFixturePath = resolve(rootDir, 'tests/fixtures/buzzdb/buzzdb-book.tokensmith.md')
export const buzzdbFixtureText = readFileSync(buzzdbFixturePath, 'utf8')

const chunkHeaderPattern = /^<!-- tokensmith:chunk id="([^"]+)" chapter="([^"]+)" section="([^"]+)" kind="([^"]+)" -->$/
const genericTerms = new Set([
  'a',
  'always',
  'also',
  'an',
  'and',
  'are',
  'as',
  'at',
  'bad',
  'be',
  'best',
  'better',
  'but',
  'by',
  'can',
  'cant',
  'cannot',
  'could',
  'did',
  'do',
  'does',
  'doing',
  'dont',
  'from',
  'for',
  'good',
  'had',
  'has',
  'have',
  'how',
  'in',
  'into',
  'important',
  'is',
  'it',
  'its',
  'just',
  'mean',
  'more',
  'need',
  'needed',
  'of',
  'on',
  'one',
  'or',
  'our',
  'over',
  'prefer',
  'preferable',
  'preferred',
  'prefers',
  'should',
  'so',
  'than',
  'that',
  'the',
  'their',
  'them',
  'then',
  'there',
  'these',
  'they',
  'those',
  'to',
  'use',
  'used',
  'uses',
  'using',
  'this',
  'was',
  'we',
  'were',
  'what',
  'when',
  'where',
  'which',
  'who',
  'why',
  'with',
  'would',
  'worse',
  'worst',
  'you',
  'your'
])

function parseChunks() {
  const lines = buzzdbFixtureText.split('\n')
  const chunks = new Map()
  let current

  function finish(endIndex) {
    if (!current) {
      return
    }

    current.text = lines.slice(current.bodyStartIndex, endIndex).join('\n').trim()
    current.lineTo = endIndex
    chunks.set(current.id, current)
  }

  for (let index = 0; index < lines.length; index += 1) {
    const match = lines[index].match(chunkHeaderPattern)
    if (!match) {
      continue
    }

    finish(index)
    current = {
      id: match[1],
      chapter: match[2],
      section: match[3],
      kind: match[4],
      bodyStartIndex: index + 1,
      lineFrom: index + 2,
      lineTo: index + 1,
      text: ''
    }
  }

  finish(lines.length)
  return chunks
}

const chunksById = parseChunks()

export function buzzdbChunk(id) {
  const chunk = chunksById.get(id)
  assert.ok(chunk, `missing BuzzDB fixture chunk ${id}`)
  return chunk
}

export function buzzdbSource(id, overrides = {}) {
  const chunk = buzzdbChunk(id)

  return {
    title: 'BuzzDB Book',
    locator: `${chunk.id} ${chunk.section}`,
    excerpt: chunk.text.slice(0, 1200),
    context: chunk.text,
    collectionName: 'BuzzDBBook',
    documentTitle: 'BuzzDB Book',
    sectionHeader: chunk.section,
    path: buzzdbFixturePath,
    lineFrom: chunk.lineFrom,
    lineTo: chunk.lineTo,
    chunkId: chunk.id,
    ...overrides
  }
}

function searchTerms(text) {
  const normalized = text
    .normalize('NFKD')
    .replace(/[\u0300-\u036f]/g, '')
    .toLowerCase()
  const tokens = normalized.match(/[a-z0-9][a-z0-9+#.-]*/g) ?? []
  const expanded = []

  for (const token of tokens) {
    expanded.push(token)

    const compact = token.replace(/[^a-z0-9]+/g, '')
    if (compact && compact !== token) {
      expanded.push(compact)
    }
  }

  return Array.from(new Set(
    expanded.filter((term) => !genericTerms.has(term) && (term.length >= 3 || /[0-9+#]/.test(term)))
  ))
}

export async function searchBuzzdbFixture(query, sources, calls, limit = 2) {
  calls.push(query)
  const queryTerms = searchTerms(query)
  if (queryTerms.length === 0) {
    return []
  }

  return sources
    .map((source) => {
      const sourceText = [
        source.chunkId,
        source.sectionHeader,
        source.excerpt,
        source.context
      ].join(' ').toLowerCase()
      const score = queryTerms.filter((term) => sourceText.includes(term)).length
      return { source, score }
    })
    .filter((hit) => hit.score > 0)
    .sort((left, right) => right.score - left.score)
    .slice(0, limit)
    .map((hit) => hit.source)
}
