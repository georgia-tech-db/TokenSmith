import type { LocalModelRole } from '../../shared/app-state'
import type { OllamaSearchResult } from '../../shared/ollama'

const ollamaLibraryBaseUrl = 'https://ollama.com'
const ollamaSearchTimeoutMs = 15_000

function decodeHtml(value: string): string {
  const namedEntities: Record<string, string> = {
    amp: '&',
    apos: "'",
    gt: '>',
    lt: '<',
    nbsp: ' ',
    quot: '"'
  }

  return value.replace(/&(#x?[0-9a-f]+|[a-z]+);/gi, (entity, name: string) => {
    const lowerName = name.toLowerCase()
    if (lowerName.startsWith('#x')) {
      return String.fromCodePoint(Number.parseInt(lowerName.slice(2), 16))
    }
    if (lowerName.startsWith('#')) {
      return String.fromCodePoint(Number.parseInt(lowerName.slice(1), 10))
    }
    return namedEntities[lowerName] ?? entity
  })
}

function stripHtml(value: string): string {
  return decodeHtml(value.replace(/<[^>]*>/g, ' '))
    .replace(/\s+/g, ' ')
    .trim()
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}

function textForAttribute(html: string, attribute: string): string | undefined {
  const pattern = new RegExp(`<[^>]*\\b${escapeRegExp(attribute)}\\b[^>]*>([\\s\\S]*?)<\\/[^>]+>`, 'i')
  const match = html.match(pattern)
  const text = match ? stripHtml(match[1]) : ''
  return text || undefined
}

function textsForAttribute(html: string, attribute: string): string[] {
  const pattern = new RegExp(`<[^>]*\\b${escapeRegExp(attribute)}\\b[^>]*>([\\s\\S]*?)<\\/[^>]+>`, 'gi')
  return Array.from(html.matchAll(pattern))
    .map((match) => stripHtml(match[1]))
    .filter(Boolean)
}

function descriptionForModel(html: string): string {
  const paragraphs = Array.from(html.matchAll(/<p\b[^>]*>([\s\S]*?)<\/p>/gi))
    .map((match) => stripHtml(match[1]))
    .filter(Boolean)

  return paragraphs[0] ?? ''
}

function modelNameForItem(html: string): string | undefined {
  const searchTitle = textForAttribute(html, 'x-test-search-response-title')
  if (searchTitle) {
    return searchTitle
  }

  const titleAttribute = html.match(/<[^>]*\bx-test-model-title\b[^>]*\btitle=["']([^"']+)["'][^>]*>/i)?.[1]
  if (titleAttribute) {
    return decodeHtml(titleAttribute).trim()
  }

  return textForAttribute(html, 'x-test-model-title')
}

function parseTagCount(value?: string): number | undefined {
  if (!value) {
    return undefined
  }

  const count = Number.parseInt(value.replace(/,/g, ''), 10)
  return Number.isFinite(count) ? count : undefined
}

function absoluteOllamaUrl(href?: string): string {
  if (!href) {
    return `${ollamaLibraryBaseUrl}/library`
  }

  return new URL(href, ollamaLibraryBaseUrl).toString()
}

function normalizeSearchText(value: string): string {
  return value.toLowerCase().replace(/[^a-z0-9]+/g, ' ').trim()
}

function searchTokens(query: string): string[] {
  return normalizeSearchText(query)
    .split(/\s+/)
    .map((token) => token.trim())
    .filter((token) => token.length >= 2)
}

function resultMatchesQuery(result: OllamaSearchResult, query: string): boolean {
  const tokens = searchTokens(query)
  if (tokens.length === 0) {
    return true
  }

  const searchableText = normalizeSearchText(
    [
      result.name,
      result.description,
      result.capabilities.join(' '),
      result.sizes.join(' ')
    ].join(' ')
  )
  const compactSearchableText = searchableText.replace(/\s+/g, '')

  return tokens.every((token) => searchableText.includes(token) || compactSearchableText.includes(token))
}

export function parseOllamaSearchResults(html: string): OllamaSearchResult[] {
  const resultItems = Array.from(html.matchAll(/<li\b[^>]*\bx-test-model\b[^>]*>([\s\S]*?)<\/li>/gi))
  const seen = new Set<string>()

  return resultItems
    .map((match): OllamaSearchResult | undefined => {
      const itemHtml = match[1]
      const name = modelNameForItem(itemHtml)
      if (!name || seen.has(name.toLowerCase())) {
        return undefined
      }

      seen.add(name.toLowerCase())
      const href = itemHtml.match(/<a\b[^>]*href=["']([^"']+)["'][^>]*>/i)?.[1]

      return {
        name,
        description: descriptionForModel(itemHtml),
        url: absoluteOllamaUrl(href),
        capabilities: textsForAttribute(itemHtml, 'x-test-capability'),
        sizes: textsForAttribute(itemHtml, 'x-test-size'),
        pulls: textForAttribute(itemHtml, 'x-test-pull-count'),
        tagCount: parseTagCount(textForAttribute(itemHtml, 'x-test-tag-count')),
        updated: textForAttribute(itemHtml, 'x-test-updated')
      }
    })
    .filter((result): result is OllamaSearchResult => Boolean(result))
}

export async function searchOllamaLibrary(
  query: string,
  role: LocalModelRole = 'generator',
  limit = 20
): Promise<OllamaSearchResult[]> {
  const normalizedQuery = query.trim()
  if (!normalizedQuery) {
    return []
  }

  const searchUrl = new URL('/library', ollamaLibraryBaseUrl)
  searchUrl.searchParams.set('q', normalizedQuery)
  if (role === 'embedder') {
    searchUrl.searchParams.set('c', 'embedding')
  }

  const controller = new AbortController()
  const timeout = setTimeout(() => controller.abort(), ollamaSearchTimeoutMs)

  try {
    const response = await fetch(searchUrl, {
      headers: {
        Accept: 'text/html,application/xhtml+xml'
      },
      signal: controller.signal
    })

    if (!response.ok) {
      throw new Error(`Ollama search failed with HTTP ${response.status}.`)
    }

    return parseOllamaSearchResults(await response.text())
      .filter((result) => resultMatchesQuery(result, normalizedQuery))
      .slice(0, Math.max(1, limit))
  } catch (error) {
    if (error instanceof Error && error.name === 'AbortError') {
      throw new Error('Ollama search timed out.')
    }
    throw error
  } finally {
    clearTimeout(timeout)
  }
}
