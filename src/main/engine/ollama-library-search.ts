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

function modelLink(html: string): { name: string; url: string } | undefined {
  for (const match of html.matchAll(/<a\b[^>]*\bhref=["']([^"']+)["'][^>]*>/gi)) {
    try {
      const url = new URL(decodeHtml(match[1]), ollamaLibraryBaseUrl)
      const name = url.pathname.match(/^\/library\/([a-z0-9][a-z0-9._-]*)\/?$/i)?.[1]
      if (url.origin === ollamaLibraryBaseUrl && name) return { name, url: url.toString() }
    } catch { /* Ignore links that are not model pages. */ }
  }
  return undefined
}

function visibleCardMetadata(html: string) {
  const paragraphs = Array.from(html.matchAll(/<p\b[^>]*>([\s\S]*?)<\/p>/gi))
  const stats = paragraphs.find((match) => /\b(?:Pulls|Tags|Updated)\b/.test(stripHtml(match[1])))
  const text = stats ? stripHtml(stats[1]) : ''
  const badges = Array.from(html.slice(0, stats?.index ?? html.length).matchAll(/<span\b[^>]*>([^<]*)<\/span>/gi))
    .map((match) => stripHtml(match[1]))
  return {
    capabilities: badges.filter((value) => /^(?:tools|thinking|vision|embedding|cloud|completion|insert)$/i.test(value)),
    sizes: badges.filter((value) => /^\d+(?:\.\d+)?[bmt]$/i.test(value)),
    pulls: text.match(/([\d.,]+[KMBT]?)\s+Pulls\b/i)?.[1],
    tagCount: parseTagCount(text.match(/([\d,]+)\s+Tags\b/i)?.[1]),
    updated: text.match(/\bUpdated\s+(.+)$/i)?.[1]
  }
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
  // Ollama's public pages no longer consistently include their internal x-test attributes.
  const resultItems = Array.from(html.matchAll(/<li\b[^>]*>([\s\S]*?)<\/li>/gi))
  const seen = new Set<string>()

  return resultItems
    .map((match): OllamaSearchResult | undefined => {
      const itemHtml = match[1]
      const link = modelLink(itemHtml)
      if (!link || !/<h[1-6]\b|\bx-test-(?:search-response-title|model-title)\b/i.test(itemHtml)) return undefined
      const name = modelNameForItem(itemHtml) ?? link.name
      if (seen.has(name.toLowerCase())) {
        return undefined
      }

      seen.add(name.toLowerCase())
      const visible = visibleCardMetadata(itemHtml)
      const capabilities = textsForAttribute(itemHtml, 'x-test-capability')
      const sizes = textsForAttribute(itemHtml, 'x-test-size')

      return {
        name,
        description: descriptionForModel(itemHtml),
        url: link.url,
        capabilities: capabilities.length ? capabilities : visible.capabilities,
        sizes: sizes.length ? sizes : visible.sizes,
        pulls: textForAttribute(itemHtml, 'x-test-pull-count') ?? visible.pulls,
        tagCount: parseTagCount(textForAttribute(itemHtml, 'x-test-tag-count')) ?? visible.tagCount,
        updated: textForAttribute(itemHtml, 'x-test-updated') ?? visible.updated
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

    const html = await response.text()
    const results = parseOllamaSearchResults(html)
    if (!results.length && !/\bid=["']repo["']|No (?:models|results) found/i.test(html)) {
      throw new Error('Could not read Ollama’s model catalog. Please try again or enter a model name directly.')
    }
    return results
      .filter((result) => role !== 'embedder' || result.capabilities.includes('embedding'))
      .filter((result) => role !== 'generator' || !result.capabilities.includes('embedding'))
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
