import type { ChatSource } from '@shared/app-state'

export function sourceEntryKey(source: ChatSource): string {
  return JSON.stringify([source.path, source.tokensmithChunkId ?? source.chunkId ?? source.chunkRowid,
    source.pageStart, source.pageEnd, source.lineFrom, source.lineTo, source.locator, source.excerpt])
}
export function navigableSource(source: ChatSource): boolean {
  return /\.(pdf|md|markdown|txt)$/i.test(source.path ?? '')
}
export function sourceNavigationList(source: ChatSource, candidates: ChatSource[]): { sources: ChatSource[]; index: number } {
  const seen = new Set<string>()
  const sources = candidates.filter(candidate => {
    const key = sourceEntryKey(candidate)
    if (!navigableSource(candidate) || seen.has(key)) return false
    seen.add(key)
    return true
  })
  let index = sources.findIndex(candidate => sourceEntryKey(candidate) === sourceEntryKey(source))
  if (index < 0 && navigableSource(source)) { sources.unshift(source); index = 0 }
  return { sources, index }
}

// Library navigation visits files, retaining the clicked passage in the first file.
export function collectionDocumentSources(source: ChatSource, documents: Array<{ path: string; title: string }>): ChatSource[] {
  return documents.map(document => document.path === source.path ? source : {
    path: document.path, title: document.title, documentTitle: document.title,
    materialId: source.materialId, excerpt: '', locator: /\.pdf$/i.test(document.path) ? 'Page 1' : 'Start of document',
    ...(/\.pdf$/i.test(document.path) ? { pageStart: 1 } : { lineFrom: 1, lineTo: 1 })
  }).filter(navigableSource)
}
