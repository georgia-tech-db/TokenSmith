import type { ChatSource } from '@shared/app-state'

export function cleanMaterialTitle(title?: string) {
  return (title ?? '').replace(/\.(pdf|md|markdown|txt)$/i, '').replace(/\s+/g, ' ').trim()
}

export function isPdfSource(source: ChatSource) {
  return Boolean(source.path?.toLowerCase().endsWith('.pdf'))
}

export function isMarkdownSource(source: ChatSource) {
  const path = source.path?.toLowerCase() ?? ''
  return path.endsWith('.md') || path.endsWith('.markdown') || path.endsWith('.txt')
}

export function canOpenSource(source: ChatSource) {
  return isPdfSource(source) || isMarkdownSource(source)
}
