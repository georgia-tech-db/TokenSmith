import type { ChatSource } from './app-state'
import type { PracticeTopic } from './practice'
import { practiceSourceKey } from './practice'

export const topicInventoryVersion = 3
export interface InventoryTopic extends PracticeTopic {
  id: string
  area: string
  aliases: string[]
  objectives: string[]
  sources: ChatSource[]
}
export interface TopicBatch { id: string; fingerprint: string; area: string; chapterId: string; sources: ChatSource[] }
export interface TopicCheckpoint { fingerprint: string; topics: InventoryTopic[] }
export interface TopicInventory {
  version: 3
  collectionId: string
  indexRevision: string
  modelId: string
  batches: Record<string, TopicCheckpoint>
  topics: InventoryTopic[]
  complete: boolean
  completed: number
  total: number
}
// Content/checkpoint identity, not a security hash.
export function topicHash(value: string): string {
  let a = 2166136261, b = 5381
  for (const c of value) { a = Math.imul(a ^ c.charCodeAt(0), 16777619); b = Math.imul(b, 33) ^ c.charCodeAt(0) }
  return `${(a >>> 0).toString(36)}${(b >>> 0).toString(36)}`
}
function nameKey(value: string) { return value.normalize('NFKC').toLowerCase().replace(/\s+/g, ' ').trim() }

/** Keep all substantive indexed text; split only when a chapter exceeds the bounded context budget. */
export function buildTopicBatches(sources: ChatSource[], maxCharacters = 65000): TopicBatch[] {
  if (maxCharacters < 1000) throw new Error('Chapter context budget is too small.')
  const chapters = new Map<string, { area: string; sources: ChatSource[] }>()
  const current = new Map<string, { id: string; area: string; exercises: boolean }>()
  for (const original of sources) {
    const doc = `${original.materialId}:${original.documentId}`
    const text = (original.context || original.excerpt || '').trim()
    const heading = (original.sectionHeader || '').replace(/^#+\s*/, '').trim()
    const chapter = heading.match(/^Chapter\s+(\d+)\.?\s+(.+)/i) || text.match(/^#{0,3}\s*Chapter\s+(\d+)\.?\s+([^\n]+)/i)
    let state = current.get(doc)
    if (chapter) state = { id: `${doc}:chapter-${chapter[1]}`, area: `Chapter ${chapter[1]} — ${chapter[2]}`, exercises: false }
    if (!state) state = { id: doc, area: original.documentTitle || original.title, exercises: false }
    if (/^Exercises\b/i.test(text) && /\.pdf$/i.test(original.path || '')) state.exercises = true
    current.set(doc, state)
    const cleanHeading = heading.replace(/^\d+(?:\.\d+)*\.?\s*/, '')
    if (!text || state.exercises || /^(code|exercise)$/.test(original.chunkKind || '') ||
      /^(check your understanding|exercises?|contents|references|bibliography)\b/i.test(cleanHeading) ||
      /\bin\s+\S+\.(cpp|py|h):\d+/.test(heading) ||
      (original.pageStart === 1 && /—\s+(?:attributed to\b|[^—\n]{1,100},\s+[^—\n]{1,100}\(\d{3,4}\))/i.test(text))) continue
    let group = chapters.get(state.id)
    if (!group) { group = { area: state.area, sources: [] }; chapters.set(state.id, group) }
    // A single unusually large chunk must not exceed a whole request's budget.
    for (let start = 0; start < text.length; start += Math.min(maxCharacters, 12000)) {
      group.sources.push({ ...original, context: undefined, excerpt: text.slice(start, start + Math.min(maxCharacters, 12000)),
        sourceUnitId: `chapter-${original.chunkId || original.chunkRowid}-${start}`, sourceUnitComplete: text.length <= Math.min(maxCharacters, 12000) })
    }
  }
  const result: TopicBatch[] = []
  for (const [chapterId, group] of chapters) {
    let part: ChatSource[] = [], size = 0, index = 0
    const finish = () => {
      if (!part.length) return
      result.push({ id: topicHash(`${chapterId}:${index++}`), chapterId, area: group.area, sources: part,
        fingerprint: topicHash('chapter-review-v1:' + JSON.stringify(part.map(s => [practiceSourceKey(s), s.sectionHeader, s.excerpt]))) })
      part = []; size = 0
    }
    for (const source of group.sources) {
      const cost = source.excerpt.length + (source.sectionHeader?.length || 0) + 160
      if (part.length && size + cost > maxCharacters) finish()
      part.push(source); size += cost
    }
    finish()
  }
  return result
}

export function parseInventoryTopics(raw: string, batch: TopicBatch, collectionId: string): InventoryTopic[] {
  const value = JSON.parse(raw)
  if (!Array.isArray(value.topics) || !value.topics.length || value.topics.length > 20) throw new Error('The chapter returned no usable review terms.')
  return value.topics.map((t: { title: string; aliases: string[]; objectives: string[]; sourceKeys: string[] }) => {
    if (typeof t.title !== 'string' || !t.title.trim() || t.title.length > 100 || /[\n?]/.test(t.title) ||
      !Array.isArray(t.aliases) || t.aliases.length > 6 || t.aliases.some(a => typeof a !== 'string' || a.length > 100) ||
      !Array.isArray(t.objectives) || t.objectives.length < 2 || t.objectives.length > 5 || t.objectives.some(o => typeof o !== 'string' || !o.trim() || o.length > 500) ||
      !Array.isArray(t.sourceKeys) || !t.sourceKeys.length || t.sourceKeys.length > 6 || t.sourceKeys.some(k => !batch.sources.some(s => practiceSourceKey(s) === k)))
      throw new Error('The chapter returned invalid terms or source references. Retry chapter topics.')
    const title = t.title.trim()
    const id = `topic-${topicHash(`${collectionId}:${batch.chapterId}:${nameKey(title)}`)}`
    return { id, title, area: batch.area, aliases: t.aliases, objectives: t.objectives,
      concept: { courseId: collectionId, id, title }, sources: batch.sources.filter(s => t.sourceKeys.includes(practiceSourceKey(s))) }
  })
}

/** Join exact repeated terms across parts of ONE chapter; never rename or merge different chapters. */
export function consolidateTopics(topics: InventoryTopic[]): InventoryTopic[] {
  const result = new Map<string, InventoryTopic>()
  for (const topic of topics) {
    const existing = result.get(topic.id)
    if (!existing) { result.set(topic.id, { ...topic, aliases: [...topic.aliases], objectives: [...topic.objectives], sources: [...topic.sources] }); continue }
    existing.aliases = [...new Set([...existing.aliases, ...topic.aliases])]
    existing.objectives = [...new Set([...existing.objectives, ...topic.objectives])]
    existing.sources = [...new Map([...existing.sources, ...topic.sources].map(s => [practiceSourceKey(s), s])).values()]
  }
  return [...result.values()]
}
