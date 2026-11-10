import { useEffect, useMemo, useRef, useState } from 'react'
import { ArrowLeft, ChevronRight, SendHorizonal } from 'lucide-react'
import type { CourseMaterial } from '@shared/app-state'
import type { PracticeSession, PracticeTopic } from '@shared/practice'
import { practiceConceptKey, practicePerformance } from '@shared/practice-study'
import { studyScopeKey, type StudyDocumentRef } from '@shared/study-scope'
import { loadSavedTopics } from '../../shared/saved-topics'
import type { InventoryTopic } from '../../shared/topic-inventory'

export function PracticeTopicPicker({ isActive, ready, disabled, scope, materials, sessions, onChoose }: {
  isActive: boolean; ready: boolean; disabled: boolean; scope: StudyDocumentRef[]; materials: CourseMaterial[]
  sessions: PracticeSession[]
  onChoose: (topic: PracticeTopic) => void
}) {
  const [text, setText] = useState('')
  const [status, setStatus] = useState<'loading' | 'ready' | 'error'>('loading')
  const [chapterId, setChapterId] = useState<string | null>(null)
  const [search, setSearch] = useState('')
  const [filter, setFilter] = useState<'all' | 'new' | 'mistakes'>('all')
  const [topics, setTopics] = useState<InventoryTopic[]>([])
  const [error, setError] = useState('')
  const heading = useRef<HTMLHeadingElement>(null)
  const focusHeading = useRef(false)
  const scopeKey = studyScopeKey(scope)
  const key = JSON.stringify([3, scopeKey, materials.filter(material => scope.some(doc => doc.materialId === material.id))
    .map(material => [material.id, material.collectionRevision, material.indexedAt, material.chunkCount])])
  const progress = useMemo(() => new Map(practicePerformance(sessions).map(row => [practiceConceptKey(row.concept), row])), [sessions])
  const topicProgress = (topic: PracticeTopic) => topic.concept ? progress.get(practiceConceptKey(topic.concept)) : undefined
  const matchesProgress = (topic: PracticeTopic) => filter === 'all' || (filter === 'new' ? !topicProgress(topic) : topicProgress(topic)?.recentMistake)
  const groups = new Map<string, { id: string; title: string; collection: string; topics: InventoryTopic[] }>()
  for (const topic of topics) {
    const source = topic.sources[0]
    const id = JSON.stringify([source?.materialId, source?.documentId, topic.area])
    let group = groups.get(id)
    if (!group) {
      group = { id, title: topic.area,
        collection: materials.find(m => m.id === source?.materialId)?.collectionTitle || source?.collectionName || '', topics: [] }
      groups.set(id, group)
    }
    group.topics.push(topic)
  }
  const chapters = [...groups.values()]
  const chapter = chapterId ? groups.get(chapterId) : undefined
  const query = search.trim().toLowerCase()
  const visibleChapters = chapters.filter(c => `${c.title} ${c.collection}`.toLowerCase().includes(query) && c.topics.some(matchesProgress))
  const visibleTopics = chapter?.topics.filter(t => matchesProgress(t) && [t.title, ...t.aliases].join(' ').toLowerCase().includes(query)) ?? []
  function navigate(id: string | null) {
    focusHeading.current = true
    setChapterId(id); setSearch('')
  }
  useEffect(() => {
    if (focusHeading.current) { heading.current?.focus(); focusHeading.current = false }
  }, [chapterId])
  useEffect(() => {
    setTopics([]); setChapterId(null); setSearch(''); setFilter('all')
    if (!ready || !isActive || !window.tokensmith) return
    let active = true
    setError(''); setStatus('loading')
    void loadSavedTopics(window.tokensmith, scope, materials).then(topics => {
      if (active) { setTopics(topics); setStatus('ready') }
    }).catch(reason => { if (active) { setError((reason as Error).message); setStatus('error') } })
    return () => { active = false }
  }, [isActive, ready, key])
  return <div className="practice-topic-picker">
    {chapter && <button className="practice-link practice-chapter-back" type="button" onClick={() => navigate(null)}><ArrowLeft size={16} aria-hidden="true" />All chapters</button>}
    <h1 ref={heading} tabIndex={-1}>{chapter?.title ?? 'What would you like to practice?'}</h1>
    <p className="practice-muted">{chapter ? 'Choose a topic from this chapter.' : 'Choose a chapter to explore its topics.'}</p>
    {ready && status === 'loading' && <p className="practice-muted" role="status">Loading saved topics…</p>}
    {topics.length > 0 && <div className="practice-topic-inventory">
      <div className="study-mode-tabs practice-topic-filters" role="group" aria-label="Filter topics by practice progress">
        {([['all', 'All'], ['new', 'Not practiced'], ['mistakes', 'Recent mistakes']] as const).map(([value, label]) =>
          <button key={value} type="button" aria-pressed={filter === value} onClick={() => setFilter(value)}>{label}</button>)}
      </div>
      <p className="practice-muted practice-topic-progress-note">{filter === 'mistakes' ? 'Topics whose latest first answer without help was incorrect.'
        : filter === 'new' ? 'Topics with no scored first answers in your saved practice.' : 'First answers across saved practice. Results include answers with help, noted separately.'}</p>
      <input className="practice-topic-search" aria-label={chapter ? 'Find a topic in this chapter' : 'Find a chapter'} placeholder={chapter ? 'Find a topic…' : 'Find a chapter…'} value={search} onChange={event => setSearch(event.target.value)} />
      {chapter ? <>
        {visibleTopics.length === 0 && <p className="practice-muted">No matching topics in this chapter.</p>}
        {visibleTopics.map(topic => {
          const result = topicProgress(topic)
          return <div className="practice-topic-entry" key={topic.concept ? practiceConceptKey(topic.concept) : topic.id}>
            <button type="button" className="practice-button" disabled={disabled} onClick={() => onChoose(topic)}>{topic.title}</button>
            <p>{topic.objectives.slice(0, 2).join(' · ')}</p>
            <p className="practice-muted practice-topic-progress">{result ? <>
              <span>{result.correctResponses} of {result.observations} correct · {result.assisted} with help</span>
              <span>Last practiced <time dateTime={result.lastPracticedAt} title={new Date(result.lastPracticedAt).toLocaleString()}>{new Date(result.lastPracticedAt).toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' })}</time></span>
            </> : 'No scored answers yet'}</p>
            <details><summary>Sources ({topic.sources.length} {topic.sources.length === 1 ? 'passage' : 'passages'})</summary>
              <small>{[...new Set(topic.sources.map(s => `${s.documentTitle}${s.sectionHeader ? ` — ${s.sectionHeader.split(' / ')[0]}` : ''}`))].join('; ')}</small>
            </details>
          </div>
        })}
      </> : <>
        {visibleChapters.length === 0 && <p className="practice-muted">No matching chapters.</p>}
        <div className="practice-chapter-list" aria-label="Chapters">
          {visibleChapters.map(item => <button key={item.id} type="button" className="practice-chapter-button" onClick={() => navigate(item.id)}>
            <span><strong>{item.title}</strong><small>{item.collection && `${item.collection} · `}{item.topics.filter(t => topicProgress(t)).length} of {item.topics.length} topics practiced</small></span><ChevronRight size={18} aria-hidden="true" />
          </button>)}
        </div>
      </>}
    </div>}
    {ready && status === 'ready' && topics.length === 0 && <p className="practice-muted">No saved topics for the selected documents.</p>}
    {ready && status === 'error' && <p className="practice-muted" role="status">{error}</p>}
    <form onSubmit={event => { event.preventDefault(); if (!disabled && text.trim()) onChoose({ title: text.trim() }) }}>
      <label htmlFor="practice-topic">Your topic</label>
      <div className="composer">
        <input id="practice-topic" value={text} maxLength={300} onChange={event => setText(event.target.value)} placeholder="Type a topic to practice..." />
        <button className="send-button" type="submit" disabled={disabled || !text.trim()} aria-label="Start practice" title="Start practice"><SendHorizonal size={21} aria-hidden="true" /></button>
      </div>
    </form>
  </div>
}
