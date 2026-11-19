import { useEffect, useMemo, useRef, useState } from 'react'
import { ArrowLeft, ChevronDown, ChevronRight, ListTree, SendHorizonal } from 'lucide-react'
import './study-buttons.css'
import type { CourseMaterial } from '@shared/app-state'
import type { PracticeMasteryRecord, PracticeSession, PracticeTopic } from '@shared/practice'
import { practiceConceptKey } from '@shared/practice-study'
import { chapterMastery, practiceMastery } from '@shared/practice-mastery'
import { MasteryMeter, TopicMasteryDetails } from './MasteryMeter'
import { studyScopeKey, type StudyDocumentRef } from '@shared/study-scope'
import { loadSavedTopics } from '../../shared/saved-topics'
import { practiceTopicKey } from '@shared/practice-question-plan'
import type { InventoryTopic } from '../../shared/topic-inventory'

export function PracticeTopicPicker({ isActive, ready, disabled, scope, materials, sessions, masteryRecords, onChoose, onChooseTopics, onExtractTopics }: {
  isActive: boolean; ready: boolean; disabled: boolean; scope: StudyDocumentRef[]; materials: CourseMaterial[]
  sessions: PracticeSession[]
  masteryRecords?: PracticeMasteryRecord[]
  onExtractTopics?: () => Promise<void>
  onChoose: (topic: PracticeTopic) => void
  onChooseTopics: (title: string, topics: PracticeTopic[]) => void
}) {
  const [extracting, setExtracting] = useState(false)
  const [reload, setReload] = useState(0)
  const [selectedKeys, setSelectedKeys] = useState<string[]>([])
  const [text, setText] = useState('')
  const [status, setStatus] = useState<'loading' | 'ready' | 'error'>('loading')
  const [chapterId, setChapterId] = useState<string | null>(null)
  const [search, setSearch] = useState('')
  const [filter, setFilter] = useState<'all' | 'new' | 'mistakes'>('all')
  const [expandedTopic, setExpandedTopic] = useState<string | null>(null)
  const [topics, setTopics] = useState<InventoryTopic[]>([])
  const [error, setError] = useState('')
  const heading = useRef<HTMLHeadingElement>(null)
  const focusHeading = useRef(false)
  const scopeKey = studyScopeKey(scope)
  const key = JSON.stringify([3, scopeKey, materials.filter(material => scope.some(doc => doc.materialId === material.id))
    .map(material => [material.id, material.collectionRevision, material.indexedAt, material.chunkCount])])
  const progress = useMemo(() => practiceMastery(sessions, [], masteryRecords), [sessions, masteryRecords])
  const topicProgress = (topic: PracticeTopic) => topic.concept ? progress.get(practiceConceptKey(topic.concept)) : undefined
  const matchesProgress = (topic: PracticeTopic) => filter === 'all' || (filter === 'new' ? topicProgress(topic)?.score == null : topicProgress(topic)?.recentMistake)
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
  const chapterProgress = chapter && chapterMastery(chapter.topics, progress)
  const selectedTopics = topics.filter(t => selectedKeys.includes(practiceTopicKey(t)))
  const selectedChapter = chapters.find(c => c.topics.length === selectedTopics.length && c.topics.every(t => selectedKeys.includes(practiceTopicKey(t))))
  function selectTopics(items: InventoryTopic[], checked: boolean) {
    const keys = items.map(practiceTopicKey)
    setSelectedKeys(current => checked ? [...new Set([...current, ...keys])] : current.filter(k => !keys.includes(k)))
  }
  function chapterCheckbox(items: InventoryTopic[], label: string) {
    const count = items.filter(t => selectedKeys.includes(practiceTopicKey(t))).length
    return <input type="checkbox" aria-label={label} checked={count === items.length && count > 0}
      ref={element => { if (element) element.indeterminate = count > 0 && count < items.length }} disabled={disabled}
      onChange={event => selectTopics(items, event.target.checked)} />
  }
  function navigate(id: string | null) {
    focusHeading.current = true
    setChapterId(id); setSearch(''); setExpandedTopic(null)
  }
  useEffect(() => {
    if (focusHeading.current) { heading.current?.focus(); focusHeading.current = false }
  }, [chapterId])
  useEffect(() => {
    setTopics([]); setSelectedKeys([]); setChapterId(null); setSearch(''); setFilter('all'); setExpandedTopic(null)
    if (!ready || !isActive || !window.tokensmith) return
    let active = true
    setError(''); setStatus('loading')
    void loadSavedTopics(window.tokensmith, scope, materials).then(topics => {
      if (active) { setTopics(topics); setStatus('ready') }
    }).catch(reason => { if (active) { setError((reason as Error).message); setStatus('error') } })
    return () => { active = false }
  }, [isActive, ready, key, reload])
  return <div className="practice-topic-picker">
    {chapter && <button className="practice-link practice-chapter-back" type="button" onClick={() => navigate(null)}><ArrowLeft size={16} aria-hidden="true" />All chapters</button>}
    <h1 ref={heading} tabIndex={-1}>{chapter?.title ?? 'What would you like to practice?'}</h1>
    {!chapter && <p className="practice-muted">Select a chapter to practice all its topics, or open it to choose individual topics.</p>}
    {chapterProgress && <div className="practice-chapter-progress">
      <div><MasteryMeter score={chapterProgress.score} label="Chapter progress" />
        <small className="practice-muted" title="Unassessed topics contribute zero to chapter progress.">{chapterProgress.assessed} of {chapterProgress.total} topics assessed</small></div>
      <label className="practice-select-chapter">{chapterCheckbox(chapter!.topics, `Select all topics in ${chapter!.title}`)}Whole chapter</label>
    </div>}
    {selectedTopics.length > 0 && <div className="practice-selection-bar" role="region" aria-label="Selected practice topics">
      <span>{selectedTopics.length} {selectedTopics.length === 1 ? 'topic' : 'topics'} selected{selectedChapter ? ` · ${selectedChapter.title}` : ''}</span>
      <button className="practice-link" type="button" onClick={() => setSelectedKeys([])}>Clear selection</button>
      <button className="practice-button" type="button" disabled={disabled} onClick={() => onChooseTopics(selectedChapter?.title ?? (selectedTopics.length === 1 ? selectedTopics[0].title : `${selectedTopics.length} topics`), selectedTopics)}>
        {selectedChapter ? 'Practice chapter' : `Practice ${selectedTopics.length === 1 ? 'topic' : `${selectedTopics.length} topics`}`}
      </button>
    </div>}
    {ready && status === 'loading' && <p className="practice-muted" role="status">Loading saved topics…</p>}
    {topics.length > 0 && <div className="practice-topic-inventory">
      <div className="practice-topic-toolbar"><div className="practice-topic-filters" role="group" aria-label="Filter topics by practice progress">
        {([['all', 'All topics'], ['new', 'Not practiced'], ['mistakes', 'Recent mistakes']] as const).map(([value, label]) =>
          <button key={value} type="button" aria-pressed={filter === value} onClick={() => setFilter(value)}>{label}</button>)}
      </div>
      <input className="practice-topic-search" aria-label={chapter ? 'Find a topic in this chapter' : 'Find a chapter'} placeholder={chapter ? 'Find a topic…' : 'Find a chapter…'} value={search} onChange={event => setSearch(event.target.value)} />
      </div>
      {filter !== 'all' && <p className="practice-muted practice-topic-progress-note">{filter === 'mistakes' ? 'Topics whose latest independent answer was incorrect.' : 'Topics without an independent mastery check yet.'}</p>}
      {chapter ? <>
        {visibleTopics.length === 0 && <p className="practice-muted">No matching topics in this chapter.</p>}
        {visibleTopics.length > 0 && <div className="practice-topic-columns" aria-hidden="true"><span>Topic</span><span>Estimated mastery</span></div>}
        {visibleTopics.map(topic => {
          const result = topicProgress(topic)
          const topicKey = topic.concept ? practiceConceptKey(topic.concept) : topic.id
          const expanded = expandedTopic === topicKey
          const detailsId = `topic-details-${encodeURIComponent(topicKey)}`
          return <div className="practice-topic-entry" key={topicKey}>
            <div className="practice-topic-heading">
              <input type="checkbox" aria-label={`Select ${topic.title}`} checked={selectedKeys.includes(practiceTopicKey(topic))} disabled={disabled}
                onChange={event => selectTopics([topic], event.target.checked)} />
              <button type="button" className="practice-topic-start" disabled={disabled} onClick={() => onChoose(topic)}>{topic.title}<ChevronRight size={16} aria-hidden="true" /></button>
              <MasteryMeter compact score={result?.score ?? null} label="Estimated mastery" empty={result?.withHelp ? 'Try without help' : 'Not practiced'} />
              <button type="button" className="practice-topic-expand" aria-label={`Details for ${topic.title}`} aria-expanded={expanded} aria-controls={detailsId}
                onClick={() => setExpandedTopic(expanded ? null : topicKey)}><ChevronDown size={16} aria-hidden="true" /></button>
            </div>
            <div id={detailsId} className="practice-topic-details" hidden={!expanded}>
            <p>{topic.objectives.slice(0, 2).join(' · ')}</p>
            <TopicMasteryDetails result={result} />
            <details><summary>Sources ({topic.sources.length} {topic.sources.length === 1 ? 'passage' : 'passages'})</summary>
              <small>{[...new Set(topic.sources.map(s => `${s.documentTitle}${s.sectionHeader ? ` — ${s.sectionHeader.split(' / ')[0]}` : ''}`))].join('; ')}</small>
            </details>
            </div>
          </div>
        })}
      </> : <>
        {visibleChapters.length === 0 && <p className="practice-muted">No matching chapters.</p>}
        <div className="practice-chapter-list" aria-label="Chapters">
          {visibleChapters.map(item => {
            const result = chapterMastery(item.topics, progress)
            return <div key={item.id} className="practice-chapter-row">
              {chapterCheckbox(item.topics, `Select chapter ${item.title}`)}
              <button type="button" className="practice-chapter-button" onClick={() => navigate(item.id)}>
              <span className="practice-chapter-overview"><strong>{item.title}</strong><small>{item.collection && `${item.collection} · `}{result.assessed} of {result.total} topics assessed</small></span>
              <MasteryMeter compact score={result.score} label="Chapter progress" /><ChevronRight size={18} aria-hidden="true" />
            </button></div>
          })}
        </div>
      </>}
    </div>}
    {ready && status === 'ready' && topics.length === 0 && <p className="practice-muted">No saved topics for the selected documents.</p>}
    {ready && (status === 'error' || (status === 'ready' && topics.length === 0)) && <div className="practice-topic-extraction" role="status"><p className="practice-muted">{error}</p>
      {onExtractTopics && <button className="practice-button is-primary" type="button" disabled={extracting || disabled} onClick={async () => {
        setExtracting(true)
        try { await onExtractTopics(); setReload(value => value + 1) }
        catch (reason) { setError(reason instanceof Error ? reason.message : 'Could not extract topics.') }
        finally { setExtracting(false) }
      }}><ListTree size={16} aria-hidden="true" />{extracting ? 'Extracting topics…' : 'Extract topics'}</button>}
    </div>}
    <form onSubmit={event => { event.preventDefault(); if (!disabled && text.trim()) onChoose({ title: text.trim() }) }}>
      <label htmlFor="practice-topic">Your topic</label>
      <div className="composer">
        <input id="practice-topic" value={text} maxLength={300} onChange={event => setText(event.target.value)} placeholder="Type a topic to practice..." />
        <button className="send-button" type="submit" disabled={disabled || !text.trim()} aria-label="Start practice" title="Start practice"><SendHorizonal size={21} aria-hidden="true" /></button>
      </div>
    </form>
  </div>
}
