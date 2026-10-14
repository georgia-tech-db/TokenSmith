import { useEffect, useState } from 'react'
import { RefreshCw, SendHorizonal } from 'lucide-react'
import type { CourseMaterial, LocalModel, ModelRuntimeSettings, TokenSmithSettings } from '@shared/app-state'
import type { PracticeState, PracticeTopic } from '@shared/practice'
import { studyScopeKey, type StudyDocumentRef } from '@shared/study-scope'
import type { GenerationProgressApi } from './hooks/useGenerationProgress'
import { createPracticeClient } from './practice-client'

export function PracticeTopicPicker({ isActive, ready, disabled, scope, materials, model, settings, modelSettings, progress, cache, onCache, onChoose }: {
  isActive: boolean; ready: boolean; disabled: boolean; scope: StudyDocumentRef[]; materials: CourseMaterial[]
  model?: LocalModel; settings: TokenSmithSettings; modelSettings: ModelRuntimeSettings; progress: GenerationProgressApi
  cache: PracticeState['topicSuggestions']; onCache: (cache: NonNullable<PracticeState['topicSuggestions']>) => void
  onChoose: (topic: PracticeTopic) => void
}) {
  const [text, setText] = useState('')
  const [status, setStatus] = useState<'loading' | 'ready' | 'error'>('loading')
  const [revision, setRevision] = useState(0)
  const scopeKey = studyScopeKey(scope)
  const key = JSON.stringify([scopeKey, materials.filter(material => scope.some(doc => doc.materialId === material.id))
    .map(material => [material.id, material.indexedAt, material.chunkCount])])
  const topics = cache?.key === key ? cache.topics : []
  useEffect(() => {
    if (!isActive || !ready) return
    if (cache?.key === key) { setStatus('ready'); return }
    if (disabled || !model || !window.tokensmith) return
    const bridge = window.tokensmith
    const id = crypto.randomUUID()
    const controller = new AbortController()
    let finished = false
    const stop = () => {
      if (finished) return
      window.clearTimeout(timer)
      controller.abort(); progress.cancel(id)
      void bridge.cancelChatRequest(id).catch(() => {})
    }
    const timer = window.setTimeout(() => {
      setStatus('loading')
      progress.begin({ id, conversationId: 'practice-topics', kind: 'starter', label: 'Finding topics', model,
        settings: { ...modelSettings, maxLength: Math.min(modelSettings.maxLength, 768), reasoningMode: 'off', thinking: false },
        onStop: () => { stop(); setStatus('error') } })
      void createPracticeClient(bridge, model, settings, { materials, embeddingModels: [] })
        .topics(scope, modelSettings, id, controller.signal).then(topics => {
          if (controller.signal.aborted) return
          finished = true; onCache({ key, topics }); setStatus('ready'); progress.finish(id, 'Topics ready')
        }).catch(() => {
          if (controller.signal.aborted) return
          finished = true; setStatus('error'); progress.cancel(id)
        })
    }, 0)
    return stop
  }, [isActive, ready, disabled, key, model?.id, revision])
  return <div className="practice-topic-picker">
    <h1>What would you like to practice?</h1>
    {ready && !disabled && status === 'loading' && <p className="practice-muted" role="status">Finding topics...</p>}
    {topics.length > 0 && <div className="answer-simplify-action practice-topic-suggestions" role="group" aria-label="Suggested topics">
      {topics.map(topic => <button key={topic.title} type="button" disabled={disabled} onClick={() => onChoose(topic)}>{topic.title}</button>)}
    </div>}
    {ready && status === 'error' && <p className="practice-muted">Topic suggestions are unavailable. You can enter your own.
      <button className="practice-link" type="button" disabled={disabled} onClick={() => setRevision(value => value + 1)}><RefreshCw size={14} />Retry topics</button></p>}
    <form onSubmit={event => { event.preventDefault(); if (!disabled && text.trim()) onChoose({ title: text.trim() }) }}>
      <label htmlFor="practice-topic">Your topic</label>
      <div className="composer">
        <input id="practice-topic" value={text} maxLength={300} onChange={event => setText(event.target.value)} placeholder="Type a topic to practice..." />
        <button className="send-button" type="submit" disabled={disabled || !text.trim()} aria-label="Start practice" title="Start practice"><SendHorizonal size={21} aria-hidden="true" /></button>
      </div>
    </form>
  </div>
}
