import { useEffect, useState } from 'react'
import { Plus, RefreshCw, Sparkles } from 'lucide-react'
import type { Conversation, LocalModel, ModelRuntimeSettings, TokenSmithSettings } from '@shared/app-state'
import type { GenerationProgressApi } from './hooks/useGenerationProgress'
import { MessageText } from './MessageText'

export function PracticeChatSuggestions({ conversation, model, settings, modelSettings, isActive, disabled, progress, onSave, onUse }: {
  conversation: Conversation; model?: LocalModel; settings: TokenSmithSettings; modelSettings: ModelRuntimeSettings
  isActive: boolean; disabled: boolean; progress: GenerationProgressApi
  onSave: (questions: string[]) => void; onUse: (question: string) => void
}) {
  const [error, setError] = useState(false)
  const [revision, setRevision] = useState(0)
  useEffect(() => {
    if (!isActive || disabled || conversation.practiceSuggestions || !model || model.status !== 'ready' || !window.tokensmith) return
    const bridge = window.tokensmith
    const id = crypto.randomUUID()
    let cancelled = false
    let finished = false
    const stop = () => {
      if (finished || cancelled) return
      window.clearTimeout(timer)
      cancelled = true; progress.cancel(id)
      void bridge.cancelChatQuestionSuggestions(id).catch(() => {})
    }
    const timer = window.setTimeout(() => {
      setError(false)
      progress.begin({ id, conversationId: conversation.id, kind: 'follow-ups', label: 'Preparing follow-ups',
        model, settings: modelSettings, count: 3, onStop: () => { stop(); setError(true) } })
      void bridge.suggestChatQuestions(id, { messages: conversation.messages, materials: [],
        documents: conversation.documentScope, retrievedSources: conversation.practiceReference?.sources ?? [], model, settings,
        applicationSettings: { ...settings.application, suggestionMode: 'on', followUpSuggestionCount: 3 },
        modelSettings: { ...modelSettings, systemMessage: '', reasoningMode: 'off', thinking: false }
      }).then(response => {
        if (cancelled) return
        const questions = response.suggestions.filter(question => question.trim()).slice(0, 3)
        if (!questions.length) throw new Error('No questions returned')
        finished = true; onSave(questions); progress.finish(id, 'Questions ready')
      }).catch(() => {
        if (cancelled) return
        finished = true; setError(true); progress.cancel(id)
      })
    }, 0)
    return stop
  }, [conversation.id, model?.id, model?.status, isActive, disabled, revision])
  return <section className="follow-up-section" aria-label="Suggested practice follow-up questions">
    <div className="follow-up-title"><Sparkles size={18} aria-hidden="true" /><span>Suggested follow-ups</span></div>
    {conversation.practiceSuggestions ? <div className="follow-up-list">
      {conversation.practiceSuggestions.map(question => <button key={question} type="button" className="follow-up-row"
        disabled={disabled || model?.status !== 'ready'} onClick={() => onUse(question)}>
        <MessageText text={question} inline /><Plus size={18} aria-hidden="true" />
      </button>)}
    </div> : error ? <p className="follow-up-error">Could not prepare suggested questions.
      <button type="button" className="practice-link" disabled={disabled || model?.status !== 'ready'} onClick={() => setRevision(value => value + 1)}><RefreshCw size={14} />Retry suggestions</button>
    </p> : model?.status !== 'ready' ? <p className="practice-muted">Choose a ready model for suggested questions.</p> : null}
  </section>
}
