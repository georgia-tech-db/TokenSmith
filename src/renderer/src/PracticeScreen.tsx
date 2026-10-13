import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { ArrowLeft, ArrowRight, ChartColumn, Loader2, Plus, RefreshCw, Square } from 'lucide-react'
import type { ChatSelectedPassage, ChatSource, LocalModel, ModelRuntimeSettings, TokenSmithSettings } from '@shared/app-state'
import { practiceAssisted, practiceMessages, practiceQuestionText, practiceTimeline, practiceReviewLabel,
  type PracticeAnswerFeedback, type PracticeQuestion, type PracticeSession, type PracticeState, type StudyMode } from '@shared/practice'
import { gradeTrueFalse } from '@shared/quiz'
import { ChatModelPicker } from './ChatModelPicker'
import { MessageText } from './MessageText'
import { StudyHeader, type StudyPanels } from './StudyHeader'
import { StudySidebar } from './StudySidebar'
import { StudyMaterialsPane } from './StudyMaterialsPane'
import { SourceTray, maxSourceTrayCards } from './SourceTray'
import type { StudyCatalog } from './hooks/useStudyDocuments'
import { studyDocumentKey, studyScopeKey, scopeRefs, type StudyDocumentRef } from '@shared/study-scope'
import { PracticeQuestionView } from './PracticeQuestionView'
import { PracticeActions, PracticeComposer } from './PracticeComposer'
import { ConversationViewport } from './ConversationViewport'
import { PracticeFeedbackView } from './PracticeFeedbackView'
import { createPracticeClient } from './practice-client'
import type { GenerationProgressApi } from './hooks/useGenerationProgress'
import './practice.css'

interface Props extends StudyPanels {
  isActive: boolean
  state: PracticeState
  onChange: (update: (state: PracticeState) => PracticeState) => void
  onModeChange: (mode: StudyMode) => void
  onViewLog: () => void
  catalog: StudyCatalog
  scope: StudyDocumentRef[]
  onScopeChange: (scope: StudyDocumentRef[]) => void
  models: LocalModel[]
  selectedModel?: LocalModel
  settings: TokenSmithSettings
  modelSettingsFor: (id: string) => ModelRuntimeSettings
  onSelectModel: (id: string) => void
  onConnectCloud: (model?: LocalModel) => void
  onManageModels: () => void
  onOpenLibrary: () => void
  onOpenSource: (source: ChatSource, sources: ChatSource[]) => void
  sourceError?: string
  progress: GenerationProgressApi
}

export function PracticeScreen(props: Props) {
  const { state, onChange, isActive, models, catalog, settings, progress } = props
  const documents = catalog.documents
  const { historyOpen: sidebarOpen, onHistoryOpenChange: setSidebarOpen, materialsOpen, onMaterialsOpenChange: setMaterialsOpen } = props
  const [error, setError] = useState('')
  const [pending, setPending] = useState<{ id: string; label: string } | null>(null)
  const [reviewOpen, setReviewOpen] = useState(false)
  const operation = useRef<{ id: string; controller: AbortController; sessionId: string; scopeKey: string } | null>(null)
  const session = state.sessions.find(item => item.id === state.activeSessionId)
  const question = session?.questions[session.currentIndex]
  const canShowSources = (item: PracticeQuestion) => settings.application.showSources && item.sources.length > 0 &&
    (item.attempts.length > 0 || item.answerRevealed || Boolean(item.discussions?.length))
  const sources = question && canShowSources(question) ? question.sources : undefined
  const selectedSources = useMemo(() => sources?.slice(0, maxSourceTrayCards) ?? [], [sources])
  const messages = useMemo(() => session?.questions.flatMap(practiceMessages) ?? [], [session?.questions])
  const navigationQuestions = useMemo(() => session?.questions.map(item => ({ id: item.id, text: practiceQuestionText(item) })) ?? [], [session?.questions])
  const model = session ? models.find(item => item.id === session.modelId) : props.selectedModel
  const available = model?.status === 'ready'
  const scope = props.scope
  const scopeKey = studyScopeKey(scope)
  const selected = scope.map(studyDocumentKey)
  function setScope(scope: StudyDocumentRef[]) {
    stop(); setError(''); setReviewOpen(false)
    props.onScopeChange(scope)
  }
  function stop() {
    const job = operation.current
    operation.current = null
    if (job) {
      job.controller.abort()
      progress.cancel(job.id)
      void window.tokensmith?.cancelChatRequest(job.id).catch(() => {})
    }
    setPending(null)
  }
  useEffect(() => () => {
    const job = operation.current
    if (job) { job.controller.abort(); progress.cancel(job.id); void window.tokensmith?.cancelChatRequest(job.id).catch(() => {}) }
    operation.current = null
  }, [progress])
  // Scope can also change in Chat while this screen is hidden.
  useLayoutEffect(() => {
    const job = operation.current
    if (job && (job.scopeKey !== scopeKey || job.sessionId !== session?.id)) stop()
  }, [scopeKey, session?.id])
  useEffect(() => { setError(''); setReviewOpen(false) }, [scopeKey, session?.id])

  function updateSession(id: string, update: (value: PracticeSession) => PracticeSession) {
    onChange(current => ({ ...current, sessions: current.sessions.map(item => item.id === id ? update(item) : item) }))
  }
  function updateQuestion(update: (value: PracticeQuestion) => PracticeQuestion, questionId = question?.id) {
    if (!session || !questionId) return
    updateSession(session.id, current => ({ ...current, questions: current.questions.map(item => item.id === questionId ? update(item) : item) }))
  }
  async function perform<T>(target: PracticeSession, label: string,
    run: (client: ReturnType<typeof createPracticeClient>, id: string, signal: AbortSignal) => Promise<T>,
    complete: (result: T, durationMs: number) => void) {
    if (operation.current) return
    const selectedModel = models.find(item => item.id === target.modelId)
    if (!window.tokensmith || !selectedModel || selectedModel.status !== 'ready') {
      setError('The model used for this session is unavailable. Reconnect it before continuing.')
      return
    }
    const id = crypto.randomUUID()
    const controller = new AbortController()
    operation.current = { id, controller, sessionId: target.id, scopeKey: studyScopeKey(target.documents) }
    setError('')
    setPending({ id, label })
    const started = performance.now()
    progress.begin({ id, kind: 'quiz', label, conversationId: target.id, model: selectedModel,
      settings: target.modelSettings, onStop: stop })
    try {
      const result = await run(createPracticeClient(window.tokensmith, selectedModel, settings), id, controller.signal)
      if (operation.current?.id !== id || controller.signal.aborted) return
      complete(result, performance.now() - started)
      progress.finish(id, 'Practice ready')
    } catch (reason) {
      if (operation.current?.id !== id || controller.signal.aborted) return
      setError(reason instanceof Error ? reason.message.replace(/^Error invoking remote method '[^']+': (?:Error: )?/, '') : String(reason))
      progress.cancel(id)
    } finally {
      if (operation.current?.id === id) { operation.current = null; setPending(null) }
    }
  }
  function generate(target: PracticeSession) {
    void perform(target, 'Preparing question', (client, id, signal) => client.question(target, id, signal), generated =>
      updateSession(target.id, current => generated
        ? { ...current, questions: [...current.questions, generated], currentIndex: current.questions.length }
        : { ...current, sourcesExhausted: true }))
  }
  function start() {
    if (!isActive || session || !model || !available || operation.current || !scopeReady) return
    const scope = documents.filter(item => selected.includes(studyDocumentKey(item)))
    if (!scope.length) return
    const names = [...new Set(scope.map(item => item.collectionName))]
    const created: PracticeSession = { id: `practice-${crypto.randomUUID()}`, title: names.join(', '), createdAt: new Date().toISOString(),
      documents: scope, modelId: model.id, modelSettings: props.modelSettingsFor(model.id), questions: [], currentIndex: 0 }
    onChange(current => ({ ...current, activeSessionId: created.id, sessions: [created, ...current.sessions] }))
    generate(created)
  }
  function chooseSession(id?: string) {
    stop(); setError(''); setReviewOpen(false)
    if (window.innerWidth <= 800) setSidebarOpen(false)
    const previous = state.sessions.find(item => item.id === id)
    if (previous) props.onScopeChange(scopeRefs(previous.documents))
    onChange(current => ({ ...current, activeSessionId: id }))
  }
  function deleteSession(id: string) {
    const target = state.sessions.find(item => item.id === id)
    if (!target || !window.confirm(`Delete "${target.title}" and its practice answers?`)) return
    if (session?.id === id) { stop(); setError('') }
    onChange(current => ({ ...current, sessions: current.sessions.filter(item => item.id !== id),
      activeSessionId: current.activeSessionId === id ? undefined : current.activeSessionId }))
  }
  function clearSessions() {
    if (!window.confirm('Clear all practice sessions and start a new practice? Your chats will not be changed.')) return
    stop(); setError(''); setReviewOpen(false)
    onChange(current => ({ ...current, sessions: [], activeSessionId: undefined }))
  }
  function checkAnswer() {
    if (!session || !question || question.composerMode === 'help' || !question.rubric || !question.draft.trim() || pending) return
    const answer = question.draft.trim()
    const assisted = practiceAssisted(question)
    const complete = (feedback: PracticeAnswerFeedback, durationMs: number) => {
      const id = crypto.randomUUID()
      updateQuestion(current => ({ ...current, reviewing: true, composerMode: 'help', attempts: [...current.attempts,
        { id, answer, feedback, assisted, durationMs }], timeline: [...practiceTimeline(current), { kind: 'attempt', id }] }))
    }
    if (question.format === 'true_false') {
      setError('')
      try { complete(gradeTrueFalse(question, answer), 0) }
      catch (reason) { setError(reason instanceof Error ? reason.message : String(reason)) }
      return
    }
    void perform(session, 'Checking your answer', (client, id) => client.feedback(session, question, answer, id), complete)
  }
  function help(questionId: string, kind: 'hint' | 'answer') {
    updateQuestion(current => {
      if (kind === 'hint' && current.hint && !current.hintUsed) return { ...current, hintUsed: true, timeline: [...practiceTimeline(current), { kind: 'hint' }] }
      if (kind === 'answer' && current.referenceAnswer && !current.answerRevealed) return { ...current, answerRevealed: true, timeline: [...practiceTimeline(current), { kind: 'explanation' }] }
      return current
    }, questionId)
  }
  function next() {
    if (!session || pending) return
    setError('')
    if (session.currentIndex + 1 < session.questions.length) updateSession(session.id, current => ({ ...current, currentIndex: current.currentIndex + 1 }))
    else if (!session.sourcesExhausted) generate(session)
  }
  function discuss() {
    if (!session || !question || question.composerMode !== 'help' || !question.helpDraft?.trim() || pending) return
    const prompt = question.helpDraft.trim()
    void perform(session, 'Explaining', (client, id) => client.discuss(session, question, prompt, id), (reply, durationMs) => {
      const id = crypto.randomUUID()
      updateQuestion(current => ({ ...current, helpDraft: '', helpPassage: undefined,
        discussions: [...(current.discussions ?? []), { id, question: prompt, answer: reply.text, sources: reply.sources,
          durationMs, selectedPassage: question.helpPassage }], timeline: [...practiceTimeline(current), { kind: 'discussion', id }] }))
    })
  }
  function quote(passage: ChatSelectedPassage) {
    if (!session || pending) return
    const index = session.questions.findIndex(item => practiceMessages(item).some(message => message.id === passage.messageId))
    if (index < 0) return
    updateSession(session.id, current => ({ ...current, currentIndex: index,
      questions: current.questions.map((item, i) => i === index ? { ...item, composerMode: 'help', helpPassage: { ...passage, question: practiceQuestionText(item) } } : item) }))
  }
  function continueQuestion(index: number) {
    if (!session || pending || index < 0 || index >= session.questions.length) return
    setReviewOpen(false)
    setError('')
    updateSession(session.id, current => ({ ...current, currentIndex: index }))
  }
  const selectedCount = documents.filter(item => selected.includes(studyDocumentKey(item))).length
  const scopeReady = catalog.status === 'ready' && selectedCount > 0 && selectedCount === scope.length
  useEffect(() => {
    if (!isActive || session || !available || !scopeReady) return
    // Coalesce quick checkbox changes and avoid duplicate starts under StrictMode.
    const timer = window.setTimeout(start, 300)
    return () => window.clearTimeout(timer)
  }, [isActive, session?.id, scopeKey, scopeReady, model, settings, documents])
  const nextDisabled = Boolean(pending) || !session || (session.currentIndex === session.questions.length - 1 && (!available || Boolean(session.sourcesExhausted)))
  const today = new Date().toDateString()
  const historyGroups = ['Today', 'Earlier'].map(label => ({ label, items: state.sessions
    .filter(item => (new Date(item.createdAt).toDateString() === today) === (label === 'Today'))
    .map(item => ({ id: item.id, title: item.title, detail: `${item.questions.length} question${item.questions.length === 1 ? '' : 's'}` })) }))
  return <div className={`view-frame study-frame practice-frame ${sidebarOpen ? 'has-history' : ''} ${materialsOpen ? 'has-materials' : ''}`} hidden={!isActive}>
    {sidebarOpen && <StudySidebar mode="practice" activeId={session?.id} groups={historyGroups}
      onNew={() => chooseSession()} onSelect={chooseSession} onRename={(id, title) => updateSession(id, current => ({ ...current, title }))}
      onDelete={deleteSession} onClear={clearSessions} onViewLog={props.onViewLog} />}
    <section className="practice-main">
      <StudyHeader mode="practice" onModeChange={props.onModeChange} historyOpen={sidebarOpen} onHistoryToggle={() => setSidebarOpen(!sidebarOpen)}
        materialsOpen={materialsOpen} onMaterialsToggle={() => setMaterialsOpen(!materialsOpen)} count={scope.length}>
        <ChatModelPicker models={models} selectedModel={model} disabled={Boolean(pending) || Boolean(session)}
          onSelect={props.onSelectModel} onConnect={props.onConnectCloud} onManage={props.onManageModels} />
        {session && <button className="icon-button subtle" type="button" aria-label="Review progress" title="Review progress"
          aria-pressed={reviewOpen} disabled={Boolean(pending)} onClick={() => setReviewOpen(open => !open)}><ChartColumn size={18} aria-hidden="true" /></button>}
      </StudyHeader>
      {session && reviewOpen ? <div className="practice-scroll"><div className="practice-content">
          <p className="practice-topic">{session.title}</p><h1>Your practice progress</h1>
          <p>{session.questions.filter(item => item.attempts.length || item.answerRevealed).length} of {session.questions.length} questions reviewed</p>
          {session.questions.map((item, index) => <details className="practice-recap-question" key={item.id}>
            <summary><span>{practiceQuestionText(item)}</span><span className="practice-muted">{practiceReviewLabel(item)}</span></summary>
            {item.rubric && <p className="practice-muted">{item.rubric.objective}</p>}
            {item.attempts.at(-1) && <><h3>Your answer</h3><MessageText text={item.attempts.at(-1)!.answer} /><h3>Feedback</h3><PracticeFeedbackView feedback={item.attempts.at(-1)!.feedback} /></>}
            {item.answerRevealed && <p className="practice-muted">Explanation viewed</p>}
            <button type="button" className="practice-link" onClick={() => continueQuestion(index)}>Review question<ArrowRight size={16} /></button>
          </details>)}
          <div className="practice-footer"><button type="button" className="practice-button" onClick={() => setReviewOpen(false)}><ArrowLeft size={16} />Return to practice</button><button type="button" className="practice-button" onClick={() => chooseSession()}><Plus size={16} />New practice</button></div>
      </div></div> : session ? <>
        <ConversationViewport key={session.id} messages={messages} navigationQuestions={navigationQuestions} pending={Boolean(pending)} isActive={isActive}
          focusQuestionId={question?.id} onNavigate={id => continueQuestion(session.questions.findIndex(item => item.id === id))}
          canQuote={!pending && available} onQuote={quote}>
          {session.questions.map((item, index) => <PracticeQuestionView key={item.id} question={item} index={index} active={index === session.currentIndex}
            busy={Boolean(pending)} onContinue={() => continueQuestion(index)} sourcesOpen={index === session.currentIndex && materialsOpen}
            onShowSources={canShowSources(item) ? () => { continueQuestion(index); setMaterialsOpen(true) } : undefined}
            onHint={() => help(item.id, 'hint')} onReveal={() => help(item.id, 'answer')}
            actions={index === session.currentIndex && <PracticeActions question={item} busy={Boolean(pending)} nextDisabled={nextDisabled}
              onChange={updateQuestion} onNext={next} />} />)}
          {!question && !pending && !session.sourcesExhausted && <button type="button" className="practice-link" disabled={!available} onClick={() => generate(session)}><RefreshCw size={17} />Retry question</button>}
        </ConversationViewport>
      </> : <div className="practice-scroll" />}
      <div className="practice-status">
        {error && <p className="practice-error" role="alert">{error}</p>}
        {!session && catalog.status === 'ready' && !scopeReady && <p className="practice-muted" role="status">{scope.length ? 'Selected documents are unavailable.' : 'No documents selected.'}</p>}
        {!session && !available && <p className="practice-muted" role="status">No ready chat model selected.</p>}
        {session && !available && <div className="practice-error" role="status"><p>This session's model is unavailable. Your questions and answers are saved.</p><button type="button" className="practice-button" onClick={props.onManageModels}>Manage models</button></div>}
        {session?.sourcesExhausted && <p className="practice-muted" role="status">No unused passages remain in these documents. Review earlier questions or start a new practice with other documents.</p>}
        {pending && <div className="practice-pending" role="status"><Loader2 className="spin" size={17} /><span>{pending.label}...</span><button type="button" className="practice-link" onClick={stop}><Square size={14} />Stop</button></div>}
      </div>
      {session && question && !reviewOpen && <PracticeComposer question={question} busy={Boolean(pending)} available={available} isActive={isActive}
        onChange={updateQuestion} onCheck={checkAnswer} onHelp={discuss} />}
    </section>
    <StudyMaterialsPane catalog={catalog} scope={scope} open={materialsOpen}
      collectionExpansion={props.collectionExpansion} onCollectionExpansionChange={props.onCollectionExpansionChange}
      onChange={setScope} onClose={() => setMaterialsOpen(false)} onManage={props.onOpenLibrary}>
      {selectedSources.length > 0 && <SourceTray sources={selectedSources} error={props.sourceError}
        onOpenSource={source => props.onOpenSource(source, sources ?? [])} />}
    </StudyMaterialsPane>
  </div>
}
