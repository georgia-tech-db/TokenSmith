import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { ArrowLeft, ArrowRight, ChartColumn, Plus, RefreshCw } from 'lucide-react'
import type { ChatSelectedPassage, ChatSource, CourseMaterial, LocalModel, ModelRuntimeSettings, TokenSmithSettings } from '@shared/app-state'
import { practiceAssisted, practiceMessages, practiceQuestionText, practiceTimeline, practiceReviewLabel,
  type PracticeTopic, type PracticeQuestion, type PracticeSession, type PracticeState, type StudyMode } from '@shared/practice'
import { gradeTrueFalse } from '@shared/quiz'
import { ChatModelPicker } from './ChatModelPicker'
import { MessageText } from './MessageText'
import { StudyHeader, type StudyPanels } from './StudyHeader'
import { StudySidebar } from './StudySidebar'
import { StudyMaterialsPane } from './StudyMaterialsPane'
import { SourceTray, maxSourceTrayCards } from './SourceTray'
import type { StudyCatalog } from './hooks/useStudyDocuments'
import { studyDocumentKey, studyScopeKey, scopeRefs, type StudyDocumentRef } from '@shared/study-scope'
import { PracticeQuestionView, PracticeTopicView } from './PracticeQuestionView'
import { PracticeTopicPicker } from './PracticeTopicPicker'
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
  onChat: (session: PracticeSession, question: PracticeQuestion, passage?: ChatSelectedPassage) => void
  onViewLog: () => void
  catalog: StudyCatalog
  scope: StudyDocumentRef[]
  onScopeChange: (scope: StudyDocumentRef[]) => void
  materials: CourseMaterial[]
  embeddingModels: LocalModel[]
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
  const [pending, setPending] = useState<{ id: string } | null>(null)
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
    setPending({ id })
    const started = performance.now()
    progress.begin({ id, kind: 'quiz', label, conversationId: target.id, model: selectedModel,
      settings: target.modelSettings, onStop: stop })
    try {
      const result = await run(createPracticeClient(window.tokensmith, selectedModel, settings, { materials: props.materials, embeddingModels: props.embeddingModels }), id, controller.signal)
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
    if (!target.topic) { chooseSession(); return }
    let prepared = target
    void perform(target, 'Preparing question', async (client, id, signal) => {
      prepared = await client.prepareTopic(target, signal)
      return client.question(prepared, id, signal)
    }, generated => updateSession(target.id, current => generated
      ? { ...current, topic: prepared.topic, questions: [...current.questions, generated], currentIndex: current.questions.length }
      : { ...current, topic: prepared.topic, sourcesExhausted: true }))
  }
  function start(topic: PracticeTopic) {
    if (!isActive || session || !model || !available || operation.current || !scopeReady || !topic.title.trim()) return
    const scope = documents.filter(item => selected.includes(studyDocumentKey(item)))
    if (!scope.length) return
    const created: PracticeSession = { id: `practice-${crypto.randomUUID()}`, title: topic.title, topic, createdAt: new Date().toISOString(),
      documents: scope, modelId: model.id, modelSettings: props.modelSettingsFor(model.id), questions: [], currentIndex: 0 }
    onChange(current => ({ ...current, draftTopic: undefined, activeSessionId: created.id, sessions: [created, ...current.sessions] }))
    generate(created)
  }
  function chooseSession(id?: string) {
    stop(); setError(''); setReviewOpen(false)
    if (window.innerWidth <= 800) setSidebarOpen(false)
    const previous = state.sessions.find(item => item.id === id)
    if (previous) props.onScopeChange(scopeRefs(previous.documents))
    onChange(current => ({ ...current, draftTopic: undefined, activeSessionId: id,
      ...(id ? {} : { draftModelId: settings.application.defaultModelId }) }))
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
    onChange(current => ({ ...current, sessions: [], activeSessionId: undefined, draftTopic: undefined }))
  }
  function checkAnswer(answer: string) {
    if (!session || !question || question.format !== 'true_false' || question.attempts.length || pending || operation.current) return
    const assisted = practiceAssisted(question)
    setError('')
    try {
      const feedback = gradeTrueFalse(question, answer)
      const id = crypto.randomUUID()
      const checked: PracticeQuestion = { ...question, draft: answer, reviewing: true, attempts: [...question.attempts,
        { id, answer, feedback, assisted, durationMs: 0 }], timeline: [...practiceTimeline(question), { kind: 'attempt', id }] }
      const updated = { ...session, questions: session.questions.map(item => item.id === question.id ? checked : item) }
      updateSession(session.id, () => updated)
      const nextIndex = updated.questions.findIndex((item, index) => index > updated.currentIndex && item.format === 'true_false')
      if (nextIndex >= 0) updateSession(session.id, current => ({ ...current, currentIndex: nextIndex }))
      else if (available && !session.sourcesExhausted) generate(updated)
    }
    catch (reason) { setError(reason instanceof Error ? reason.message : String(reason)) }
  }
  function chatAbout(item: PracticeQuestion, passage?: ChatSelectedPassage) {
    if (!session) return
    stop()
    props.onChat(session, item, passage)
  }
  function quote(passage: ChatSelectedPassage) {
    const item = session?.questions.find(item => practiceMessages(item).some(message => message.id === passage.messageId))
    if (item) chatAbout(item, { ...passage, question: practiceQuestionText(item) })
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
    if (state.draftTopic && isActive && !session && available && scopeReady) start(state.draftTopic)
  }, [state.draftTopic, isActive, session?.id, scopeKey, scopeReady, model])
  const today = new Date().toDateString()
  const historyGroups = ['Today', 'Earlier'].map(label => ({ label, items: state.sessions
    .filter(item => (new Date(item.createdAt).toDateString() === today) === (label === 'Today'))
    .map(item => ({ id: item.id, title: item.title, detail: `${item.questions.length} question${item.questions.length === 1 ? '' : 's'}` })) }))
  return <div className={`view-frame study-frame practice-frame ${sidebarOpen ? 'has-history' : ''} ${materialsOpen ? 'has-materials' : ''}`} hidden={!isActive} aria-busy={Boolean(pending)}>
    {sidebarOpen && <StudySidebar mode="practice" activeId={session?.id} groups={historyGroups}
      onNew={() => chooseSession()} onSelect={chooseSession} onRename={(id, title) => updateSession(id, current => ({ ...current, title }))}
      onDelete={deleteSession} onClear={clearSessions} onViewLog={props.onViewLog} />}
    <section className="practice-main">
      <StudyHeader mode="practice" onModeChange={props.onModeChange} historyOpen={sidebarOpen} onHistoryToggle={() => setSidebarOpen(!sidebarOpen)}
        materialsOpen={materialsOpen} onMaterialsToggle={() => setMaterialsOpen(!materialsOpen)} count={scope.length}>
        <ChatModelPicker models={models} selectedModel={model} selectedModelId={session ? session.modelId : state.draftModelId} disabled={Boolean(pending) || Boolean(session)}
          onSelect={props.onSelectModel} onConnect={props.onConnectCloud} onManage={props.onManageModels} />
        {session && <button className="icon-button subtle" type="button" aria-label="Review progress" title="Review progress"
          aria-pressed={reviewOpen} disabled={Boolean(pending)} onClick={() => setReviewOpen(open => !open)}><ChartColumn size={18} aria-hidden="true" /></button>}
      </StudyHeader>
      {session && <div className="practice-topic-bar"><span title={session.topic?.title}>{session.topic?.title ?? 'Saved practice'}</span>
        <button className="study-action-button" type="button" onClick={() => chooseSession()}>Practice another topic</button></div>}
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
          canQuote={!pending} onQuote={quote}>
          {session.topic?.question && session.topic.answer && <PracticeTopicView topic={session.topic} />}
          {session.questions.map((item, index) => <PracticeQuestionView key={item.id} question={item} index={index} active={index === session.currentIndex}
            busy={Boolean(pending)} sourcesOpen={index === session.currentIndex && materialsOpen}
            onShowSources={canShowSources(item) ? () => { continueQuestion(index); setMaterialsOpen(true) } : undefined}
            actions={index === session.currentIndex && <PracticeActions onChat={() => chatAbout(item)} />} />)}
          {(!question || error || (question.attempts.length > 0 && question === session.questions.at(-1))) && !pending && !session.sourcesExhausted &&
            <div className="practice-retry"><button type="button" className="study-action-button" disabled={!available} onClick={() => generate(session)}><RefreshCw size={16} aria-hidden="true" />Retry question</button></div>}
        </ConversationViewport>
      </> : <div className="practice-scroll"><PracticeTopicPicker isActive={isActive && !state.draftTopic} ready={scopeReady}
        disabled={!scopeReady || !available} scope={scope} materials={props.materials} model={model} settings={settings}
        modelSettings={model ? props.modelSettingsFor(model.id) : settings.modelDefaults} progress={progress}
        cache={state.topicSuggestions} onCache={cache => onChange(current => ({ ...current, topicSuggestions: cache }))} onChoose={start} /></div>}
      <div className="practice-status">
        {error && <p className="practice-error" role="alert">{error}</p>}
        {!session && catalog.status === 'ready' && !scopeReady && <p className="practice-muted" role="status">{scope.length ? 'Selected documents are unavailable.' : 'No documents selected.'}</p>}
        {!session && !available && <p className="practice-muted" role="status">No ready chat model selected.</p>}
        {session && !available && <div className="practice-error" role="status"><p>This session's model is unavailable. Your questions and answers are saved.</p><button type="button" className="practice-button" onClick={props.onManageModels}>Manage models</button></div>}
        {session?.sourcesExhausted && <p className="practice-muted" role="status">No supporting passages remain for this topic. Choose Practice another topic or change the documents in Library.</p>}
      </div>
      {session && question?.format === 'true_false' && !reviewOpen && <PracticeComposer question={question} busy={Boolean(pending)} isActive={isActive} onCheck={checkAnswer} />}
    </section>
    <StudyMaterialsPane catalog={catalog} scope={scope} open={materialsOpen}
      collectionExpansion={props.collectionExpansion} onCollectionExpansionChange={props.onCollectionExpansionChange}
      onChange={setScope} onClose={() => setMaterialsOpen(false)} onManage={props.onOpenLibrary}>
      {selectedSources.length > 0 && <SourceTray sources={selectedSources} error={props.sourceError}
        onOpenSource={source => props.onOpenSource(source, sources ?? [])} />}
    </StudyMaterialsPane>
  </div>
}
