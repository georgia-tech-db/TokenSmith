import { assessmentSession, assessmentLocked, assessmentDisplayQuestion, parseAssessmentPack, exportPractice, practiceConceptKey } from '@shared/practice-study'
import { practiceMastery, removePracticeHistory } from '@shared/practice-mastery'
import { practiceQuestionPlan } from '@shared/practice-question-plan'
import { MasteryMeter, TopicMasteryDetails } from './MasteryMeter'
import { usePracticeClock } from './hooks/usePracticeClock'
import { useEffect, useLayoutEffect, useMemo, useRef, useState, type ComponentType } from 'react'
import { ArrowLeft, ArrowRight, ChartColumn, Plus, RefreshCw, Upload } from 'lucide-react'
import type { ChatSelectedPassage, ChatSource, CourseMaterial, LocalModel, ModelRuntimeSettings, TokenSmithSettings } from '@shared/app-state'
import { practiceMessages, practiceQuestionText, practiceReviewLabel,
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
import { useSourceReader, type SourceReaderProps } from './hooks/useSourceReader'
import type { AnswerSourceDocument } from './answer-source-preview'
import { useSourcePane } from './hooks/useSourcePane'
import { SourcePane } from './SourcePane'
import { navigableSource } from './source-navigation'
import './practice.css'

interface Props extends StudyPanels {
  isActive: boolean
  state: PracticeState
  onChange: (update: (state: PracticeState) => PracticeState) => void
  onCommit?: (update: (state: PracticeState) => PracticeState) => Promise<PracticeState>
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
  readSource: (source: ChatSource) => Promise<AnswerSourceDocument>
  SourceReader: ComponentType<SourceReaderProps>
  progress: GenerationProgressApi
}

type SaveAction = { sessionId: string; kind: 'import' | 'response' | 'completion' }
export function PracticeScreen(props: Props) {
  const { state, onChange, isActive, models, catalog, settings, progress } = props
  const documents = catalog.documents
  const { historyOpen: sidebarOpen, onHistoryOpenChange: setSidebarOpen, materialsOpen, onMaterialsOpenChange: setMaterialsOpen } = props
  const [error, setError] = useState('')
  const [saving, setSaving] = useState(false)
  const quizFileInput = useRef<HTMLInputElement>(null)
  const [unsaved, setUnsaved] = useState<SaveAction>()
  const submitting = useRef(false)
  const [pending, setPending] = useState<{ id: string } | null>(null)
  const [pendingSources, setPendingSources] = useState<ChatSource[]>([])
  const reader = useSourceReader(props.readSource)
  const { state: passage, controller: sourcePreview } = useSourcePane()
  const [reviewOpen, setReviewOpen] = useState(false)
  const operation = useRef<{ id: string; controller: AbortController; sessionId: string; scopeKey: string } | null>(null)
  const session = state.sessions.find(item => item.id === state.activeSessionId)
  const mastery = useMemo(() => practiceMastery(state.sessions, [], state.masteryRecords), [state.sessions, state.masteryRecords])
  const topicMastery = session?.topic?.concept ? mastery.get(practiceConceptKey(session.topic.concept)) : undefined
  const activeSession = useRef({ id: session?.id, active: isActive })
  activeSession.current = { id: session?.id, active: isActive }
  const question = session?.questions[session.currentIndex]
  const questionPlan = useMemo(() => session && !session.assessment ? practiceQuestionPlan(session) : undefined, [session])
  const topicComplete = Boolean(questionPlan?.totalQuestions && !questionPlan.source && session?.questions.every(q => q.attempts.length))
  const locked = Boolean(session && assessmentLocked(session))
  const canShowSources = (item: PracticeQuestion) => !locked && settings.application.showSources && item.sources.length > 0
  const sources = pending && !locked && settings.application.showSources ? pendingSources
    : question && canShowSources(question) ? question.sources : undefined
  const selectedSources = useMemo(() => sources?.slice(0, maxSourceTrayCards) ?? [], [sources])
  const messages = useMemo(() => session?.questions.flatMap(q => practiceMessages(locked ? assessmentDisplayQuestion(q) : q)) ?? [], [session?.questions, locked])
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
      sourcePreview.finish(job.id)
      job.controller.abort()
      progress.cancel(job.id)
      void window.tokensmith?.cancelChatRequest(job.id).catch(() => {})
    }
    setPending(null)
    setPendingSources([])
  }
  useEffect(() => () => {
    const job = operation.current
    if (job) { job.controller.abort(); progress.cancel(job.id); void window.tokensmith?.cancelChatRequest(job.id).catch(() => {}) }
    operation.current = null
    sourcePreview.dismiss()
  }, [progress])
  // Scope can also change in Chat while this screen is hidden.
  useLayoutEffect(() => {
    const job = operation.current
    if (job && (job.scopeKey !== scopeKey || job.sessionId !== session?.id)) stop()
  }, [scopeKey, session?.id])
  useEffect(() => { setError(''); setReviewOpen(false) }, [scopeKey, session?.id])
  useLayoutEffect(() => {
    reader.close()
    // Starting a session begins generation before React commits its new ID.
    // Keep that request's preview; switching away invalidates it.
    if (!isActive || operation.current?.sessionId !== session?.id || operation.current?.scopeKey !== scopeKey) sourcePreview.dismiss()
  }, [isActive, scopeKey, session?.id, reader.close, sourcePreview])
  useEffect(() => {
    if (!settings.application.showSources) { sourcePreview.dismiss(); reader.close() }
    else if (!settings.application.readWhileWaiting) sourcePreview.dismissAutomatic()
  }, [settings.application.showSources, settings.application.readWhileWaiting, sourcePreview])
  useEffect(() => {
    if (isActive && question && !locked && settings.application.showSources && window.innerWidth > 1200) setMaterialsOpen(true)
  }, [isActive, question?.id, locked, settings.application.showSources, setMaterialsOpen])

  function openSource(source: ChatSource) {
    reader.close()
    sourcePreview.select(source, sources ?? [], question ? practiceQuestionText(question) : undefined)
    setMaterialsOpen(true)
  }
  useEffect(() => { if (locked) { sourcePreview.dismiss(); reader.close() } }, [locked, sourcePreview, reader.close])

  function updateSession(id: string, update: (value: PracticeSession) => PracticeSession) {
    onChange(current => ({ ...current, sessions: current.sessions.map(item => item.id === id ? update(item) : item) }))
  }
  useEffect(() => {
    const attempt = question?.attempts[0]
    if (!session || !question || !attempt || !isActive || locked || reviewOpen) return
    const id = `feedback:${attempt.id}`
    const record = () => {
      if (document.visibilityState !== 'visible' || !document.hasFocus()) return
      updateSession(session.id, s => s.observations?.some(e => e.id === id) ? s : ({ ...s, observations: [...(s.observations ?? []),
        { id, questionId: question.id, kind: 'feedback_shown', at: new Date().toISOString(), referenceId: attempt.id }] }))
    }
    record()
    window.addEventListener('focus', record); document.addEventListener('visibilitychange', record)
    return () => { window.removeEventListener('focus', record); document.removeEventListener('visibilitychange', record) }
  }, [session?.id, question?.id, question?.attempts.length, isActive, locked, reviewOpen])
  const timing = usePracticeClock(session, question, isActive && !reviewOpen && !pending && !saving && !unsaved, updateSession)
  async function commit(update: (value: PracticeState) => PracticeState) {
    if (props.onCommit) return props.onCommit(update)
    const next = update(state); onChange(() => next); return next
  }
  async function persist(action: SaveAction, update: (value: PracticeState) => PracticeState) {
    if (submitting.current) return
    submitting.current = true; setSaving(true); setError('')
    try {
      const saved = await commit(update)
      setUnsaved(undefined)
      if (activeSession.current.id !== action.sessionId || !activeSession.current.active) return
      const updated = saved.sessions.find(s => s.id === action.sessionId)
      if (!updated) return
      if (action.kind === 'completion') setReviewOpen(true)
      if (action.kind !== 'response') return
      const next = updated.questions.findIndex(q => !q.attempts.length && (updated.assessment || q.format === 'true_false'))
      if (next >= 0) updateSession(updated.id, s => ({ ...s, currentIndex: next }))
      else if (!updated.assessment && !practiceQuestionPlan(updated).source) {
        updateSession(updated.id, s => ({ ...s, sourcesExhausted: true }))
      } else if (!updated.assessment && available && scopeReady && !updated.sourcesExhausted) generate(updated)
    } catch (reason) {
      setUnsaved(action); setError(reason instanceof Error ? reason.message : String(reason))
    } finally { submitting.current = false; setSaving(false) }
  }
  async function importPack(file?: File) {
    if (!file || submitting.current || unsaved) return
    try {
      if (file.size > 2_000_000) throw new Error('Assessment file exceeds 2 MB.')
      const pack = parseAssessmentPack(JSON.parse(await file.text()))
      const created = assessmentSession(pack, `assessment-${crypto.randomUUID()}`, settings.modelDefaults, new Date().toISOString())
      props.onScopeChange([])
      await persist({ sessionId: created.id, kind: 'import' }, current => ({ ...current, activeSessionId: created.id, sessions: [created, ...current.sessions] }))
    } catch (e) { setError(e instanceof Error ? e.message : String(e)) }
  }
  function exportSession() {
    if (!session || unsaved) return
    const participant = window.prompt('Participant code (optional; avoid names or email addresses):', '')
    if (participant === null) return
    const url = URL.createObjectURL(new Blob([JSON.stringify(exportPractice([session], participant), null, 2)], { type: 'application/json' }))
    const link = document.createElement('a'); link.href = url; link.download = 'tokensmith-practice.json'; link.click()
    setTimeout(() => URL.revokeObjectURL(url), 1000)
  }
  async function finishAssessment() {
    if (!session || !locked || submitting.current || unsaved || session.questions.some(q => !q.attempts.length)) return
    const completedAt = new Date().toISOString()
    await persist({ sessionId: session.id, kind: 'completion' }, current => ({ ...current, sessions: current.sessions.map(s => s.id !== session.id ? s : {
      ...s, assessment: { ...s.assessment!, completedAt },
      questions: s.questions.map(q => ({ ...q, attempts: q.attempts.map(a => ({ ...a, feedback: q.format === 'true_false'
        ? { kind: 'true_false' as const, correct: (a.answer === 'True') === q.correctAnswer, explanation: q.referenceAnswer ?? '' }
        : 'Awaiting independent review.' })) })) }) }))
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
    reader.close()
    sourcePreview.begin(id)
    setPendingSources([])
    setError('')
    setPending({ id })
    const started = performance.now()
    progress.begin({ id, materialIds: target.documents.map(d => d.materialId), kind: 'quiz', label, conversationId: target.id, model: selectedModel,
      settings: target.modelSettings, onStop: stop })
    try {
      const result = await run(createPracticeClient(window.tokensmith, selectedModel, settings, { materials: props.materials, embeddingModels: props.embeddingModels }, state.sessions), id, controller.signal)
      if (operation.current?.id !== id || controller.signal.aborted) return
      sourcePreview.finish(id)
      complete(result, performance.now() - started)
      progress.finish(id, 'Practice ready')
    } catch (reason) {
      if (operation.current?.id !== id || controller.signal.aborted) return
      setError(reason instanceof Error ? reason.message.replace(/^Error invoking remote method '[^']+': (?:Error: )?/, '') : String(reason))
      progress.cancel(id)
    } finally {
      sourcePreview.finish(id)
      if (operation.current?.id === id) { operation.current = null; setPending(null); setPendingSources([]) }
    }
  }
  function generate(target: PracticeSession) {
    if (target.assessment) return
    if (!target.topic) { chooseSession(); return }
    if (target.topic.prepared && !practiceQuestionPlan(target).source) {
      updateSession(target.id, current => ({ ...current, sourcesExhausted: true }))
      return
    }
    if (!target.documents.every(doc => documents.some(available => studyDocumentKey(available) === studyDocumentKey(doc)))) {
      setError('Open the library to continue this practice.'); return
    }
    let prepared = target
    sourcePreview.dismiss()
    void perform(target, 'Preparing question', async (client, id, signal) => {
      prepared = await client.prepareTopic(target, signal, id)
      return client.question(prepared, id, signal, (questionSourceCursor, candidates, questionTargetAnswer) => {
        if (operation.current?.id !== id || signal.aborted) return
        // Retry advances the passage but retains the same randomly chosen key.
        updateSession(target.id, current => ({ ...current, topic: prepared.topic, questionSourceCursor, questionTargetAnswer }))
        progress.stage(id, 'Preparing question')
        setPendingSources(candidates)
        const source = candidates[0]
        if (activeSession.current.active && settings.application.showSources && settings.application.readWhileWaiting && source && navigableSource(source)) {
          if (sourcePreview.open(id, candidates)) setMaterialsOpen(true)
        }
      })
    }, (generated, generationMs) => {
      updateSession(target.id, current => generated
        ? { ...current, topic: prepared.topic, questionTargetAnswer: undefined,
          questions: [...current.questions, { ...generated, generationMs, item: { id: generated.id, version: '1', keyVersion: '1', origin: 'generated' as const } }], currentIndex: current.questions.length }
        : { ...current, topic: prepared.topic, sourcesExhausted: true })
    })
  }
  function start(topic: PracticeTopic) {
    if (!isActive || session || !model || !available || operation.current || submitting.current || unsaved || !scopeReady || !topic.title.trim()) return
    const scope = documents.filter(item => selected.includes(studyDocumentKey(item)))
    if (!scope.length) return
    const concept = topic.concept ?? { courseId: studyScopeKey(scope), id: crypto.randomUUID(), title: topic.title }
    const created: PracticeSession = { observations: [], id: `practice-${crypto.randomUUID()}`, title: topic.title, topic: { ...topic, concept }, createdAt: new Date().toISOString(),
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
    if (!target || !window.confirm(`Delete "${target.title}" and its practice answers? Your topic mastery will be kept.`)) return
    if (session?.id === id) { stop(); setError('') }
    onChange(current => removePracticeHistory(current, id))
  }
  function clearSessions() {
    if (!window.confirm('Clear all practice conversations? Your topic mastery and Chat conversations will be kept.')) return
    stop(); setError(''); setReviewOpen(false)
    onChange(current => removePracticeHistory(current))
  }
  async function checkAnswer(answer: string) {
    if (!session || !question || submitting.current || unsaved || pending || operation.current || question.attempts.length) return
    if (!answer.trim() || (question.format === 'true_false' && !['True', 'False'].includes(answer))) return
    const at = timing()
    const help = (session.observations ?? []).filter(e => e.questionId === question.id && e.kind === 'help_shown')
    try {
      const feedback = locked ? 'Response saved. Feedback is available after completing the assessment.'
        : gradeTrueFalse(question, answer)
      const id = crypto.randomUUID()
      await persist({ sessionId: session.id, kind: 'response' }, current => ({ ...current, sessions: current.sessions.map(s => s.id !== session.id ? s : {
        ...s, observations: [...(s.observations ?? []), { id, questionId: question.id, kind: 'submitted' as const, at: at.submittedAt }],
        questions: s.questions.map(q => q.id !== question.id || q.attempts.length ? q : { ...q, draft: answer,
          attempts: [...q.attempts, { id, answer, feedback, ...at, observationVersion: 1 as const,
            assistanceIds: help.map(e => e.id), assisted: help.length > 0 }] }) }) }))
    } catch (reason) { setError(reason instanceof Error ? reason.message : String(reason)) }
  }
  function chatAbout(item: PracticeQuestion, passage?: ChatSelectedPassage) {
    if (!session || locked) return
    stop()
    props.onChat(session, item, passage)
  }
  function quote(passage: ChatSelectedPassage) {
    const item = session?.questions.find(item => practiceMessages(item).some(message => message.id === passage.messageId))
    if (item) chatAbout(item, { ...passage, question: practiceQuestionText(item) })
  }
  function continueQuestion(index: number) {
    if (!session || pending || saving || unsaved || index < 0 || index >= session.questions.length) return
    reader.close()
    if (index !== session.currentIndex) sourcePreview.dismiss()
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
      {session && <div className="practice-topic-bar"><span title={session.topic?.title}>{session.topic?.title ?? session.title}</span>
        {!session.assessment && <div className="practice-live-mastery" aria-live="polite"><MasteryMeter score={topicMastery?.score ?? null} empty={topicMastery?.withHelp ? 'Try without help' : 'Not practiced'} /></div>}
        <div className="practice-session-actions" role="group" aria-label="Practice session actions">
          <button className="study-action-button" type="button" onClick={() => chooseSession()}>Practice another topic</button>
          <button className="study-action-button" type="button" disabled={Boolean(unsaved) || saving} onClick={exportSession}>Export practice record</button>
          {!session.assessment && <button className="study-action-button" type="button" disabled={Boolean(pending)} aria-expanded={reviewOpen} onClick={() => setReviewOpen(open => !open)}>View mastery</button>}
        </div></div>}
      {session && locked && <div className="practice-topic-bar">
        <span>Assessment: help and feedback are available after completion.</span><button type="button" disabled={saving || Boolean(unsaved) || session.questions.some(q => !q.attempts.length)} onClick={() => void finishAssessment()}>Complete assessment</button>
      </div>}
      {session && reviewOpen ? <div className="practice-scroll"><div className="practice-content">
          <p className="practice-topic">{session.title}</p><h1>Your practice progress</h1>
          {!session.assessment && <div className="practice-review-mastery"><MasteryMeter score={topicMastery?.score ?? null} /><TopicMasteryDetails result={topicMastery} /></div>}
          <p>{session.questions.filter(item => item.attempts.length).length} of {session.questions.length} questions reviewed</p>
          {session.questions.map((item, index) => <details className="practice-recap-question" key={item.id}>
            <summary><span>{practiceQuestionText(item)}</span><span className="practice-muted">{practiceReviewLabel(item)}</span></summary>
            {!locked && item.rubric && <p className="practice-muted">{item.rubric.objective}</p>}
            {item.attempts.at(-1) && <><h3>Your answer</h3><MessageText text={item.attempts.at(-1)!.answer} /><h3>Feedback</h3><PracticeFeedbackView feedback={locked ? 'Response saved.' : item.attempts.at(-1)!.feedback} /></>}
            <button type="button" className="practice-link" onClick={() => continueQuestion(index)}>Review question<ArrowRight size={16} /></button>
          </details>)}
          <div className="practice-footer"><button type="button" className="practice-button" onClick={() => setReviewOpen(false)}><ArrowLeft size={16} />Return to practice</button><button type="button" className="practice-button" onClick={() => chooseSession()}><Plus size={16} />New practice</button></div>
      </div></div> : session ? <>
        <ConversationViewport key={session.id} messages={messages} navigationQuestions={navigationQuestions} pending={Boolean(pending)} isActive={isActive}
          focusQuestionId={question?.id} onNavigate={id => continueQuestion(session.questions.findIndex(item => item.id === id))}
          canQuote={!pending && !locked} onQuote={quote}>
          {session.topic?.question && session.topic.answer && <PracticeTopicView topic={session.topic} />}
          {session.questions.map((item, index) => <PracticeQuestionView key={item.id} question={locked ? assessmentDisplayQuestion(item) : item} index={index} active={index === session.currentIndex}
            busy={Boolean(pending)} sourcesOpen={index === session.currentIndex && materialsOpen}
            onShowSources={canShowSources(item) ? () => { continueQuestion(index); sourcePreview.dismiss(); setMaterialsOpen(true) } : undefined}
            actions={!locked && index === session.currentIndex && <PracticeActions onChat={() => chatAbout(item)} />} />)}
          {(!question || error || (question.attempts.length > 0 && question === session.questions.at(-1))) && !pending && !session.sourcesExhausted && !topicComplete &&
            <div className="practice-retry"><button type="button" className="study-action-button" disabled={!available || !scopeReady || saving || Boolean(unsaved)} onClick={() => generate(session)}><RefreshCw size={16} aria-hidden="true" />Retry question</button></div>}
        </ConversationViewport>
      </> : <div className="practice-scroll"><PracticeTopicPicker isActive={isActive && !state.draftTopic} ready={scopeReady}
        disabled={!scopeReady || !available || saving || Boolean(unsaved)} scope={scope} materials={props.materials} sessions={state.sessions} masteryRecords={state.masteryRecords}
        onChoose={start} /></div>}
      <div className="practice-status">
        {unsaved && <p role="alert">Changes have not been saved. <button type="button" disabled={saving} onClick={() => void persist(unsaved, current => current)}>Retry saving</button></p>}
        {error && <p className="practice-error" role="alert">{error}</p>}
        {!session && catalog.status === 'ready' && !scopeReady && <p className="practice-muted" role="status">{scope.length ? 'Selected documents are unavailable.' : 'No documents selected.'}</p>}
        {!session && !available && <p className="practice-muted" role="status">No ready chat model selected.</p>}
        {session && !session.assessment && catalog.status === 'ready' && !scopeReady && <p className="practice-muted" role="status">Open the library to continue this practice.</p>}
        {session && !session.assessment && !available && <div className="practice-error" role="status"><p>This session's model is unavailable. Your questions and answers are saved.</p><button type="button" className="practice-button" onClick={props.onManageModels}>Manage models</button></div>}
        {topicComplete ? <p className="practice-muted" role="status">Practice complete. You've finished this topic's {session!.questions.length}-question session. Choose Practice another topic to continue.</p>
          : session?.sourcesExhausted && !session.assessment && <p className="practice-muted" role="status">No supporting passages remain for this topic. Choose Practice another topic or change the selected documents.</p>}
      </div>
      {session && question && (question.format === 'true_false' || session.assessment) && !reviewOpen && <PracticeComposer question={locked ? assessmentDisplayQuestion(question) : question} assessment={Boolean(session.assessment)} busy={Boolean(pending) || saving || Boolean(unsaved)} isActive={isActive} onDraft={draft => updateSession(session.id, s => ({ ...s, questions: s.questions.map(q => q.id === question.id ? { ...q, draft } : q) }))} onCheck={checkAnswer} />}
    </section>
    {passage ? <SourcePane state={passage} open={materialsOpen} error={reader.error}
      loading={reader.loading} onMove={direction => { reader.close(); sourcePreview.move(direction) }} onClose={() => { sourcePreview.dismiss(); reader.close(); setMaterialsOpen(false) }}
      onDocuments={() => { sourcePreview.dismiss(); reader.close() }}
      onOpenDocument={source => { void reader.open(source, passage.sources) }} /> : <StudyMaterialsPane catalog={catalog} scope={scope} open={materialsOpen}
      collectionExpansion={props.collectionExpansion} onCollectionExpansionChange={props.onCollectionExpansionChange}
      onChange={setScope} onClose={() => setMaterialsOpen(false)} onManage={props.onOpenLibrary}
      footerActions={!session && <>
        <button type="button" className="study-action-button" disabled={saving || Boolean(unsaved)} onClick={() => quizFileInput.current?.click()}><Upload size={17} aria-hidden="true" />Import Quiz</button>
        <input ref={quizFileInput} type="file" hidden accept=".json,application/json" aria-label="Import Quiz" disabled={saving || Boolean(unsaved)} onChange={e => { void importPack(e.target.files?.[0]); e.target.value = '' }} />
      </>}>
      {selectedSources.length > 0 && <SourceTray sources={selectedSources} error={reader.error} onOpenSource={openSource} />}
    </StudyMaterialsPane>}
    {isActive && reader.state && <props.SourceReader state={reader.state} navigation={reader.navigation} onClose={reader.close} />}
  </div>
}
