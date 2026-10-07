import { useEffect, useRef, useState } from 'react'
import { ArrowLeft, ArrowRight, Loader2, Plus, Square, Trash2 } from 'lucide-react'
import type { ChatSource, LocalModel, ModelRuntimeSettings, TokenSmithSettings } from '@shared/app-state'
import { practiceAssisted, practiceReference, practiceReviewLabel,
  type PracticeQuestion, type PracticeReference, type PracticeSession, type PracticeState, type StudyMode } from '@shared/practice'
import { quizTotalQuestions } from '@shared/quiz'
import { ChatModelPicker } from './ChatModelPicker'
import { MessageText } from './MessageText'
import { StudyHeader } from './StudyHeader'
import { StudyMaterialsPane } from './StudyMaterialsPane'
import type { StudyCatalog } from './hooks/useStudyDocuments'
import { studyDocumentKey, scopeRefs, type StudyDocumentRef } from '@shared/study-scope'
import { PracticeQuestionView } from './PracticeQuestionView'
import { PracticeFeedbackView } from './PracticeFeedbackView'
import { createPracticeClient } from './practice-client'
import type { GenerationProgressApi } from './hooks/useGenerationProgress'
import './practice.css'

interface Props {
  isActive: boolean
  state: PracticeState
  onChange: (update: (state: PracticeState) => PracticeState) => void
  onModeChange: (mode: StudyMode) => void
  catalog: StudyCatalog
  inheritScope: StudyDocumentRef[]
  models: LocalModel[]
  selectedModel?: LocalModel
  settings: TokenSmithSettings
  modelSettingsFor: (id: string) => ModelRuntimeSettings
  onSelectModel: (id: string) => void
  onConnectCloud: (model?: LocalModel) => void
  onManageModels: () => void
  onOpenLibrary: () => void
  onDiscuss: (reference: PracticeReference) => void
  onOpenSource: (source: ChatSource, sources: ChatSource[]) => void
  sourceError?: string
  progress: GenerationProgressApi
}

export function PracticeScreen(props: Props) {
  const { state, onChange, isActive, models, catalog, settings, progress } = props
  const documents = catalog.documents
  const [sidebarOpen, setSidebarOpen] = useState(() => window.innerWidth > 800)
  const [materialsOpen, setMaterialsOpen] = useState(() => window.innerWidth > 1200)
  const [error, setError] = useState('')
  const [pending, setPending] = useState<{ id: string; label: string } | null>(null)
  const scroll = useRef<HTMLDivElement>(null)
  const operation = useRef<{ id: string; controller: AbortController } | null>(null)
  const session = state.sessions.find(item => item.id === state.activeSessionId)
  const question = session?.questions[session.currentIndex]
  useEffect(() => { if (scroll.current) scroll.current.scrollTop = 0 }, [session?.id, question?.id, session?.completed])
  const model = session ? models.find(item => item.id === session.modelId) : props.selectedModel
  const available = model?.status === 'ready'
  const scope = session ? scopeRefs(session.documents) : state.draftScope ?? props.inheritScope
  const selected = scope.map(studyDocumentKey)
  const setScope = (draftScope: StudyDocumentRef[]) => onChange(current => ({ ...current, draftScope }))
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

  function updateSession(id: string, update: (value: PracticeSession) => PracticeSession) {
    onChange(current => ({ ...current, sessions: current.sessions.map(item => item.id === id ? update(item) : item) }))
  }
  function updateQuestion(update: (value: PracticeQuestion) => PracticeQuestion) {
    if (!session || !question) return
    updateSession(session.id, current => ({ ...current, questions: current.questions.map(item => item.id === question.id ? update(item) : item) }))
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
    operation.current = { id, controller }
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
      updateSession(target.id, current => ({ ...current, questions: [...current.questions, generated], currentIndex: current.questions.length })))
  }
  function start() {
    if (!model || !available || pending || !scopeReady) return
    const scope = documents.filter(item => selected.includes(studyDocumentKey(item)))
    if (!scope.length) return
    const names = [...new Set(scope.map(item => item.collectionName))]
    const created: PracticeSession = { id: `practice-${crypto.randomUUID()}`, title: names.join(', '), createdAt: new Date().toISOString(),
      documents: scope, modelId: model.id, modelSettings: props.modelSettingsFor(model.id), totalQuestions: quizTotalQuestions,
      questions: [], currentIndex: 0, completed: false }
    onChange(current => ({ ...current, activeSessionId: created.id, sessions: [created, ...current.sessions] }))
    generate(created)
  }
  function chooseSession(id?: string) {
    stop(); setError('')
    if (window.innerWidth <= 800) setSidebarOpen(false)
    onChange(current => ({ ...current, activeSessionId: id, ...(id ? {} : { draftScope: scopeRefs(scope) }) }))
  }
  function checkAnswer() {
    if (!session || !question || !question.rubric || !question.draft.trim() || pending) return
    const answer = question.draft.trim()
    const assisted = practiceAssisted(question)
    void perform(session, 'Checking your answer', (client, id) => client.feedback(session, question, answer, id), (feedback, durationMs) =>
      updateQuestion(current => ({ ...current, reviewing: true, attempts: [...current.attempts,
        { id: crypto.randomUUID(), answer, feedback, assisted, durationMs }] })))
  }
  function help(kind: 'hint' | 'answer') {
    if (!session || !question || pending) return
    if (kind === 'hint' && question.hint) updateQuestion(current => ({ ...current, hintUsed: true }))
    if (kind === 'answer' && question.referenceAnswer) updateQuestion(current => ({ ...current, answerRevealed: true, reviewing: true }))
  }
  function next() {
    if (!session || pending) return
    setError('')
    if (session.currentIndex + 1 < session.questions.length) updateSession(session.id, current => ({ ...current, currentIndex: current.currentIndex + 1 }))
    else if (session.questions.length >= session.totalQuestions) updateSession(session.id, current => ({ ...current, completed: true }))
    else generate(session)
  }
  function discuss() {
    if (!session || !question) return
    const reference = practiceReference(session, question)
    props.onDiscuss(reference)
  }
  const selectedCount = documents.filter(item => selected.includes(studyDocumentKey(item))).length
  const scopeReady = catalog.status === 'ready' && selectedCount > 0 && selectedCount === scope.length
  return <div className={`view-frame study-frame practice-frame ${sidebarOpen ? 'has-history' : ''} ${materialsOpen ? 'has-materials' : ''}`} hidden={!isActive}>
    {sidebarOpen && <aside className="practice-sidebar" aria-label="Practice sessions">
      <button type="button" className="practice-button" onClick={() => chooseSession()}><Plus size={17} aria-hidden="true" />New practice</button>
      <div className="practice-session-list">{state.sessions.map(item => <div className="practice-session-row" key={item.id}>
        <button type="button" className={`practice-session ${session?.id === item.id ? 'is-selected' : ''}`} onClick={() => chooseSession(item.id)}>
          <strong>{item.title}</strong><span>{item.completed ? 'Finished' : item.questions.length ? `Question ${item.currentIndex + 1} of ${item.totalQuestions}` : 'Not started'}</span>
        </button>
        <button type="button" className="practice-delete" title="Delete practice session" aria-label={`Delete practice ${item.title}`} onClick={() => {
          if (!window.confirm('Delete this practice session and its answers?')) return
          if (session?.id === item.id) { stop(); setError('') }
          onChange(current => ({ ...current, sessions: current.sessions.filter(value => value.id !== item.id), activeSessionId: current.activeSessionId === item.id ? undefined : current.activeSessionId }))
        }}><Trash2 size={15} aria-hidden="true" /></button>
      </div>)}</div>
    </aside>}
    <section className="practice-main">
      <StudyHeader mode="practice" onModeChange={props.onModeChange} historyOpen={sidebarOpen} onHistoryToggle={() => setSidebarOpen(value => !value)}
        materialsOpen={materialsOpen} onMaterialsToggle={() => setMaterialsOpen(value => !value)} count={scope.length}>
        <ChatModelPicker models={models} selectedModel={model} disabled={Boolean(pending) || Boolean(session)}
          onSelect={props.onSelectModel} onConnect={props.onConnectCloud} onManage={props.onManageModels} />
      </StudyHeader>
      <div className="practice-scroll" ref={scroll}><div className="practice-content">
        {error && <p className="practice-error" role="alert">{error}</p>}
        {props.sourceError && <p className="practice-error" role="alert">{props.sourceError}</p>}
        {session && !available && <div className="practice-error" role="status"><p>This session's model is unavailable. Your questions and answers are saved.</p><button type="button" className="practice-button" onClick={props.onManageModels}>Manage models</button></div>}
        {!session ? <>
          <h1>What would you like to practice?</h1>
          <div className="study-scope-summary">{scope.length ? <>{scope.length} document{scope.length === 1 ? '' : 's'} selected</> : 'No materials selected'}
            <br /><button className="practice-link" type="button" onClick={() => setMaterialsOpen(true)}>Choose documents</button></div>
          <div className="practice-footer"><span><strong>{quizTotalQuestions} questions</strong></span>
            <button type="button" className="practice-button is-primary" onClick={start} disabled={!available || !scopeReady || Boolean(pending)}>Start practice<ArrowRight size={17} /></button>
          </div>
        </> : session.completed ? <>
          <p className="practice-topic">{session.title}</p><h1>Your practice recap</h1>
          <p>{session.questions.filter(item => item.attempts.length || item.answerRevealed).length} of {session.totalQuestions} questions reviewed</p>
          {session.questions.map((item, index) => <details className="practice-recap-question" key={item.id}>
            <summary><span>Question {index + 1}</span><span className="practice-muted">{practiceReviewLabel(item)}</span></summary>
            <MessageText text={item.text} />{item.rubric && <p className="practice-muted">{item.rubric.objective}</p>}
            {item.attempts.at(-1) && <><h3>Your answer</h3><MessageText text={item.attempts.at(-1)!.answer} /><h3>Feedback</h3><PracticeFeedbackView feedback={item.attempts.at(-1)!.feedback} /></>}
            {item.answerRevealed && <p className="practice-muted">Explanation viewed</p>}
            <button type="button" className="practice-link" onClick={() => updateSession(session.id, current => ({ ...current, completed: false, currentIndex: index }))}>Review question<ArrowRight size={16} /></button>
          </details>)}
          <div className="practice-footer"><button type="button" className="practice-button" onClick={() => updateSession(session.id, current => ({ ...current, completed: false }))}><ArrowLeft size={16} />Return to session</button><button type="button" className="practice-button is-primary" onClick={() => chooseSession()}><Plus size={16} />New practice</button></div>
        </> : question ? <PracticeQuestionView session={session} question={question} busy={Boolean(pending)} available={available}
          onDraft={draft => updateQuestion(current => ({ ...current, draft }))} onCheck={checkAnswer}
          onHint={() => help('hint')} onReveal={() => help('answer')} onRetry={() => { setError(''); updateQuestion(current => ({ ...current, draft: current.attempts.at(-1)?.answer ?? current.draft, reviewing: false })) }}
          onCancelRevision={() => updateQuestion(current => ({ ...current, draft: current.attempts.at(-1)?.answer ?? current.draft, reviewing: true }))}
          onDiscuss={discuss} onNext={next} onPrevious={() => updateSession(session.id, current => ({ ...current, currentIndex: current.currentIndex - 1 }))}
          onFinish={() => updateSession(session.id, current => ({ ...current, completed: true }))} onOpenSource={props.onOpenSource} />
          : <><h1>{session.title}</h1>{!pending && <button type="button" className="practice-button is-primary" disabled={!available} onClick={() => generate(session)}>Prepare first question<ArrowRight size={17} /></button>}</>}
        {pending && <div className="practice-pending" role="status"><Loader2 className="spin" size={17} /><span>{pending.label}...</span><button type="button" className="practice-link" onClick={stop}><Square size={14} />Stop</button></div>}
      </div></div>
    </section>
    <StudyMaterialsPane catalog={catalog} scope={scope} open={materialsOpen} disabled={Boolean(pending)} locked={Boolean(session)}
      onChange={setScope} onClose={() => setMaterialsOpen(false)} onManage={props.onOpenLibrary} onNewPractice={() => chooseSession()} />
  </div>
}
