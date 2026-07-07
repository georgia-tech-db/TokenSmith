import { app, BrowserWindow, shell } from 'electron'
import { spawn } from 'node:child_process'
import { existsSync } from 'node:fs'
import { join } from 'node:path'
import type { LocalModel, ModelRuntimeSettings } from '../../shared/app-state'
import type {
  EngineChatRequest,
  EngineChatResponse,
  EngineQuestionSuggestionRequest,
  EngineQuestionSuggestionResponse
} from '../../shared/engine'
import {
  defaultOllamaBaseUrl,
  recommendedOllamaChatModel,
  recommendedOllamaEmbeddingModel,
  type OllamaDeleteResult,
  type OllamaModelInfo,
  type OllamaOpenResult,
  type OllamaPullProgress,
  type OllamaPullResult,
  type OllamaStatus
} from '../../shared/ollama'
import {
  answerWithOrderedSources,
  defaultFollowUpPrompt,
  followUpSuggestionCount,
  formatFollowUpInstruction,
  parseFollowUpSuggestions,
  questionSuggestionCount,
  questionSuggestionMessages,
  shouldGenerateFollowUps,
  studyChatMessages,
  type StudyChatMessage
} from './study-chat-format'

interface OllamaTagsResponse {
  models?: Array<{
    name?: string
    model?: string
    modified_at?: string
    size?: number
    digest?: string
    details?: {
      family?: string
      parameter_size?: string
      quantization_level?: string
    }
  }>
}

interface OllamaChatResponse {
  message?: {
    content?: string
  }
}

interface OllamaPullResponse {
  status?: string
}

interface OllamaPullStreamMessage {
  status?: string
  digest?: string
  total?: number
  completed?: number
  error?: string
}

interface ActiveOllamaPull {
  baseUrl: string
  controller: AbortController
  latestProgress?: OllamaPullProgress
  model: string
  reader?: ReadableStreamDefaultReader<Uint8Array>
}

const activeOllamaPulls = new Map<string, ActiveOllamaPull>()

function normalizeOllamaBaseUrl(baseUrl = defaultOllamaBaseUrl): string {
  let normalized = baseUrl.trim().replace(/\/+$/, '')
  if (normalized.endsWith('/api')) {
    normalized = normalized.slice(0, -4).replace(/\/+$/, '')
  }
  if (!normalized || normalized === 'http://localhost:11434') {
    return defaultOllamaBaseUrl
  }
  return normalized
}

function ollamaApiBaseUrl(baseUrl = defaultOllamaBaseUrl): string {
  const normalized = normalizeOllamaBaseUrl(baseUrl)
  return normalized.endsWith('/api') ? normalized : `${normalized}/api`
}

function ollamaAppCandidates(): string[] {
  if (process.platform !== 'darwin') {
    return []
  }

  return [
    '/Applications/Ollama.app',
    join(app.getPath('home'), 'Applications', 'Ollama.app')
  ]
}

function installedOllamaAppPath(): string | undefined {
  return ollamaAppCandidates().find((candidate) => existsSync(candidate))
}

function ollamaCliCandidates(): string[] {
  const candidates = [
    join('/Applications', 'Ollama.app', 'Contents', 'Resources', 'ollama'),
    join(app.getPath('home'), 'Applications', 'Ollama.app', 'Contents', 'Resources', 'ollama'),
    '/opt/homebrew/bin/ollama',
    '/usr/local/bin/ollama'
  ]

  const installedCandidates = candidates.filter((candidate) => existsSync(candidate))
  installedCandidates.push(process.platform === 'win32' ? 'ollama.exe' : 'ollama')
  return Array.from(new Set(installedCandidates))
}

function wait(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms))
}

function pullProgressPercent(completed?: number, total?: number): number {
  if (!total || total <= 0 || !completed) {
    return 0
  }

  return Math.max(0, Math.min(100, Math.round((completed / total) * 100)))
}

function emitPullProgress(progress: OllamaPullProgress): void {
  for (const window of BrowserWindow.getAllWindows()) {
    window.webContents.send('ollama:pull-progress', progress)
  }
}

async function fetchWithTimeout(url: string, init: RequestInit = {}, timeoutMs = 10_000): Promise<Response> {
  const controller = new AbortController()
  const inputSignal = init.signal
  const timeout = setTimeout(() => controller.abort(), timeoutMs)
  const abortFromInput = () => controller.abort(inputSignal?.reason)

  if (inputSignal?.aborted) {
    controller.abort(inputSignal.reason)
  } else {
    inputSignal?.addEventListener('abort', abortFromInput, { once: true })
  }

  try {
    const { signal: _signal, ...fetchInit } = init
    return await fetch(url, {
      ...fetchInit,
      signal: controller.signal
    })
  } finally {
    inputSignal?.removeEventListener('abort', abortFromInput)
    clearTimeout(timeout)
  }
}

async function responseErrorDetail(response: Response): Promise<string> {
  try {
    const text = await response.text()
    const trimmed = text.trim()
    return trimmed ? `: ${trimmed.slice(0, 500)}` : ''
  } catch {
    return ''
  }
}

function errorMessage(error: unknown, fallback: string): string {
  return error instanceof Error ? error.message : fallback
}

function isMissingOllamaModelError(status: number, detail: string): boolean {
  return status === 404 || /file does not exist|model.*not found|not found|does not exist/i.test(detail)
}

function normalizeModelInfo(model: NonNullable<OllamaTagsResponse['models']>[number]): OllamaModelInfo | undefined {
  const name = model.model || model.name
  if (!name) {
    return undefined
  }

  return {
    name,
    model: model.model,
    size: model.size,
    digest: model.digest,
    modifiedAt: model.modified_at,
    details: {
      family: model.details?.family,
      parameterSize: model.details?.parameter_size,
      quantizationLevel: model.details?.quantization_level
    }
  }
}

function normalizedModelName(name: string): string {
  return name.trim().toLowerCase().replace(/:latest$/, '')
}

function ollamaPullKey(model: string, baseUrl = defaultOllamaBaseUrl): string {
  return `${normalizeOllamaBaseUrl(baseUrl)}::${normalizedModelName(model)}`
}

function activeOllamaPullFor(model: string, baseUrl = defaultOllamaBaseUrl): ActiveOllamaPull | undefined {
  const exactPull = activeOllamaPulls.get(ollamaPullKey(model, baseUrl))
  if (exactPull) {
    return exactPull
  }

  const requestedModel = normalizedModelName(model)
  return Array.from(activeOllamaPulls.values()).find((pull) => normalizedModelName(pull.model) === requestedModel)
}

function runOllamaCli(args: string[], timeoutMs = 120_000): Promise<string> {
  const candidates = ollamaCliCandidates()

  return new Promise((resolve, reject) => {
    let candidateIndex = 0
    const errors: string[] = []

    const tryNextCandidate = () => {
      const cliPath = candidates[candidateIndex]
      candidateIndex += 1

      if (!cliPath) {
        reject(new Error(errors.join('\n') || 'The Ollama command was not found.'))
        return
      }

      const child = spawn(cliPath, args, {
        stdio: ['ignore', 'pipe', 'pipe']
      })
      let stdout = ''
      let stderr = ''
      const timeout = setTimeout(() => {
        child.kill('SIGTERM')
      }, timeoutMs)

      child.stdout?.on('data', (chunk) => {
        stdout += chunk.toString()
      })
      child.stderr?.on('data', (chunk) => {
        stderr += chunk.toString()
      })
      child.once('error', (error) => {
        clearTimeout(timeout)
        errors.push(`${cliPath}: ${error.message}`)
        tryNextCandidate()
      })
      child.once('close', (code, signal) => {
        clearTimeout(timeout)
        if (code === 0) {
          resolve(stdout.trim())
          return
        }

        const detail = [stderr.trim(), stdout.trim()].filter(Boolean).join('\n')
        errors.push(`${cliPath}: ${detail || `exited with ${signal ?? code}`}`)
        tryNextCandidate()
      })
    }

    tryNextCandidate()
  })
}

async function deleteOllamaModelWithCli(model: string): Promise<'removed' | 'missing'> {
  try {
    await runOllamaCli(['rm', model])
    return 'removed'
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error)
    if (isMissingOllamaModelError(404, message)) {
      return 'missing'
    }
    throw error
  }
}

export function ollamaModelMatches(installedModel: OllamaModelInfo, requestedModel: string): boolean {
  const requested = normalizedModelName(requestedModel)
  return [installedModel.name, installedModel.model]
    .filter((name): name is string => Boolean(name))
    .some((name) => {
      const normalized = normalizedModelName(name)
      return normalized === requested || normalized.startsWith(`${requested}:`)
    })
}

export async function getOllamaStatus(baseUrl = defaultOllamaBaseUrl): Promise<OllamaStatus> {
  const normalizedBaseUrl = normalizeOllamaBaseUrl(baseUrl)

  try {
    const response = await fetchWithTimeout(`${ollamaApiBaseUrl(normalizedBaseUrl)}/tags`, {
      headers: { Accept: 'application/json' }
    }, 2_500)

    if (!response.ok) {
      throw new Error(`Ollama returned HTTP ${response.status}${await responseErrorDetail(response)}.`)
    }

    const payload = (await response.json()) as OllamaTagsResponse
    const models = (payload.models ?? [])
      .map(normalizeModelInfo)
      .filter((model): model is OllamaModelInfo => Boolean(model))
    const hasRecommendedChatModel = models.some((model) => ollamaModelMatches(model, recommendedOllamaChatModel))
    const hasRecommendedEmbeddingModel = models.some((model) => ollamaModelMatches(model, recommendedOllamaEmbeddingModel))

    return {
      baseUrl: normalizedBaseUrl,
      running: true,
      installedApp: Boolean(installedOllamaAppPath()),
      models,
      recommendedChatModel: recommendedOllamaChatModel,
      recommendedEmbeddingModel: recommendedOllamaEmbeddingModel,
      hasRecommendedChatModel,
      hasRecommendedEmbeddingModel
    }
  } catch (error) {
    return {
      baseUrl: normalizedBaseUrl,
      running: false,
      installedApp: Boolean(installedOllamaAppPath()),
      models: [],
      recommendedChatModel: recommendedOllamaChatModel,
      recommendedEmbeddingModel: recommendedOllamaEmbeddingModel,
      hasRecommendedChatModel: false,
      hasRecommendedEmbeddingModel: false,
      error: error instanceof Error ? error.message : 'Ollama is not running.'
    }
  }
}

export async function openOllamaDownloadPage(): Promise<void> {
  await shell.openExternal('https://ollama.com/download')
}

export async function openOllamaApp(): Promise<OllamaOpenResult> {
  const appPath = installedOllamaAppPath()
  if (!appPath) {
    return {
      opened: false,
      message: 'Ollama was not found in Applications.'
    }
  }

  const message = await shell.openPath(appPath)
  return {
    opened: !message,
    message: message || undefined
  }
}

export async function startOllamaService(): Promise<OllamaOpenResult> {
  const currentStatus = await getOllamaStatus()
  if (currentStatus.running) {
    return { opened: true }
  }

  const cliPath = ollamaCliCandidates()[0]
  if (!cliPath) {
    return {
      opened: false,
      message: 'The Ollama command was not found. Install Ollama, then try again.'
    }
  }

  let spawnError: Error | undefined

  try {
    const child = spawn(cliPath, ['serve'], {
      detached: true,
      stdio: 'ignore'
    })
    child.once('error', (error) => {
      spawnError = error
    })
    child.unref()
  } catch (error) {
    return {
      opened: false,
      message: error instanceof Error ? error.message : 'Could not start the Ollama service.'
    }
  }

  for (let attempt = 0; attempt < 20; attempt += 1) {
    await wait(500)
    if (spawnError) {
      return {
        opened: false,
        message: spawnError.message || 'Could not start the Ollama service.'
      }
    }

    const status = await getOllamaStatus()
    if (status.running) {
      return { opened: true }
    }
  }

  return {
    opened: false,
    message: 'Ollama started, but its local API did not respond on 127.0.0.1:11434.'
  }
}

export async function pullOllamaModel(
  model = recommendedOllamaChatModel,
  baseUrl = defaultOllamaBaseUrl
): Promise<OllamaPullResult> {
  const normalizedBaseUrl = normalizeOllamaBaseUrl(baseUrl)
  const key = ollamaPullKey(model, normalizedBaseUrl)
  if (activeOllamaPulls.has(key)) {
    throw new Error('This Ollama model is already downloading.')
  }

  const controller = new AbortController()
  const activePull: ActiveOllamaPull = {
    baseUrl: normalizedBaseUrl,
    controller,
    model
  }
  let latestStatus = 'starting'
  let latestProgress: OllamaPullProgress = {
    model,
    status: 'starting',
    percent: 0,
    message: 'Starting download'
  }

  const emit = (progress: OllamaPullProgress) => {
    latestProgress = progress
    activePull.latestProgress = progress
    emitPullProgress(progress)
  }

  emit(latestProgress)
  activeOllamaPulls.set(key, activePull)

  try {
    const response = await fetchWithTimeout(`${ollamaApiBaseUrl(normalizedBaseUrl)}/pull`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Accept: 'application/x-ndjson, application/json'
      },
      body: JSON.stringify({ model, stream: true }),
      signal: controller.signal
    }, 60 * 60_000)

    if (!response.ok) {
      throw new Error(`Ollama model download failed with HTTP ${response.status}${await responseErrorDetail(response)}.`)
    }

    if (!response.body) {
      const payload = (await response.json()) as OllamaPullResponse
      latestStatus = payload.status || 'success'
      emit({
        model,
        status: 'complete',
        percent: 100,
        message: latestStatus
      })
      return { model, status: latestStatus }
    }

    const reader = response.body.getReader()
    activePull.reader = reader
    const decoder = new TextDecoder()
    let buffer = ''
    const layerProgress = new Map<string, { completed: number; total: number }>()

    const processLine = (line: string) => {
      const trimmed = line.trim()
      if (!trimmed) {
        return
      }

      const payload = JSON.parse(trimmed) as OllamaPullStreamMessage
      if (payload.error) {
        throw new Error(payload.error)
      }

      latestStatus = payload.status || latestStatus
      let completed = latestProgress.completed
      let total = latestProgress.total

      if (typeof payload.completed === 'number' && typeof payload.total === 'number' && Number.isFinite(payload.completed) && Number.isFinite(payload.total)) {
        layerProgress.set(payload.digest || latestStatus, {
          completed: payload.completed,
          total: payload.total
        })

        completed = 0
        total = 0
        for (const progress of layerProgress.values()) {
          completed += progress.completed
          total += progress.total
        }
      }

      const percent = total ? pullProgressPercent(completed, total) : latestProgress.percent
      const isComplete = latestStatus.toLowerCase() === 'success'

      emit({
        model,
        status: isComplete ? 'complete' : 'downloading',
        percent: isComplete ? 100 : percent,
        completed,
        total,
        digest: payload.digest || latestProgress.digest,
        message: latestStatus
      })
    }

    while (true) {
      const { value, done } = await reader.read()
      if (done) {
        break
      }

      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split(/\r?\n/)
      buffer = lines.pop() ?? ''
      for (const line of lines) {
        processLine(line)
      }
    }

    buffer += decoder.decode()
    processLine(buffer)

    if (controller.signal.aborted) {
      throw new Error('Download paused.')
    }

    if (latestStatus.toLowerCase() !== 'success') {
      emit({
        ...latestProgress,
        model,
        status: 'incomplete',
        message: latestStatus === 'starting' ? 'Download did not complete' : latestStatus
      })
      throw new Error('Ollama model download did not complete.')
    }

    emit({
      ...latestProgress,
      model,
      status: 'complete',
      percent: 100,
      message: latestStatus === 'starting' ? 'Downloaded' : latestStatus
    })

    return {
      model,
      status: latestStatus
    }
  } catch (error) {
    if (controller.signal.aborted) {
      emit({
        ...latestProgress,
        model,
        status: 'incomplete',
        message: 'Download paused'
      })
      throw new Error('Download paused.')
    }

    const message = error instanceof Error ? error.message : 'Ollama model download failed.'
    emit({
      ...latestProgress,
      model,
      status: 'error',
      error: message,
      message
    })
    throw error
  } finally {
    activeOllamaPulls.delete(key)
  }
}

export async function cancelOllamaPullModel(model: string, baseUrl = defaultOllamaBaseUrl): Promise<void> {
  const activePull = activeOllamaPullFor(model, baseUrl)

  if (activePull) {
    const latestProgress = activePull.latestProgress ?? {
      model,
      status: 'starting' as const,
      percent: 0,
      message: 'Starting download'
    }
    activePull.controller.abort()
    await activePull.reader?.cancel().catch(() => undefined)
    emitPullProgress({
      ...latestProgress,
      model: latestProgress.model || model,
      status: 'incomplete',
      message: 'Download paused'
    })
    return
  }

  emitPullProgress({
    model,
    status: 'incomplete',
    percent: 0,
    message: 'Download paused'
  })
}

export async function deleteOllamaModel(
  model: string,
  baseUrl = defaultOllamaBaseUrl
): Promise<OllamaDeleteResult> {
  const normalizedModel = model.trim()
  if (!normalizedModel) {
    throw new Error('Ollama model name is required.')
  }

  await cancelOllamaPullModel(normalizedModel, baseUrl)

  const normalizedBaseUrl = normalizeOllamaBaseUrl(baseUrl)
  let response = await fetchWithTimeout(`${ollamaApiBaseUrl(normalizedBaseUrl)}/delete`, {
    method: 'DELETE',
    headers: {
      'Content-Type': 'application/json',
      Accept: 'application/json'
    },
    body: JSON.stringify({ model: normalizedModel })
  }, 30_000)

  let deleteStatus: 'removed' | 'missing' = response.status === 404 ? 'missing' : 'removed'

  if (!response.ok && (response.status === 404 || response.status === 405)) {
    response = await fetchWithTimeout(`${ollamaApiBaseUrl(normalizedBaseUrl)}/delete`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Accept: 'application/json'
      },
      body: JSON.stringify({ model: normalizedModel })
    }, 30_000)
    deleteStatus = response.status === 404 ? 'missing' : 'removed'
  }

  if (!response.ok) {
    const detail = await responseErrorDetail(response)
    if (isMissingOllamaModelError(response.status, detail)) {
      deleteStatus = 'missing'
    } else {
      try {
        deleteStatus = await deleteOllamaModelWithCli(normalizedModel)
      } catch {
        throw new Error(`Ollama model removal failed with HTTP ${response.status}${detail}.`)
      }
    }
  }

  emitPullProgress({
    model: normalizedModel,
    status: 'removed',
    percent: 0,
    message: 'Removed'
  })

  return {
    model: normalizedModel,
    status: deleteStatus
  }
}

function assertOllamaModel(model: LocalModel): asserts model is LocalModel & { ollamaModelName: string } {
  if (model.engine !== 'ollama' || !model.ollamaModelName) {
    throw new Error('Ollama model configuration is incomplete.')
  }
}

function ollamaOptions(settings?: ModelRuntimeSettings): Record<string, number> {
  if (!settings) {
    return {}
  }

  return {
    num_ctx: settings.contextLength,
    num_predict: settings.maxLength,
    temperature: settings.temperature,
    top_p: settings.topP,
    top_k: settings.topK,
    min_p: settings.minP,
    repeat_penalty: settings.repeatPenalty
  }
}

async function runOllamaChatCompletion(
  baseUrl: string,
  modelName: string,
  messages: StudyChatMessage[],
  settings?: ModelRuntimeSettings,
  overrides: { maxTokens?: number; temperature?: number } = {}
): Promise<string> {
  const response = await fetchWithTimeout(`${ollamaApiBaseUrl(baseUrl)}/chat`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Accept: 'application/json'
    },
    body: JSON.stringify({
      model: modelName,
      messages,
      options: {
        ...ollamaOptions(settings),
        ...(overrides.maxTokens ? { num_predict: overrides.maxTokens } : {}),
        ...(overrides.temperature ? { temperature: overrides.temperature } : {})
      },
      stream: false,
      think: false
    })
  }, 180_000)

  if (!response.ok) {
    throw new Error(`Ollama chat failed with HTTP ${response.status}${await responseErrorDetail(response)}.`)
  }

  const payload = (await response.json()) as OllamaChatResponse
  const text = payload.message?.content ?? ''
  if (!text.trim()) {
    throw new Error('Ollama returned an empty response.')
  }

  return text.trim()
}

async function generateOllamaFollowUpSuggestions(
  request: EngineChatRequest,
  answer: string,
  baseUrl: string,
  modelName: string
): Promise<string[]> {
  if (!shouldGenerateFollowUps(request)) {
    return []
  }

  const count = followUpSuggestionCount(request)
  if (count === 0) {
    return []
  }

  const prompt = formatFollowUpInstruction(
    request.modelSettings?.suggestedFollowUpPrompt?.trim() || defaultFollowUpPrompt(),
    count
  )
  const maxTokens = Math.min(request.modelSettings?.maxLength ?? 160, 160)
  const temperature = Math.min(Math.max(request.modelSettings?.temperature ?? 0.2, 0.2), 0.8)

  const text = await runOllamaChatCompletion(
    baseUrl,
    modelName,
    [...studyChatMessages(request), { role: 'assistant', content: answer }, { role: 'user', content: prompt }],
    request.modelSettings,
    { maxTokens, temperature }
  )
  const suggestions = parseFollowUpSuggestions(text, count)
  if (suggestions.length === 0) {
    throw new Error('Ollama did not return any suggested questions.')
  }
  return suggestions
}

export async function runOllamaStudyEngine(request: EngineChatRequest): Promise<EngineChatResponse> {
  assertOllamaModel(request.model)

  const baseUrl = request.model.ollamaBaseUrl || defaultOllamaBaseUrl
  const modelName = request.model.ollamaModelName
  const text = await runOllamaChatCompletion(baseUrl, modelName, studyChatMessages(request), request.modelSettings)
  const answer = answerWithOrderedSources(text, request.retrievedSources ?? [])
  let followUpSuggestions: string[] | undefined
  let followUpError: string | undefined
  try {
    followUpSuggestions = await generateOllamaFollowUpSuggestions(request, answer.text, baseUrl, modelName)
  } catch (error) {
    followUpError = `Suggested follow-ups failed: ${errorMessage(error, 'Ollama could not generate suggestions.')}`
  }

  return {
    engineId: 'tokensmith',
    modelName: request.model.name,
    text: answer.text,
    sources: answer.sources,
    followUpSuggestions,
    followUpError
  }
}

export async function generateOllamaStudyQuestionSuggestions(
  request: EngineQuestionSuggestionRequest
): Promise<EngineQuestionSuggestionResponse> {
  assertOllamaModel(request.model)

  const count = questionSuggestionCount(request.applicationSettings)
  if (count === 0) {
    return { suggestions: [] }
  }

  const baseUrl = request.model.ollamaBaseUrl || defaultOllamaBaseUrl
  const modelName = request.model.ollamaModelName
  const maxTokens = Math.min(request.modelSettings?.maxLength ?? 160, 160)
  const temperature = Math.min(Math.max(request.modelSettings?.temperature ?? 0.2, 0.2), 0.8)

  const text = await runOllamaChatCompletion(
    baseUrl,
    modelName,
    questionSuggestionMessages(request),
    request.modelSettings,
    { maxTokens, temperature }
  )
  const suggestions = parseFollowUpSuggestions(text, count)
  if (suggestions.length === 0) {
    throw new Error('Ollama did not return any suggested questions.')
  }
  return { suggestions }
}
