import { app, BrowserWindow, shell } from 'electron'
import { spawn } from 'node:child_process'
import { existsSync } from 'node:fs'
import { join } from 'node:path'
import type { LocalModel, ModelRuntimeSettings } from '../../shared/app-state'
import type { EngineChatRequest, EngineChatResponse } from '../../shared/engine'
import {
  defaultOllamaBaseUrl,
  recommendedOllamaChatModel,
  recommendedOllamaEmbeddingModel,
  type OllamaModelInfo,
  type OllamaOpenResult,
  type OllamaPullProgress,
  type OllamaPullResult,
  type OllamaStatus
} from '../../shared/ollama'
import {
  defaultFollowUpPrompt,
  followUpSuggestionCount,
  formatFollowUpInstruction,
  parseFollowUpSuggestions,
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
  const timeout = setTimeout(() => controller.abort(), timeoutMs)

  try {
    return await fetch(url, {
      ...init,
      signal: controller.signal
    })
  } finally {
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
  let latestStatus = 'starting'
  let latestProgress: OllamaPullProgress = {
    model,
    status: 'starting',
    percent: 0,
    message: 'Starting download'
  }

  const emit = (progress: OllamaPullProgress) => {
    latestProgress = progress
    emitPullProgress(progress)
  }

  emit(latestProgress)

  try {
    const response = await fetchWithTimeout(`${ollamaApiBaseUrl(normalizedBaseUrl)}/pull`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Accept: 'application/x-ndjson, application/json'
      },
      body: JSON.stringify({ model, stream: true })
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
    const decoder = new TextDecoder()
    let buffer = ''

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
      const completed = Number.isFinite(payload.completed) ? payload.completed : latestProgress.completed
      const total = Number.isFinite(payload.total) ? payload.total : latestProgress.total
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
    const message = error instanceof Error ? error.message : 'Ollama model download failed.'
    emit({
      ...latestProgress,
      model,
      status: 'error',
      error: message,
      message
    })
    throw error
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

  try {
    const text = await runOllamaChatCompletion(
      baseUrl,
      modelName,
      [...studyChatMessages(request), { role: 'assistant', content: answer }, { role: 'user', content: prompt }],
      request.modelSettings,
      { maxTokens, temperature }
    )
    return parseFollowUpSuggestions(text, count)
  } catch {
    return []
  }
}

export async function runOllamaStudyEngine(request: EngineChatRequest): Promise<EngineChatResponse> {
  assertOllamaModel(request.model)

  const baseUrl = request.model.ollamaBaseUrl || defaultOllamaBaseUrl
  const modelName = request.model.ollamaModelName
  const text = await runOllamaChatCompletion(baseUrl, modelName, studyChatMessages(request), request.modelSettings)
  const followUpSuggestions = await generateOllamaFollowUpSuggestions(request, text, baseUrl, modelName)

  return {
    engineId: 'tokensmith',
    modelName: request.model.name,
    text,
    sources: request.retrievedSources ?? [],
    followUpSuggestions
  }
}
