import type { LocalModel, LocalModelRole, ModelRuntimeSettings } from '../../shared/app-state'
import { writeTokenSmithLog } from '../python/python-engine-service'
import type {
  EngineRunOptions,
  EngineChatRequest,
  EngineChatResponse,
  EngineQuestionSuggestionRequest,
  EngineQuestionSuggestionResponse
} from '../../shared/engine'
import type { EngineQuestionRewriteRequest, QuestionRewrite } from '../../shared/engine'
import { lastChatExchange } from '../../shared/study-chat-pipeline'
import { createLatencyTrace, startLatencySpan, type LatencySpan } from '../../shared/latency-trace'
import { remoteChatParameters } from './remote-chat-parameters'
import { remoteGeneratorFetch } from './remote-generator-network'
import { parseQuestionRewrite, questionRewriteMessages } from './question-rewrite'
import {
  answerWithOrderedSources,
  followUpSuggestionMessages,
  followUpSuggestionCount,
  modelAwareRuntimeSettings,
  parseFollowUpSuggestions,
  questionSuggestionCount,
  questionSuggestionMessages,
  suggestionMaxTokens,
  shouldGenerateFollowUps,
  prepareStudyChatMessages,
  type StudyChatMessage,
  filterSuggestedQuestions
} from './study-chat-format'
import { promptPreparationSpan } from './study-engine-latency'

interface OpenAiCompatibleModelList {
  data?: Array<{ id?: string }>
}

interface OpenAiCompatibleChatResponse {
  choices?: Array<{
    finish_reason?: string
    message?: {
      content?: string
    }
    text?: string
  }>
  usage?: {
    prompt_tokens?: number
    completion_tokens?: number
  }
}

interface RemoteCompletionOverrides {
  maxTokens?: number
  temperature?: number
  requireComplete?: boolean
  signal?: AbortSignal
  onComplete?: (payload: OpenAiCompatibleChatResponse) => void
}

interface RemoteCompletionConfig {
  endpoint: string
  modelName: string
  apiKey: string
  settings?: ModelRuntimeSettings
}

function normalizeBaseUrl(baseUrl: string): string {
  return baseUrl.trim().replace(/\/+$/, '')
}

function errorMessage(error: unknown, fallback: string): string {
  return error instanceof Error ? error.message : fallback
}

function remoteCompletionDetails(payload?: OpenAiCompatibleChatResponse): Record<string, string | number> | undefined {
  if (!payload) return undefined
  const details: Record<string, string | number> = {}
  const promptTokens = payload.usage?.prompt_tokens
  const generatedTokens = payload.usage?.completion_tokens
  const finishReason = payload.choices?.[0]?.finish_reason
  if (typeof promptTokens === 'number' && Number.isFinite(promptTokens) && promptTokens >= 0) {
    details.prompt_tokens = promptTokens
  }
  if (typeof generatedTokens === 'number' && Number.isFinite(generatedTokens) && generatedTokens >= 0) {
    details.generated_tokens = generatedTokens
  }
  if (finishReason) details.finish_reason = finishReason
  return Object.keys(details).length ? details : undefined
}

function isGeminiOpenAiBaseUrl(baseUrl: string): boolean {
  try {
    return new URL(baseUrl).hostname === 'generativelanguage.googleapis.com'
  } catch {
    return false
  }
}

function normalizeListedModelId(modelId: string, baseUrl: string): string {
  const trimmedModelId = modelId.trim()

  if (isGeminiOpenAiBaseUrl(baseUrl) && trimmedModelId.startsWith('models/')) {
    return trimmedModelId.slice('models/'.length)
  }

  return trimmedModelId
}

function assertRemoteModel(model: LocalModel): asserts model is LocalModel & {
  apiKey: string
  baseUrl: string
  remoteModelName: string
} {
  if (model.engine !== 'remote' || !model.apiKey || !model.baseUrl || !model.remoteModelName) {
    throw new Error('Remote model configuration is incomplete.')
  }
}

function redactSecret(text: string, secret?: string): string {
  const cleanedSecret = secret?.trim()
  return cleanedSecret ? text.split(cleanedSecret).join('[redacted]') : text
}

async function responseErrorDetail(response: Response, secret?: string): Promise<string> {
  try {
    const text = await response.text()
    const trimmed = text.trim()
    return trimmed ? `: ${redactSecret(trimmed.slice(0, 500), secret)}` : ''
  } catch {
    return ''
  }
}

function numericGroups(value: string): number[] {
  return [...value.matchAll(/\d+(?:\.\d+)*/g)].flatMap((match) => match[0].split('.').map(Number))
}

function compareNumberListsDescending(left: number[], right: number[]): number {
  const length = Math.max(left.length, right.length)

  for (let index = 0; index < length; index += 1) {
    const leftValue = left[index] ?? -1
    const rightValue = right[index] ?? -1

    if (leftValue !== rightValue) {
      return rightValue - leftValue
    }
  }

  return 0
}

function compareModelIds(left: string, right: string, preferredPrefix?: string): number {
  if (preferredPrefix) {
    const leftPreferred = left.startsWith(preferredPrefix)
    const rightPreferred = right.startsWith(preferredPrefix)

    if (leftPreferred !== rightPreferred) {
      return leftPreferred ? -1 : 1
    }
  }

  return compareNumberListsDescending(numericGroups(left), numericGroups(right)) || left.localeCompare(right)
}

function isLikelyEmbeddingModelId(modelId: string): boolean {
  const lower = modelId.toLowerCase()
  return lower.includes('embedding') || lower.includes('embed')
}

function listedModelMatchesRole(modelId: string, baseUrl: string, role: LocalModelRole): boolean {
  if (!isGeminiOpenAiBaseUrl(baseUrl)) {
    return true
  }

  if (role === 'both') {
    return true
  }

  const embeddingModel = isLikelyEmbeddingModelId(modelId)
  return role === 'embedder' ? embeddingModel : !embeddingModel
}

export async function listOpenAiCompatibleModels(
  apiKey: string,
  baseUrl: string,
  role: LocalModelRole = 'generator'
): Promise<string[]> {
  const normalizedBaseUrl = normalizeBaseUrl(baseUrl)
  if (!apiKey.trim() || !normalizedBaseUrl) {
    return []
  }

  const response = await fetch(`${normalizedBaseUrl}/models`, {
    headers: {
      Authorization: `Bearer ${apiKey.trim()}`,
      Accept: 'application/json'
    }
  })

  if (!response.ok) {
    throw new Error(`Model list failed with HTTP ${response.status}${await responseErrorDetail(response, apiKey)}.`)
  }

  const payload = (await response.json()) as OpenAiCompatibleModelList
  const modelIds = (payload.data ?? [])
    .map((model) => model.id)
    .filter((id): id is string => Boolean(id))
    .map((id) => normalizeListedModelId(id, normalizedBaseUrl))
    .filter((id) => Boolean(id))
    .filter((id) => listedModelMatchesRole(id, normalizedBaseUrl, role))

  return Array.from(new Set(modelIds))
    .sort((left, right) => compareModelIds(left, right, isGeminiOpenAiBaseUrl(normalizedBaseUrl) ? 'gemini-' : undefined))
}

async function runRemoteChatCompletion(
  config: RemoteCompletionConfig,
  messages: StudyChatMessage[],
  overrides: RemoteCompletionOverrides = {}
): Promise<string> {
  const response = await remoteGeneratorFetch(config.endpoint, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${config.apiKey.trim()}`,
      'Content-Type': 'application/json',
      Accept: 'application/json'
    },
    signal: overrides.signal
      ? AbortSignal.any([overrides.signal, AbortSignal.timeout(180_000)])
      : AbortSignal.timeout(180_000),
    body: JSON.stringify({
      model: config.modelName,
      messages,
      ...remoteChatParameters(config.endpoint, config.modelName,
        overrides.maxTokens ?? config.settings?.maxLength,
        overrides.temperature ?? config.settings?.temperature, config.settings?.topP)
    })
  })

  if (!response.ok) {
    throw new Error(
      `Remote model request failed with HTTP ${response.status} at POST ${config.endpoint} using model ${config.modelName}${await responseErrorDetail(response, config.apiKey)}.`
    )
  }

  const payload = (await response.json()) as OpenAiCompatibleChatResponse
  overrides.onComplete?.(payload)
  if (overrides.requireComplete && payload.choices?.[0]?.finish_reason === 'length') {
    throw new Error('The model response exceeded its output limit.')
  }
  const text = payload.choices?.[0]?.message?.content ?? payload.choices?.[0]?.text ?? ''

  if (!text.trim()) {
    throw new Error('Remote model returned an empty response.')
  }

  return text.trim()
}

export async function resolveRemoteChatQuestion(request: EngineQuestionRewriteRequest, signal?: AbortSignal): Promise<QuestionRewrite> {
  assertRemoteModel(request.model)
  if (!request.selectedPassage && !lastChatExchange(request.messages)) return { mode: 'standalone', query: request.prompt, clarification: '', reasoning: false }
  const started = performance.now()
  const settings = modelAwareRuntimeSettings(request) ?? request.modelSettings
  const messages = questionRewriteMessages({ ...request, modelSettings: settings })
  const modelName = normalizeListedModelId(request.model.remoteModelName, request.model.baseUrl)
  const text = await runRemoteChatCompletion({
    endpoint: `${normalizeBaseUrl(request.model.baseUrl)}/chat/completions`,
    modelName, apiKey: request.model.apiKey, settings
  }, messages, { maxTokens: 512, temperature: 0, requireComplete: true, signal })
  const resolution = parseQuestionRewrite(text, request.prompt, Boolean(request.selectedPassage))
  writeTokenSmithLog('chat_question_rewrite', {
    modelName, prompt: request.prompt, modelMessages: messages,
    rawResponse: text, resolution, query: resolution.query, conversationContextMode: resolution.mode,
    durationMs: performance.now() - started
  })
  return resolution
}

async function generateRemoteFollowUpSuggestions(
  request: EngineChatRequest,
  answer: string,
  config: RemoteCompletionConfig,
  signal?: AbortSignal,
  onComplete?: (payload: OpenAiCompatibleChatResponse) => void
): Promise<string[]> {
  if (!shouldGenerateFollowUps(request)) {
    return []
  }

  const count = followUpSuggestionCount(request)
  if (count === 0) {
    return []
  }
  const maxTokens = suggestionMaxTokens
  const temperature = Math.min(Math.max(config.settings?.temperature ?? 0.2, 0.2), 0.8)

  const text = await runRemoteChatCompletion(
    config,
    followUpSuggestionMessages(request, answer),
    { maxTokens, temperature, requireComplete: true, signal, onComplete }
  )
  const referenceQuestions = [
    ...request.messages.filter((message) => message.role === 'user').map((message) => message.text),
    request.prompt
  ].filter((question) => question.trim().length > 0)
  const suggestions = filterSuggestedQuestions(
    parseFollowUpSuggestions(text, count * 2),
    referenceQuestions,
    count
  )
  writeTokenSmithLog('follow_up_suggestions', {
    provider: 'remote', modelName: config.modelName, prompt: request.prompt,
    rawResponse: text, suggestions, requestedCount: count
  })
  return suggestions
}

export async function runRemoteStudyEngine(request: EngineChatRequest, options: EngineRunOptions = {}): Promise<EngineChatResponse> {
  assertRemoteModel(request.model)

  const settings = modelAwareRuntimeSettings(request) ?? request.modelSettings
  const runtimeRequest = settings ? { ...request, modelSettings: settings } : request
  const endpoint = `${normalizeBaseUrl(request.model.baseUrl)}/chat/completions`
  const modelName = normalizeListedModelId(request.model.remoteModelName, request.model.baseUrl)
  const config = {
    endpoint,
    modelName,
    apiKey: request.model.apiKey,
    settings
  }
  if (request.practiceTask) {
    const prepared = prepareStudyChatMessages(runtimeRequest)
    const text = await runRemoteChatCompletion(config, prepared.messages, {
      temperature: request.practiceTask === 'feedback' ? 0 : 0.3, requireComplete: true, signal: options.signal
    })
    return { engineId: 'tokensmith', modelName: request.model.name, text, sources: prepared.sources }
  }
  const preparationStartedAt = Date.now()
  const preparationStarted = performance.now()
  const prepared = prepareStudyChatMessages(runtimeRequest)
  const preparationSpan = promptPreparationSpan(
    preparationStartedAt,
    performance.now() - preparationStarted,
    runtimeRequest.retrievedSources?.length ?? 0,
    prepared.budget
  )
  let generationPayload: OpenAiCompatibleChatResponse | undefined
  const generationTimer = startLatencySpan('Generation')
  const text = await runRemoteChatCompletion(config, prepared.messages, {
    signal: options.signal,
    onComplete: (payload) => { generationPayload = payload }
  })
  const generationSpan = generationTimer.finish({
    details: remoteCompletionDetails(generationPayload)
  })
  const answerTrace = createLatencyTrace([preparationSpan, generationSpan], request.requestId)
  const answer = answerWithOrderedSources(text, request.retrievedSources ?? [])
  options.signal?.throwIfAborted()
  const hasFollowUps = shouldGenerateFollowUps(runtimeRequest) && followUpSuggestionCount(runtimeRequest) > 0
  options.onAnswer?.({
    engineId: 'tokensmith',
    modelName: request.model.name,
    text: answer.text,
    sources: answer.sources,
    latencyTrace: answerTrace
  }, hasFollowUps)
  let followUpSuggestions: string[] | undefined
  let followUpError: string | undefined
  let suggestionPayload: OpenAiCompatibleChatResponse | undefined
  const suggestionTimer = startLatencySpan('Suggestions')
  let suggestionSpan: LatencySpan
  try {
    followUpSuggestions = hasFollowUps
      ? await generateRemoteFollowUpSuggestions(
        runtimeRequest,
        answer.text,
        config,
        options.signal,
        (payload) => { suggestionPayload = payload }
      )
      : []
    suggestionSpan = suggestionTimer.finish({
      status: hasFollowUps ? 'ok' : 'skipped',
      outCount: followUpSuggestions.length,
      details: remoteCompletionDetails(suggestionPayload)
    })
  } catch (error) {
    options.signal?.throwIfAborted()
    followUpError = `Suggested follow-ups failed: ${errorMessage(error, 'The remote provider could not generate suggestions.')}`
    suggestionSpan = suggestionTimer.finish({
      status: 'error',
      details: { reason: 'follow_up_generation_failed' }
    })
  }

  return {
    engineId: 'tokensmith',
    modelName: request.model.name,
    text: answer.text,
    sources: answer.sources,
    latencyTrace: createLatencyTrace(
      [preparationSpan, generationSpan, suggestionSpan],
      request.requestId
    ),
    followUpSuggestions,
    followUpError
  }
}

export async function generateRemoteStudyQuestionSuggestions(
  request: EngineQuestionSuggestionRequest,
  signal?: AbortSignal
): Promise<EngineQuestionSuggestionResponse> {
  assertRemoteModel(request.model)

  const count = questionSuggestionCount(request.applicationSettings)
  if (count === 0) {
    return { suggestions: [] }
  }

  const settings = modelAwareRuntimeSettings(request) ?? request.modelSettings
  const runtimeRequest = settings ? { ...request, modelSettings: settings } : request
  const config = {
    endpoint: `${normalizeBaseUrl(request.model.baseUrl)}/chat/completions`,
    modelName: normalizeListedModelId(request.model.remoteModelName, request.model.baseUrl),
    apiKey: request.model.apiKey,
    settings
  }
  const maxTokens = suggestionMaxTokens
  const temperature = Math.min(Math.max(config.settings?.temperature ?? 0.2, 0.2), 0.8)

  const text = await runRemoteChatCompletion(config, questionSuggestionMessages(runtimeRequest), {
    maxTokens, temperature, requireComplete: true, signal
  })
  const referenceQuestions = runtimeRequest.messages
    .filter((message) => message.role === 'user')
    .map((message) => message.text)
  const suggestions = filterSuggestedQuestions(
    parseFollowUpSuggestions(text, count * 2),
    referenceQuestions,
    count
  )
  writeTokenSmithLog('initial_question_suggestions', {
    provider: 'remote', modelName: config.modelName, rawResponse: text, suggestions, requestedCount: count
  })
  return { suggestions }
}
