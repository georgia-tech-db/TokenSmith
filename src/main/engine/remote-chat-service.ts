import type { LocalModel, LocalModelRole, ModelRuntimeSettings } from '../../shared/app-state'
import { writeTokenSmithLog } from '../python/python-engine-service'
import type {
  EngineChatRequest,
  EngineChatResponse,
  EngineQuestionSuggestionRequest,
  EngineQuestionSuggestionResponse
} from '../../shared/engine'
import type { EngineQuestionRewriteRequest, QuestionRewrite } from '../../shared/engine'
import { lastChatExchange } from '../../shared/study-chat-pipeline'
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
  studyChatMessages,
  type StudyChatMessage,
  filterSuggestedQuestions
} from './study-chat-format'

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
  overrides: { maxTokens?: number; temperature?: number; requireComplete?: boolean } = {}
): Promise<string> {
  const response = await fetch(config.endpoint, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${config.apiKey.trim()}`,
      'Content-Type': 'application/json',
      Accept: 'application/json'
    },
    signal: AbortSignal.timeout(180_000),
    body: JSON.stringify({
      model: config.modelName,
      messages,
      max_tokens: overrides.maxTokens ?? config.settings?.maxLength,
      temperature: overrides.temperature ?? config.settings?.temperature,
      top_p: config.settings?.topP
    })
  })

  if (!response.ok) {
    throw new Error(
      `Remote model request failed with HTTP ${response.status} at POST ${config.endpoint} using model ${config.modelName}${await responseErrorDetail(response, config.apiKey)}.`
    )
  }

  const payload = (await response.json()) as OpenAiCompatibleChatResponse
  if (overrides.requireComplete && payload.choices?.[0]?.finish_reason === 'length') {
    throw new Error('The model response exceeded its output limit.')
  }
  const text = payload.choices?.[0]?.message?.content ?? payload.choices?.[0]?.text ?? ''

  if (!text.trim()) {
    throw new Error('Remote model returned an empty response.')
  }

  return text.trim()
}

export async function resolveRemoteChatQuestion(request: EngineQuestionRewriteRequest): Promise<QuestionRewrite> {
  assertRemoteModel(request.model)
  if (!lastChatExchange(request.messages)) return { mode: 'standalone', query: request.prompt, clarification: '' }
  const started = performance.now()
  const settings = modelAwareRuntimeSettings(request) ?? request.modelSettings
  const messages = questionRewriteMessages({ ...request, modelSettings: settings })
  const modelName = normalizeListedModelId(request.model.remoteModelName, request.model.baseUrl)
  const text = await runRemoteChatCompletion({
    endpoint: `${normalizeBaseUrl(request.model.baseUrl)}/chat/completions`,
    modelName, apiKey: request.model.apiKey, settings
  }, messages, { maxTokens: 512, temperature: 0, requireComplete: true })
  const resolution = parseQuestionRewrite(text, request.prompt)
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
  config: RemoteCompletionConfig
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
    { maxTokens, temperature, requireComplete: true }
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

export async function runRemoteStudyEngine(request: EngineChatRequest): Promise<EngineChatResponse> {
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
  const text = await runRemoteChatCompletion(config, studyChatMessages(runtimeRequest))
  const answer = answerWithOrderedSources(text, request.retrievedSources ?? [])
  let followUpSuggestions: string[] | undefined
  let followUpError: string | undefined
  try {
    followUpSuggestions = await generateRemoteFollowUpSuggestions(runtimeRequest, answer.text, config)
  } catch (error) {
    followUpError = `Suggested follow-ups failed: ${errorMessage(error, 'The remote provider could not generate suggestions.')}`
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

export async function generateRemoteStudyQuestionSuggestions(
  request: EngineQuestionSuggestionRequest
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

  const text = await runRemoteChatCompletion(config, questionSuggestionMessages(runtimeRequest), { maxTokens, temperature, requireComplete: true })
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
