import type { LocalModel, LocalModelRole, ModelRuntimeSettings } from '../../shared/app-state'
import type {
  EngineChatRequest,
  EngineChatResponse,
  EngineQuestionSuggestionRequest,
  EngineQuestionSuggestionResponse
} from '../../shared/engine'
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

interface OpenAiCompatibleModelList {
  data?: Array<{ id?: string }>
}

interface GeminiModelList {
  models?: Array<{
    name?: string
    supportedGenerationMethods?: string[]
  }>
}

interface OpenAiCompatibleChatResponse {
  choices?: Array<{
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

function geminiNativeBaseUrl(baseUrl: string): string {
  const url = new URL(normalizeBaseUrl(baseUrl))
  url.pathname = url.pathname.replace(/\/openai$/, '')
  return url.toString().replace(/\/+$/, '')
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

export async function listOpenAiCompatibleModels(
  apiKey: string,
  baseUrl: string,
  role: LocalModelRole = 'generator'
): Promise<string[]> {
  const normalizedBaseUrl = normalizeBaseUrl(baseUrl)
  if (!apiKey.trim() || !normalizedBaseUrl) {
    return []
  }

  if (isGeminiOpenAiBaseUrl(normalizedBaseUrl)) {
    const nativeBaseUrl = geminiNativeBaseUrl(normalizedBaseUrl)
    const response = await fetch(`${nativeBaseUrl}/models`, {
      headers: {
        'x-goog-api-key': apiKey.trim(),
        Accept: 'application/json'
      }
    })

    if (!response.ok) {
      throw new Error(`Model list failed with HTTP ${response.status}.`)
    }

    const payload = (await response.json()) as GeminiModelList
    const requiredMethod = role === 'embedder' ? 'embedContent' : 'generateContent'
    const candidates = Array.from(
      new Set(
        (payload.models ?? [])
          .filter((model) => model.supportedGenerationMethods?.includes(requiredMethod))
          .map((model) => model.name)
          .filter((id): id is string => Boolean(id))
          .map((id) => normalizeListedModelId(id, normalizedBaseUrl))
          .filter((id) => Boolean(id))
      )
    ).sort((left, right) => compareModelIds(left, right, 'gemini-'))

    return candidates
  }

  const response = await fetch(`${normalizedBaseUrl}/models`, {
    headers: {
      Authorization: `Bearer ${apiKey.trim()}`,
      Accept: 'application/json'
    }
  })

  if (!response.ok) {
    throw new Error(`Model list failed with HTTP ${response.status}.`)
  }

  const payload = (await response.json()) as OpenAiCompatibleModelList
  const modelIds = (payload.data ?? [])
    .map((model) => model.id)
    .filter((id): id is string => Boolean(id))
    .map((id) => normalizeListedModelId(id, normalizedBaseUrl))
    .filter((id) => Boolean(id))

  return Array.from(new Set(modelIds))
    .sort(compareModelIds)
}

async function runRemoteChatCompletion(
  config: RemoteCompletionConfig,
  messages: StudyChatMessage[],
  overrides: { maxTokens?: number; temperature?: number } = {}
): Promise<string> {
  const response = await fetch(config.endpoint, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${config.apiKey.trim()}`,
      'Content-Type': 'application/json',
      Accept: 'application/json'
    },
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
  const text = payload.choices?.[0]?.message?.content ?? payload.choices?.[0]?.text ?? ''

  if (!text.trim()) {
    throw new Error('Remote model returned an empty response.')
  }

  return text.trim()
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
  const prompt = formatFollowUpInstruction(
    request.modelSettings?.suggestedFollowUpPrompt?.trim() || defaultFollowUpPrompt(),
    count
  )
  const maxTokens = Math.min(config.settings?.maxLength ?? 160, 160)
  const temperature = Math.min(Math.max(config.settings?.temperature ?? 0.2, 0.2), 0.8)

  const text = await runRemoteChatCompletion(
    config,
    [...studyChatMessages(request), { role: 'assistant', content: answer }, { role: 'user', content: prompt }],
    { maxTokens, temperature }
  )
  const suggestions = parseFollowUpSuggestions(text, count)
  if (suggestions.length === 0) {
    throw new Error('The remote model did not return any suggested questions.')
  }
  return suggestions
}

export async function runRemoteStudyEngine(request: EngineChatRequest): Promise<EngineChatResponse> {
  assertRemoteModel(request.model)

  const settings = request.modelSettings
  const endpoint = `${normalizeBaseUrl(request.model.baseUrl)}/chat/completions`
  const modelName = normalizeListedModelId(request.model.remoteModelName, request.model.baseUrl)
  const config = {
    endpoint,
    modelName,
    apiKey: request.model.apiKey,
    settings
  }
  const text = await runRemoteChatCompletion(config, studyChatMessages(request))
  const answer = answerWithOrderedSources(text, request.retrievedSources ?? [])
  let followUpSuggestions: string[] | undefined
  let followUpError: string | undefined
  try {
    followUpSuggestions = await generateRemoteFollowUpSuggestions(request, answer.text, config)
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

  const config = {
    endpoint: `${normalizeBaseUrl(request.model.baseUrl)}/chat/completions`,
    modelName: normalizeListedModelId(request.model.remoteModelName, request.model.baseUrl),
    apiKey: request.model.apiKey,
    settings: request.modelSettings
  }
  const maxTokens = Math.min(config.settings?.maxLength ?? 160, 160)
  const temperature = Math.min(Math.max(config.settings?.temperature ?? 0.2, 0.2), 0.8)

  const text = await runRemoteChatCompletion(config, questionSuggestionMessages(request), { maxTokens, temperature })
  const suggestions = parseFollowUpSuggestions(text, count)
  if (suggestions.length === 0) {
    throw new Error('The remote model did not return any suggested questions.')
  }
  return { suggestions }
}
