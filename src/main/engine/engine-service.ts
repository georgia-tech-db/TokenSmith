import {
  getPythonEngineHealth,
  starterSourcesWithPython,
  writeTokenSmithLog
} from '../python/python-engine-service'
import type {
  EngineChatRequest,
  EngineChatResponse,
  EngineInfo,
  EngineQuestionSuggestionRequest,
  EngineQuestionSuggestionResponse
} from '../../shared/engine'
import type { EngineQuestionRewriteRequest, QuestionRewrite } from '../../shared/engine'
import { modelWithRememberedRemoteApiKey } from './remote-model-secrets'
import { resolveRemoteChatQuestion } from './remote-chat-service'
import type { ChatSource } from '../../shared/app-state'
import { generateStudyQuestionSuggestions, listStudyEngines, sendStudyChatMessage } from './study-engine-core'
import { generateOllamaStudyQuestionSuggestions, resolveOllamaChatQuestion, runOllamaStudyEngine } from './ollama-service'
import { questionSuggestionMessages, sourceContextBudgetForRequest, studyChatMessages } from './study-chat-format'

async function withStarterSources(
  request: EngineQuestionSuggestionRequest
): Promise<EngineQuestionSuggestionRequest> {
  if (request.messages.length > 0 || (request.retrievedSources?.length ?? 0) > 0) {
    return request
  }

  const starterSources = await starterSourcesWithPython(request.materials, 4)
  if (starterSources.length === 0) {
    throw new Error('No indexed PDF text was available for starter questions.')
  }

  return { ...request, retrievedSources: starterSources }
}

export async function listEngines(): Promise<EngineInfo[]> {
  return listStudyEngines({
    getPythonEngineHealth,
    generateOllamaStudyQuestionSuggestions,
    runOllamaStudyEngine
  })
}

export async function sendChatMessage(request: EngineChatRequest): Promise<EngineChatResponse> {
  writeTokenSmithLog('chat_request_context', chatRequestLogDetails(request))

  const response = await sendStudyChatMessage(request, {
    getPythonEngineHealth,
    generateOllamaStudyQuestionSuggestions,
    runOllamaStudyEngine
  })

  writeTokenSmithLog('chat_response_context', {
    modelName: response.modelName,
    text: response.text,
    sourceCount: response.sources.length,
    sources: response.sources.map(logSource)
  })

  return response
}

export async function resolveChatQuestion(request: EngineQuestionRewriteRequest): Promise<QuestionRewrite> {
  if (request.model.engine === 'ollama') return resolveOllamaChatQuestion(request)
  if (request.model.engine === 'remote') {
    return resolveRemoteChatQuestion({ ...request, model: modelWithRememberedRemoteApiKey(request.model) })
  }
  throw new Error('Question rewriting requires an Ollama or remote chat model.')
}

export async function suggestChatQuestions(
  request: EngineQuestionSuggestionRequest
): Promise<EngineQuestionSuggestionResponse> {
  const suggestionRequest = await withStarterSources(request)
  writeTokenSmithLog('question_suggestion_request_context', questionSuggestionLogDetails(suggestionRequest))
  return generateStudyQuestionSuggestions(suggestionRequest, {
    getPythonEngineHealth,
    generateOllamaStudyQuestionSuggestions,
    runOllamaStudyEngine
  })
}

function logSource(source: ChatSource): Record<string, unknown> {
  return {
    title: source.title,
    locator: source.locator,
    sourceId: source.sourceId,
    chunkId: source.chunkId,
    chunkRowid: source.chunkRowid,
    chunkKind: source.chunkKind,
    tokensmithChunkId: source.tokensmithChunkId,
    tokensmithChapter: source.tokensmithChapter,
    tokensmithChunkKind: source.tokensmithChunkKind,
    queryTerms: source.queryTerms,
    keywordTerms: source.keywordTerms,
    documentTitle: source.documentTitle,
    collectionName: source.collectionName,
    sectionHeader: source.sectionHeader,
    path: source.path,
    pageStart: source.pageStart,
    pageEnd: source.pageEnd,
    score: source.score,
    retrievalMode: source.retrievalMode,
    embeddingModel: source.embeddingModel,
    chunkEmbeddingModel: source.chunkEmbeddingModel,
    excerpt: source.excerpt
  }
}

function logModel(request: EngineChatRequest): Record<string, unknown> {
  return {
    id: request.model.id,
    name: request.model.name,
    engine: request.model.engine,
    source: request.model.source,
    role: request.model.role,
    status: request.model.status,
    ollamaModelName: request.model.ollamaModelName,
    remoteModelName: request.model.remoteModelName,
    providerId: request.model.providerId
  }
}

function chatRequestLogDetails(request: EngineChatRequest): Record<string, unknown> {
  const messages = studyChatMessages(request)
  const systemPrompt = request.modelSettings?.systemMessage?.trim() ?? ''

  return {
    prompt: request.prompt,
    answerPrompt: request.answerPrompt,
    retrievalQuery: request.retrievalQuery,
    conversationContextMode: request.conversationContextMode ?? 'standalone',
    referenceExchange: request.referenceExchange,
    model: logModel(request),
    contextBudget: sourceContextBudgetForRequest(request),
    systemPrompt,
    sourceCount: request.retrievedSources?.length ?? 0,
    sources: (request.retrievedSources ?? []).map(logSource),
    modelMessages: messages.map((message) => ({
      role: message.role,
      content: message.content
    }))
  }
}

function questionSuggestionLogDetails(request: EngineQuestionSuggestionRequest): Record<string, unknown> {
  const messages = questionSuggestionMessages(request)
  const systemPrompt = request.modelSettings?.systemMessage?.trim() ?? ''

  return {
    model: {
      id: request.model.id,
      name: request.model.name,
      engine: request.model.engine,
      source: request.model.source,
      role: request.model.role,
      status: request.model.status,
      ollamaModelName: request.model.ollamaModelName,
      remoteModelName: request.model.remoteModelName,
      providerId: request.model.providerId
    },
    systemPrompt,
    contextBudget: sourceContextBudgetForRequest(request),
    sourceCount: request.retrievedSources?.length ?? 0,
    sources: (request.retrievedSources ?? []).map(logSource),
    modelMessages: messages.map((message) => ({
      role: message.role,
      content: message.content
    }))
  }
}
