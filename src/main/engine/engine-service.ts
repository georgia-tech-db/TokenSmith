import {
  getPythonEngineHealth,
  runPythonStudyEngine,
  starterSourcesWithPython
} from '../python/python-engine-service'
import type {
  EngineChatRequest,
  EngineChatResponse,
  EngineInfo,
  EngineQuestionSuggestionRequest,
  EngineQuestionSuggestionResponse
} from '../../shared/engine'
import { generateStudyQuestionSuggestions, listStudyEngines, sendStudyChatMessage } from './study-engine-core'
import { generateOllamaStudyQuestionSuggestions, runOllamaStudyEngine } from './ollama-service'

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
    runOllamaStudyEngine,
    runPythonStudyEngine
  })
}

export async function sendChatMessage(request: EngineChatRequest): Promise<EngineChatResponse> {
  return sendStudyChatMessage(request, {
    getPythonEngineHealth,
    generateOllamaStudyQuestionSuggestions,
    runOllamaStudyEngine,
    runPythonStudyEngine
  })
}

export async function suggestChatQuestions(
  request: EngineQuestionSuggestionRequest
): Promise<EngineQuestionSuggestionResponse> {
  const suggestionRequest = await withStarterSources(request)
  return generateStudyQuestionSuggestions(suggestionRequest, {
    getPythonEngineHealth,
    generateOllamaStudyQuestionSuggestions,
    runOllamaStudyEngine,
    runPythonStudyEngine
  })
}
