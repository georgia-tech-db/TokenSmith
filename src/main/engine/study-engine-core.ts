import type {
  EngineChatRequest,
  EngineChatResponse,
  EngineInfo,
  EngineQuestionSuggestionRequest,
  EngineQuestionSuggestionResponse
} from '../../shared/engine'
import { generateRemoteStudyQuestionSuggestions, runRemoteStudyEngine } from './remote-chat-service'
import {
  modelWithRememberedRemoteApiKey,
  rememberRemoteModelApiKey
} from './remote-model-secrets'

export interface PythonEngineHealth {
  ok?: boolean
}

export interface StudyEngineDependencies {
  getPythonEngineHealth: () => Promise<PythonEngineHealth>
  generateOllamaStudyQuestionSuggestions: (request: EngineQuestionSuggestionRequest) => Promise<EngineQuestionSuggestionResponse>
  runOllamaStudyEngine: (request: EngineChatRequest) => Promise<EngineChatResponse>
}

export async function listStudyEngines(dependencies: StudyEngineDependencies): Promise<EngineInfo[]> {
  try {
    await dependencies.getPythonEngineHealth()
    return [
      {
        id: 'tokensmith',
        name: 'TokenSmith',
        status: 'ready',
        detail: 'Local indexing and vector retrieval are available. Ollama chat and embedding models are supported alongside cloud-based providers.'
      }
    ]
  } catch {
    return [
      {
        id: 'tokensmith',
        name: 'TokenSmith',
        status: 'unavailable',
        detail: 'The local TokenSmith runtime is not available. Material indexing and source-backed chat need it.'
      }
    ]
  }
}

export async function generateStudyQuestionSuggestions(
  request: EngineQuestionSuggestionRequest,
  dependencies: StudyEngineDependencies
): Promise<EngineQuestionSuggestionResponse> {
  if (request.model.engine === 'ollama') {
    return dependencies.generateOllamaStudyQuestionSuggestions(request)
  }

  if (request.model.engine === 'remote') {
    const remoteModel = modelWithRememberedRemoteApiKey(request.model)
    rememberRemoteModelApiKey(remoteModel)
    return generateRemoteStudyQuestionSuggestions({
      ...request,
      model: remoteModel
    })
  }

  throw new Error('Question suggestions require an Ollama or remote chat model.')
}

export async function sendStudyChatMessage(
  request: EngineChatRequest,
  dependencies: StudyEngineDependencies
): Promise<EngineChatResponse> {
  if (request.model.engine === 'ollama') {
    const response = await dependencies.runOllamaStudyEngine(request)
    return {
      ...response,
      engineId: 'tokensmith',
      modelName: response.modelName || request.model.name
    }
  }

  if (request.model.engine === 'remote') {
    const remoteModel = modelWithRememberedRemoteApiKey(request.model)
    rememberRemoteModelApiKey(remoteModel)
    return await runRemoteStudyEngine({
      ...request,
      model: remoteModel
    })
  }

  throw new Error('Local GGUF chat models are not supported in this app version. Use Ollama for local chat or add a remote chat model from Models.')
}
