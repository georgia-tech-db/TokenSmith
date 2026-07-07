import type { EngineChatRequest, EngineChatResponse, EngineInfo } from '../../shared/engine'
import { runRemoteStudyEngine } from './remote-chat-service'
import {
  modelWithRememberedRemoteApiKey,
  rememberRemoteModelApiKey
} from './remote-model-secrets'

export interface PythonEngineHealth {
  llamaCppAvailable: boolean
}

export interface StudyEngineDependencies {
  getPythonEngineHealth: () => Promise<PythonEngineHealth>
  runOllamaStudyEngine: (request: EngineChatRequest) => Promise<EngineChatResponse>
  runPythonStudyEngine: (request: EngineChatRequest) => Promise<EngineChatResponse>
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

export async function sendStudyChatMessage(
  request: EngineChatRequest,
  dependencies: StudyEngineDependencies
): Promise<EngineChatResponse> {
  if (request.model.engine === 'ollama') {
    try {
      const response = await dependencies.runOllamaStudyEngine(request)
      return {
        ...response,
        engineId: 'tokensmith',
        modelName: response.modelName || request.model.name
      }
    } catch (error) {
      const reason = error instanceof Error ? error.message : 'Ollama could not answer.'

      return {
        engineId: 'tokensmith',
        modelName: request.model.name,
        text: `The Ollama chat model was not available: ${reason}\n\nOpen Ollama, make sure llama3 is downloaded, then try again.`,
        sources: request.retrievedSources ?? []
      }
    }
  }

  if (request.model.engine === 'remote') {
    try {
      const remoteModel = modelWithRememberedRemoteApiKey(request.model)
      rememberRemoteModelApiKey(remoteModel)
      return await runRemoteStudyEngine({
        ...request,
        model: remoteModel
      })
    } catch (error) {
      const reason = error instanceof Error ? error.message : 'The remote provider could not answer.'

      return {
        engineId: 'tokensmith',
        modelName: request.model.name,
        text: `The remote model was not available: ${reason}`,
        sources: request.retrievedSources ?? []
      }
    }
  }

  return {
    engineId: 'tokensmith',
    modelName: request.model.name,
    text: 'Python/GGUF chat models are not packaged in this app version. Use Ollama for local chat or add a cloud-based chat model from Models.',
    sources: request.retrievedSources ?? []
  }
}
