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
  runPythonStudyEngine: (request: EngineChatRequest) => Promise<EngineChatResponse>
}

export async function listStudyEngines(dependencies: StudyEngineDependencies): Promise<EngineInfo[]> {
  try {
    const health = await dependencies.getPythonEngineHealth()
    return [
      {
        id: 'tokensmith',
        name: 'TokenSmith',
        status: 'ready',
        detail: health.llamaCppAvailable
          ? 'Local indexing, vector retrieval, local GGUF models, and configured remote models are available.'
          : 'Local indexing and vector retrieval are available. Local GGUF models need llama-cpp-python.'
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

  try {
    const response = await dependencies.runPythonStudyEngine(request)
    return {
      ...response,
      engineId: 'tokensmith',
      modelName: response.modelName || request.model.name
    }
  } catch (error) {
    const reason = error instanceof Error ? error.message : 'The Python engine could not start.'

    return {
      engineId: 'tokensmith',
      modelName: request.model.name,
      text: `The local TokenSmith runtime was not available: ${reason}\n\nOpen Settings or restart the app after choosing a working Python runtime.`,
      sources: []
    }
  }
}
