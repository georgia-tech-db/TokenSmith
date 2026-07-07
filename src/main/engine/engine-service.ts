import {
  getPythonEngineHealth,
  runPythonStudyEngine
} from '../python/python-engine-service'
import type { EngineChatRequest, EngineChatResponse, EngineInfo } from '../../shared/engine'
import { listStudyEngines, sendStudyChatMessage } from './study-engine-core'
import { runOllamaStudyEngine } from './ollama-service'

export async function listEngines(): Promise<EngineInfo[]> {
  return listStudyEngines({
    getPythonEngineHealth,
    runOllamaStudyEngine,
    runPythonStudyEngine
  })
}

export async function sendChatMessage(request: EngineChatRequest): Promise<EngineChatResponse> {
  return sendStudyChatMessage(request, {
    getPythonEngineHealth,
    runOllamaStudyEngine,
    runPythonStudyEngine
  })
}
