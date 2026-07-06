import {
  getPythonEngineHealth,
  runPythonStudyEngine
} from '../python/python-engine-service'
import type { EngineChatRequest, EngineChatResponse, EngineInfo } from '../../shared/engine'
import { listStudyEngines, sendStudyChatMessage } from './study-engine-core'

export async function listEngines(): Promise<EngineInfo[]> {
  return listStudyEngines({
    getPythonEngineHealth,
    runPythonStudyEngine
  })
}

export async function sendChatMessage(request: EngineChatRequest): Promise<EngineChatResponse> {
  return sendStudyChatMessage(request, {
    getPythonEngineHealth,
    runPythonStudyEngine
  })
}
