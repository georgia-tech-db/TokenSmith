import type { LocalModel } from './app-state'

// Only runtime identity crosses IPC; never credentials or prompts.
export interface ModelLoadingEvent {
  requestId: string
  loadId: string
  state: 'loading' | 'ready' | 'failed'
  model: Pick<LocalModel, 'id' | 'name' | 'engine' | 'ollamaModelName' | 'ollamaBaseUrl' | 'role'>
  contextLength?: number
  durationMs?: number
}
