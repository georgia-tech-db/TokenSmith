import type { LocalModel } from './app-state'
import type { RemoteProviderId } from './model-providers'

export interface CloudConnection {
  id: string
  providerId: RemoteProviderId
  baseUrl: string
  remembered: boolean
  connected: boolean
}

export interface CloudConnectionInput {
  requestId?: string
  providerId: RemoteProviderId
  connectionId?: string
  baseUrl?: string
  apiKey?: string
}

export interface CloudGeneratorInput extends CloudConnectionInput {
  modelName: string
  modelId?: string
  remember: boolean
}

export type CloudErrorCode = 'credentials' | 'access' | 'quota' | 'rate_limit' | 'network' | 'model' | 'storage' | 'configuration'
export interface CloudSetupError { code: CloudErrorCode; message: string }
export type CloudResult<T> = { ok: true; value: T } | { ok: false; error: CloudSetupError }
export interface CloudConnectionStatus {
  connections: CloudConnection[]
  secureStorageAvailable: boolean
}

// Keep disconnected generators selectable for repair, without changing embedder eligibility.
export function isCloudGenerator(model: LocalModel): boolean {
  return model.engine === 'remote' && (model.role === 'generator' || model.role === undefined)
}

export function mergeCloudGenerator(models: LocalModel[], incoming: LocalModel): LocalModel[] {
  if (!isCloudGenerator(incoming)) return models
  if (models.some(model => model.id === incoming.id && !isCloudGenerator(model))) return models
  const existing = models.find(model => isCloudGenerator(model) && (
    model.id === incoming.id || (model.connectionId === incoming.connectionId &&
      model.baseUrl === incoming.baseUrl && model.remoteModelName === incoming.remoteModelName)
  ))
  const added = { ...incoming, id: existing?.id ?? incoming.id }
  return [added, ...models.filter(model => model.id !== added.id).map(model =>
    isCloudGenerator(model) && model.connectionId === added.connectionId
      ? { ...model, cloudCredentialStatus: 'connected' as const, status: 'ready' as const }
      : model
  )]
}
