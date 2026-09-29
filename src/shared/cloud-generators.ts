import type { LocalModel, LocalModelRole } from './app-state'
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
  role?: LocalModelRole
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

export interface ForgottenCloudConnections {
  connectionIds: string[]
}

// Keep disconnected generators selectable for repair, without changing embedder eligibility.
export function isCloudGenerator(model: LocalModel): boolean {
  return model.engine === 'remote' && (model.role === 'generator' || model.role === undefined)
}

export function isCloudConnectionModel(model: LocalModel): boolean {
  return model.engine === 'remote' && Boolean(model.connectionId)
}

function normalizedEndpoint(baseUrl?: string): string | undefined {
  return baseUrl?.trim().replace(/\/+$/, '')
}

/**
 * A fixed provider owns one credential. Older saved models may still point to
 * separate per-model connections, so attach them to the connection the user
 * has just verified. Custom providers remain separate for each endpoint.
 */
export function attachCloudProviderConnection(models: LocalModel[], connection: LocalModel): LocalModel[] {
  if (!isCloudConnectionModel(connection)) return models
  const endpoint = normalizedEndpoint(connection.baseUrl)
  return models.map(model =>
    model.engine === 'remote' && model.providerId === connection.providerId &&
    (connection.providerId !== 'custom' || normalizedEndpoint(model.baseUrl) === endpoint)
      ? {
          ...model,
          apiKey: undefined,
          connectionId: connection.connectionId,
          cloudCredentialStatus: 'connected' as const,
          status: 'ready' as const
        }
      : model
  )
}

export function mergeCloudConnectionModel(models: LocalModel[], incoming: LocalModel): LocalModel[] {
  if (!isCloudConnectionModel(incoming)) return models
  if (models.some(model => model.id === incoming.id && !isCloudConnectionModel(model))) return models
  const existing = models.find(model => isCloudConnectionModel(model) && (
    model.id === incoming.id || (model.connectionId === incoming.connectionId &&
      normalizedEndpoint(model.baseUrl) === normalizedEndpoint(incoming.baseUrl) &&
      model.remoteModelName === incoming.remoteModelName && model.role === incoming.role)
  ))
  const added = { ...incoming, id: existing?.id ?? incoming.id }
  return [added, ...models.filter(model => model.id !== added.id).map(model =>
    isCloudConnectionModel(model) && model.connectionId === added.connectionId
      ? { ...model, cloudCredentialStatus: 'connected' as const, status: 'ready' as const }
      : model
  )]
}

export function disconnectCloudConnectionModels(models: LocalModel[], connectionIds: string[]): LocalModel[] {
  const disconnected = new Set(connectionIds)
  return models.map(model => model.connectionId && disconnected.has(model.connectionId)
    ? { ...model, cloudCredentialStatus: 'reconnect' as const, status: 'needsRuntime' as const }
    : model)
}

// Kept for callers outside the provider-connection flow while they migrate.
export const mergeCloudGenerator = mergeCloudConnectionModel
