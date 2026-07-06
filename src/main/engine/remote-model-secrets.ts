import type { AppStateSnapshot, LocalModel } from '../../shared/app-state'

const remoteApiKeys = new Map<string, string>()

function cleanApiKey(apiKey?: string): string | undefined {
  const trimmed = apiKey?.trim()
  return trimmed ? trimmed : undefined
}

function isRemoteModel(model: LocalModel): boolean {
  return model.engine === 'remote' || model.source === 'remote'
}

export function rememberRemoteModelApiKey(model: LocalModel): void {
  if (!isRemoteModel(model)) {
    return
  }

  const apiKey = cleanApiKey(model.apiKey)
  if (apiKey) {
    remoteApiKeys.set(model.id, apiKey)
  }
}

export function rememberRemoteModelApiKeys(models: LocalModel[] = []): void {
  for (const model of models) {
    rememberRemoteModelApiKey(model)
  }
}

export function modelWithRememberedRemoteApiKey(model: LocalModel): LocalModel {
  if (!isRemoteModel(model)) {
    return model
  }

  const apiKey = cleanApiKey(model.apiKey) ?? remoteApiKeys.get(model.id)
  if (!apiKey) {
    return model
  }

  return {
    ...model,
    apiKey,
    status: model.status === 'needsRuntime' ? 'ready' : model.status
  }
}

export function sanitizeRemoteModelSecrets(model: LocalModel): LocalModel {
  if (!isRemoteModel(model) || model.apiKey === undefined) {
    return model
  }

  const { apiKey: _apiKey, ...safeModel } = model
  return safeModel
}

export function sanitizeAppStateSecrets(state: AppStateSnapshot): AppStateSnapshot {
  return {
    ...state,
    models: state.models.map(sanitizeRemoteModelSecrets)
  }
}
