import type { ApplicationSettings, ChatModelMode, Conversation, LocalModel } from './app-state'

export function isLocalAnswerModel(model: LocalModel): boolean {
  const role = model.role ?? 'generator'
  return model.engine === 'ollama' && (role === 'generator' || role === 'both')
}

export function isOnlineAnswerModel(model: LocalModel): boolean {
  return model.engine === 'remote' && (model.role === 'generator' || model.role === undefined)
}

export function normalizeAnswerModelDefaults(
  settings: Partial<ApplicationSettings> | undefined,
  models: LocalModel[]
): Pick<ApplicationSettings, 'defaultLocalModelId' | 'defaultOnlineModelId' | 'defaultChatMode'> {
  const localModels = models.filter(isLocalAnswerModel)
  const onlineModels = models.filter(isOnlineAnswerModel)
  const legacyModel = models.find((model) => model.id === settings?.defaultModelId)
  const defaultLocalModelId = settings?.defaultLocalModelId && localModels.some((model) => model.id === settings.defaultLocalModelId)
    ? settings.defaultLocalModelId
    : legacyModel && isLocalAnswerModel(legacyModel) ? legacyModel.id : localModels[0]?.id ?? ''
  const defaultOnlineModelId = settings?.defaultOnlineModelId && onlineModels.some((model) => model.id === settings.defaultOnlineModelId)
    ? settings.defaultOnlineModelId
    : legacyModel && isOnlineAnswerModel(legacyModel) ? legacyModel.id : onlineModels[0]?.id ?? ''
  const requestedMode: ChatModelMode = settings?.defaultChatMode === 'online' ? 'online' : 'local'
  const defaultChatMode: ChatModelMode = requestedMode === 'online' && defaultOnlineModelId
    ? 'online'
    : defaultLocalModelId ? 'local' : defaultOnlineModelId ? 'online' : 'local'

  return { defaultLocalModelId, defaultOnlineModelId, defaultChatMode }
}

export function normalizeConversationModelSelection(
  conversation: Conversation,
  defaults: Pick<ApplicationSettings, 'defaultLocalModelId' | 'defaultOnlineModelId' | 'defaultChatMode'>,
  models: LocalModel[],
  legacySelectedModelId = ''
): Conversation {
  const localIds = new Set(models.filter(isLocalAnswerModel).map((model) => model.id))
  const onlineIds = new Set(models.filter(isOnlineAnswerModel).map((model) => model.id))
  const legacyModel = models.find((model) => model.id === legacySelectedModelId)
  const localModelId = conversation.localModelId && localIds.has(conversation.localModelId)
    ? conversation.localModelId
    : legacyModel && isLocalAnswerModel(legacyModel) ? legacyModel.id : defaults.defaultLocalModelId
  const onlineModelId = conversation.onlineModelId && onlineIds.has(conversation.onlineModelId)
    ? conversation.onlineModelId
    : legacyModel && isOnlineAnswerModel(legacyModel) ? legacyModel.id : defaults.defaultOnlineModelId
  const requestedMode = conversation.chatModelMode ?? (legacyModel && isOnlineAnswerModel(legacyModel) ? 'online' : defaults.defaultChatMode)
  const chatModelMode: ChatModelMode = requestedMode === 'online' && onlineModelId
    ? 'online'
    : localModelId ? 'local' : onlineModelId ? 'online' : defaults.defaultChatMode

  return { ...conversation, chatModelMode, localModelId, onlineModelId }
}
