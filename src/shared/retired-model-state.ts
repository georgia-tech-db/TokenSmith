import type { AppStateSnapshot, ChatSource } from './app-state'

// Drop records for the removed direct-GGUF backend, not models managed by Ollama.
export function removeRetiredModelState(state: AppStateSnapshot): AppStateSnapshot {
  const retiredIds = new Set((state.models ?? []).filter(model => String(model.engine) === 'python').map(model => model.id))
  const isRetiredKey = (key?: string) => key?.startsWith('llama-cpp:') === true
  const retiredMaterials = new Set((state.materials ?? []).filter(material =>
    isRetiredKey(material.embeddingModel) || retiredIds.has(material.embeddingModelId ?? '')
  ).map(material => material.id))
  const isRetiredSource = (source: ChatSource) => isRetiredKey(source.embeddingModel) ||
    isRetiredKey(source.chunkEmbeddingModel) || retiredMaterials.has(source.materialId ?? '')
  const conversations = (state.conversations ?? []).filter(conversation => !conversation.messages.some(message =>
    [...(message.sources ?? []), ...(message.simplerExplanation?.sources ?? [])].some(isRetiredSource)
  ) && !(conversation.quizState?.currentSources ?? []).some(isRetiredSource))
  return {
    ...state,
    models: (state.models ?? []).filter(model => !retiredIds.has(model.id)),
    materials: (state.materials ?? []).filter(material => !retiredMaterials.has(material.id)),
    conversations,
    activeConversationId: conversations.some(item => item.id === state.activeConversationId) ? state.activeConversationId : conversations[0]?.id ?? '',
    selectedModelId: retiredIds.has(state.selectedModelId) ? '' : state.selectedModelId,
    selectedEmbeddingModelId: retiredIds.has(state.selectedEmbeddingModelId) ? '' : state.selectedEmbeddingModelId,
    settings: state.settings ? {
      ...state.settings,
      application: {
        ...state.settings.application,
        defaultModelId: retiredIds.has(state.settings.application?.defaultModelId ?? '') ? '' : state.settings.application?.defaultModelId
      },
      modelSettingsById: Object.fromEntries(Object.entries(state.settings.modelSettingsById ?? {}).filter(([id]) => !retiredIds.has(id)))
    } : state.settings
  }
}
