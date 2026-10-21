import type { AppStateSnapshot, Conversation, LocalModel } from './app-state'

export type ModelSelectionTarget = { kind: 'chat'; conversationId: string } | { kind: 'practice' }

export function isAnswerModel(model: LocalModel): boolean {
  const role = model.role ?? 'generator'
  return (model.engine === 'ollama' || model.engine === 'remote') && (role === 'generator' || role === 'both')
}

export function answerModel(models: LocalModel[], id: string | undefined): LocalModel | undefined {
  return models.find(model => model.id === id && isAnswerModel(model))
}

// Older saves have one selection for all chats. Consume it once, not at request time.
export function migrateConversationModels(conversations: Conversation[], legacyModelId: string): Conversation[] {
  return conversations.map(conversation => typeof conversation.modelId === 'string'
    ? conversation : { ...conversation, modelId: legacyModelId })
}

export function studyModelTarget(state: AppStateSnapshot): ModelSelectionTarget {
  return state.practice?.mode === 'practice' ? { kind: 'practice' }
    : { kind: 'chat', conversationId: state.activeConversationId }
}

export function selectStudyModel(state: AppStateSnapshot, modelId: string, target: ModelSelectionTarget): AppStateSnapshot {
  if (!answerModel(state.models, modelId)) return state
  if (target.kind === 'chat' && !state.conversations.some(chat => chat.id === target.conversationId)) return state
  if (target.kind === 'practice' && (!state.practice || state.practice.activeSessionId)) return state
  return {
    ...state,
    conversations: target.kind === 'chat' ? state.conversations.map(chat => chat.id === target.conversationId
      ? { ...chat, modelId } : chat) : state.conversations,
    practice: target.kind === 'practice' ? { ...state.practice!, draftModelId: modelId } : state.practice,
    // The first explicit choice also establishes the default for new sessions.
    settings: state.settings.application.defaultModelId ? state.settings : { ...state.settings,
      application: { ...state.settings.application, defaultModelId: modelId } }
  }
}
