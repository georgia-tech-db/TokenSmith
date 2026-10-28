import type { ModelRuntimeSettings, ReasoningMode } from './app-state'

export const minimumReasoningOutputTokens = 4096

export function reasoningMode(settings?: Partial<ModelRuntimeSettings>): ReasoningMode {
  return settings?.reasoningMode ?? (settings?.thinking === true ? 'on' : 'off')
}

export function answerReasoningSettings(
  settings: ModelRuntimeSettings | undefined, selected: boolean
): ModelRuntimeSettings | undefined {
  if (!settings) return settings
  const mode = reasoningMode(settings)
  const thinking = mode === 'on' || (mode === 'auto' && selected)
  return {
    ...settings,
    thinking,
    maxLength: thinking ? Math.max(settings.maxLength, minimumReasoningOutputTokens) : settings.maxLength
  }
}
