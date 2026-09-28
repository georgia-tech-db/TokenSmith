import { gibibytes, type DeviceCapabilities } from './device-capabilities'
import { classifyDeviceTier, type DeviceTierAssessment } from './device-tier'
import type { DeviceTierPolicy } from './device-tier-policy'
import type { LocalModelProfile } from './model-catalog'

export interface ModelRecommendation {
  recommendedModelId: string
  recommendedModelName: string
  kind: 'local' | 'cloud'
  assumedContextTokens: number
  tierAssessment: DeviceTierAssessment
  reasons: string[]
  warnings: string[]
  alternatives: LocalModelProfile[]
}

export function recommendModel(
  device: DeviceCapabilities,
  models: LocalModelProfile[],
  policy: DeviceTierPolicy
): ModelRecommendation {
  const tierAssessment = classifyDeviceTier(device, policy)

  if (tierAssessment.tier === 0) {
    return {
      recommendedModelId: policy.cloudModel.id,
      recommendedModelName: policy.cloudModel.displayName,
      kind: 'cloud',
      assumedContextTokens: policy.assumedContextTokens,
      tierAssessment,
      reasons: tierAssessment.reasons,
      warnings: tierAssessment.warnings,
      alternatives: []
    }
  }

  const tier = policy.tiers.find((candidate) => candidate.tier === tierAssessment.tier)
  const model = models.find((candidate) => candidate.id === tier?.recommendedModelId)
  if (!tier || !model) {
    throw new Error(`No model is configured for device Tier ${tierAssessment.tier}.`)
  }

  const warnings = [...tierAssessment.warnings]
  if (model.id === 'gemma4:26b' && tierAssessment.executionPath === 'unified-gpu'
    && device.memory.totalBytes < gibibytes(48)) {
    warnings.push('26B completed our trial on a 32 GB Mac, but memory pressure increased. If other apps become sluggish, choose 12B or E4B.')
  }

  return {
    recommendedModelId: model.id,
    recommendedModelName: model.displayName,
    kind: 'local',
    assumedContextTokens: policy.assumedContextTokens,
    tierAssessment,
    reasons: [model.description, ...tierAssessment.reasons],
    warnings,
    alternatives: models.filter((candidate) => candidate.id !== model.id &&
      tierAssessment.evaluatedTiers.some((tier) => tier.eligible && tier.recommendedModelId === candidate.id))
  }
}
