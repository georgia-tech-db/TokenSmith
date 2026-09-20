import type { DeviceCapabilities } from './device-capabilities'
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
      warnings: tierAssessment.warnings
    }
  }

  const tier = policy.tiers.find((candidate) => candidate.tier === tierAssessment.tier)
  const model = models.find((candidate) => candidate.id === tier?.recommendedModelId)
  if (!tier || !model) {
    throw new Error(`No model is configured for device Tier ${tierAssessment.tier}.`)
  }

  return {
    recommendedModelId: model.id,
    recommendedModelName: model.displayName,
    kind: 'local',
    assumedContextTokens: policy.assumedContextTokens,
    tierAssessment,
    reasons: tierAssessment.reasons,
    warnings: tierAssessment.warnings
  }
}
