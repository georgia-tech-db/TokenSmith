import {
  acceleratorIsUsable,
  bytesToRoundedGibibytes,
  type AcceleratorCapabilities,
  type DeviceCapabilities
} from './device-capabilities'
import {
  type DeviceTierDefinition,
  type DeviceTierPolicy,
  type LocalDeviceTier
} from './device-tier-policy'

export type ExecutionPath = 'dedicated-gpu' | 'unified-gpu' | 'cpu'

export interface TierEvaluation {
  tier: LocalDeviceTier
  name: string
  recommendedModelId: string
  eligible: boolean
  executionPath: ExecutionPath | null
  rejectionReasons: string[]
}

export interface DeviceTierAssessment {
  tier: 0 | LocalDeviceTier
  name: string
  executionPath: ExecutionPath | 'cloud'
  reasons: string[]
  warnings: string[]
  evaluatedTiers: TierEvaluation[]
}

interface EligiblePath {
  kind: ExecutionPath
  accelerator?: AcceleratorCapabilities
}

function platformReleaseParts(release: string): number[] {
  return release.split('.').map((part) => Number.parseInt(part, 10))
}

function localRuntimeSupport(
  device: DeviceCapabilities,
  policy: DeviceTierPolicy
): { supported: boolean; reason?: string } {
  const target = policy.localInferenceTargets.find(
    (candidate) =>
      candidate.platform === device.platform &&
      candidate.architectures.includes(device.architecture)
  )

  if (!target) {
    return {
      supported: false,
      reason: `Local inference is not supported for ${device.platform} ${device.architecture}.`
    }
  }

  const release = platformReleaseParts(device.platformRelease)
  if (
    target.minimumReleaseMajor !== undefined &&
    (!Number.isFinite(release[0]) || release[0] < target.minimumReleaseMajor)
  ) {
    return {
      supported: false,
      reason: `This ${device.platform} release is below the local Ollama requirement.`
    }
  }

  if (
    target.minimumWindowsBuild !== undefined &&
    (!Number.isFinite(release[2]) || release[2] < target.minimumWindowsBuild)
  ) {
    return {
      supported: false,
      reason: 'This Windows build is below the local Ollama requirement.'
    }
  }

  return { supported: true }
}

function supportedAccelerators(device: DeviceCapabilities): AcceleratorCapabilities[] {
  return device.accelerators.filter(acceleratorIsUsable)
}

function eligiblePath(
  device: DeviceCapabilities,
  tier: DeviceTierDefinition
): EligiblePath | null {
  if (device.memory.totalBytes < tier.minimumHostMemoryBytes) {
    return null
  }

  const dedicatedGpu = supportedAccelerators(device).find(
    (accelerator) =>
      accelerator.memoryTopology === 'dedicated' &&
      accelerator.totalMemoryBytes !== null &&
      accelerator.totalMemoryBytes >= tier.minimumDedicatedVramBytes
  )
  if (dedicatedGpu) {
    return { kind: 'dedicated-gpu', accelerator: dedicatedGpu }
  }

  const unifiedGpu = supportedAccelerators(device).find(
    (accelerator) => accelerator.memoryTopology === 'unified'
  )
  if (unifiedGpu && device.memory.totalBytes >= tier.minimumUnifiedMemoryBytes) {
    return { kind: 'unified-gpu', accelerator: unifiedGpu }
  }

  if (
    tier.minimumCpuMemoryBytes !== null &&
    tier.minimumCpuParallelism !== null &&
    device.memory.totalBytes >= tier.minimumCpuMemoryBytes &&
    device.cpu.availableParallelism >= tier.minimumCpuParallelism
  ) {
    return { kind: 'cpu' }
  }

  return null
}

function rejectionReasons(
  device: DeviceCapabilities,
  tier: DeviceTierDefinition,
  runtime: { supported: boolean; reason?: string }
): string[] {
  const reasons: string[] = []

  if (!runtime.supported) {
    reasons.push(runtime.reason ?? 'The local Ollama runtime is unsupported.')
  }
  if (device.storage.availableBytes < tier.minimumFreeDiskBytes) {
    reasons.push(
      `Requires ${bytesToRoundedGibibytes(tier.minimumFreeDiskBytes)} GiB of free storage.`
    )
  }
  if (device.memory.totalBytes < tier.minimumHostMemoryBytes) {
    reasons.push(
      `Requires ${bytesToRoundedGibibytes(tier.minimumHostMemoryBytes)} GiB of host memory.`
    )
  }

  if (!eligiblePath(device, tier)) {
    if (tier.minimumCpuMemoryBytes !== null && tier.minimumCpuParallelism !== null) {
      if (device.memory.totalBytes < tier.minimumCpuMemoryBytes) {
        reasons.push(
          `CPU execution requires ${bytesToRoundedGibibytes(tier.minimumCpuMemoryBytes)} GiB of system memory.`
        )
      } else if (device.cpu.availableParallelism < tier.minimumCpuParallelism) {
        reasons.push(`CPU execution requires ${tier.minimumCpuParallelism} available threads.`)
      }
    }

    reasons.push(
      `The accelerator path requires ${bytesToRoundedGibibytes(tier.minimumUnifiedMemoryBytes)} GiB unified memory or ${bytesToRoundedGibibytes(tier.minimumDedicatedVramBytes)} GiB dedicated VRAM.`
    )
  }

  return [...new Set(reasons)]
}

function reasonsFor(
  device: DeviceCapabilities,
  tier: DeviceTierDefinition,
  path: EligiblePath
): string[] {
  const totalRam = bytesToRoundedGibibytes(device.memory.totalBytes)

  if (path.kind === 'dedicated-gpu') {
    const vram = bytesToRoundedGibibytes(path.accelerator?.totalMemoryBytes ?? 0)
    return [
      `${vram} GiB of dedicated GPU memory meets the Tier ${tier.tier} threshold.`,
      `${totalRam} GiB of system memory meets the host-memory threshold.`
    ]
  }

  if (path.kind === 'unified-gpu') {
    return [
      `${totalRam} GiB of unified memory meets the Tier ${tier.tier} threshold.`,
      `${path.accelerator?.name ?? 'The detected accelerator'} is supported by the local runtime.`
    ]
  }

  return [
    `${totalRam} GiB of system memory meets the Tier ${tier.tier} CPU threshold.`,
    `${device.cpu.availableParallelism} available CPU threads meet the ${tier.minimumCpuParallelism}-thread minimum.`
  ]
}

function warningsFor(
  device: DeviceCapabilities,
  tier: DeviceTierDefinition,
  path: EligiblePath
): string[] {
  const warnings = [...device.detectionWarnings]

  if (
    path.kind === 'dedicated-gpu' &&
    path.accelerator?.availableMemoryBytes !== null &&
    path.accelerator?.availableMemoryBytes !== undefined &&
    path.accelerator.availableMemoryBytes < tier.minimumDedicatedVramBytes
  ) {
    warnings.push('GPU memory is currently busy. Close other GPU-heavy applications before loading this model.')
  }

  return [...new Set(warnings)]
}

export function classifyDeviceTier(
  device: DeviceCapabilities,
  policy: DeviceTierPolicy
): DeviceTierAssessment {
  const runtime = localRuntimeSupport(device, policy)
  const evaluatedTiers = policy.tiers.map((tier) => {
    const path = runtime.supported ? eligiblePath(device, tier) : null
    const hasDisk = device.storage.availableBytes >= tier.minimumFreeDiskBytes
    const eligible = Boolean(path) && hasDisk

    return {
      tier: tier.tier,
      name: tier.name,
      recommendedModelId: tier.recommendedModelId,
      eligible,
      executionPath: eligible ? path?.kind ?? null : null,
      rejectionReasons: eligible ? [] : rejectionReasons(device, tier, runtime)
    }
  })

  const highestEligible = [...evaluatedTiers]
    .sort((left, right) => right.tier - left.tier)
    .find((evaluation) => evaluation.eligible)

  if (!highestEligible) {
    return {
      tier: 0,
      name: 'Cloud',
      executionPath: 'cloud',
      reasons: [
        runtime.reason ?? 'This device does not meet the requirements for our local models.'
      ],
      warnings: [...device.detectionWarnings],
      evaluatedTiers
    }
  }

  const tier = policy.tiers.find((candidate) => candidate.tier === highestEligible.tier)
  const path = tier ? eligiblePath(device, tier) : null
  if (!tier || !path) {
    throw new Error('The selected device tier is missing from the policy.')
  }

  return {
    tier: tier.tier,
    name: tier.name,
    executionPath: path.kind,
    reasons: reasonsFor(device, tier, path),
    warnings: warningsFor(device, tier, path),
    evaluatedTiers
  }
}
