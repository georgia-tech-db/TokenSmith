import { gibibytes, type DevicePlatform } from './device-capabilities'

export type LocalDeviceTier = 1 | 2 | 3

export interface LocalInferenceTarget {
  platform: Exclude<DevicePlatform, 'unknown'>
  architectures: string[]
  minimumReleaseMajor?: number
  minimumWindowsBuild?: number
}

export interface DeviceTierDefinition {
  tier: LocalDeviceTier
  name: string
  recommendedModelId: string
  minimumHostMemoryBytes: number
  minimumCpuMemoryBytes: number | null
  minimumCpuParallelism: number | null
  minimumUnifiedMemoryBytes: number
  minimumDedicatedVramBytes: number
  minimumFreeDiskBytes: number
}

export interface DeviceTierPolicy {
  assumedContextTokens: number
  localInferenceTargets: LocalInferenceTarget[]
  tiers: DeviceTierDefinition[]
  cloudModel: {
    id: string
    displayName: string
  }
}

// Eligibility estimates include room for an 8K context, Nomic, and other apps.
// Tier numbers identify options, not an automatic quality ranking.
export const defaultDeviceTierPolicy: DeviceTierPolicy = {
  assumedContextTokens: 8_192,
  localInferenceTargets: [
    { platform: 'macos', architectures: ['arm64', 'x64'], minimumReleaseMajor: 23 },
    { platform: 'windows', architectures: ['x64', 'arm64'], minimumWindowsBuild: 19045 },
    { platform: 'linux', architectures: ['x64', 'arm64'] }
  ],
  tiers: [
    {
      tier: 1, name: 'Everyday study', recommendedModelId: 'gemma4:e4b',
      minimumHostMemoryBytes: gibibytes(16),
      minimumCpuMemoryBytes: gibibytes(24), minimumCpuParallelism: 8,
      minimumUnifiedMemoryBytes: gibibytes(16), minimumDedicatedVramBytes: gibibytes(10),
      minimumFreeDiskBytes: gibibytes(14)
    },
    {
      tier: 2, name: 'Slower alternative', recommendedModelId: 'gemma4:12b',
      minimumHostMemoryBytes: gibibytes(24),
      minimumCpuMemoryBytes: null, minimumCpuParallelism: null,
      minimumUnifiedMemoryBytes: gibibytes(24), minimumDedicatedVramBytes: gibibytes(12),
      minimumFreeDiskBytes: gibibytes(14)
    },
    {
      tier: 3, name: 'More memory required', recommendedModelId: 'gemma4:26b',
      minimumHostMemoryBytes: gibibytes(32),
      minimumCpuMemoryBytes: null, minimumCpuParallelism: null,
      minimumUnifiedMemoryBytes: gibibytes(48), minimumDedicatedVramBytes: gibibytes(24),
      minimumFreeDiskBytes: gibibytes(25)
    }
  ],
  cloudModel: { id: 'gemini:gemini-2.5-flash', displayName: 'Gemini 2.5 Flash (Google)' }
}
