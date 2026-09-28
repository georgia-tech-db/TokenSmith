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

// Eligibility estimates assume an 8K context and Nomic; 32 GiB unified devices
// may page with 26B, as observed in the benchmark.
// Evaluate options in preference order: 26B for interactive study, then 12B and E4B.
// Tier numbers identify hardware requirements, not recommendation priority.
export const defaultDeviceTierPolicy: DeviceTierPolicy = {
  assumedContextTokens: 8_192,
  localInferenceTargets: [
    { platform: 'macos', architectures: ['arm64', 'x64'], minimumReleaseMajor: 23 },
    { platform: 'windows', architectures: ['x64', 'arm64'], minimumWindowsBuild: 19045 },
    { platform: 'linux', architectures: ['x64', 'arm64'] }
  ],
  tiers: [
    {
      tier: 3, name: 'Interactive study', recommendedModelId: 'gemma4:26b',
      minimumHostMemoryBytes: gibibytes(32),
      minimumCpuMemoryBytes: null, minimumCpuParallelism: null,
      minimumUnifiedMemoryBytes: gibibytes(32), minimumDedicatedVramBytes: gibibytes(24),
      minimumFreeDiskBytes: gibibytes(25)
    },
    {
      tier: 2, name: 'Lower-memory alternative', recommendedModelId: 'gemma4:12b',
      minimumHostMemoryBytes: gibibytes(24),
      minimumCpuMemoryBytes: null, minimumCpuParallelism: null,
      minimumUnifiedMemoryBytes: gibibytes(24), minimumDedicatedVramBytes: gibibytes(12),
      minimumFreeDiskBytes: gibibytes(14)
    },
    {
      tier: 1, name: 'Lighter fallback', recommendedModelId: 'gemma4:e4b',
      minimumHostMemoryBytes: gibibytes(16),
      minimumCpuMemoryBytes: gibibytes(24), minimumCpuParallelism: 8,
      minimumUnifiedMemoryBytes: gibibytes(16), minimumDedicatedVramBytes: gibibytes(10),
      minimumFreeDiskBytes: gibibytes(14)
    }
  ],
  cloudModel: { id: 'gemini:gemini-2.5-flash', displayName: 'Gemini 2.5 Flash (Google)' }
}
