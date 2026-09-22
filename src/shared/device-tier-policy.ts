import { gibibytes, type DevicePlatform } from './device-capabilities'

export type LocalDeviceTier = 1 | 2 | 3 | 4 | 5

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

// Conservative starting thresholds for TokenSmith's existing 2,048-token context.
// Memory and performance thresholds must be validated on representative hardware.
export const defaultDeviceTierPolicy: DeviceTierPolicy = {
  assumedContextTokens: 2_048,
  localInferenceTargets: [
    { platform: 'macos', architectures: ['arm64', 'x64'], minimumReleaseMajor: 23 },
    { platform: 'windows', architectures: ['x64', 'arm64'], minimumWindowsBuild: 19045 },
    { platform: 'linux', architectures: ['x64', 'arm64'] }
  ],
  tiers: [
    {
      tier: 1,
      name: 'Light Local',
      recommendedModelId: 'gemma3:4b-it-q4_K_M',
      minimumHostMemoryBytes: gibibytes(8),
      minimumCpuMemoryBytes: gibibytes(16),
      minimumCpuParallelism: 8,
      minimumUnifiedMemoryBytes: gibibytes(8),
      minimumDedicatedVramBytes: gibibytes(4),
      minimumFreeDiskBytes: gibibytes(5)
    },
    {
      tier: 2,
      name: 'Standard Local',
      recommendedModelId: 'gemma3:4b-it-q8_0',
      minimumHostMemoryBytes: gibibytes(12),
      minimumCpuMemoryBytes: gibibytes(24),
      minimumCpuParallelism: 12,
      minimumUnifiedMemoryBytes: gibibytes(16),
      minimumDedicatedVramBytes: gibibytes(8),
      minimumFreeDiskBytes: gibibytes(8)
    },
    {
      tier: 3,
      name: 'Enhanced Local',
      recommendedModelId: 'gemma3:12b-it-q4_K_M',
      minimumHostMemoryBytes: gibibytes(16),
      minimumCpuMemoryBytes: null,
      minimumCpuParallelism: null,
      minimumUnifiedMemoryBytes: gibibytes(20),
      minimumDedicatedVramBytes: gibibytes(10),
      minimumFreeDiskBytes: gibibytes(10)
    },
    {
      tier: 4,
      name: 'High Precision Local',
      recommendedModelId: 'gemma3:12b-it-q8_0',
      minimumHostMemoryBytes: gibibytes(16),
      minimumCpuMemoryBytes: null,
      minimumCpuParallelism: null,
      minimumUnifiedMemoryBytes: gibibytes(32),
      minimumDedicatedVramBytes: gibibytes(16),
      minimumFreeDiskBytes: gibibytes(16)
    },
    {
      tier: 5,
      name: 'Workstation Local',
      recommendedModelId: 'gemma3:27b-it-q8_0',
      minimumHostMemoryBytes: gibibytes(32),
      minimumCpuMemoryBytes: null,
      minimumCpuParallelism: null,
      minimumUnifiedMemoryBytes: gibibytes(48),
      minimumDedicatedVramBytes: gibibytes(32),
      minimumFreeDiskBytes: gibibytes(35)
    }
  ],
  cloudModel: {
    id: 'gemini:gemini-2.5-flash',
    displayName: 'Gemini 2.5 Flash (Google)'
  }
}
