export type DevicePlatform = 'windows' | 'macos' | 'linux' | 'unknown'

export type AcceleratorBackend = 'metal' | 'cuda' | 'rocm' | 'unknown'

export type MemoryTopology = 'dedicated' | 'unified' | 'shared' | 'unknown'

export type AcceleratorRuntimeSupport = 'supported' | 'unsupported' | 'unverified'

export interface AcceleratorCapabilities {
  name: string
  backend: AcceleratorBackend
  memoryTopology: MemoryTopology
  totalMemoryBytes: number | null
  availableMemoryBytes: number | null
  runtimeSupport: AcceleratorRuntimeSupport
  source: string
}

export interface DeviceCapabilities {
  platform: DevicePlatform
  platformRelease: string
  architecture: string
  cpu: {
    model: string
    logicalCores: number
    availableParallelism: number
  }
  memory: {
    totalBytes: number
  }
  storage: {
    availableBytes: number
  }
  accelerators: AcceleratorCapabilities[]
  detectionWarnings: string[]
}

export const bytesPerGibibyte = 1024 ** 3

export function gibibytes(value: number): number {
  return value * bytesPerGibibyte
}

export function bytesToRoundedGibibytes(value: number): number {
  return Math.round((value / bytesPerGibibyte) * 10) / 10
}

export function acceleratorIsUsable(accelerator: AcceleratorCapabilities): boolean {
  return accelerator.runtimeSupport === 'supported'
}
