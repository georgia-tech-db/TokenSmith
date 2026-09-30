import type { AcceleratorCapabilities } from '../../../shared/device-capabilities'
import type { ElectronGpuDevice } from './electron-gpu'
import { detectNvidiaAccelerators } from './nvidia-gpu'

export function applyWindowsRuntimeSupport(
  architecture: string,
  accelerators: AcceleratorCapabilities[]
): AcceleratorCapabilities[] {
  if (architecture !== 'arm64') {
    return accelerators
  }

  return accelerators.map((accelerator) => ({
    ...accelerator,
    runtimeSupport: 'unverified'
  }))
}

export async function detectWindowsAccelerators(
  architecture: string,
  electronDevices: ElectronGpuDevice[]
): Promise<AcceleratorCapabilities[]> {
  const nvidiaAccelerators = await detectNvidiaAccelerators()
  if (nvidiaAccelerators.length > 0) {
    return applyWindowsRuntimeSupport(architecture, nvidiaAccelerators)
  }

  return electronDevices.map((device) => ({
    name: device.name,
    backend: 'unknown',
    memoryTopology: 'unknown',
    totalMemoryBytes: null,
    availableMemoryBytes: null,
    runtimeSupport: 'unverified',
    source: 'electron'
  }))
}
