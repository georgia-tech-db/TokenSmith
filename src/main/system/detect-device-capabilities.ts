import {
  acceleratorIsUsable,
  type AcceleratorCapabilities,
  type DeviceCapabilities
} from '../../shared/device-capabilities'
import { parseElectronGpuDevices, type ElectronGpuDevice } from './detectors/electron-gpu'
import { detectLinuxAccelerators } from './detectors/linux-gpu-detector'
import { detectMacosAccelerators } from './detectors/macos-gpu-detector'
import { detectPortableCapabilities } from './detectors/portable-detector'
import { detectWindowsAccelerators } from './detectors/windows-gpu-detector'

async function detectAccelerators(
  device: Omit<DeviceCapabilities, 'accelerators' | 'detectionWarnings'>,
  electronDevices: ElectronGpuDevice[]
) {
  if (device.platform === 'macos') {
    return detectMacosAccelerators(
      device.architecture,
      device.memory.totalBytes,
      electronDevices
    )
  }
  if (device.platform === 'windows') {
    return detectWindowsAccelerators(device.architecture, electronDevices)
  }
  if (device.platform === 'linux') {
    return detectLinuxAccelerators(electronDevices)
  }
  return []
}

export function acceleratorDetectionWarning(
  accelerators: AcceleratorCapabilities[]
): string | null {
  if (accelerators.length === 0 || accelerators.some(acceleratorIsUsable)) {
    return null
  }

  const supportIsKnown = accelerators.every(
    (accelerator) => accelerator.runtimeSupport === 'unsupported'
  )
  return supportIsKnown
    ? null
    : 'A graphics device was detected, but runtime support or inference memory could not be verified; recommendations use the CPU fallback policy.'
}

export async function detectDeviceCapabilities(
  storagePath: string,
  getElectronGpuInfo: () => Promise<unknown>
): Promise<DeviceCapabilities> {
  const portable = await detectPortableCapabilities(storagePath)
  const warnings: string[] = []

  let electronGpuInfo: unknown = null
  try {
    electronGpuInfo = await getElectronGpuInfo()
  } catch {
    warnings.push('GPU identity could not be read from Electron.')
  }

  const electronDevices = parseElectronGpuDevices(electronGpuInfo)
  const accelerators = await detectAccelerators(portable, electronDevices)
  const acceleratorWarning = acceleratorDetectionWarning(accelerators)
  if (acceleratorWarning) {
    warnings.push(acceleratorWarning)
  }

  return {
    ...portable,
    accelerators,
    detectionWarnings: warnings
  }
}
