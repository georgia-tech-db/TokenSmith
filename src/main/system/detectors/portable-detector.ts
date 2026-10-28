import { statfs } from 'node:fs/promises'
import os from 'node:os'
import type { DeviceCapabilities, DevicePlatform } from '../../../shared/device-capabilities'

export function normalizePlatform(platform: NodeJS.Platform): DevicePlatform {
  if (platform === 'win32') {
    return 'windows'
  }
  if (platform === 'darwin') {
    return 'macos'
  }
  if (platform === 'linux') {
    return 'linux'
  }
  return 'unknown'
}

async function availableStorageBytes(storagePath: string): Promise<number> {
  try {
    const fileSystem = await statfs(storagePath, { bigint: true })
    return Number(fileSystem.bsize * fileSystem.bavail)
  } catch {
    const homeFileSystem = await statfs(os.homedir(), { bigint: true })
    return Number(homeFileSystem.bsize * homeFileSystem.bavail)
  }
}

export async function detectPortableCapabilities(
  storagePath: string
): Promise<Omit<DeviceCapabilities, 'accelerators' | 'detectionWarnings'>> {
  const cpuInfo = os.cpus()
  const storageAvailableBytes = await availableStorageBytes(storagePath)

  return {
    platform: normalizePlatform(process.platform),
    platformRelease: os.release(),
    architecture: os.arch(),
    cpu: {
      model: cpuInfo[0]?.model?.trim() || 'Unknown CPU',
      logicalCores: Math.max(1, cpuInfo.length),
      availableParallelism: Math.max(1, os.availableParallelism())
    },
    memory: {
      totalBytes: os.totalmem()
    },
    storage: {
      availableBytes: storageAvailableBytes
    }
  }
}
