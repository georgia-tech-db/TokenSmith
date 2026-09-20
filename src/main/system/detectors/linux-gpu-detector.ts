import type { AcceleratorCapabilities } from '../../../shared/device-capabilities'
import type { ElectronGpuDevice } from './electron-gpu'
import { runDetectorCommand } from './command'
import { detectNvidiaAccelerators } from './nvidia-gpu'

interface UnknownRecord {
  [key: string]: unknown
}

function findNumericValue(record: UnknownRecord, keyPart: string): number | null {
  for (const [key, value] of Object.entries(record)) {
    if (key.toLowerCase().includes(keyPart) && Number.isFinite(Number(value))) {
      return Number(value)
    }
  }

  return null
}

export function parseRocmSmiJson(output: string): AcceleratorCapabilities[] {
  try {
    const parsed = JSON.parse(output) as UnknownRecord

    return Object.entries(parsed).flatMap(([cardName, rawCard]) => {
      if (typeof rawCard !== 'object' || rawCard === null) {
        return []
      }

      const card = rawCard as UnknownRecord
      const totalMemoryBytes = findNumericValue(card, 'vram total memory')
      const usedMemoryBytes = findNumericValue(card, 'vram total used')
      if (totalMemoryBytes === null) {
        return []
      }

      const productName = Object.entries(card).find(([key]) =>
        key.toLowerCase().includes('card series')
      )?.[1]

      return [
        {
          name: typeof productName === 'string' ? productName : cardName,
          backend: 'rocm' as const,
          memoryTopology: 'dedicated' as const,
          totalMemoryBytes,
          availableMemoryBytes:
            usedMemoryBytes === null ? null : Math.max(0, totalMemoryBytes - usedMemoryBytes),
          runtimeSupport: 'supported' as const,
          source: 'rocm-smi'
        }
      ]
    })
  } catch {
    return []
  }
}

async function detectRocmAccelerators(): Promise<AcceleratorCapabilities[]> {
  const output = await runDetectorCommand('rocm-smi', ['--showproductname', '--showmeminfo', 'vram', '--json'])
  return output ? parseRocmSmiJson(output) : []
}

export async function detectLinuxAccelerators(
  electronDevices: ElectronGpuDevice[]
): Promise<AcceleratorCapabilities[]> {
  const nvidiaAccelerators = await detectNvidiaAccelerators()
  if (nvidiaAccelerators.length > 0) {
    return nvidiaAccelerators
  }

  const rocmAccelerators = await detectRocmAccelerators()
  if (rocmAccelerators.length > 0) {
    return rocmAccelerators
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
