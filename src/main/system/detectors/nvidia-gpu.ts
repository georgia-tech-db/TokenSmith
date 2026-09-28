import type { AcceleratorCapabilities } from '../../../shared/device-capabilities'
import { runDetectorCommand } from './command'

const bytesPerMebibyte = 1024 ** 2

export function parseNvidiaSmiCsv(output: string): AcceleratorCapabilities[] {
  return output
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .flatMap((line) => {
      const fields = line.split(',').map((field) => field.trim())
      if (fields.length < 3) {
        return []
      }

      const availableMemoryMib = Number(fields.pop())
      const totalMemoryMib = Number(fields.pop())
      const name = fields.join(', ')

      if (!name || !Number.isFinite(totalMemoryMib) || !Number.isFinite(availableMemoryMib)) {
        return []
      }

      return [
        {
          name,
          backend: 'cuda' as const,
          memoryTopology: 'dedicated' as const,
          totalMemoryBytes: totalMemoryMib * bytesPerMebibyte,
          availableMemoryBytes: availableMemoryMib * bytesPerMebibyte,
          runtimeSupport: 'supported' as const,
          source: 'nvidia-smi'
        }
      ]
    })
}

export async function detectNvidiaAccelerators(): Promise<AcceleratorCapabilities[]> {
  const output = await runDetectorCommand('nvidia-smi', [
    '--query-gpu=name,memory.total,memory.free',
    '--format=csv,noheader,nounits'
  ])

  return output ? parseNvidiaSmiCsv(output) : []
}
