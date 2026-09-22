import type { AcceleratorCapabilities } from '../../../shared/device-capabilities'
import type { ElectronGpuDevice } from './electron-gpu'
import { runDetectorCommand } from './command'

interface MacosDisplayRecord {
  name: string
  metalSupported: boolean
  dedicatedMemoryBytes: number | null
  sharedMemoryBytes: number | null
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return typeof value === 'object' && value !== null ? (value as Record<string, unknown>) : null
}

function memoryBytes(value: unknown): number | null {
  if (typeof value !== 'string') {
    return null
  }

  const match = value.match(/([\d.]+)\s*(KB|MB|GB|TB)/i)
  if (!match) {
    return null
  }

  const amount = Number.parseFloat(match[1])
  const unit = match[2].toUpperCase()
  const multiplier = { KB: 1024, MB: 1024 ** 2, GB: 1024 ** 3, TB: 1024 ** 4 }[unit]
  return Number.isFinite(amount) && multiplier ? Math.round(amount * multiplier) : null
}

export function parseMacosDisplays(systemProfilerJson: string | null): MacosDisplayRecord[] {
  if (!systemProfilerJson) {
    return []
  }

  try {
    const parsed = asRecord(JSON.parse(systemProfilerJson))
    const displays = Array.isArray(parsed?.SPDisplaysDataType) ? parsed.SPDisplaysDataType : []

    return displays.flatMap((value) => {
      const display = asRecord(value)
      if (!display) {
        return []
      }

      const name = display.sppci_model ?? display._name
      return [
        {
          name: typeof name === 'string' && name.trim() ? name.trim() : 'Unknown GPU',
          metalSupported: display.spdisplays_metal === 'spdisplays_supported',
          dedicatedMemoryBytes: memoryBytes(display._spdisplays_vram),
          sharedMemoryBytes: memoryBytes(display.spdisplays_vram_shared)
        }
      ]
    })
  } catch {
    return []
  }
}

export async function detectMacosAccelerators(
  architecture: string,
  totalSystemMemoryBytes: number,
  electronDevices: ElectronGpuDevice[],
  systemProfilerJson?: string | null
): Promise<AcceleratorCapabilities[]> {
  const activeDevice = electronDevices.find((device) => device.active) ?? electronDevices[0]

  if (architecture === 'arm64') {
    return [
      {
        name: activeDevice?.name ?? 'Apple Silicon GPU',
        backend: 'metal',
        memoryTopology: 'unified',
        totalMemoryBytes: totalSystemMemoryBytes,
        availableMemoryBytes: null,
        runtimeSupport: 'supported',
        source: activeDevice ? 'electron+node' : 'node-platform'
      }
    ]
  }

  const profilerOutput =
    systemProfilerJson === undefined
      ? await runDetectorCommand('system_profiler', ['SPDisplaysDataType', '-json'])
      : systemProfilerJson
  const displays = parseMacosDisplays(profilerOutput)

  if (displays.length > 0) {
    return displays.map((display) => {
      const hasSharedMemory = display.sharedMemoryBytes !== null
      const isDedicated = display.dedicatedMemoryBytes !== null && !hasSharedMemory

      return {
        name: display.name,
        backend: display.metalSupported ? 'metal' : 'unknown',
        memoryTopology: isDedicated ? 'dedicated' : hasSharedMemory ? 'shared' : 'unknown',
        totalMemoryBytes: isDedicated ? display.dedicatedMemoryBytes : display.sharedMemoryBytes,
        availableMemoryBytes: null,
        // Ollama supports x86 macOS through its CPU runtime, even when the Mac has
        // Metal-capable graphics. Preserve GPU facts without using them for tiering.
        runtimeSupport: 'unsupported',
        source: 'system_profiler'
      }
    })
  }

  return electronDevices.map((device) => ({
    name: device.name,
    backend: 'unknown',
    memoryTopology: 'unknown',
    totalMemoryBytes: null,
    availableMemoryBytes: null,
    runtimeSupport: 'unsupported',
    source: 'electron'
  }))
}
