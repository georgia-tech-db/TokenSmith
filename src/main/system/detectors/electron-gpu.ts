interface UnknownRecord {
  [key: string]: unknown
}

export interface ElectronGpuDevice {
  name: string
  active: boolean
}

function asRecord(value: unknown): UnknownRecord | null {
  return typeof value === 'object' && value !== null ? (value as UnknownRecord) : null
}

export function parseElectronGpuDevices(gpuInfo: unknown): ElectronGpuDevice[] {
  const info = asRecord(gpuInfo)
  const rawDevices = Array.isArray(info?.gpuDevice) ? info.gpuDevice : []

  return rawDevices.flatMap((rawDevice) => {
    const device = asRecord(rawDevice)
    if (!device) {
      return []
    }

    const nameValue = device.deviceString ?? device.name
    const name = typeof nameValue === 'string' && nameValue.trim() ? nameValue.trim() : 'Unknown GPU'

    return [
      {
        name,
        active: device.active !== false
      }
    ]
  })
}
