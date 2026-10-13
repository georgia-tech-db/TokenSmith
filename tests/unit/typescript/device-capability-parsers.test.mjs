import assert from 'node:assert/strict'
import { test } from 'node:test'
import { requireTranspiledTs } from './ts-module-loader.mjs'

const { parseElectronGpuDevices } = requireTranspiledTs('src/main/system/detectors/electron-gpu.ts')
const { parseNvidiaSmiCsv } = requireTranspiledTs('src/main/system/detectors/nvidia-gpu.ts')
const { parseRocmSmiJson } = requireTranspiledTs('src/main/system/detectors/linux-gpu-detector.ts')
const { applyWindowsRuntimeSupport } = requireTranspiledTs(
  'src/main/system/detectors/windows-gpu-detector.ts'
)
const { detectMacosAccelerators, parseMacosDisplays } = requireTranspiledTs(
  'src/main/system/detectors/macos-gpu-detector.ts'
)
const { normalizePlatform } = requireTranspiledTs('src/main/system/detectors/portable-detector.ts')
const { acceleratorDetectionWarning } = requireTranspiledTs(
  'src/main/system/detect-device-capabilities.ts'
)

test('normalizes supported Node platform names', () => {
  assert.equal(normalizePlatform('win32'), 'windows')
  assert.equal(normalizePlatform('darwin'), 'macos')
  assert.equal(normalizePlatform('linux'), 'linux')
  assert.equal(normalizePlatform('freebsd'), 'unknown')
})

test('parses Electron GPU devices without assuming optional fields exist', () => {
  const devices = parseElectronGpuDevices({
    gpuDevice: [
      {
        active: true,
        deviceString: 'Example GPU',
        vendorId: 4318,
        deviceId: 1234
      }
    ]
  })

  assert.deepEqual(devices, [
    {
      name: 'Example GPU',
      active: true
    }
  ])
})

test('parses NVIDIA memory values reported in MiB', () => {
  const accelerators = parseNvidiaSmiCsv('NVIDIA RTX Test, 12288, 10240')

  assert.equal(accelerators.length, 1)
  assert.equal(accelerators[0].backend, 'cuda')
  assert.equal(accelerators[0].memoryTopology, 'dedicated')
  assert.equal(accelerators[0].totalMemoryBytes, 12288 * 1024 ** 2)
  assert.equal(accelerators[0].availableMemoryBytes, 10240 * 1024 ** 2)
})

test('parses ROCm total and used VRAM values', () => {
  const accelerators = parseRocmSmiJson(
    JSON.stringify({
      card0: {
        'Card series': 'AMD Radeon Test',
        'VRAM Total Memory (B)': '17179869184',
        'VRAM Total Used Memory (B)': '2147483648'
      }
    })
  )

  assert.equal(accelerators.length, 1)
  assert.equal(accelerators[0].backend, 'rocm')
  assert.equal(accelerators[0].totalMemoryBytes, 17179869184)
  assert.equal(accelerators[0].availableMemoryBytes, 15032385536)
})

test('returns no accelerators for malformed detector output', () => {
  assert.deepEqual(parseNvidiaSmiCsv('not enough fields'), [])
  assert.deepEqual(parseRocmSmiJson('not json'), [])
})

test('models Apple Silicon memory as a unified Metal accelerator', async () => {
  const accelerators = await detectMacosAccelerators('arm64', 16 * 1024 ** 3, [
    {
      name: 'Apple M-series GPU',
      active: true
    }
  ])

  assert.equal(accelerators.length, 1)
  assert.equal(accelerators[0].backend, 'metal')
  assert.equal(accelerators[0].memoryTopology, 'unified')
  assert.equal(accelerators[0].totalMemoryBytes, 16 * 1024 ** 3)
  assert.equal(accelerators[0].runtimeSupport, 'supported')
})

test('reads shared Intel graphics memory without treating it as separate VRAM', () => {
  const displays = parseMacosDisplays(
    JSON.stringify({
      SPDisplaysDataType: [
        {
          sppci_model: 'Intel Iris Plus Graphics 645',
          spdisplays_vendor: 'Intel',
          spdisplays_metal: 'spdisplays_supported',
          _spdisplays_vram: '1536 MB',
          spdisplays_vram_shared: '1536 MB'
        }
      ]
    })
  )

  assert.deepEqual(displays, [
    {
      name: 'Intel Iris Plus Graphics 645',
      metalSupported: true,
      dedicatedMemoryBytes: 1536 * 1024 ** 2,
      sharedMemoryBytes: 1536 * 1024 ** 2
    }
  ])
})

test('keeps an Intel Mac graphics device on the CPU/RAM recommendation path', async () => {
  const accelerators = await detectMacosAccelerators('x64', 16 * 1024 ** 3, [
    {
      name: 'Intel Graphics',
      active: true
    }
  ], JSON.stringify({
    SPDisplaysDataType: [
      {
        sppci_model: 'Intel Iris Graphics',
        spdisplays_vendor: 'Intel',
        spdisplays_metal: 'spdisplays_supported',
        _spdisplays_vram: '1536 MB',
        spdisplays_vram_shared: '1536 MB'
      }
    ]
  }))

  assert.equal(accelerators[0].runtimeSupport, 'unsupported')
  assert.equal(accelerators[0].backend, 'metal')
  assert.equal(accelerators[0].memoryTopology, 'shared')
})

test('keeps a discrete Intel Mac GPU visible but unsupported by the Ollama runtime', async () => {
  const accelerators = await detectMacosAccelerators('x64', 32 * 1024 ** 3, [], JSON.stringify({
    SPDisplaysDataType: [
      {
        sppci_model: 'AMD Radeon Pro',
        spdisplays_metal: 'spdisplays_supported',
        _spdisplays_vram: '8 GB'
      }
    ]
  }))

  assert.equal(accelerators[0].memoryTopology, 'dedicated')
  assert.equal(accelerators[0].totalMemoryBytes, 8 * 1024 ** 3)
  assert.equal(accelerators[0].runtimeSupport, 'unsupported')
})

test('marks Windows ARM accelerator support as unverified', () => {
  const accelerators = applyWindowsRuntimeSupport('arm64', [
    {
      name: 'NVIDIA Test GPU',
      backend: 'cuda',
      memoryTopology: 'dedicated',
      totalMemoryBytes: 12 * 1024 ** 3,
      availableMemoryBytes: 10 * 1024 ** 3,
      runtimeSupport: 'supported',
      source: 'test'
    }
  ])

  assert.equal(accelerators[0].runtimeSupport, 'unverified')
})

test('warns only when accelerator runtime support is unverified', () => {
  const accelerator = {
    name: 'Test GPU',
    backend: 'unknown',
    memoryTopology: 'unknown',
    totalMemoryBytes: null,
    availableMemoryBytes: null,
    source: 'test'
  }

  assert.equal(
    acceleratorDetectionWarning([{ ...accelerator, runtimeSupport: 'unsupported' }]),
    null
  )
  assert.match(
    acceleratorDetectionWarning([{ ...accelerator, runtimeSupport: 'unverified' }]),
    /could not be verified/i
  )
})
