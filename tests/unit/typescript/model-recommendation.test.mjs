import assert from 'node:assert/strict'
import { test } from 'node:test'
import { requireTranspiledTs } from './ts-module-loader.mjs'

const { gibibytes } = requireTranspiledTs('src/shared/device-capabilities.ts')
const { defaultDeviceTierPolicy } = requireTranspiledTs('src/shared/device-tier-policy.ts')
const { localModelCatalog } = requireTranspiledTs('src/shared/model-catalog.ts')
const { recommendModel } = requireTranspiledTs('src/shared/model-recommendation.ts')

function device({
  ramGb,
  cpuThreads = 8,
  diskGb = 100,
  accelerators = [],
  detectionWarnings = [],
  platform = 'linux',
  platformRelease = '6.8.0',
  architecture = 'x64'
}) {
  return {
    platform,
    platformRelease,
    architecture,
    cpu: {
      model: 'Test CPU',
      logicalCores: cpuThreads,
      availableParallelism: cpuThreads
    },
    memory: {
      totalBytes: gibibytes(ramGb)
    },
    storage: {
      availableBytes: gibibytes(diskGb)
    },
    accelerators,
    detectionWarnings
  }
}

function unifiedAccelerator(ramGb) {
  return {
    name: 'Test unified GPU',
    backend: 'metal',
    memoryTopology: 'unified',
    totalMemoryBytes: gibibytes(ramGb),
    availableMemoryBytes: gibibytes(ramGb),
    runtimeSupport: 'supported',
    source: 'test'
  }
}

function dedicatedAccelerator(totalVramGb, availableVramGb = totalVramGb, runtimeSupport = 'supported') {
  return {
    name: 'Test dedicated GPU',
    backend: 'cuda',
    memoryTopology: 'dedicated',
    totalMemoryBytes: gibibytes(totalVramGb),
    availableMemoryBytes: gibibytes(availableVramGb),
    runtimeSupport,
    source: 'test'
  }
}

function recommendationFor(testDevice) {
  return recommendModel(testDevice, localModelCatalog, defaultDeviceTierPolicy)
}


test('small machines receive an optional cloud recommendation', () => {
  for (const ramGb of [4, 8]) {
    const result = recommendationFor(device({ramGb, accelerators: [unifiedAccelerator(ramGb)]}))
    assert.equal(result.kind, 'cloud')
    assert.deepEqual(result.alternatives, [])
  }
})

test('12B is preferred when eligible, without promoting larger models', () => {
  for (const ramGb of [24, 32, 48, 64, 128]) {
    const result = recommendationFor(device({ramGb, accelerators: [unifiedAccelerator(ramGb)]}))
    assert.equal(result.recommendedModelId, 'gemma4:12b')
    assert.equal(result.tierAssessment.executionPath, 'unified-gpu')
  }
})

test('32 GiB defaults to 12B and offers E4B, not memory-heavy 26B', () => {
  const result = recommendationFor(device({ramGb: 32, accelerators: [unifiedAccelerator(32)]}))
  assert.deepEqual(result.alternatives.map(m => m.id), ['gemma4:e4b'])
})

test('48 GiB offers both alternatives without changing the default', () => {
  const result = recommendationFor(device({ramGb: 48, accelerators: [unifiedAccelerator(48)]}))
  assert.equal(result.recommendedModelId, 'gemma4:12b')
  assert.deepEqual(result.alternatives.map(m => m.id), ['gemma4:e4b', 'gemma4:26b'])
})

test('dedicated GPUs need both host memory and VRAM headroom', () => {
  assert.equal(recommendationFor(device({ramGb: 8, accelerators: [dedicatedAccelerator(48)]})).kind, 'cloud')
  const result = recommendationFor(device({ramGb: 32, accelerators: [dedicatedAccelerator(24)]}))
  assert.equal(result.tierAssessment.executionPath, 'dedicated-gpu')
  assert.equal(result.alternatives.length, 2)
})

test('busy GPUs do not qualify using total VRAM alone', () => {
  const result = recommendationFor(device({ramGb: 16, accelerators: [dedicatedAccelerator(24, 4)]}))
  assert.equal(result.kind, 'cloud')
})

test('unknown free VRAM permits a tentative recommendation', () => {
  const gpu = dedicatedAccelerator(12)
  gpu.availableMemoryBytes = null
  const result = recommendationFor(device({ramGb: 16, accelerators: [gpu]}))
  assert.equal(result.kind, 'local')
  assert.ok(result.warnings.some(w => /estimate/.test(w)))
})

test('CPU fallback needs 24 GiB and eight threads, and discloses unmeasured latency', () => {
  assert.equal(recommendationFor(device({ramGb: 16})).kind, 'cloud')
  assert.equal(recommendationFor(device({ramGb: 24, cpuThreads: 7})).kind, 'cloud')
  const result = recommendationFor(device({ramGb: 128, cpuThreads: 32}))
  assert.equal(result.recommendedModelId, 'gemma4:e4b')
  assert.deepEqual(result.alternatives, [])
  assert.match(result.warnings.join(' '), /CPU-only answers may be slow/)
})

test('unverified and shared GPUs cannot qualify as dedicated memory', () => {
  for (const gpu of [dedicatedAccelerator(48, 48, 'unverified'), {...dedicatedAccelerator(48), memoryTopology: 'shared'}]) {
    assert.equal(recommendationFor(device({ramGb: 16, accelerators: [gpu]})).kind, 'cloud')
  }
})

test('disk includes headroom for Nomic and working files', () => {
  const base = {ramGb: 64, accelerators: [unifiedAccelerator(64)]}
  assert.equal(recommendationFor(device({...base, diskGb: 13})).kind, 'cloud')
  assert.equal(recommendationFor(device({...base, diskGb: 14})).alternatives.length, 1)
  assert.equal(recommendationFor(device({...base, diskGb: 25})).alternatives.length, 2)
})

test('unsupported platforms cannot qualify even with ample memory', () => {
  for (const extra of [{platform: 'unknown'}, {platform: 'macos', platformRelease: '22.6.0', architecture: 'arm64'}, {platform: 'windows', platformRelease: '10.0.18000'}]) {
    assert.equal(recommendationFor(device({ramGb: 64, accelerators: [unifiedAccelerator(64)], ...extra})).kind, 'cloud')
  }
})

test('Windows ARM and Intel Macs can use the conservative CPU fallback', () => {
  for (const extra of [{platform: 'windows', platformRelease: '10.0.26100', architecture: 'arm64'}, {platform: 'macos', platformRelease: '24.6.0', architecture: 'x64'}]) {
    const result = recommendationFor(device({ramGb: 32, ...extra}))
    assert.equal(result.tierAssessment.executionPath, 'cpu')
  }
})

test('catalog, fallback, and context agree with the measured setup', () => {
  const {recommendedOllamaChatModel} = requireTranspiledTs('src/shared/ollama.ts')
  assert.equal(recommendedOllamaChatModel, 'gemma4:e4b')
  assert.equal(localModelCatalog.length, 3)
  assert.equal(localModelCatalog.some(m => /31b|gemma3/.test(m.id)), false)
  assert.equal(defaultDeviceTierPolicy.assumedContextTokens, 8192)
  assert.ok(defaultDeviceTierPolicy.tiers.every(t => localModelCatalog.some(m => m.id === t.recommendedModelId)))
})


test('first-run recommendation falls back to E4B below the 12B memory threshold', () => {
  for (const ramGb of [16, 23.9]) {
    const result = recommendationFor(device({ramGb, accelerators: [unifiedAccelerator(ramGb)]}))
    assert.equal(result.recommendedModelId, 'gemma4:e4b')
    assert.deepEqual(result.alternatives, [])
  }
})

test('12B requires available VRAM as well as host RAM on dedicated GPUs', () => {
  const base = {ramGb: 24, accelerators: [dedicatedAccelerator(24, 12)]}
  assert.equal(recommendationFor(device(base)).recommendedModelId, 'gemma4:12b')
  assert.equal(recommendationFor(device({...base, accelerators: [dedicatedAccelerator(24, 10)]})).recommendedModelId, 'gemma4:e4b')
})
