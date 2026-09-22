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

test('assigns Tier 0 and cloud when no local tier fits', () => {
  const recommendation = recommendationFor(device({ ramGb: 4 }))

  assert.equal(recommendation.kind, 'cloud')
  assert.equal(recommendation.recommendedModelId, 'gemini:gemini-2.5-flash')
  assert.equal(recommendation.recommendedModelName, 'Gemini 2.5 Flash (Google)')
  assert.equal(recommendation.tierAssessment.tier, 0)
  assert.deepEqual(recommendation.warnings, [])
})

test('sends an 8 GiB Intel Mac to cloud instead of recommending a 4B model', () => {
  const recommendation = recommendationFor(device({
    ramGb: 8,
    cpuThreads: 8,
    platform: 'macos',
    platformRelease: '24.6.0',
    architecture: 'x64'
  }))

  assert.equal(recommendation.kind, 'cloud')
  assert.equal(recommendation.tierAssessment.tier, 0)
})

test('assigns Tier 1 to an 8 GiB Apple Silicon device', () => {
  const recommendation = recommendationFor(device({
    ramGb: 8,
    platform: 'macos',
    platformRelease: '24.6.0',
    architecture: 'arm64',
    accelerators: [unifiedAccelerator(8)]
  }))

  assert.equal(recommendation.recommendedModelId, 'gemma3:4b-it-q4_K_M')
  assert.equal(recommendation.tierAssessment.tier, 1)
  assert.equal(recommendation.tierAssessment.executionPath, 'unified-gpu')
})

test('assigns Tier 1 to a 16 GiB CPU device with eight threads', () => {
  const recommendation = recommendationFor(device({ ramGb: 16, cpuThreads: 8 }))

  assert.equal(recommendation.recommendedModelId, 'gemma3:4b-it-q4_K_M')
  assert.equal(recommendation.tierAssessment.tier, 1)
  assert.equal(recommendation.tierAssessment.executionPath, 'cpu')
})

test('sends a 16 GiB seven-thread CPU device to cloud', () => {
  const recommendation = recommendationFor(device({ ramGb: 16, cpuThreads: 7 }))

  assert.equal(recommendation.tierAssessment.tier, 0)
  assert.equal(
    recommendation.tierAssessment.evaluatedTiers[0].rejectionReasons.some((reason) =>
      reason.includes('8 available threads')
    ),
    true
  )
})

test('assigns Tier 2 to a 24 GiB CPU device with twelve threads', () => {
  const recommendation = recommendationFor(device({ ramGb: 24, cpuThreads: 12 }))

  assert.equal(recommendation.recommendedModelId, 'gemma3:4b-it-q8_0')
  assert.equal(recommendation.tierAssessment.tier, 2)
})

test('caps CPU-only devices at Tier 2', () => {
  const recommendation = recommendationFor(device({ ramGb: 128, cpuThreads: 32 }))

  assert.equal(recommendation.tierAssessment.tier, 2)
})

test('assigns Tier 3 to a 20 GiB unified-memory device', () => {
  const recommendation = recommendationFor(
    device({ ramGb: 20, accelerators: [unifiedAccelerator(20)] })
  )

  assert.equal(recommendation.recommendedModelId, 'gemma3:12b-it-q4_K_M')
  assert.equal(recommendation.tierAssessment.tier, 3)
})

test('assigns Tier 4 to a 32 GiB unified-memory device', () => {
  const recommendation = recommendationFor(
    device({ ramGb: 32, accelerators: [unifiedAccelerator(32)] })
  )

  assert.equal(recommendation.recommendedModelId, 'gemma3:12b-it-q8_0')
  assert.equal(recommendation.tierAssessment.tier, 4)
  assert.equal(recommendation.tierAssessment.executionPath, 'unified-gpu')
})

test('assigns Tier 5 to a 64 GiB unified-memory device', () => {
  const recommendation = recommendationFor(
    device({ ramGb: 64, accelerators: [unifiedAccelerator(64)] })
  )

  assert.equal(recommendation.recommendedModelId, 'gemma3:27b-it-q8_0')
  assert.equal(recommendation.tierAssessment.tier, 5)
})

test('uses supported dedicated VRAM for Tier 2', () => {
  const recommendation = recommendationFor(
    device({ ramGb: 16, accelerators: [dedicatedAccelerator(8)] })
  )

  assert.equal(recommendation.tierAssessment.tier, 2)
  assert.equal(recommendation.tierAssessment.executionPath, 'dedicated-gpu')
})

test('requires both Tier 5 host memory and dedicated VRAM', () => {
  const recommendation = recommendationFor(
    device({ ramGb: 16, accelerators: [dedicatedAccelerator(48)] })
  )

  assert.equal(recommendation.tierAssessment.tier, 4)
})

test('falls back to the highest tier with enough storage', () => {
  const recommendation = recommendationFor(
    device({ ramGb: 16, diskGb: 5, accelerators: [unifiedAccelerator(16)] })
  )

  assert.equal(recommendation.tierAssessment.tier, 1)
})

test('does not use an unverified GPU for tier selection', () => {
  const recommendation = recommendationFor(
    device({ ramGb: 24, accelerators: [dedicatedAccelerator(48, 48, 'unverified')] })
  )

  assert.equal(recommendation.tierAssessment.tier, 1)
  assert.equal(recommendation.tierAssessment.executionPath, 'cpu')
})

test('does not count shared graphics memory as separate model memory', () => {
  const recommendation = recommendationFor(
    device({
      ramGb: 24,
      accelerators: [{
        name: 'Shared GPU',
        backend: 'unknown',
        memoryTopology: 'shared',
        totalMemoryBytes: gibibytes(16),
        availableMemoryBytes: null,
        runtimeSupport: 'supported',
        source: 'test'
      }]
    })
  )

  assert.equal(recommendation.tierAssessment.tier, 1)
  assert.equal(recommendation.tierAssessment.executionPath, 'cpu')
})

test('assigns Tier 0 to unsupported operating systems', () => {
  const recommendation = recommendationFor(
    device({ ramGb: 64, platform: 'unknown', architecture: 'x64' })
  )

  assert.equal(recommendation.tierAssessment.tier, 0)
  assert.equal(
    recommendation.tierAssessment.evaluatedTiers.every((tier) =>
      tier.rejectionReasons.some((reason) => reason.includes('not supported'))
    ),
    true
  )
})

test('assigns Tier 0 to macOS versions below the Ollama minimum', () => {
  const recommendation = recommendationFor(
    device({
      ramGb: 64,
      platform: 'macos',
      platformRelease: '22.6.0',
      architecture: 'arm64',
      accelerators: [unifiedAccelerator(64)]
    })
  )

  assert.equal(recommendation.tierAssessment.tier, 0)
  assert.match(recommendation.reasons[0], /below the local Ollama requirement/i)
})

test('allows Windows ARM to qualify through the CPU path', () => {
  const recommendation = recommendationFor(
    device({
      ramGb: 64,
      platform: 'windows',
      platformRelease: '10.0.26100',
      architecture: 'arm64'
    })
  )

  assert.equal(recommendation.tierAssessment.tier, 1)
  assert.equal(recommendation.tierAssessment.executionPath, 'cpu')
})

test('assigns Tier 0 below the Tier 1 memory boundary', () => {
  const recommendation = recommendationFor(device({ ramGb: 15.9, cpuThreads: 8 }))

  assert.equal(recommendation.tierAssessment.tier, 0)
})

test('keeps a device below the Tier 2 memory boundary in Tier 1', () => {
  const recommendation = recommendationFor(device({ ramGb: 23.9, cpuThreads: 12 }))

  assert.equal(recommendation.tierAssessment.tier, 1)
})

test('assigns Tier 0 below the minimum Windows build', () => {
  const recommendation = recommendationFor(
    device({
      ramGb: 64,
      platform: 'windows',
      platformRelease: '10.0.19044',
      architecture: 'x64'
    })
  )

  assert.equal(recommendation.tierAssessment.tier, 0)
})

test('does not expose a confidence field', () => {
  const recommendation = recommendationFor(device({ ramGb: 8 }))

  assert.equal('confidence' in recommendation, false)
  assert.equal('confidence' in recommendation.tierAssessment, false)
})

test('keeps exactly five Gemma local tiers with a 4B entry model', () => {
  const modelIds = new Set(localModelCatalog.map((model) => model.id))

  assert.equal(localModelCatalog.length, 5)
  assert.equal(defaultDeviceTierPolicy.tiers.length, 5)
  assert.equal(localModelCatalog.every((model) => model.id.startsWith('gemma3:')), true)
  assert.equal(localModelCatalog.some((model) => model.parameterLabel === '1B'), false)
  assert.equal(localModelCatalog.every((model) => Number.parseInt(model.parameterLabel, 10) >= 4), true)
  assert.equal(
    defaultDeviceTierPolicy.tiers.every((tier) => modelIds.has(tier.recommendedModelId)),
    true
  )
})

test('records the fixed recommendation context without changing runtime settings', () => {
  const recommendation = recommendationFor(device({ ramGb: 8 }))

  assert.equal(recommendation.assumedContextTokens, 2048)
})
