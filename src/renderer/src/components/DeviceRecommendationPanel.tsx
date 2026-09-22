import { Loader2, Sparkles } from 'lucide-react'
import {
  bytesToRoundedGibibytes,
  type AcceleratorCapabilities,
  type DeviceCapabilities
} from '@shared/device-capabilities'
import type { ModelRecommendation } from '@shared/model-recommendation'
import type { DeviceCapabilityState } from '../hooks/useDeviceCapabilities'

interface DeviceRecommendationPanelProps {
  capabilities: DeviceCapabilities | null
  error: string | null
  recommendation: ModelRecommendation | null
  state: DeviceCapabilityState
  onRefresh: () => void
}

const executionPathLabels = {
  cloud: 'Cloud',
  cpu: 'CPU',
  'dedicated-gpu': 'Dedicated GPU',
  'unified-gpu': 'Unified GPU'
} as const

function gibibytes(bytes: number | null): string {
  return bytes === null ? 'Unavailable' : `${bytesToRoundedGibibytes(bytes)} GiB`
}

function acceleratorSummary(accelerator: AcceleratorCapabilities): string {
  const support =
    accelerator.runtimeSupport === 'supported'
      ? 'Supported by Ollama'
      : accelerator.runtimeSupport === 'unsupported'
        ? 'Not supported by Ollama'
        : 'Ollama support could not be verified'

  return `${accelerator.name} · ${support}`
}

export function DeviceRecommendationPanel({
  capabilities,
  error,
  recommendation,
  state,
  onRefresh
}: DeviceRecommendationPanelProps) {
  if (state === 'checking') {
    return (
      <section className="model-recommendation-panel is-loading" aria-live="polite">
        <Loader2 size={20} aria-hidden="true" className="spin" />
        <div>
          <strong>Checking this device</strong>
          <p>Inspecting memory, processor capacity, and available acceleration.</p>
        </div>
      </section>
    )
  }

  if (state === 'error' || !capabilities || !recommendation) {
    return (
      <section className="model-recommendation-panel is-unavailable" aria-live="polite">
        <div>
          <strong>Device recommendation unavailable</strong>
          <p>{error ?? 'TokenSmith could not inspect this device.'}</p>
        </div>
      </section>
    )
  }

  const assessment = recommendation.tierAssessment
  const isCloud = recommendation.kind === 'cloud'
  const detail = isCloud
    ? `Suggested model: ${recommendation.recommendedModelName}`
    : `Expected execution: ${executionPathLabels[assessment.executionPath]}`

  return (
    <section
      className={`model-recommendation-panel ${isCloud ? 'is-cloud' : ''}`}
      aria-live="polite"
    >
      <div className="model-recommendation-summary">
        <div className="model-recommendation-icon" aria-hidden="true">
          <Sparkles size={22} />
        </div>
        <div className="model-recommendation-copy">
          <span>Device Tier {assessment.tier} · {assessment.name}</span>
          <h2>{isCloud ? 'Cloud model recommended' : recommendation.recommendedModelName}</h2>
          <p>{recommendation.reasons[0]}</p>
          <small>{detail}</small>
        </div>
        <div className="model-recommendation-actions">
          <button className="secondary-action" type="button" onClick={onRefresh}>
            <span>Recheck device</span>
          </button>
        </div>
      </div>

      {recommendation.warnings.length > 0 && (
        <ul className="model-recommendation-warnings">
          {recommendation.warnings.map((warning) => <li key={warning}>{warning}</li>)}
        </ul>
      )}

      <details className="device-details">
        <summary>Detected device details</summary>
        <dl>
          <div><dt>Platform</dt><dd>{capabilities.platform} {capabilities.architecture} · {capabilities.platformRelease}</dd></div>
          <div><dt>Processor</dt><dd>{capabilities.cpu.model}</dd></div>
          <div><dt>CPU</dt><dd>{capabilities.cpu.logicalCores} logical cores · {capabilities.cpu.availableParallelism} available threads</dd></div>
          <div><dt>Memory</dt><dd>{gibibytes(capabilities.memory.totalBytes)} total</dd></div>
          <div><dt>Storage</dt><dd>{gibibytes(capabilities.storage.availableBytes)} available</dd></div>
          <div>
            <dt>Graphics</dt>
            <dd>{capabilities.accelerators.length > 0 ? capabilities.accelerators.map(acceleratorSummary).join('; ') : 'No graphics accelerator detected'}</dd>
          </div>
        </dl>
      </details>
    </section>
  )
}
