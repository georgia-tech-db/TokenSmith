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
}

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
  state
}: DeviceRecommendationPanelProps) {
  if (state === 'checking') {
    return (
      <section className="model-recommendation-panel is-loading" aria-live="polite">
        <Loader2 size={20} aria-hidden="true" className="spin" />
        <div>
          <strong>Checking this device</strong>
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

  const isCloud = recommendation.kind === 'cloud'

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
          <span>{isCloud ? 'Cloud option' : 'Recommended model'}</span>
          <h2>{recommendation.recommendedModelName}</h2>
        </div>
      </div>

      {recommendation.alternatives.length > 0 && <details className="device-details">
        <summary>Other local options</summary>
        {recommendation.alternatives.map((model) => <p key={model.id}>{model.displayName}</p>)}
      </details>}
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
