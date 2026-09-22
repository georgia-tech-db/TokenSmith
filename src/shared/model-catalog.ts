export interface LocalModelProfile {
  id: string
  displayName: string
  description: string
  parameterLabel: string
  quantizationLabel: string
  artifactSizeBytes: number
  maximumContextTokens: number
}

export function formatModelArtifactSize(bytes: number): string {
  return bytes < 1_000_000_000
    ? `${Math.round(bytes / 1_000_000)} MB`
    : `${Math.round((bytes / 1_000_000_000) * 10) / 10} GB`
}

// Objective metadata for the exact Ollama artifacts in the recommendation universe.
// Device thresholds belong to device-tier-policy.ts, not this catalog.
export const localModelCatalog: LocalModelProfile[] = [
  {
    id: 'gemma3:4b-it-q4_K_M',
    displayName: 'Gemma 3 4B Q4',
    description: 'Lightweight Gemma option for CPU-only study questions.',
    parameterLabel: '4B',
    quantizationLabel: 'Q4',
    artifactSizeBytes: 3_300_000_000,
    maximumContextTokens: 131_072
  },
  {
    id: 'gemma3:4b-it-q8_0',
    displayName: 'Gemma 3 4B Q8',
    description: 'Higher-precision Gemma option for stronger CPU-only and accelerated devices.',
    parameterLabel: '4B',
    quantizationLabel: 'Q8_0',
    artifactSizeBytes: 5_000_000_000,
    maximumContextTokens: 131_072
  },
  {
    id: 'gemma3:12b-it-q4_K_M',
    displayName: 'Gemma 3 12B Q4',
    description: 'Accelerated Gemma option with stronger study-task capacity.',
    parameterLabel: '12B',
    quantizationLabel: 'Q4_K_M',
    artifactSizeBytes: 8_100_000_000,
    maximumContextTokens: 131_072
  },
  {
    id: 'gemma3:12b-it-q8_0',
    displayName: 'Gemma 3 12B Q8',
    description: 'Higher-precision Gemma option for high-memory accelerators.',
    parameterLabel: '12B',
    quantizationLabel: 'Q8_0',
    artifactSizeBytes: 13_000_000_000,
    maximumContextTokens: 131_072
  },
  {
    id: 'gemma3:27b-it-q8_0',
    displayName: 'Gemma 3 27B Q8',
    description: 'High-capacity Gemma option for accelerated workstations.',
    parameterLabel: '27B',
    quantizationLabel: 'Q8_0',
    artifactSizeBytes: 30_000_000_000,
    maximumContextTokens: 131_072
  }
]
