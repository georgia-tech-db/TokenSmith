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

// Approximate download sizes, not runtime memory. Ollama tags checked 2026-09-28.
// These Q4_K_M packages match the local BuzzDB comparison; aliases can change.
export const localModelCatalog: LocalModelProfile[] = [
  {
    id: 'gemma4:e4b', displayName: 'Gemma 4 E4B',
    description: 'Lighter, faster option. In our small trial, E4B answered faster than 12B but made more errors.',
    parameterLabel: 'E4B', quantizationLabel: 'Q4_K_M',
    artifactSizeBytes: 9_600_000_000, maximumContextTokens: 131_072
  },
  {
    id: 'gemma4:12b', displayName: 'Gemma 4 12B',
    description: 'Recommended when memory allows: fewer errors in our small trial, but answers took about two minutes on the test Mac.',
    parameterLabel: '12B', quantizationLabel: 'Q4_K_M',
    artifactSizeBytes: 7_600_000_000, maximumContextTokens: 262_144
  },
  {
    id: 'gemma4:26b', displayName: 'Gemma 4 26B A4B',
    description: 'Optional: more complete answers in our trial, with a much larger memory footprint. Try with your other apps open.',
    parameterLabel: '26B total / 4B active', quantizationLabel: 'Q4_K_M',
    artifactSizeBytes: 19_000_000_000, maximumContextTokens: 262_144
  }
]
