export interface EmbeddingOptions {
  embeddingGpuEnabled?: boolean
}

export function normalizeEmbeddingGpuEnabled(value: unknown): boolean {
  return value !== false
}
