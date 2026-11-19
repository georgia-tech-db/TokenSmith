import type { CourseMaterial, MaterialIndexProgress } from './app-state'
import { waitDisplay } from './generation-estimate'

export function preparationProgress(status?: MaterialIndexProgress, sinceUpdateMs = 0) {
  const staged = Boolean(status?.stage)
  const percent = status?.stagePercent ?? (staged ? undefined : status?.percent ?? 0)
  const label = status?.stage || 'Overall preparation'
  const remaining = status?.estimatedRemainingMs
  let time = ''
  if (staged && status?.stage !== 'Model setup' && (percent ?? 0) < 100) {
    time = remaining !== undefined && remaining > 0
      ? waitDisplay({ expectedMs: remaining, lowerMs: remaining * .6, upperMs: remaining * 1.4, samples: 1 }, sinceUpdateMs).text
      : 'Estimating time…'
  }
  return { percent, label, text: `${staged ? label : ''}${percent === undefined ? '' : `${staged ? ' · ' : ''}${Math.round(percent)}%${staged ? '' : ' overall'}`}${time ? ` · ${time}` : ''}` }
}

export function normalizeIndexingProgress(material: CourseMaterial): MaterialIndexProgress | undefined {
  if (material.status !== 'indexing' && material.status !== 'paused') {
    return material.indexing
  }

  const progress = material.indexing as (Partial<MaterialIndexProgress> & { phase?: string }) | undefined
  const rawPhase = progress?.phase as string | undefined
  const materialId = progress?.materialId ?? material.id
  const migratedPhase: MaterialIndexProgress['phase'] =
    rawPhase === 'chunking' ||
    rawPhase === 'embedding' ||
    rawPhase === 'saving' ||
    rawPhase === 'topics' ||
    rawPhase === 'complete' ||
    rawPhase === 'error'
      ? rawPhase
      : 'parsing'
  const rawMessage = progress?.message
  const migratedMessage =
    rawPhase === 'queued'
      ? 'Parsing'
      : rawPhase === 'extracting'
        ? rawMessage?.replace(/^Reading/, 'Parsing') || 'Parsing'
        : rawMessage || 'Parsing'

  return {
    ...progress,
    materialId,
    phase: migratedPhase,
    percent: Math.max(rawPhase === 'queued' ? 1 : 0, Math.min(100, Math.round(progress?.percent ?? 1))),
    processedFiles: progress?.processedFiles ?? 0,
    totalFiles: progress?.totalFiles ?? material.fileCount ?? 0,
    processedEmbeddings: progress?.processedEmbeddings ?? 0,
    totalEmbeddings: progress?.totalEmbeddings ?? 0,
    message: migratedMessage
  }
}
