import { useMemo, useState } from 'react'
import { createSourcePane, type SourcePaneState } from '../answer-source-preview'

export function useSourcePane() {
  const [state, setState] = useState<SourcePaneState | null>(null)
  const controller = useMemo(() => createSourcePane(setState), [])
  return { state, controller }
}
