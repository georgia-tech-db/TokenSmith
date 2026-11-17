import { useCallback, useEffect, useRef, useState } from 'react'
import type { ChatSource } from '@shared/app-state'
import type { AnswerSourceDocument } from '../answer-source-preview'
import type { SourceNavigationProps } from '../SourceNavigation'
import { sourceNavigationList } from '../source-navigation'

export interface SourceReaderState {
  document: AnswerSourceDocument
  sources: ChatSource[]
  index: number
}
export interface SourceReaderProps {
  state: SourceReaderState
  navigation?: SourceNavigationProps
  onClose: () => void
}
export function useSourceReader(load: (source: ChatSource) => Promise<AnswerSourceDocument>) {
  const [state, setState] = useState<SourceReaderState | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const sequence = useRef(0)
  useEffect(() => () => { sequence.current++ }, [])
  const close = useCallback(() => {
    sequence.current++
    setState(null)
    setLoading(false)
    setError('')
  }, [])
  const open = useCallback(async (source: ChatSource, candidates: ChatSource[]) => {
    const navigation = sourceNavigationList(source, candidates)
    if (navigation.index < 0) return
    const current = ++sequence.current
    setLoading(true)
    setError('')
    try {
      const document = await load(source)
      if (sequence.current !== current) return
      setState({ document, ...navigation })
    } catch (reason) {
      if (sequence.current === current) setError(reason instanceof Error ? reason.message : 'Could not open this source.')
    } finally {
      if (sequence.current === current) setLoading(false)
    }
  }, [load])
  const move = (direction: number) => {
    if (!state || loading) return
    const source = state.sources[state.index + direction]
    if (source) void open(source, state.sources)
  }
  const navigation: SourceNavigationProps | undefined = state ? {
    index: state.index, count: state.sources.length, loading, error,
    onPrevious: () => move(-1), onNext: () => move(1)
  } : undefined
  return { state, open, close, navigation, error, loading }
}
