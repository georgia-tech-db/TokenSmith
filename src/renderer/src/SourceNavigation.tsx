import { ChevronLeft, ChevronRight } from 'lucide-react'

export interface SourceNavigationProps {
  index: number
  count: number
  loading?: boolean
  error?: string
  onPrevious: () => void
  onNext: () => void
}
export function SourceNavigation({ navigation }: { navigation?: SourceNavigationProps }) {
  if (!navigation || navigation.count < 2) return null
  return <div className="source-navigation" role="group" aria-label="Source navigation">
    <button type="button" aria-label="Previous source" title="Previous source" disabled={navigation.loading || navigation.index <= 0} onClick={navigation.onPrevious}>
      <ChevronLeft size={16} aria-hidden="true" /><span>Previous</span>
    </button>
    <span className="source-navigation-position" aria-live="polite">{navigation.loading ? 'Opening…' : `Source ${navigation.index + 1} of ${navigation.count}`}</span>
    <button type="button" aria-label="Next source" title="Next source" disabled={navigation.loading || navigation.index >= navigation.count - 1} onClick={navigation.onNext}>
      <span>Next</span><ChevronRight size={16} aria-hidden="true" />
    </button>
    {navigation.error && <span className="source-navigation-error" role="alert">{navigation.error}</span>}
  </div>
}
