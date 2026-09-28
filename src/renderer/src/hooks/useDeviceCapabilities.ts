import { useEffect, useState } from 'react'
import type { DeviceCapabilities } from '@shared/device-capabilities'

export type DeviceCapabilityState = 'idle' | 'checking' | 'error'

export function useDeviceCapabilities(active: boolean) {
  const [capabilities, setCapabilities] = useState<DeviceCapabilities | null>(null)
  const [state, setState] = useState<DeviceCapabilityState>('checking')
  const [error, setError] = useState<string | null>(null)
  const [revision, setRevision] = useState(0)

  useEffect(() => {
    if (!active) {
      return
    }

    if (!window.tokensmith?.getDeviceCapabilities) {
      setState('error')
      setError('Device inspection is only available in the desktop app.')
      return
    }

    let cancelled = false
    setState('checking')
    setError(null)

    window.tokensmith
      .getDeviceCapabilities()
      .then((detected) => {
        if (!cancelled) {
          setCapabilities(detected)
          setState('idle')
        }
      })
      .catch((reason) => {
        if (!cancelled) {
          setState('error')
          setError(reason instanceof Error ? reason.message : 'Could not inspect this device.')
        }
      })

    return () => {
      cancelled = true
    }
  }, [active, revision])

  return {
    capabilities,
    state,
    error,
    refresh: () => setRevision((current) => current + 1)
  }
}
