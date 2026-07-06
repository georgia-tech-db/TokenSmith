import type { TokenSmithBridge } from '@shared/bridge'

declare global {
  interface Window {
    tokensmith?: TokenSmithBridge
  }
}

export {}
