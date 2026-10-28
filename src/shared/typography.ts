import type { AppFontSize } from './app-state'

export const fontSizeOptions: { label: string; value: AppFontSize }[] = [
  { label: 'Extra small', value: 'extra-small' },
  { label: 'Small', value: 'small' },
  { label: 'Normal', value: 'normal' },
  { label: 'Large', value: 'large' },
  { label: 'Extra large', value: 'extra-large' }
]

export function normalizeFontSize(value: unknown): AppFontSize {
  // Preserve saved preferences from the earlier three/four-size control.
  if (value === 'medium') return 'normal'
  return fontSizeOptions.find(option => option.value === value)?.value ?? 'normal'
}
