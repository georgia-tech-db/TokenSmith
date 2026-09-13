import { Check } from 'lucide-react'
import type { AppTheme } from '@shared/app-state'

const themes: { id: AppTheme; name: string; description: string }[] = [
  { id: 'light', name: 'Gray', description: 'The original, quiet gray palette.' },
  { id: 'sarah-and-duck', name: 'Sarah & Duck', description: 'Sage green, soft pink, and a little blue sky.' }
]

export function ThemePicker({ value, onChange }: { value: AppTheme; onChange: (theme: AppTheme) => void }) {
  return (
    <fieldset className="theme-picker">
      <legend>Theme</legend>
      <p className="theme-picker-help">Make yourself at home. Your choice is saved automatically.</p>
      <div className="theme-options">
        {themes.map((theme) => (
          <label className="theme-option" key={theme.id}>
            <input
              type="radio"
              name="app-theme"
              value={theme.id}
              checked={value === theme.id}
              onChange={() => onChange(theme.id)}
              aria-labelledby={`theme-name-${theme.id}`}
              aria-describedby={`theme-description-${theme.id}`}
            />
            <span className="theme-option-card">
              <span className="theme-preview" data-preview-theme={theme.id} aria-hidden="true">
                <span className="theme-preview-rail"><i /><i /><i /></span>
                <span className="theme-preview-sidebar"><i /><i /><i /></span>
                <span className="theme-preview-chat">
                  <span className="theme-preview-heading" />
                  <span className="theme-preview-line" />
                  <span className="theme-preview-line is-short" />
                  <span className="theme-preview-suggestion" />
                  <span className="theme-preview-composer"><i /></span>
                </span>
                <span className="theme-preview-moon" />
              </span>
              <span className="theme-option-name">
                <strong id={`theme-name-${theme.id}`}>{theme.name}</strong>
                <span className="theme-option-check" aria-hidden="true"><Check size={13} strokeWidth={3} /></span>
              </span>
              <span className="theme-option-description" id={`theme-description-${theme.id}`}>{theme.description}</span>
            </span>
          </label>
        ))}
      </div>
    </fieldset>
  )
}
