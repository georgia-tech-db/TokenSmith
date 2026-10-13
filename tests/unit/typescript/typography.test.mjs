import assert from 'node:assert/strict'
import { test } from 'node:test'
import { requireTranspiledTs } from './ts-module-loader.mjs'

const { normalizeFontSize } = requireTranspiledTs('src/shared/typography.ts')

test('font settings survive persistence including both smallest and largest options', () => {
  for (const value of ['extra-small', 'small', 'normal', 'large', 'extra-large']) {
    assert.equal(normalizeFontSize(JSON.parse(JSON.stringify({fontSize: value})).fontSize), value)
  }
})

test('legacy medium maps to normal and invalid saved settings have a readable fallback', () => {
  assert.equal(normalizeFontSize('medium'), 'normal')
  for (const value of [undefined, null, '', 'huge', 'toString', 20, {}, []]) {
    assert.equal(normalizeFontSize(value), 'normal')
  }
})
