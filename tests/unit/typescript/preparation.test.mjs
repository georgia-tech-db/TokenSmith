import assert from 'node:assert/strict'
import test from 'node:test'
import { requireTranspiledTs } from './ts-module-loader.mjs'

const { defaultPreparation } = requireTranspiledTs('src/shared/preparation.ts')

test('new collections default to Basic without a preparation model', () => {
  assert.deepEqual(defaultPreparation(), {
    mode: 'basic', instructions: '', documentInstructions: {}
  })
})

test('changing one collection does not change defaults for the next', () => {
  const settings = defaultPreparation()
  settings.mode = 'ai'
  settings.modelId = 'chosen-model'
  settings.documentInstructions['notes.md'] = 'Keep examples together.'
  assert.deepEqual(defaultPreparation(), {
    mode: 'basic', instructions: '', documentInstructions: {}
  })
})
