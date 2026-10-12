import assert from 'node:assert/strict'
import test from 'node:test'
import { requireTranspiledTs } from './ts-module-loader.mjs'

const { defaultPreparation } = requireTranspiledTs('src/shared/preparation.ts')

test('new collections default to Basic without a preparation model', () => {
  assert.deepEqual(defaultPreparation(), {
    mode: 'basic', instructions: '', documentInstructions: {}
  })
})

test('plain-language search terms are off until a collection turns them on', () => {
  assert.equal(defaultPreparation().plainLanguageTerms, undefined)
  const settings = { ...defaultPreparation(), plainLanguageTerms: true }
  assert.equal(settings.plainLanguageTerms, true)
  assert.equal(defaultPreparation().plainLanguageTerms, undefined)
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
