import assert from 'node:assert/strict'
import test from 'node:test'
import { requireTranspiledTs } from './ts-module-loader.mjs'

const { defaultPreparation } = requireTranspiledTs('src/shared/preparation.ts')

test('new collections default to AI with the selected chat model resolved at import', () => {
  assert.deepEqual(defaultPreparation(), {
    mode: 'ai', ocr: 'none', extractTopics: true, instructions: '', documentInstructions: {}
  })
})

test('changing one collection does not change defaults for the next', () => {
  const settings = defaultPreparation()
  settings.mode = 'basic'
  settings.modelId = 'chosen-model'
  settings.documentInstructions['notes.md'] = 'Keep examples together.'
  assert.deepEqual(defaultPreparation(), {
    mode: 'ai', ocr: 'none', extractTopics: true, instructions: '', documentInstructions: {}
  })
})

test('Gemma OCR resolves a Gemma model and never silently selects another model family', () => {
  const { preparationGenerator } = requireTranspiledTs('src/shared/preparation.ts')
  const gemma = { id: 'gemma', name: 'Gemma', engine: 'ollama', ollamaModelName: 'gemma4:e4b', status: 'ready' }
  const other = { id: 'other', name: 'Other', engine: 'remote', status: 'ready' }
  assert.equal(preparationGenerator([other, gemma], undefined, 'other', 'gemma'), gemma)
  assert.equal(preparationGenerator([other], undefined, 'other', 'gemma'), undefined)
  assert.equal(preparationGenerator([other, gemma], 'other', undefined, 'gemma'), undefined)
})
