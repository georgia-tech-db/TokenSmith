import assert from 'node:assert/strict'
import test from 'node:test'
import { requireTranspiledTs } from './ts-module-loader.mjs'
const { createAnswerSourcePreview } = requireTranspiledTs('src/renderer/src/answer-source-preview.ts')
function deferred() { let resolve; const promise = new Promise(r => {resolve=r}); return {promise,resolve} }

test('opens the first source once and closes when the answer arrives', async () => {
  const changes=[]; const preview=createAnswerSourcePreview(value=>changes.push(value))
  preview.begin('answer')
  let loads=0
  await preview.open('answer',async()=>{loads++;return 'first source'})
  await preview.open('answer',async()=>{loads++;return 'second source'})
  assert.equal(changes.at(-1),'first source')
  assert.equal(loads,1)
  preview.finish('answer')
  assert.equal(changes.at(-1),null)
  await preview.open('answer',async()=>{loads++;return 'too late'})
  assert.equal(loads,1)
})
test('a source that finishes loading after the answer cannot reopen the preview', async () => {
  const changes=[]; const preview=createAnswerSourcePreview(value=>changes.push(value)),load=deferred()
  preview.begin('answer')
  const pending=preview.open('answer',()=>load.promise)
  preview.finish('answer')
  load.resolve('late PDF')
  await pending
  assert.ok(changes.every(value=>value===null))
})
test('dismissal, navigation and Stop invalidate a pending preview without reopening it', async () => {
  const changes=[]; const preview=createAnswerSourcePreview(value=>changes.push(value)),load=deferred()
  preview.begin('answer')
  const pending=preview.open('answer',()=>load.promise)
  preview.dismiss()
  load.resolve('late Markdown')
  await pending
  await preview.open('answer',async()=> 'reopened')
  assert.ok(changes.every(value=>value===null))
})
test('old answer completion and out-of-order source loads cannot close a newer preview', async () => {
  const changes=[];const preview=createAnswerSourcePreview(value=>changes.push(value)),old=deferred()
  preview.begin('old')
  const pending=preview.open('old',()=>old.promise)
  preview.begin('new')
  await preview.open('new',async()=> 'new source')
  preview.finish('old')
  old.resolve('old source')
  await pending
  assert.equal(changes.at(-1),'new source')
})
test('answer completion leaves a manually chosen source alone', async () => {
  let displayed;const preview=createAnswerSourcePreview(value=>{displayed=value})
  preview.begin('answer')
  await preview.open('answer',async()=> 'automatic')
  preview.dismiss()
  displayed='manual source'
  preview.finish('answer')
  assert.equal(displayed,'manual source')
})
test('source load failure is optional and does not interrupt generation', async () => {
  const changes=[];const preview=createAnswerSourcePreview(value=>changes.push(value))
  preview.begin('answer')
  await preview.open('answer',async()=>{throw new Error('Missing source')})
  assert.equal(changes.at(-1),null)
  preview.finish('answer')
})
