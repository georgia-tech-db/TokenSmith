import { test } from 'node:test'
import assert from 'node:assert/strict'
import { gradeDecision, localChatOptions, indexFingerprint } from './local-eval-policy.mjs'
const good = { score: .9, verdict: 'correct', core_claim_satisfied: true, contradictions: [], rationale: 'Synthetic evidence supports the answer.' }
test('only consistent supported judgments pass', () => {
  assert.equal(gradeDecision(good,.8).passed,true)
  for (const patch of [{score:.4},{verdict:'incorrect'},{core_claim_satisfied:false},{contradictions:['Synthetic counterexample']},{score:2},{score:NaN},{verdict:'manual_review'}]) assert.equal(gradeDecision({...good,...patch},.8).passed,false)
  assert.equal(gradeDecision({},.8).verdict,'manual_review')
})
test('actual model options honor formatter budgets', () => {
  assert.deepEqual(localChatOptions(8192,1024,0,42),{num_ctx:8192,num_predict:1024,temperature:0,seed:42})
  assert.throws(()=>localChatOptions(1024,1024,0,42))
  assert.throws(()=>localChatOptions(NaN,100,0,42))
})

test('index reuse depends on content, model, endpoint, and engine version', () => {
  const args = [Buffer.from('synthetic PDF'), 'model-v1', 'http://localhost', [['engine.py', Buffer.from('version1')]]]
  const key = indexFingerprint(...args)
  assert.equal(indexFingerprint(...args), key)
  for (const [i, replacement] of [[0,Buffer.from('edited PDF')],[1,'model-v2'],[2,'http://127.0.0.1'],[3,[['engine.py',Buffer.from('version2')]]]]) {
    const changed = [...args]; changed[i] = replacement
    assert.notEqual(indexFingerprint(...changed), key)
  }
})
