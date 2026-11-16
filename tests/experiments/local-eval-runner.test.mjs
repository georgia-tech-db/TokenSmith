import { test } from 'node:test'
import assert from 'node:assert/strict'
import { createServer } from 'node:http'
import { mkdtempSync, writeFileSync, readFileSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { spawn } from 'node:child_process'

test('synthetic local run rejects contradictory judge and forwards budgets and history', async () => {
  const dir = mkdtempSync(join(tmpdir(), 'local-eval-test-'))
  const calls = []
  const server = createServer(async (req,res) => {
    let raw=''; for await (const chunk of req) raw+=chunk
    const body=raw ? JSON.parse(raw) : {}; calls.push(body)
    let value
    if (req.url === '/api/tags') value={models:[{name:'qwen2.5:3b',digest:'synthetic'}]}
    else if (req.url === '/api/embed') value={embeddings:[[1,0]]}
    else if (body.format?.properties?.mode) value={message:{content:JSON.stringify({mode:'contextual',query:'When does the synthetic lamp turn off?',clarification:'',reasoning:false})}}
    else if (body.format) value={message:{content:JSON.stringify({score:.95,verdict:'correct',core_claim_satisfied:false,contradictions:['The timer is omitted.'],rationale:'Synthetic mismatch.'})}}
    else value={message:{content:'The synthetic lamp turns off after ten minutes.'}}
    res.setHeader('Content-Type','application/json'); res.end(JSON.stringify(value))
  })
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve))
  try {
    const input=join(dir,'sources.json'), out=join(dir,'report.json')
    writeFileSync(input,JSON.stringify({cases:[{id:'synthetic',question:'When does it turn off?',referenceAnswer:'After ten minutes.',messages:[{role:'user',text:'Describe the synthetic lamp.'},{role:'assistant',text:'It has a timer.'}],sources:[{title:'Synthetic manual',excerpt:'The lamp turns off after ten minutes.',context:'The lamp turns off after ten minutes.'}]}]}))
    const child=spawn(process.execPath,['tests/experiments/run-eval-pipeline-local.mjs','--sources-from',input,'--out',out,'--summary',join(dir,'report.md'),'--ollama-base-url',`http://127.0.0.1:${server.address().port}`],{stdio:['ignore','pipe','pipe']})
    let log=''; child.stdout.on('data',x=>log+=x);child.stderr.on('data',x=>log+=x)
    const code=await new Promise(resolve=>child.on('close',resolve))
    assert.equal(code,1,log)
    const result=JSON.parse(readFileSync(out));assert.equal(result.cases[0].verdict,'incorrect',log)
    assert.equal(result.cases[0].resolution.mode,'contextual')
    const chat=calls.filter(c=>c.messages)
    assert.equal(chat.length,3)
    assert.ok(chat.every(c=>c.options.num_ctx===8192 && c.options.num_predict===1024))
    assert.ok(chat[0].messages.some(m=>m.content.includes('It has a timer.')))
  } finally { await new Promise(resolve=>server.close(resolve));rmSync(dir,{recursive:true,force:true}) }
})
