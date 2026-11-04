// Deterministic Electron integration check; no installed model or user profile is used.
import { createServer } from 'node:http'
import { mkdtemp, writeFile, readFile, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { spawn } from 'node:child_process'
import { createRequire } from 'node:module'
import assert from 'node:assert/strict'
const dir = await mkdtemp(join(tmpdir(), 'practice-foundation-runner-'))
const calls = []
const server = createServer(async (req, res) => {
  let body = ''; for await (const chunk of req) body += chunk
  const request = body ? JSON.parse(body) : null
  calls.push({ path: req.url, request })
  res.setHeader('Content-Type', 'application/json')
  if (req.url === '/api/chat' && calls.filter(c => c.path === '/api/chat').length === 3) {
    res.statusCode = 503; res.end(JSON.stringify({ error: 'Injected smoke-test failure' })); return
  }
  const reply = req.url === '/api/show' ? { capabilities: ['completion', 'thinking'] }
    : req.url === '/api/tags' ? { models: [{ name: 'fixture:latest', digest: 'fixture-digest' }] }
    : req.url === '/api/ps' ? { models: [] }
    : { message: { role: 'assistant', content: 'Hash indexes scatter keys across buckets [1].' }, done: true,
        done_reason: 'stop', prompt_eval_count: 256, eval_count: 12, total_duration: 1e6 }
  res.end(JSON.stringify(reply))
})
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve))
try {
  const source = { materialId:'m', documentId:1, chunkId:'c', title:'Fixture', excerpt:'Hash indexes scatter keys.', context:'Hash indexes scatter keys.' }
  const manifest = { version:1, cases:[{id:'smoke',prompt:'How are keys stored?', model:{id:'fixture',name:'Fixture',engine:'ollama',status:'ready',contextLength:8192,ollamaModelName:'fixture',ollamaBaseUrl:`http://127.0.0.1:${server.address().port}`},
    modelSettings:{contextLength:8192,contextLengthMode:'manual',maxLength:1024,temperature:0}, evidence:{baseline:[source],expanded:[source],reviewed:[source]}}] }
  const input = join(dir,'cases.json'), out = join(dir,'out'); await writeFile(input,JSON.stringify(manifest))
  const env = {...process.env}; delete env.ELECTRON_RUN_AS_NODE
  const child = spawn(createRequire(import.meta.url)('electron'),['tests/experiments/proposal-comparison.mjs','--input',input,'--out',out],{env,stdio:['ignore','pipe','pipe']})
  let output=''; child.stdout.on('data',c=>output+=c);child.stderr.on('data',c=>output+=c)
  const timeout = setTimeout(() => child.kill('SIGTERM'), 30_000)
  let code
  try { code = await new Promise((resolve, reject) => { child.on('close',resolve); child.on('error',reject) }) }
  finally { clearTimeout(timeout) }
  assert.equal(code,0,output)
  const results = JSON.parse(await readFile(join(out,'results.json'),'utf8'))
  assert.equal(results.length,6)
  assert.equal(results.filter(r=>r.error).length,1)
  assert.equal(results.filter(r=>r.result?.text).length,5)
  assert.deepEqual(new Set(calls.filter(c=>c.path==='/api/chat').map(c=>c.request.think)), new Set([false,true]))
  console.log('PASS: Electron runner completed six conditions, recorded one injected failure and five answers through the production path.')
} finally { server.closeAllConnections(); await new Promise(resolve=>server.close(resolve)); await rm(dir,{recursive:true,force:true}) }
