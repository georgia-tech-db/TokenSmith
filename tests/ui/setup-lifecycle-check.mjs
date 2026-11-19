import assert from 'node:assert/strict'
import { createServer } from 'vite'
import react from '@vitejs/plugin-react'
import { resolve } from 'node:path'
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright')
const server=await createServer({configFile:false,cacheDir:'node_modules/.vite-setup-regression',optimizeDeps:{entries:['tests/ui/setup-lifecycle.html']},plugins:[react()],resolve:{alias:{'@shared':resolve('src/shared')}},server:{host:'127.0.0.1',port:0}})
await server.listen()
const base=`http://127.0.0.1:${server.httpServer.address().port}/tests/ui/setup-lifecycle.html`
const browser=await chromium.launch({headless:true,executablePath:process.env.CHROME_PATH})
const failures=[]
async function scenario(name,query,run){
 const context=await browser.newContext();const page=await context.newPage();page.setDefaultTimeout(10000)
 const errors=[];page.on('pageerror',e=>errors.push(e.message))
 await context.route('**/*',route=>new URL(route.request().url()).hostname==='127.0.0.1'?route.continue():route.abort())
 try{await page.goto(`${base}?${query}`);await page.getByRole('heading',{name:'Setup TokenSmith'}).waitFor({timeout:30000});await run(page);assert.deepEqual(errors,[]);console.log(`PASS ${name}`)}
 catch(error){failures.push(name);console.error(name,error,await page.locator('body').innerText())}
 finally{await context.close()}
}
const button=(p,name)=>p.getByRole('button',{name,exact:true})
const configure=(p,value)=>p.evaluate(v=>window.setupTest.configure(v),value)
const finish=(p,model,options={})=>p.evaluate(({model,options})=>window.setupTest.finish(model,options),{model,options})
const registered=(p,model)=>p.waitForFunction(name=>window.setupTest.state()?.models.some(m=>m.ollamaModelName===name&&m.status==='ready'),model)
try{
 await scenario('missing Ollama: install, recheck, failed start, recovery','',async p=>{
  await button(p,'Install Ollama').click()
  assert.equal(await p.evaluate(()=>window.setupTest.calls.filter(c=>c.kind==='install').length),1)
  assert.equal(await p.evaluate(()=>window.setupTest.calls.filter(c=>c.kind==='pull').length),0)
  await configure(p,{installed:true,startError:'Ollama could not start. Try again.'});await button(p,'Recheck').click()
  await button(p,'Start Ollama').click();await p.getByText('Ollama could not start. Try again.',{exact:true}).waitFor()
  await configure(p,{startError:''});await button(p,'Start Ollama').click()
  await button(p,'Download Nomic').waitFor();assert.equal(await button(p,'Download Nomic').isEnabled(),true)
 })
 await scenario('installed Ollama: open app and recover status check','installed',async p=>{
  await configure(p,{statusError:'Connection unavailable'});await button(p,'Recheck').click()
  await p.getByText('Connection unavailable',{exact:true}).waitFor()
  await configure(p,{statusError:''});await button(p,'Open App').click()
  await button(p,'Download Nomic').waitFor()
 })
 for(const [query,model,label] of [['running','gemma4:e4b','Gemma 4 E4B'],['running&strong','gemma4:26b','Gemma 4 26B A4B']]){
  await scenario(`download lifecycle and device recommendation: ${model}`,query,async p=>{
   const chat=button(p,`Download ${label}`)
   await chat.waitFor();await button(p,'Download Nomic').click()
   assert.equal(await chat.isDisabled(),true,'Do not start two setup downloads at once')
   await p.evaluate(()=>window.setupTest.progress('nomic-embed-text',40))
   await p.getByLabel('Download progress 40%').waitFor()
   await button(p,'Pause').click();await button(p,'Resume').waitFor()
   assert.equal(await p.evaluate(()=>window.setupTest.pending().length),0)
   await button(p,'Resume').click();await finish(p,'nomic-embed-text',{error:'Disk full'})
   await p.getByText('Disk full',{exact:true}).first().waitFor()
   assert.equal(await p.evaluate(()=>window.setupTest.state().models.some(m=>m.status==='ready')),false)
   await button(p,'Download Nomic').click();await finish(p,'nomic-embed-text',{verified:false})
   await p.getByText(/could not verify the Nomic/).first().waitFor()
   assert.equal(await button(p,'Delete Embedding Model').count(),1)
   await button(p,'Download Nomic').click();await finish(p,'nomic-embed-text');await registered(p,'nomic-embed-text')
   await chat.click();await finish(p,model);await registered(p,model)
   await p.getByText('Models are ready',{exact:true}).waitFor()
   await p.reload();await p.getByText('Models are ready',{exact:true}).waitFor()
   assert.equal(await p.evaluate(()=>window.setupTest.state().models.filter(m=>m.status==='ready').length),2)

  })
 }
 for(const [model,download,remove] of [['nomic-embed-text','Download Nomic','Delete Embedding Model'],['gemma4:e4b','Download Gemma 4 E4B','Delete Chat Model']]){
  await scenario(`delete ${model}: failure restores model, successful removal persists`,'running',async p=>{
   await button(p,download).click();await finish(p,model);await registered(p,model)
   await configure(p,{deleteError:'Model in use'});await button(p,remove).click();await registered(p,model)
   await button(p,remove).waitFor()
   await configure(p,{deleteError:''});await button(p,remove).click();await button(p,download).waitFor()
   await p.waitForFunction(name=>!window.setupTest.state().models.some(m=>m.ollamaModelName===name),model)
   await p.reload();await button(p,download).waitFor()
   assert.equal(await p.evaluate(()=>window.setupTest.state().models.length),0)
  })
 }
 await scenario('deleting an in-progress download cancels it without registering a model','running',async p=>{
  await button(p,'Download Nomic').click();await button(p,'Delete Embedding Model').click()
  await p.waitForFunction(()=>window.setupTest.pending().length===0)
  assert.equal(await p.evaluate(()=>window.setupTest.state().models.some(m=>m.status==='ready')),false)
 })
 await scenario('document catalog retries an IPC revision race without showing raw error','running&catalog-race',async p=>{
  await p.waitForFunction(()=>window.setupTest.catalogCalls()>=3)
  await p.getByText('No documents yet.',{exact:true}).waitFor()
  assert.equal(await p.getByText(/Error invoking remote method/).count(),0)
 })
}finally{await browser.close();await server.close()}
assert.deepEqual(failures,[],'Setup regressions failed')
