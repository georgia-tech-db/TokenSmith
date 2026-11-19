import assert from 'node:assert/strict'
import {createServer} from 'vite'
import react from '@vitejs/plugin-react'
import {resolve} from 'node:path'
import {chromium} from 'playwright'
const server=await createServer({configFile:false,plugins:[react()],resolve:{alias:{'@shared':resolve('src/shared')}},server:{host:'127.0.0.1',port:0}})
await server.listen()
const browser=await chromium.launch({headless:true,executablePath:process.env.CHROME_PATH})
try {
 const page=await browser.newPage({viewport:{width:1440,height:1000}})
 await page.goto(`http://127.0.0.1:${server.httpServer.address().port}/tests/ui/generation-progress.html?preview=pdf&wide-pdf`)
 await page.locator('#reset').click()
 await page.getByLabel('Message',{exact:true}).fill('Explain atomicity')
 await page.locator('.chat-frame .send-button').click()
 const pane=page.getByRole('complementary',{name:'Source passage',exact:true})
 await pane.getByRole('button',{name:'Open document',exact:true}).click()
 await page.locator('.pdf-js-page img').waitFor()
 for(const width of [1440,1024,390,1440]) {
  await page.setViewportSize({width,height:1000})
  await page.waitForFunction(()=>{
   const stage=document.querySelector('.pdf-js-stage'),image=document.querySelector('.pdf-js-page')
   return stage && image && stage.scrollWidth<=stage.clientWidth+1 && image.getBoundingClientRect().width<=stage.clientWidth
  })
  const size=await page.locator('.pdf-js-page').boundingBox()
  assert.ok(Math.abs(size.width/size.height-1600/900)<.01,'Preserve the page aspect ratio')
 }
 await page.screenshot({path:'/tmp/tokensmith-pdf-fit-width.png'})
 console.log('PASS: landscape PDF fits reader width at desktop/mobile and after resizing, with correct aspect ratio.')
} finally {await browser.close();await server.close()}
