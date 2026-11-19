import test from 'node:test'
import assert from 'node:assert/strict'
import {requireTranspiledTs} from './ts-module-loader.mjs'
const {preparationProgress}=requireTranspiledTs('src/shared/preparation-progress.ts')
test('stage progress displays OCR completion independently of overall progress',()=>{
 const result=preparationProgress({stage:'OCR',stagePercent:60,percent:20,estimatedRemainingMs:10000})
 assert.equal(result.percent,60);assert.match(result.text,/OCR · 60% · About/)
 assert.doesNotMatch(result.text,/overall/)
 assert.match(preparationProgress({stage:'Chunking',stagePercent:0}).text,/Chunking · 0% · Estimating/)
})
test('setup is indeterminate and completed stages do not count down',()=>{
 assert.equal(preparationProgress({stage:'Model setup',percent:1}).percent,undefined)
 assert.equal(preparationProgress({stage:'OCR',stagePercent:100,estimatedRemainingMs:0}).text,'OCR · 100%')
 assert.match(preparationProgress({stage:'OCR',stagePercent:50,estimatedRemainingMs:1000},5000).text,/Taking longer/)
})

test('app normalization preserves stage counts and estimates through progress updates',()=>{
 const {normalizeIndexingProgress}=requireTranspiledTs('src/shared/preparation-progress.ts')
 const progress={materialId:'m',phase:'chunking',percent:7,stage:'OCR',stagePercent:60,stageCompleted:6,stageTotal:10,stageElapsedMs:12000,estimatedRemainingMs:8000,message:'Rapid OCR'}
 for(const status of ['indexing','paused']) assert.deepEqual(normalizeIndexingProgress({id:'m',status,indexing:progress}),{...progress,processedFiles:0,totalFiles:0,processedEmbeddings:0,totalEmbeddings:0})
 assert.match(preparationProgress(normalizeIndexingProgress({id:'m',status:'indexing',indexing:progress})).text,/OCR · 60% · About/)
})
