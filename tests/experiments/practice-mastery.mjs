// Opt-in real-model diagnostic with fixed BuzzDB passages. Does not modify app history.
// Scripted scoring checks software behavior; it does not measure a student's mastery.
import { execFileSync } from 'node:child_process'
import { writeFileSync } from 'node:fs'
import { requireTranspiledTs } from '../unit/typescript/ts-module-loader.mjs'
const { runOllamaStudyEngine } = requireTranspiledTs('src/main/engine/ollama-service.ts')
const { createPracticeClient } = requireTranspiledTs('src/renderer/src/practice-client.ts')
const { checkPracticeQuestionAnswer, gradeTrueFalse } = requireTranspiledTs('src/shared/quiz.ts')
const { practiceMastery } = requireTranspiledTs('src/shared/practice-mastery.ts')
const chunks = JSON.parse(execFileSync('app_runtime/python/bin/python', ['-E','-s','-B','-c',
  'import json; from tests.benchmarks.test_buzzdb_embeddings import all_chunks; print(json.dumps(all_chunks()))'], {maxBuffer:10000000}))
function source(id) {
  const c = chunks.find(c => c.tokensmithChunkId === id)
  if (!c) throw new Error(`Missing chunk ${id}`)
  return { title:'BuzzDBBook', documentTitle:'buzzdb-book.tokensmith.md', materialId:'book', documentId:1,
    chunkId:id, sectionHeader:c.sectionHeader, excerpt:c.text, context:c.text }
}
const memory = source('ch05.002'), policy = source('ch06.024')
const modelName = process.env.PRACTICE_MODEL || 'gemma4:e4b'
const model = { id:'live-mastery',name:modelName,engine:'ollama',status:'ready',ollamaModelName:modelName,contextLength:8192 }
const modelSettings = {contextLength:8192,contextLengthMode:'manual',maxLength:2048,reasoningMode:'off',thinking:false,temperature:0.3,topP:0.95}
const settings = {application:{suggestionMode:'off'}}
const topic = {title:'All-in-Memory Strategy',objectives:['Identify limitations of loading entire databases into RAM','Understand the memory ceiling and scaling problems'],concept:{courseId:'book',id:'all-in-memory',title:'All-in-Memory Strategy'},sources:[memory],prepared:true}
const original = 'When a small database is loaded entirely into RAM, its queries can operate directly on in-memory data structures.'
const cases = [
  {id:'valid-topic',statement:original,source:memory,previousQuestions:[],expected:{answer:'true',topicMatch:'direct',relationship:'new_claim'}},
  {id:'off-topic',statement:'The replacement policy interface tells the buffer manager which page should be evicted when the pool is full.',source:policy,previousQuestions:[],expected:{answer:'true',topicMatch:'related',relationship:'new_claim'}},
  {id:'paraphrased-claim',statement:'Loading a small database into RAM allows queries to work directly with its in-memory structures.',source:memory,previousQuestions:[original],expected:{answer:'true',topicMatch:'direct',relationship:'restatement'}},
  {id:'policy-on-topic',topic:'Replacement Policies',objectives:['Understand the interface between a buffer manager and its replacement policy.'],statement:'The replacement policy interface tells the buffer manager which page should be evicted when the pool is full.',source:policy,previousQuestions:[],expected:{answer:'true',topicMatch:'direct',relationship:'new_claim'}},
  {id:'negated-prior-claim',topic:'Replacement Policies',objectives:['Understand the interface between a buffer manager and its replacement policy.'],statement:'The replacement policy interface does not tell the buffer manager which page to evict when the pool is full.',source:policy,previousQuestions:['The replacement policy interface tells the buffer manager which page should be evicted when the pool is full.'],expected:{answer:'false',topicMatch:'direct',relationship:'negation'}},
  {id:'new-claim',statement:'An all-in-memory design becomes impractical when the database grows beyond the available RAM.',source:memory,previousQuestions:[original],expected:{answer:'true',topicMatch:'direct',relationship:'new_claim'}}
]
const rows=[], calls=[]
const save=()=>writeFileSync(process.env.PRACTICE_OUTPUT || '/tmp/tokensmith-mastery-live.json',JSON.stringify({model:model.name,rows,calls},null,2))
async function send(request) {
  const start=performance.now(); const reply=await runOllamaStudyEngine(request)
  calls.push({task:request.practiceTask,prompt:request.prompt,seconds:(performance.now()-start)/1000,reply});save();return reply
}
for (const c of cases) {
  if (process.env.PRACTICE_CASE_FILTER && !c.id.includes(process.env.PRACTICE_CASE_FILTER)) continue
  const reply=await send({requestId:c.id,practiceTask:'verify_question',prompt:JSON.stringify({statement:c.statement,topic:c.topic ?? topic.title,learningObjectives:c.objectives ?? topic.objectives,previousQuestions:c.previousQuestions}),messages:[],materials:[],retrievedSources:[c.source],model,settings,modelSettings:{...modelSettings,maxLength:512},conversationContextMode:'standalone',reasoning:false})
  const actual=JSON.parse(reply.text)
  let accepted=true;try{checkPracticeQuestionAnswer(reply.text,c.expected.answer === 'true')}catch{accepted=false}
  rows.push({id:c.id,actual,expected:c.expected,accepted,pass:Object.entries(c.expected).every(([k,v])=>actual[k]===v)})
  save();console.log(JSON.stringify(rows.at(-1)))
}
const session={id:'isolated-live-test',title:topic.title,createdAt:new Date().toISOString(),modelId:model.id,modelSettings,questions:[],documents:[{materialId:'book',documentId:1,title:'BuzzDBBook'}],topic,currentIndex:0}
const client=createPracticeClient({sendChatMessage:send,practiceSources:async()=>[memory]},model,settings)
if (!process.env.PRACTICE_CASE_FILTER) try {
  const question=await client.question(session,'generated',new AbortController().signal)
  if(!question)throw new Error('No question')
  question.attempts.push({id:'scripted-correct-answer',answer:question.correctAnswer?'True':'False',feedback:gradeTrueFalse(question,question.correctAnswer?'True':'False'),assisted:false,durationMs:0,submittedAt:new Date().toISOString(),observationVersion:1})
  session.questions.push(question)
  const score=[...practiceMastery([session]).values()][0]
  rows.push({id:'generated-and-scored',question:question.text,answer:question.correctAnswer,masteryCheck:question.masteryCheck,score,pass:Math.abs(score.score-0.3103448275862069)<1e-10})
} catch(error){rows.push({id:'generated-and-scored',pass:false,error:error.message})}
save();console.log(JSON.stringify(rows.at(-1)));console.log(`Results: ${process.env.PRACTICE_OUTPUT || '/tmp/tokensmith-mastery-live.json'}`)
if(rows.some(r=>!r.pass))process.exitCode=1
