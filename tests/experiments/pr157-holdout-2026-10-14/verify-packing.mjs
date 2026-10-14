// Offline check: the bundled loader and PR source recreate all recorded prompts.
import {readFileSync} from 'node:fs';
import {dirname,resolve} from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';
import {parseArgs} from 'node:util';
import assert from 'node:assert/strict';
const out=dirname(fileURLToPath(import.meta.url));
const {values}=parseArgs({options:{source:{type:'string'}}});
if(!values.source)throw new Error('--source is required');
process.chdir(resolve(values.source));
const {requireTranspiledTs}=await import(pathToFileURL(resolve(out,'ts-module-loader.mjs')));
const {prepareStudyChatMessages}=requireTranspiledTs('src/main/engine/study-chat-format.ts');
const answers=JSON.parse(readFileSync(resolve(out,'answers.json'),'utf8'));
const retrieval=JSON.parse(readFileSync(resolve(out,'retrieval.json'),'utf8'));
const model={id:'ollama:gemma4:e4b',name:'gemma4:e4b',engine:'ollama',role:'generator',status:'ready',
  ollamaModelName:'gemma4:e4b',ollamaBaseUrl:'http://127.0.0.1:11434',contextLength:8192};
const modelSettings={contextLengthMode:'manual',contextLength:8192,maxLength:1024,temperature:0,
  topP:1,topK:40,minP:0,repeatPenalty:1.1,reasoningMode:'off',thinking:false,
  systemMessage:'Answer the student in at most 180 words. Address each requested part using the supplied course sources. If those sources do not support a needed detail, say so; do not invent it.'};
for(const a of answers){
  const sources=retrieval.find(r=>r.id===a.id&&r.arm===a.arm&&r.mode==='hybrid'&&r.k===4).result.sources;
  const prepared=prepareStudyChatMessages({prompt:a.question,messages:[],materials:[],settings:{},model,modelSettings,
    applicationSettings:{suggestionMode:'off'},retrievedSources:sources});
  assert.deepEqual(prepared.messages,a.calls[0].request.messages,`${a.id} ${a.arm}: prompt differs`);
  // The heuristic counts metadata paths that were shortened for publication.
  // Keep historical estimates intact; compare limits/selection and actual prompts.
  for(const key of Object.keys(a.budget)){
    if(['usedSourceTokens','estimatedPromptTokens'].includes(key))continue;
    assert.equal(prepared.budget[key],a.budget[key],`${a.id} ${a.arm}: ${key} differs`);
  }
}
console.log(`PASS: all ${answers.length} recorded prompts, source counts, and budget limits reproduced without model calls.`);
console.log('Historical heuristic token estimates are preserved; metadata path redaction can change their recomputed values.');
