import assert from 'node:assert/strict'
import test from 'node:test'
import { requireTranspiledTs } from './ts-module-loader.mjs'
const { sourceNavigationList, collectionDocumentSources } = requireTranspiledTs('src/renderer/src/source-navigation.ts')
const sources = ['one.md','two.pdf','three.markdown','four.PDF'].map((name,index)=>({path:`/collection/${name}`,title:name,excerpt:`Source ${index+1}`,locator:'Passage'}))
test('answer navigation preserves mixed-format retrieval order and starts at the selected source', () => {
  const list=sourceNavigationList(sources[2],sources)
  assert.deepEqual(list.sources,sources)
  assert.equal(list.index,2)
})
test('removes identical repeats and unopenable entries while retaining distinct passages in one file', () => {
  const other={...sources[0],excerpt:'Another passage'}
  const list=sourceNavigationList(other,[sources[0],sources[0],{title:'Text only',excerpt:'No path'},other,{path:'/book.docx',excerpt:'unsupported'}])
  assert.deepEqual(list.sources,[sources[0],other])
  assert.equal(list.index,1)
})
test('a source missing from the candidate list remains available', () => {
  const list=sourceNavigationList(sources[0],[sources[2]])
  assert.deepEqual(list.sources,[sources[0],sources[2]])
  assert.equal(list.index,0)
})
test('Library navigation uses collection files, preserves the clicked passage and opens other files at the start', () => {
  const clicked={...sources[1],materialId:'collection',pageStart:7,pageEnd:8}
  const list=collectionDocumentSources(clicked,sources.map(({path,title})=>({path,title})))
  assert.deepEqual(list.map(source=>source.path),sources.map(source=>source.path))
  assert.equal(list[1],clicked)
  assert.equal(list[0].lineFrom,1)
  assert.equal(list[3].pageStart,1)
  assert.equal(list[3].materialId,'collection')
})
