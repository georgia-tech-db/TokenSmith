import React, { useEffect, useState } from 'react'
import {createRoot} from 'react-dom/client'
import {PracticeTopicPicker} from '../../src/renderer/src/PracticeTopicPicker'
import {LibraryPracticeTopics} from '../../src/renderer/src/LibraryPracticeTopics'
import '../../src/renderer/src/styles.css'
import '../../src/renderer/src/practice.css'
import '../../src/renderer/src/library-workspace.css'

const source={materialId:'library/1',documentId:1,documentTitle:'Book',excerpt:'Buffer pools cache database pages.'}
const collection={id:'library',indexRevision:'r1'}
const models=[{id:'small',name:'Gemma 4 E4B',engine:'ollama',ollamaModelName:'gemma4:e4b',status:'ready'},
  {id:'large',name:'Gemma 4 26B',engine:'ollama',ollamaModelName:'gemma4:26b',status:'ready'}]
let saved=new URLSearchParams(location.search).has('paused')
  ? {version:3,indexRevision:'r1',complete:false,completed:1,topics:[]} : null
const listeners=new Set()
window.topicFixture={calls:[],manualTopic:''}
window.tokensmith={
  loadTopicInventory:async()=>saved,
  onLibraryTopicProgress(callback) { listeners.add(callback);return ()=>listeners.delete(callback) },
  async extractLibraryTopics(collectionId,modelId) {
    window.topicFixture.calls.push([collectionId,modelId])
    saved={version:3,indexRevision:'r1',complete:true,modelId,topics:[{id:'buffers',title:'Buffer pools',area:'Chapter 1',aliases:[],objectives:['Caching'],sources:[source]}]}
    for(const listener of listeners) listener({collectionId,status:'complete'})
  }
}

function Fixture() {
  const [screen,setScreen]=useState('practice')
  const [progress,setProgress]=useState()
  useEffect(()=>window.tokensmith.onLibraryTopicProgress(setProgress),[])
  return screen==='library' ? <div className="library-workspace">
    <LibraryPracticeTopics collection={collection} models={models} defaultModelId="small" canGenerate progress={progress} />
    <button onClick={()=>setScreen('practice')}>Practice</button>
  </div> : <PracticeTopicPicker isActive ready disabled={false} scope={[source]} materials={[{id:'library/1',collectionRevision:'r1'}]} sessions={[]}
    onChoose={topic=>{window.topicFixture.manualTopic=topic.title}} onChooseTopics={()=>{}} onOpenLibrary={()=>setScreen('library')} />
}

createRoot(document.getElementById('root')).render(<Fixture />)
