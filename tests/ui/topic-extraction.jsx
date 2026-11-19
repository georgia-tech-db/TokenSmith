import React from 'react'
import {createRoot} from 'react-dom/client'
import {PracticeTopicPicker} from '../../src/renderer/src/PracticeTopicPicker'
import '../../src/renderer/src/styles.css'
import '../../src/renderer/src/practice.css'
const source={materialId:'library/1',documentId:1,documentTitle:'Book',excerpt:'Buffer pools cache database pages.'}
window.extractionCalls=0
window.tokensmith={loadTopicInventory:async()=>window.extractionCalls?{version:3,indexRevision:'r1',complete:true,topics:[{id:'buffers',title:'Buffer pools',area:'Chapter 1',aliases:[],objectives:['Caching'],sources:[source]}]}:null}
createRoot(document.getElementById('root')).render(<PracticeTopicPicker isActive ready disabled={false} scope={[source]} materials={[{id:'library/1',collectionRevision:'r1',preparation:{mode:'basic',extractTopics:false}}]} sessions={[]}
 onChoose={()=>{}} onChooseTopics={()=>{}}
 onExtractTopics={async()=>{window.extractionCalls++}} />)
