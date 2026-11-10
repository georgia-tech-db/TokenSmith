import React, { useState } from 'react'
import { createRoot } from 'react-dom/client'
import { PracticeTopicPicker } from '../../src/renderer/src/PracticeTopicPicker'
import { normalizePracticeState } from '../../src/shared/practice'
import { practiceConceptKey } from '../../src/shared/practice-study'
import { practiceMastery, removePracticeHistory } from '../../src/shared/practice-mastery'
import { MasteryMeter } from '../../src/renderer/src/MasteryMeter'
import '../../src/renderer/src/styles.css'
import '../../src/renderer/src/practice.css'

// Real picker and saved-state normalization; fake books and no model access.
const storageKey = `topic-progress:${new URLSearchParams(location.search).get('run') || 'preview'}`
const materials = ['a', 'b'].map(id => ({ id: `${id}/1`, collectionTitle: `Book ${id.toUpperCase()}`, collectionRevision: 'r1',
  title: 'Course book', status: 'ready', kind: 'document', addedAt: '2026-11-01T00:00:00Z' }))
const documents = materials.map(m => ({ materialId: m.id, documentId: 1 }))
const topic = (course, id, title, area) => ({ id, title, area, aliases: [], objectives: [`Explain ${title.toLowerCase()}.`],
  concept: { courseId: course, id, title }, sources: [{ materialId: `${course}/1`, documentId: 1, chunkId: `${id}-passage`,
    title: 'Course book', documentTitle: 'Course book', sectionHeader: area, excerpt: `Evidence for ${title}.` }] })
const topics = {
  a: [topic('a', 'trees', 'Tree insertion', 'Chapter 1 — Indexes'), topic('a', 'hash', 'Hash indexes', 'Chapter 1 — Indexes'),
    topic('a', 'ranges', 'Range queries', 'Chapter 1 — Indexes'), topic('a', 'pages', 'Slotted pages', 'Chapter 2 — Storage')],
  b: [topic('b', 'trees', 'Tree insertion', 'Chapter 1 — Indexes')]
}
function answered(id, course, topicId, correct, assisted, at) {
  const picked = topics[course].find(t => t.id === topicId)
  return { id, title: picked.title, createdAt: at, documents: [documents[course === 'a' ? 0 : 1]], modelId: 'fixture', modelSettings: { contextLength: 8192 }, currentIndex: 0,
    topic: picked, questions: [{ id: 'q1', text: `Claim ${id} is true.`, format: 'true_false', correctAnswer: true,
      masteryCheck: { version: 1, conceptKey: practiceConceptKey(picked.concept) },
      rubric: { objective: 'A distinct claim.', assumptions: [], criteria: [{ id: 'c1', description: 'Understand the claim.', evidence: [{ sourceKey: 'fixture', quote: `Evidence for ${id}.` }] }] },
      concept: picked.concept, sources: picked.sources, draft: '', reviewing: false, attempts: [{ id: `${id}:first`, answer: correct ? 'True' : 'False',
        feedback: { kind: 'true_false', correct, explanation: 'Fixture feedback.' }, assisted, observationVersion: 1, submittedAt: at, durationMs: 1000 }] }] }
}
const seed = { mode: 'practice', sessions: [answered('mistake', 'a', 'trees', false, false, '2026-11-09T12:00:00Z'),
  answered('help', 'a', 'trees', true, true, '2026-11-10T12:00:00Z'), answered('hash', 'a', 'hash', true, false, '2026-11-10T13:00:00Z')] }
const reads = []
window.tokensmith = { loadTopicInventory: async id => {
  reads.push(id)
  return { version: 3, collectionId: id, indexRevision: 'r1', complete: true, topics: topics[id] }
} }

function Preview() {
  const [state, setState] = useState(() => normalizePracticeState(JSON.parse(sessionStorage.getItem(storageKey) || JSON.stringify(seed))))
  const [scope, setScope] = useState([documents[0]])
  const [selected, setSelected] = useState(null)
  function update(change) {
    setState(current => {
      const next = change(current)
      sessionStorage.setItem(storageKey, JSON.stringify(next))
      return next
    })
  }
  window.topicProgressTest = { reads, selected, state,
    answer: (correct, assisted = false) => update(current => ({ ...current, sessions: [...current.sessions,
      answered(crypto.randomUUID(), 'a', 'trees', correct, assisted, new Date(Date.parse('2026-11-10T15:00:00Z') + current.sessions.length * 1000).toISOString())] })),
    deleteSession: id => update(current => removePracticeHistory(current, id)),
    clear: () => update(current => removePracticeHistory(current))
  }
  const live = selected?.concept && practiceMastery(state.sessions, [], state.masteryRecords).get(practiceConceptKey(selected.concept))
  return <main className="practice-frame" style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
    <div role="group" aria-label="Fixture documents" style={{ display: 'flex', flexWrap: 'wrap', gap: 8, padding: 12 }}>
      <button type="button" onClick={() => setScope([documents[0]])}>Book A</button>
      <button type="button" onClick={() => setScope([documents[1]])}>Book B</button>
      <button type="button" onClick={() => setScope(documents)}>Both books</button>
    </div>
    {selected && <div className="practice-topic-bar" aria-label="Active topic"><span>{selected.title}</span><div className="practice-live-mastery" aria-live="polite"><MasteryMeter score={live?.score ?? null} /></div></div>}
    <div className="practice-scroll"><PracticeTopicPicker isActive ready disabled={false} scope={scope} materials={materials} sessions={state.sessions} masteryRecords={state.masteryRecords} onChoose={setSelected} /></div>
  </main>
}
createRoot(document.getElementById('root')).render(<Preview />)
