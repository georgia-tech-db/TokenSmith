// Exercise trace arrival/removal on the same mounted answer (the hook-order regression).
import React, { useState } from 'react'
import { createRoot } from 'react-dom/client'
import { AnswerExplanation } from '../../src/renderer/src/AnswerExplanation'
import '../../src/renderer/src/styles.css'

function Fixture() {
  const [message, setMessage] = useState({ id: 'late-plan', role: 'assistant', text: 'An existing saved answer.' })
  window.queryPlanTest = { setTrace: latencyTrace => setMessage(current => ({ ...current, latencyTrace })) }
  return <main style={{ maxWidth: 800, padding: 20, margin: 'auto' }}>
    <AnswerExplanation message={message} pending={false} disabled={false} onSimplify={() => {}}
      onSelectView={() => {}} />
  </main>
}
createRoot(document.getElementById('root')).render(<Fixture />)
