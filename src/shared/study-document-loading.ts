import type { StudyDocument } from './study-scope'

export function documentLoadError(reason: unknown): string {
  const message = reason instanceof Error ? reason.message : String(reason)
  return message.replace(/^Error invoking remote method '[^']+':\s*(?:Error:\s*)?/, '')
}

/** A catalog is a live view: retry a fresh snapshot when indexing or closing invalidates it. */
export async function loadStudyDocuments(read: () => Promise<StudyDocument[]>, signal: AbortSignal,
  wait: (ms: number) => Promise<void> = ms => new Promise(resolve => setTimeout(resolve, ms))): Promise<StudyDocument[]> {
  for (let attempt = 0; ; attempt++) {
    signal.throwIfAborted()
    try {
      const documents = await read()
      signal.throwIfAborted()
      return documents
    } catch (reason) {
      signal.throwIfAborted()
      const message = documentLoadError(reason)
      if (!/library (?:changed during this request|is updating|was closed|is closed)/i.test(message)) {
        throw new Error(message || 'Could not load documents. Please try again.')
      }
      if (attempt === 4) throw new Error('Documents are still updating. Please try again in a moment.')
      await wait(150 * (attempt + 1))
    }
  }
}
