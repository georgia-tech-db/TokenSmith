import type { ClaimConfidence } from '@shared/confidence'

/** Link target a citation marker carries through the Markdown renderer; MessageText matches it. */
export const citationHrefPrefix = '#cite-'

/**
 * Letters and digits only, with a map back to where each one sits in `text`.
 * Claims come back from the engine with Markdown and spacing stripped, so they
 * are found by their letters, not their exact characters.
 */
function comparable(text: string): { letters: string; offsets: number[] } {
  let letters = ''
  const offsets: number[] = []

  for (let index = 0; index < text.length; index += 1) {
    const char = text[index]
    if (/[\p{L}\p{N}]/u.test(char)) {
      letters += char.toLowerCase()
      offsets.push(index)
    }
  }

  return { letters, offsets }
}

/**
 * Puts a numbered citation after each sentence the check traced to a source.
 * The marker is a Markdown link to `#cite-<claim index>`; MessageText turns it
 * into a button. Sentences it cannot find in the text are left unmarked.
 */
export function withCitationMarkers(text: string, claims: ClaimConfidence[]): string {
  const { letters, offsets } = comparable(text)
  const insertions: { at: number; marker: string }[] = []
  let searchFrom = 0

  claims.forEach((claim, claimIndex) => {
    if (claim.sourceIndex === undefined) {
      return
    }

    const needle = comparable(claim.text).letters
    const found = needle ? letters.indexOf(needle, searchFrom) : -1
    if (found < 0) {
      return
    }

    searchFrom = found + needle.length
    let at = offsets[found + needle.length - 1] + 1
    // Step past the sentence's closing punctuation and emphasis so the marker
    // sits after "word." rather than inside "**word**".
    while (at < text.length && /[.!?:;,*_)"'’”]/.test(text[at])) {
      at += 1
    }
    insertions.push({ at, marker: ` [${claim.sourceIndex + 1}](${citationHrefPrefix}${claimIndex})` })
  })

  let marked = text
  for (const { at, marker } of insertions.sort((left, right) => right.at - left.at)) {
    marked = `${marked.slice(0, at)}${marker}${marked.slice(at)}`
  }

  return marked
}
