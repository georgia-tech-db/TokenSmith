import { validateChatAttachment, type AttachmentRun, type ChatAttachment } from './chat-attachment'

const evidenceLimit = 6000
const stopWords = new Set('a an and are as at be by can could do does explain file for from give how i in is it me of on or pdf please several tell text that the this to what when where which why with would you your'.split(' '))
const words = (text: string) => text.normalize('NFKC').toLowerCase().match(/[\p{L}\p{N}]+/gu) ?? []
const numberedProblem = /\b(?:question|problem|exercise) \d+\b/g

function keywords(text: string) {
  return words(text).filter(word => !stopWords.has(word) && (word.length > 1 || /^\d$/u.test(word)))
}

function attachmentBlocks(attachment: ChatAttachment) {
  return attachment.pages.flatMap(page => {
    const text = page.text.replace(/\r\n?/g, '\n').trim()
    const hasQuestions = /(?:^|\n)\s*(?:#{1,6}\s*)?(?:question|problem|exercise)\s+\d+\b/i.test(text)
    const paragraphs = text.split(hasQuestions
      ? /\n(?=\s*(?:#{1,6}\s*)?(?:question|problem|exercise)\s+\d+\b)/i : /\n\s*\n/)
    const blocks: string[] = []
    for (const paragraph of paragraphs) {
      if (hasQuestions || paragraph.length <= 1800) {
        blocks.push(paragraph)
        continue
      }
      // Keep sentences intact, including numeric values such as 3.14.
      const sentences = paragraph.split(/(?<=[.!?])\s+(?=[\p{Lu}\d])/u)
      let block = ''
      for (const sentence of sentences) {
        if (block && block.length + sentence.length + 1 > 1800) {
          blocks.push(block)
          block = ''
        }
        block += (block ? '\n' : '') + sentence
      }
      if (block) blocks.push(block)
    }
    return blocks.filter(text => text.trim()).map((text, index) => ({
      id: `p${page.page}.b${index + 1}`, page: page.page, text: text.trim(),
      problem: hasQuestions
    }))
  })
}

export function rankAttachmentPassages(attachment: ChatAttachment, question: string) {
  validateChatAttachment(attachment)
  const blocks = attachmentBlocks(attachment)
  const query = keywords(question).slice(0, 32)
  const terms = [...new Set(query)]
  const problemReferences = words(question).join(' ').match(numberedProblem) ?? []
  const passages = blocks.map((block, index) => {
    const tokens = keywords(block.text)
    return { ...block, index, terms: new Set(tokens), phrase: ` ${tokens.join(' ')} ` }
  })
  const weights = terms.map(term => 1 + Math.log((passages.length + 1) /
    (1 + passages.filter(passage => passage.terms.has(term)).length)))
  const phrases = query.slice(1).map((term, index) => ` ${query[index]} ${term} `)
  const ranked = passages.map((passage, index) => {
    const matches = terms.reduce((sum, term, i) => sum + (passage.terms.has(term) ? weights[i] : 0), 0)
    const exactPhrase = phrases.filter(phrase => passage.phrase.includes(phrase)).length
    const heading = words(passage.text.split(/[:\n]/, 1)[0]).join(' ')
    const isReferenced = passage.problem && problemReferences.some(ref =>
      heading === ref || heading.startsWith(ref + ' '))
    return { index, passage, isReferenced, score: matches + 3 * exactPhrase }
  })
  const anchors = ranked.filter(hit => hit.isReferenced)
  const matches = (anchors.length ? anchors : ranked.filter(hit => hit.score > 0))
    .sort((a, b) => b.score - a.score || a.index - b.index)
  if (!matches.length) return { terms, passages: [], problemText: undefined }
  const selected = new Map<number, { index: number; id: string; page: number; text: string }>()
  let size = 0
  for (const hit of matches) {
    // Weak matches add noise when a substantially stronger passage exists.
    if (hit.score < matches[0].score * 0.45) break
    const next = passages[hit.index + 1]
    const continuation = !hit.passage.problem && next && !next.problem && next.text.length <= 600 &&
      ((next.page === hit.passage.page && /^(?:Depending|Therefore|Thus|As a result)\b/i.test(next.text)) ||
        (!/[.!?]$/.test(hit.passage.text) && /^\p{Ll}/u.test(next.text)))
    const group = continuation ? [hit.passage, next] : [hit.passage]
    const toAdd = group.filter(passage => !selected.has(passage.index))
    const characters = toAdd.reduce((sum, passage) => sum + passage.text.length + passage.id.length + 15, 0)
    if (characters + size > evidenceLimit) {
      if (!selected.size) throw new Error('The matching attachment passage is too long. Attach a shorter excerpt.')
      continue
    }
    for (const { index, id, page, text } of toAdd) selected.set(index, { index, id, page, text })
    size += characters
  }
  return { terms, passages: [...selected.values()], problemText: anchors[0]?.passage.text }
}

export function packAttachmentEvidence(
  attachment: ChatAttachment,
  ranking: ReturnType<typeof rankAttachmentPassages>,
  maxBlocks = ranking.passages.length
) {
  const passages = ranking.passages.slice(0, maxBlocks).sort((a, b) => a.index - b.index)
  const pages = attachment.pages.flatMap(page => {
    const matches = passages.filter(passage => passage.page === page.page)
    return matches.length ? [{ ...page, text: matches.map(passage => `[Passage ${passage.id}]\n${passage.text}`).join('\n\n') }] : []
  })
  const run: AttachmentRun = {
    sourceCharacters: attachment.pages.reduce((sum, page) => sum + page.text.length, 0),
    selectedCharacters: passages.reduce((sum, passage) => sum + passage.text.length, 0),
    blockCount: passages.length,
    terms: ranking.terms
  }
  return { attachment: pages.length ? { ...attachment, pages } : undefined, run }
}

export function attachmentSearchQuestion(prompt: string, selection?: string, previousQuestion?: string) {
  const reference = selection || (/\b(it|this|that|those|these|they|them)\b/i.test(prompt) ? previousQuestion : undefined)
  return reference ? `${prompt}\n${reference.slice(0, 800)}` : prompt
}
