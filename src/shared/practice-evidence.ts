const whitespace = (value: string) => value.replace(/\s+/g, ' ').trim()
// PDF extraction often separates punctuation from the word before it, as in "N3 ."
const punctuationSpacing = (value: string) => whitespace(value).replace(/ ([.,;:!?)\]])/g, '$1').replace(/([(\[]) /g, '$1')

// Compare presentation syntax, never paraphrases or algebraic equivalence. Keep
// identifiers, numbers, operators, subscripts and superscripts distinguishable.
function presentationText(value: string): string {
  return whitespace(value.split(/(```[\s\S]*?```|`[^`\n]+`)/g).map((part, index) => {
    if (index % 2) return part
    return part
      .replace(/<sub>([\w]+)<\/sub>/gi, '_{$1}')
      .replace(/<sup>([\w]+)<\/sup>/gi, '^{$1}')
      .replace(/\$\$?([^$\n]+)\$\$?/g, '$1')
      .replace(/\\\(([\s\S]*?)\\\)/g, '$1')
      .replace(/\\(?:mathrm|mathit|mathbf|text)\{([\w ]+)\}/g, '$1')
      .replace(/\\times\b/g, '×')
      .replace(/\\(?:leq|le)\b/g, '≤')
      .replace(/\\(?:geq|ge)\b/g, '≥')
      .replace(/\\(?:neq|ne)\b/g, '≠')
      .replace(/(?<!\w)(\*{1,2})(\S(?:[^*]*?\S)?)\1(?!\w)/g, '$2')
      .replace(/\s*([<>=≤≥≠×])\s*/g, '$1')
  }).join(''))
}

export function hasSupportingQuote(passage: string, quote: string): boolean {
  const exact = whitespace(quote)
  if (!exact) return false
  if (whitespace(passage).includes(exact)) return true
  if (punctuationSpacing(passage).includes(punctuationSpacing(quote))) return true
  const presented = presentationText(quote)
  return Boolean(presented) && presentationText(passage).includes(presented)
}
