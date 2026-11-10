// A shape check, not a truth check. Evidence validation still determines whether
// a generated statement has supporting passages; this rejects obvious questions.
export function trueFalseStatementIssue(value: string): string {
  const prose = value.trim().replace(/^true\s*\/\s*false\s*:\s*/i, '')
    .replace(/`[^`]*`|"[^"\n]*"|“[^”\n]*”/g, 'quoted text')
    .replace(/^[\s*_#]+/, '')
  if (!prose || /[?？]/u.test(prose) ||
      /^(?:why|what|which|who|whom|whose|how|is|are|was|were|does|do|did|can|could|would|should|will|has|have)\b/i.test(prose) ||
      /^(?:when|where)\s+(?:is|are|was|were|does|do|did|can|could|would|should|will|has|have)\b/i.test(prose) ||
      /^(?:(?:please\s+)?(?:explain|describe|identify|compare|discuss|calculate|determine|justify|list|choose|select|name)\b|(?:state|give)\s+(?:why|how|which|what|an?\s+example)\b)/i.test(prose)) {
    return 'This is an open-ended question, not a True/False statement. It cannot be graded. Prepare another question.'
  }
  return ''
}
