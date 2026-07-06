export type CleaningProfileId = 'minimal' | 'course' | 'article'
export type CleaningRuleId =
  | 'normalize_text'
  | 'remove_repeated_edges'
  | 'repair_hyphenated_breaks'
  | 'merge_wrapped_lines'
  | 'detect_article_section_headers'
  | 'detect_chapter_section_headers'

export interface CleaningProfileInfo {
  id: CleaningProfileId
  name: string
  description: string
  defaultRuleIds: CleaningRuleId[]
}

export interface CleaningRuleInfo {
  id: CleaningRuleId
  name: string
  description: string
  locked?: boolean
}

export const defaultCleaningProfileId: CleaningProfileId = 'course'

export const cleaningProfiles: CleaningProfileInfo[] = [
  {
    id: 'course',
    name: 'Textbook',
    description: 'Repairs wrapped lines, removes repeated page headers, and records chapter or section headings.',
    defaultRuleIds: [
      'normalize_text',
      'remove_repeated_edges',
      'repair_hyphenated_breaks',
      'merge_wrapped_lines',
      'detect_chapter_section_headers'
    ]
  },
  {
    id: 'article',
    name: 'Article',
    description: 'Builds cleaner paragraphs for web articles, encyclopedia PDFs, and reports.',
    defaultRuleIds: [
      'normalize_text',
      'remove_repeated_edges',
      'repair_hyphenated_breaks',
      'merge_wrapped_lines',
      'detect_article_section_headers'
    ]
  },
  {
    id: 'minimal',
    name: 'Minimal',
    description: 'Only normalizes spacing and blank lines.',
    defaultRuleIds: ['normalize_text']
  }
]

export const cleaningRules: CleaningRuleInfo[] = [
  {
    id: 'normalize_text',
    name: 'Normalize text',
    description: 'Removes null characters and normalizes spaces, line endings, and blank lines.',
    locked: true
  },
  {
    id: 'remove_repeated_edges',
    name: 'Remove repeated page headers and footers',
    description: 'Drops repeated edge lines that appear across multiple pages.'
  },
  {
    id: 'repair_hyphenated_breaks',
    name: 'Repair hyphenated line breaks',
    description: 'Joins words split across PDF line breaks.'
  },
  {
    id: 'merge_wrapped_lines',
    name: 'Merge wrapped paragraph lines',
    description: 'Builds paragraphs from lines that were wrapped by PDF extraction.'
  },
  {
    id: 'detect_article_section_headers',
    name: 'Detect article section headers',
    description: 'Records common wiki/article headings such as Career, References, and External links as chunk context.'
  },
  {
    id: 'detect_chapter_section_headers',
    name: 'Detect chapter section headers',
    description: 'Records textbook headings such as Chapter 12 or 12.1 Storage as chunk context.'
  }
]

export function cleaningProfileLabel(profileId?: string): string {
  return cleaningProfiles.find((profile) => profile.id === profileId)?.name ?? 'Textbook'
}

export function defaultCleaningRuleIdsForProfile(profileId?: string): CleaningRuleId[] {
  return [
    ...(cleaningProfiles.find((profile) => profile.id === profileId)?.defaultRuleIds ??
      cleaningProfiles.find((profile) => profile.id === defaultCleaningProfileId)?.defaultRuleIds ??
      ['normalize_text'])
  ]
}

export function normalizeCleaningRuleIds(
  ruleIds: readonly string[] | undefined,
  profileId?: string
): CleaningRuleId[] {
  const selected = ruleIds?.length ? [...ruleIds] : defaultCleaningRuleIdsForProfile(profileId)
  const knownIds = new Set(cleaningRules.map((rule) => rule.id))
  const normalized = selected.filter((ruleId): ruleId is CleaningRuleId =>
    knownIds.has(ruleId as CleaningRuleId)
  )

  if (!normalized.includes('normalize_text')) {
    normalized.unshift('normalize_text')
  }

  return cleaningRules
    .map((rule) => rule.id)
    .filter((ruleId): ruleId is CleaningRuleId => normalized.includes(ruleId))
}
