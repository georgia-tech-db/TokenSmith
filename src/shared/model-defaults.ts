export const followUpSuggestionCountOptions = [2, 4] as const
export const minFollowUpSuggestionCount = followUpSuggestionCountOptions[0]
export const defaultFollowUpSuggestionCount = followUpSuggestionCountOptions[1]

export const defaultSuggestedFollowUpPrompt =
  'Suggest {count} very short factual follow-up questions that have not been answered yet or cannot be found inspired by the previous conversation and excerpts.'
