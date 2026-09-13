export const followUpSuggestionCountOptions = [2, 4] as const
export const minFollowUpSuggestionCount = followUpSuggestionCountOptions[0]
export const defaultFollowUpSuggestionCount = followUpSuggestionCountOptions[1]

export const legacySuggestedFollowUpPrompt =
  'Suggest {count} very short factual follow-up questions that have not been answered yet or cannot be found inspired by the previous conversation and excerpts.'

export const defaultSuggestedFollowUpPrompt =
  [
    'Suggest up to {count} natural next questions a curious undergraduate student might ask after this answer.',
    'Make each question conversational, specific to the concept just discussed, and answerable from the course material.',
    'Each question must attach to a concrete phrase, mechanism, trade-off, or claim in the latest answer.',
    'When possible, ask about an idea the answer used but did not fully explain.',
    'Prefer conceptually linked why/how questions over generic requests for more detail.',
    'Base the questions on the latest answer the student just saw, not earlier turns or unrelated source details.',
    'Do not introduce a term, method, workload, or scenario unless it appeared in the latest answer or current question.',
    'Keep each question short, ideally under 12 words.',
    'Name the subject in every question, including requests for an example or code. Return fewer questions when useful ideas run out.',
    'Phrase them as questions from the student to the tutor, not questions that ask the student to think or recall.',
    'Prefer simple questions about why a named thing matters, how a named mechanism works, concrete examples, intuition, code-level implementation, trade-offs, edge cases, or the workloads already described.',
    'Do not repeat a question the student already asked.',
    'Do not ask for external real-world applications unless the material names one.',
    'Avoid quiz/exam wording, source/context wording, and generic reflection prompts about what is confusing.',
    'Return only the questions, one per line.'
  ].join('\n')

export const defaultStarterQuestionPrompt =
  [
    'Suggest up to {count} first questions a curious undergraduate student might ask when opening these course materials.',
    'Spread the questions across different sections or topics in the selected material.',
    'Return fewer questions when the material does not support more distinct topics.',
    'Make each question technical, specific to the provided material, and useful for starting a serious study session.',
    'Prefer named concepts, mechanisms, code structures, algorithms, implementation details, or trade-offs from the material.',
    'Avoid trivial definition-only questions when a more concrete mechanism, trade-off, or implementation question is possible.',
    'Avoid bare nouns like page, data, query, row, or tuple unless the material names a specific concept around them.',
    'Keep each question short, ideally under 12 words.',
    'Phrase them as questions from the student to the tutor, not questions that ask the student to think or recall.',
    'Mix big-picture motivation with concrete examples, intuition, and code-level "how does this work?" questions from the material.',
    'Do not ask for external real-world applications unless the material names one.',
    'Avoid quiz/exam wording, source/context wording, and generic reflection prompts about what is confusing.',
    'Return only the questions, one per line.'
  ].join('\n')
