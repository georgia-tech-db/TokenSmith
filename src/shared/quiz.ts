export const quizTotalQuestions = 5
export const quizSourceLimit = 4
export const quizSourcePoolSize = 32

const quizFocusAreas = [
  'core definitions and central concepts',
  'important properties, theorems, conditions, or rules',
  'worked examples, procedures, or applications',
  'comparisons, contrasts, or relationships between concepts',
  'edge cases, limitations, common mistakes, or consequences'
]

function quizFocusArea(questionNumber: number): string {
  return quizFocusAreas[Math.max(0, questionNumber - 1) % quizFocusAreas.length]
}

function previousQuestionBlock(previousQuestions: string[]): string[] {
  if (previousQuestions.length === 0) {
    return ['Previous quiz questions: none.']
  }

  return [
    'Previous quiz questions to avoid repeating or paraphrasing:',
    ...previousQuestions.map((question, index) => `${index + 1}. ${question}`)
  ]
}

export function quizQuestionPrompt({
  questionNumber,
  previousQuestions = [],
  totalQuestions
}: {
  questionNumber: number
  previousQuestions?: string[]
  totalQuestions: number
}): string {
  return [
    `Create quiz question ${questionNumber} of ${totalQuestions}.`,
    `Focus on ${quizFocusArea(questionNumber)}.`,
    'Make it one short-answer question from the provided PDF excerpts.',
    'Ask about one concrete course concept, theorem, definition, example, procedure, or relationship that the excerpts support.',
    'Prefer questions that test understanding, such as why, how, explain, compare, apply, or what condition must hold.',
    'Avoid broad book-overview questions such as "What is this book about?" or "What is this book on?"',
    'Avoid trivial wording, document metadata, title pages, copyright or licensing text, tables of contents, prefaces, acknowledgments, author names, file names, and page-location facts unless they are central to the course content.',
    ...previousQuestionBlock(previousQuestions),
    'Do not repeat or lightly rephrase a previous quiz question.',
    'Do not answer the question.',
    'Do not mention page numbers, page ranges, locators, source labels, excerpt labels, or where the answer appears.',
    'Do not ask questions whose answer depends on knowing a page number or location.',
    'Return only the question.'
  ].join('\n')
}

export function quizFeedbackPrompt({
  answer,
  question,
  questionNumber,
  totalQuestions
}: {
  answer: string
  question: string
  questionNumber: number
  totalQuestions: number
}): string {
  return [
    'You are running a short-answer study quiz.',
    'Grade the student answer using only the provided PDF excerpts.',
    'Be concise and specific.',
    'Address the learner directly as "you". Do not say "the student".',
    'Do not start with "Feedback:".',
    'Use this structure exactly:',
    'Good/Partial/Needs work. One concise sentence explaining the grade.',
    'Expected answer:',
    '- One concise answer.',
    'Do not ask a new quiz question.',
    'Do not mention page numbers, page ranges, locators, source labels, excerpt labels, or where the answer appears.',
    'Refer to the concept directly instead of saying phrases like "as stated on page 36".',
    '',
    `Quiz question ${questionNumber} of ${totalQuestions}: ${question}`,
    `Student answer: ${answer}`
  ].join('\n')
}
