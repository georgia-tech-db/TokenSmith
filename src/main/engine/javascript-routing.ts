export type JavascriptQuestionRoute = 'direct' | 'tool' | 'model'

const number = String.raw`[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?`
const singleOperation = new RegExp(
  String.raw`^\s*(?:(?:what(?:'s| is)|calculate|compute|evaluate)\s+)?${number}\s*(?:[+×*÷/−-]|plus|minus|times|multiplied by|divided by)\s*${number}\s*[?.!]?\s*$`,
  'i'
)
const derivedMetric = /\b(?:average|mean|median|percentage|percent|weighted|consecutive|difference|gap|variance|standard deviation|subtotal|tax|expected value|break[ -]even|ratio|speed|sum|total)\b/i
const computationVerb = /\b(?:calculate|compute|convert|conversion|solve|evaluate|derive)\b/i
const sourceLookup = /\b(?:according to|which|who|when|where|identify|name|list|how many)\b/i
const explanationRequest = /\b(?:explain|describe|define|what (?:is|are) (?:the )?formula|give (?:me )?the formula|state (?:the )?formula)\b/i
const socialMessage = /^\s*(?:(?:hi|hello|hey)(?: there)?|good (?:morning|afternoon|evening)|how are you|thank you|thanks|goodbye|bye)[.!?]*\s*$/i

// Only take deterministic routes for clear requests. The model chooses when
// neither the direct-answer nor the multi-step calculation intent is clear.
export function javascriptRouteForQuestion(question: string, hasSources: boolean): JavascriptQuestionRoute {
  if (socialMessage.test(question)) return 'direct'
  if (singleOperation.test(question)) return 'direct'
  if (explanationRequest.test(question) && !computationVerb.test(question)) return 'direct'
  if (/\b(?:how many|count)\b/i.test(question) &&
    !derivedMetric.test(question) && !computationVerb.test(question)) return 'direct'
  if (derivedMetric.test(question) || computationVerb.test(question)) return 'tool'
  if (hasSources && sourceLookup.test(question)) return 'direct'
  return 'model'
}
