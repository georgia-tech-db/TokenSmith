import type { EngineChatRequest } from '../../shared/engine'
import { attachmentSearchQuestion, packAttachmentEvidence, rankAttachmentPassages } from '../../shared/attachment-search'
import { answerTokenAllowance, effectiveContextLength } from '../../shared/model-context'
import { reasoningMode } from '../../shared/reasoning'
import { startLatencySpan } from '../../shared/latency-trace'
import { AttachmentContextError, prepareStudyChatMessages } from './study-chat-format'

export function fitAttachmentContext(request: EngineChatRequest) {
  const { attachmentRun: run, attachmentSource: source, modelSettings } = request
  if (!run?.blockCount || !source || !modelSettings || request.practiceTask) {
    return { request, span: undefined }
  }

  const timer = startLatencySpan('Attachment context fit')
  const requested = answerTokenAllowance(request.model, modelSettings)
  const smallContext = effectiveContextLength(request.model, modelSettings) <= 8192
  const standardAnswer = reasoningMode(modelSettings) !== 'on' &&
    !(request.applicationSettings?.explanationDepthEnabled && request.applicationSettings.explanationDepth === 'detailed')
  // On small models, reserve room for evidence alongside ordinary study answers.
  const allowance = smallContext && standardAnswer ? Math.min(requested, 1024) : requested
  const limits = [...new Set([allowance, 2048, 1024, 512, 256])].filter(limit => limit <= allowance)
  const searchQuestion = attachmentSearchQuestion(request.prompt, request.selectedPassage?.text, request.referenceExchange?.question)
  const ranking = rankAttachmentPassages(source, searchQuestion)
  let lastError: AttachmentContextError | undefined

  for (const answerTokens of limits) {
    for (let count = ranking.passages.length; count > 0; count = count === 1 ? 0 : Math.ceil(count / 2)) {
      const selected = packAttachmentEvidence(source, ranking, count)
      const candidate: EngineChatRequest = {
        ...request,
        attachment: selected.attachment,
        modelSettings: { ...modelSettings, maxLength: answerTokens },
        attachmentRun: { ...selected.run, answerTokens }
      }
      try {
        prepareStudyChatMessages(candidate)
        return {
          request: candidate,
          span: timer.finish({ outCount: count, details: {
            model_requests: 0, source_characters: run.sourceCharacters,
            selected_characters: selected.run.selectedCharacters,
            requested_answer_tokens: requested, answer_tokens: answerTokens
          } })
        }
      } catch (error) {
        if (!(error instanceof AttachmentContextError)) throw error
        lastError = error
      }
    }
  }
  throw lastError ?? new Error('The best attachment passage cannot fit. Use a shorter excerpt or a larger context window.')
}
