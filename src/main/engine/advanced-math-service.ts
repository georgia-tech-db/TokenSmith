import type { AdvancedMathPlan, EngineChatRequest, EngineQuestionRewriteRequest, QuestionRewrite } from '../../shared/engine'
import type { LocalModel, ModelRuntimeSettings } from '../../shared/app-state'
import type { OllamaStatus } from '../../shared/ollama'
import { defaultOllamaBaseUrl } from '../../shared/ollama'
import { advancedMathBucket, advancedMathModelName, advancedMathModelLabel, advancedMathModelInstalled } from '../../shared/advanced-math'
import { lastChatExchange } from '../../shared/study-chat-pipeline'
import { reasoningMode } from '../../shared/reasoning'

interface Dependencies {
  resolve: (request: EngineQuestionRewriteRequest, signal?: AbortSignal) => Promise<QuestionRewrite>
  status: () => Promise<OllamaStatus>
  log?: (event: string, detail: Record<string, unknown>) => void
}

export function createAdvancedMathService(dependencies: Dependencies) {
  let cachedStatus: { until: number; status: OllamaStatus } | undefined
  async function mathModel(settings: EngineQuestionRewriteRequest['settings'], signal?: AbortSignal) {
    signal?.throwIfAborted()
    let status = cachedStatus && cachedStatus.until > Date.now() ? cachedStatus.status : undefined
    if (!status) {
      status = await dependencies.status()
      cachedStatus = { status, until: Date.now() + (advancedMathModelInstalled(status) ? 30_000 : 1_000) }
    }
    signal?.throwIfAborted()
    const installed = status.models.find(m => m.name.toLowerCase() === advancedMathModelName)
    if (!status.running) throw new Error('Advanced Math needs Ollama running. Start Ollama and try again; your default chat model is unchanged.')
    if (!installed) throw new Error(`Advanced Math requires ${advancedMathModelName}. Install it in Models; your default chat model is unchanged.`)
    const model: LocalModel = {
      id: `ollama:${advancedMathModelName}`, name: advancedMathModelLabel, engine: 'ollama', source: 'ollama', role: 'generator',
      status: 'ready', ollamaModelName: advancedMathModelName, ollamaBaseUrl: defaultOllamaBaseUrl,
      sizeBytes: installed.size, contextLength: installed.details?.contextLength, quant: installed.details?.quantizationLevel,
      addedAt: installed.modifiedAt ?? ''
    }
    // Dedicated profile: never inherit the default chat model's persona,
    // endpoint or credentials. Preserve the tested loop's sampling overrides.
    const overrides = settings?.modelSettingsById?.[model.id]
    const modelSettings: ModelRuntimeSettings = {
      systemMessage: '', suggestedFollowUpPrompt: '', contextLength: 8192, contextLengthMode: 'manual', reasoningMode: 'off', thinking: false, maxLength: 1536,
      temperature: 0, topP: 0.4, topK: 40, minP: 0, repeatPenalty: 1.18, ...overrides
    }
    return { model, modelSettings }
  }

  async function resolve(request: EngineQuestionRewriteRequest, signal?: AbortSignal): Promise<QuestionRewrite> {
    if (!request.advancedMathEnabled) return dependencies.resolve(request, signal)
    const started = performance.now()
    const bucket = advancedMathBucket(request.prompt)
    const hasContext = Boolean(request.selectedPassage || lastChatExchange(request.messages))
    const suppliedNumbers = (request.prompt.match(/\d+(?:\.\d+)?/g) ?? []).length
    const referencesContext = /\b(?:it|that|those|their|above|same)\b/i.test(request.prompt) ||
      suppliedNumbers < 2 && /\b(?:this|these)\b/i.test(request.prompt)
    const needsContext = hasContext && (bucket === 'ambiguous' || bucket === 'math' && suppliedNumbers === 0 ||
      request.resolveContext === true || Boolean(request.selectedPassage) || referencesContext)
    const resolveReasoning = bucket === 'chat' && reasoningMode(request.modelSettings) === 'auto'
    const resolution: QuestionRewrite = needsContext || bucket === 'ambiguous' || resolveReasoning
      ? await dependencies.resolve({ ...request, messages: needsContext ? request.messages : [], classifyAdvancedMath: bucket === 'ambiguous' }, signal)
      : { mode: 'standalone', query: request.prompt, clarification: '', reasoning: false }
    signal?.throwIfAborted()
    if (resolution.mode === 'clarify') {
      dependencies.log?.('advanced_math_route', { requestId: request.requestId, prompt: request.prompt, bucket, route: 'clarify',
        reason: resolution.routeReason ?? 'The task or referent needs clarification.', defaultModel: request.model.name,
        classificationCall: bucket === 'ambiguous', sharedContextResolution: needsContext, latencyMs: performance.now() - started })
      return resolution
    }
    const route = bucket === 'ambiguous' ? resolution.answerRoute : bucket
    if (route !== 'chat' && route !== 'math') throw new Error('Could not determine whether this question needs Advanced Math. Please clarify the requested task.')
    const reason = bucket === 'ambiguous' ? resolution.routeReason ?? 'Default model classified task intent.' : `Clear ${route === 'math' ? 'multi-step computation' : 'ordinary chat'} intent.`
    const selected = route === 'math' ? await mathModel(request.settings, signal) : {}
    const plan: AdvancedMathPlan = { prompt: request.prompt, route, reason, ...selected }
    dependencies.log?.('advanced_math_route', {
      prompt: request.prompt, bucket, route, reason, defaultModel: request.model.name,
      requestId: request.requestId, defaultModelId: request.model.id,
      answerModel: plan.model?.name ?? request.model.name, classificationCall: bucket === 'ambiguous',
      sharedContextResolution: needsContext, latencyMs: performance.now() - started
    })
    return { ...resolution, advancedMathPlan: plan }
  }

  async function prepare(request: EngineChatRequest, signal?: AbortSignal): Promise<EngineChatRequest | QuestionRewrite> {
    if (!(request.advancedMathEnabled ?? request.applicationSettings?.advancedMathEnabled) || request.answerToSimplify || request.practiceTask) return request
    let plan = request.advancedMathPlan
    if (!plan || plan.prompt !== request.prompt) {
      const resolution = await resolve({ ...request, advancedMathEnabled: true }, signal)
      if (resolution.mode === 'clarify') return resolution
      plan = resolution.advancedMathPlan!
      request = { ...request, advancedMathPlan: plan, conversationContextMode: resolution.mode, reasoning: resolution.reasoning,
        referenceExchange: resolution.mode === 'contextual' ? request.referenceExchange ?? lastChatExchange(request.messages) : undefined }
    }
    if (plan.route === 'chat') return { ...request, advancedMathPlan: plan }
    // Revalidate installation using the cached local inventory, rather than
    // trusting model descriptors received through IPC or swapping UI selection.
    const selected = await mathModel(request.settings, signal)
    return { ...request, ...selected, advancedMathPlan: { ...plan, ...selected } }
  }
  return { resolve, prepare }
}
