import { PracticeScreen } from './PracticeScreen'
import { StudyHeader, type StudyPanels } from './StudyHeader'
import { StudySidebar } from './StudySidebar'
import { StudyMaterialsPane } from './StudyMaterialsPane'
import { SourceTray, maxSourceTrayCards } from './SourceTray'
import { canOpenSource, cleanMaterialTitle, isMarkdownSource, isPdfSource } from './source-display'
import { useStudyDocuments, type StudyCatalog } from './hooks/useStudyDocuments'
import { changeStudyScope, initializeStudyScopes, studyScopeKey, sourcesInScope, scopeRefs, type StudyDocumentRef } from '@shared/study-scope'
import { PracticeChatSuggestions } from './PracticeChatSuggestions'
import { emptyPracticeState, normalizePracticeState, practiceDiscussionPassage, practiceChatConversation, type PracticeTopic, type PracticeSession, type PracticeQuestion, type PracticeState, type StudyMode } from '@shared/practice'
import { SourceNavigation, type SourceNavigationProps } from './SourceNavigation'
import { sourceEntryKey, sourceNavigationList } from './source-navigation'
import { useSourceReader, type SourceReaderState } from './hooks/useSourceReader'
import { createAnswerSourcePreview, type AnswerSourceDocument } from './answer-source-preview'
import { GenerationStatus } from './GenerationStatus'
import { useGenerationProgress, type GenerationProgressApi } from './hooks/useGenerationProgress'
import type { GenerationKind } from '@shared/generation-estimate'
import { LibraryWorkspace } from './LibraryWorkspace'
import { defaultPreparation, type IndexMaterialOptions } from '../../shared/preparation'
import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import type { CSSProperties, FormEvent, ReactNode } from 'react'
import { MessageText } from './MessageText'
import { MessageAvatar } from './MessageAvatar'
import { MessageComposer } from './MessageComposer'
import { MarkdownSourceViewer } from './MarkdownSourceViewer'
import { ConversationViewport } from './ConversationViewport'
import { QuestionEditor } from './QuestionEditor'
import { normalizeEmbeddingGpuEnabled } from '@shared/embedding-settings'
import { canExplainSimpler, canRetryWithReasoning, reasoningRetryRequest, replaceQuestion, simplerExplanationRequest, type ChatDraft } from './chat-interactions'
import { SelectedPassagePreview } from './SelectedPassagePreview'
import './chat-interactions.css'
import { ThemePicker } from './ThemePicker'
import { ChatModelPicker } from './ChatModelPicker'
import { ChatDepthPicker } from './ChatDepthPicker'
import { AnswerExplanation } from './AnswerExplanation'
import { answerForDisplay } from '../../shared/study-chat-pipeline'
import { CloudGeneratorDialog } from './CloudGeneratorDialog'
import { answerModel, isAnswerModel, migrateConversationModels, selectStudyModel, studyModelTarget, type ModelSelectionTarget } from '@shared/chat-model-selection'
import { attachCloudProviderConnection, disconnectCloudConnectionModels, isCloudGenerator, mergeCloudConnectionModel } from '@shared/cloud-generators'
import './cloud-generators.css'
import * as pdfjsLib from 'pdfjs-dist'
import pdfWorkerUrl from 'pdfjs-dist/build/pdf.worker.mjs?url'
import type { PDFDocumentProxy, TextItem } from 'pdfjs-dist/types/src/display/api'
import type {
  ApplicationSettings,
  AnswerView,
  AppStateSnapshot,
  ChatMessage,
  ChatSelectedPassage,
  ChatSource,
  Conversation,
  CourseMaterial,
  ExplanationDepth,
  LocalModel,
  LocalModelRole,
  MaterialIndexProgress,
  ModelDownloadProgress,
  ModelRuntimeSettings,
  ScreenId,
  SuggestionMode,
  TokenSmithSettings
} from '@shared/app-state'
import type {
  CleaningPreviewResult,
  EngineInfo,
  EngineChatRequest,
  EngineChatResponse,
  MarkdownSourceDocument,
  PdfSourceDocument,
  TokenSmithLogFile
} from '@shared/engine'
import {
  cleaningRules,
  cleaningProfileLabel,
  cleaningProfiles,
  defaultCleaningProfileId,
  defaultCleaningRuleIdsForProfile,
  normalizeCleaningRuleIds,
  type CleaningProfileId,
  type CleaningRuleId
} from '@shared/cleaning'
import {
  defaultOllamaBaseUrl,
  recommendedOllamaChatModel,
  recommendedOllamaEmbeddingModel,
  type OllamaModelInfo,
  type OllamaPullProgress,
  type OllamaSearchResult,
  type OllamaStatus
} from '@shared/ollama'
import { formatModelArtifactSize, localModelCatalog } from '@shared/model-catalog'
import { recommendModel, type ModelRecommendation } from '@shared/model-recommendation'
import { defaultDeviceTierPolicy } from '@shared/device-tier-policy'
import { fontSizeOptions, normalizeFontSize } from '@shared/typography'
import {
  defaultFollowUpSuggestionCount,
  defaultStarterQuestionPrompt,
  defaultSuggestedFollowUpPrompt,
  followUpSuggestionCountOptions,
  normalizeSuggestedFollowUpPrompt,
  normalizeStarterQuestionPrompt,
  minFollowUpSuggestionCount
} from '@shared/model-defaults'
import { DeviceRecommendationPanel } from './components/DeviceRecommendationPanel'
import { useDeviceCapabilities } from './hooks/useDeviceCapabilities'
import { modelAwareRetrievalLimit } from '@shared/retrieval-budget'
import { contextMetadataIdentity, fallbackContextLength, mergeModelContextMetadata, migratedContextMode, normalizeTokenLimit,
  reportedContextLimit, usesAutomaticContext } from '@shared/model-context'
import { prepareRewrittenStudyChat } from '@shared/study-chat-pipeline'
import { combineLatencyTrace, type LatencySpan } from '@shared/latency-trace'
import tokensmithRailWordmark from './assets/tokensmith-rail-wordmark.png'
import {
  BookOpen,
  Check,
  ChevronDown,
  Copy,
  Database,
  Download as DownloadIcon,
  FileText,
  Library,
  Loader2,
  MessageSquare,
  Pause,
  Pencil,
  Plus,
  RefreshCw,
  Search,
  Settings,
  Sparkles,
  Trash2,
  X
} from 'lucide-react'

pdfjsLib.GlobalWorkerOptions.workerSrc = pdfWorkerUrl

interface NavItem {
  id: ScreenId
  label: string
  icon: typeof MessageSquare
}

type PdfViewerState = PdfSourceDocument & { searchTerm?: string }
type PdfRenderedTextItem = {
  index: number
  text: string
  left: number
  top: number
  width: number
  height: number
}
type PdfRenderedPage = {
  pageNumber: number
  width: number
  height: number
  imageDataUrl: string
  textItems: PdfRenderedTextItem[]
}
type PdfSearchMatch = {
  pageNumber: number
  itemIndex: number
}
type ParsedLogEntry = {
  id: string
  event: string
  time?: string
  data?: Record<string, unknown>
  raw?: string
}

const navItems: NavItem[] = [
  { id: 'chat', label: 'Chat', icon: MessageSquare },
  { id: 'library', label: 'Library', icon: Library },
  { id: 'models', label: 'Models', icon: Database },
  { id: 'settings', label: 'Settings', icon: Settings }
]

const stateStorageKey = 'tokensmith-app-state-v1'
const defaultCollectionChunkSize = 1000
const pdfViewerRenderScale = 1.45
// React Strict Mode immediately runs an effect setup/cleanup cycle in development.
// This gives that cleanup time to cancel before a starter-question request is sent.
const starterQuestionDebounceMs = 150

const defaultMaterials: CourseMaterial[] = []

const defaultModelRuntimeSettings: ModelRuntimeSettings = {
  systemMessage: '',
  suggestedFollowUpPrompt: defaultSuggestedFollowUpPrompt,
  starterQuestionPrompt: defaultStarterQuestionPrompt,
  contextLength: 8192,
  contextLengthMode: 'auto',
  maxLength: 1536,
  reasoningMode: 'auto',
  temperature: 0.7,
  topP: 0.4,
  topK: 40,
  minP: 0,
  repeatPenalty: 1.18,
}

const defaultApplicationSettings: ApplicationSettings = {
  theme: 'light',
  fontSize: 'normal',
  defaultModelId: '',
  suggestionMode: 'on',
  searchMode: 'hybrid',
  explanationDepthEnabled: false,
  explanationDepth: 'standard',
  followUpSuggestionCount: defaultFollowUpSuggestionCount,
  showSources: true,
  embeddingGpuEnabled: true
}

const defaultSettings: TokenSmithSettings = {
  maxSources: 4,
  application: defaultApplicationSettings,
  modelDefaults: defaultModelRuntimeSettings,
  modelSettingsById: {}
}

const defaultModels: LocalModel[] = []

const starterConversations: Conversation[] = [
  {
    id: 'starter-study-chat',
    modelId: '',
    title: 'New Study Chat',
    period: 'Today',
    messages: []
  }
]

const defaultAppState: AppStateSnapshot = {
  version: 1,
  appVersion: 'dev',
  activeScreen: 'chat',
  activeConversationId: starterConversations[0].id,
  conversations: starterConversations,
  materials: defaultMaterials,
  models: defaultModels,
  selectedEmbeddingModelId: '',
  settings: defaultSettings,
  updatedAt: new Date(0).toISOString()
}

function createFreshConversation(modelId: string): Conversation {
  return {
    id: createId('conversation'),
    modelId,
    title: 'New Study Chat',
    period: 'Today',
    messages: []
  }
}

function createId(prefix: string) {
  return `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`
}

function formatBytes(bytes?: number) {
  if (bytes === undefined || bytes === null) {
    return 'Local'
  }

  if (bytes < 1024) {
    return `${bytes} B`
  }

  if (bytes < 1024 * 1024 * 1024) {
    return `${(bytes / 1024 / 1024).toFixed(1)} MB`
  }

  return `${(bytes / 1024 / 1024 / 1024).toFixed(2)} GB`
}

function formatNumber(value: number) {
  return new Intl.NumberFormat().format(value)
}

function pathLeaf(path?: string) {
  return path?.replace(/^file:\/\//, '').split(/[\\/]/).filter(Boolean).pop()
}

function parseTokenSmithLog(text: string): ParsedLogEntry[] {
  return text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line, index) => {
      try {
        const parsed = JSON.parse(line) as Record<string, unknown>
        const { event, time, ...detail } = parsed
        const eventName = String(event ?? 'log_event')

        return {
          id: `${index}-${String(time ?? '')}-${String(event ?? 'event')}`,
          event: eventName,
          time: typeof time === 'string' ? time : undefined,
          data: detail
        }
      } catch {
        return {
          id: `${index}-raw`,
          event: 'raw_log_line',
          raw: line
        }
      }
    })
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value && typeof value === 'object' && !Array.isArray(value))
}

function logString(value: unknown): string | undefined {
  return typeof value === 'string' && value.trim() ? value : undefined
}

function logRecords(value: unknown): Record<string, unknown>[] {
  return Array.isArray(value) ? value.filter(isRecord) : []
}

function logJson(value: unknown) {
  return JSON.stringify(value, null, 2)
}

function prettyLogEventName(event: string) {
  return event.replaceAll('_', ' ')
}

function logModelLabel(model: unknown) {
  if (!isRecord(model)) {
    return undefined
  }

  return [model.name, model.ollamaModelName, model.remoteModelName, model.engine]
    .map((value) => logString(value))
    .find(Boolean)
}

function logSourceTitle(source: Record<string, unknown>, index: number) {
  return (
    logString(source.title) ??
    logString(source.documentTitle) ??
    logString(source.collectionName) ??
    `Source ${index + 1}`
  )
}

function logSourceLocator(source: Record<string, unknown>) {
  const locator = logString(source.locator)
  const pageStart = typeof source.pageStart === 'number' ? source.pageStart : undefined
  const pageEnd = typeof source.pageEnd === 'number' ? source.pageEnd : undefined

  if (locator) {
    return locator
  }

  if (pageStart && pageEnd && pageEnd !== pageStart) {
    return `Pages ${pageStart}-${pageEnd}`
  }

  if (pageStart) {
    return `Page ${pageStart}`
  }

  return undefined
}

function logSourceScore(source: Record<string, unknown>) {
  return typeof source.score === 'number' ? source.score.toFixed(3) : undefined
}

function modelFilename(model: LocalModel) {
  return model.filename ?? pathLeaf(model.path)
}

function normalizeModelFilename(filename?: string) {
  return filename?.toLowerCase()
}

function normalizeCollectionPath(path: string) {
  return path.trim().replace(/^file:\/\//, '')
}

function materialIdentityKey(material: Pick<CourseMaterial, 'detail' | 'embeddingModel' | 'embeddingModelId' | 'path' | 'title'>) {
  const locationKey = material.path ?? `${material.title}:${material.detail}`
  const embeddingKey = material.embeddingModelId ?? material.embeddingModel ?? ''
  return `${locationKey}::${embeddingKey}`
}

function normalizeIndexingProgress(material: CourseMaterial): MaterialIndexProgress | undefined {
  if (material.status !== 'indexing' && material.status !== 'paused') {
    return material.indexing
  }

  const progress = material.indexing as (Partial<MaterialIndexProgress> & { phase?: string }) | undefined
  const rawPhase = progress?.phase as string | undefined
  const materialId = progress?.materialId ?? material.id
  const migratedPhase: MaterialIndexProgress['phase'] =
    rawPhase === 'chunking' ||
    rawPhase === 'embedding' ||
    rawPhase === 'saving' ||
    rawPhase === 'complete' ||
    rawPhase === 'error'
      ? rawPhase
      : 'parsing'
  const rawMessage = progress?.message
  const migratedMessage =
    rawPhase === 'queued'
      ? 'Parsing'
      : rawPhase === 'extracting'
        ? rawMessage?.replace(/^Reading/, 'Parsing') || 'Parsing'
        : rawMessage || 'Parsing'

  return {
    materialId,
    phase: migratedPhase,
    percent: Math.max(rawPhase === 'queued' ? 1 : 0, Math.min(100, Math.round(progress?.percent ?? 1))),
    processedFiles: progress?.processedFiles ?? 0,
    totalFiles: progress?.totalFiles ?? material.fileCount ?? 0,
    processedEmbeddings: progress?.processedEmbeddings ?? 0,
    totalEmbeddings: progress?.totalEmbeddings ?? 0,
    message: migratedMessage
  }
}

function titleCaseModelToken(token: string) {
  const upperToken = token.toUpperCase()

  if (upperToken === 'BGE') {
    return upperToken
  }

  if (/^\d+(?:\.\d+)?B$/i.test(token)) {
    return upperToken
  }

  if (/^v\d+(?:\.\d+)?$/i.test(token)) {
    return token.toLowerCase()
  }

  return token.charAt(0).toUpperCase() + token.slice(1)
}

function modelNameFromPath(modelPath?: string) {
  if (!modelPath) {
    return undefined
  }

  const fileName = modelPath
    .split(/[\\/]/)
    .pop()
    ?.replace(/\.(gguf|bin)$/i, '')

  if (!fileName) {
    return undefined
  }

  const withoutQuant = fileName
    .replace(/[-_. ]q\d(?:[-_.][a-z0-9]+)*$/i, '')
    .replace(/[-_. ](?:f16|fp16|bf16|int8|int4)$/i, '')
  const tokens = withoutQuant
    .split(/[-_ ]+/)
    .map((token) => token.trim())
    .filter(Boolean)

  if (tokens.length === 0) {
    return undefined
  }

  return tokens.map(titleCaseModelToken).join(' ')
}

function displayModelName(model: LocalModel) {
  return modelNameFromPath(model.path) ?? model.name
}

function modelRole(model: LocalModel): LocalModelRole {
  if (model.role) {
    return model.role
  }

  if (model.engine === 'remote') {
    return 'generator'
  }

  return 'generator'
}

function modelCanGenerate(model: LocalModel) {
  const role = modelRole(model)
  return model.status !== 'needsRuntime' && model.status !== 'missing' && (role === 'generator' || role === 'both')
}

function modelCanEmbed(model: LocalModel) {
  const role = modelRole(model)
  return model.status !== 'needsRuntime' && model.status !== 'missing' && (role === 'embedder' || role === 'both')
}

function ollamaChatModelId(modelName: string) {
  return `ollama:${modelName.trim().toLowerCase() || recommendedOllamaChatModel}`
}

function ollamaEmbedderModelId(modelName: string) {
  return `ollama:${modelName.trim().toLowerCase() || recommendedOllamaEmbeddingModel}`
}

function ollamaModelNameMatches(installedModelName: string, requestedModelName: string) {
  const installed = installedModelName.trim().toLowerCase().replace(/:latest$/, '')
  const requested = requestedModelName.trim().toLowerCase().replace(/:latest$/, '')
  return installed === requested || installed.startsWith(`${requested}:`)
}

function ollamaModelInfoMatches(modelInfo: OllamaModelInfo, modelName: string) {
  return [modelInfo.name, modelInfo.model]
    .filter((name): name is string => Boolean(name))
    .some((name) => ollamaModelNameMatches(name, modelName))
}

function createOllamaChatModel(
  modelName = recommendedOllamaChatModel,
  baseUrl = defaultOllamaBaseUrl,
  modelInfo?: OllamaModelInfo,
  status: LocalModel['status'] = 'ready',
  download?: ModelDownloadProgress
): LocalModel {
  const normalizedModelName = modelName.trim() || recommendedOllamaChatModel

  return {
    id: ollamaChatModelId(normalizedModelName),
    name: `Ollama ${normalizedModelName}`,
    engine: 'ollama',
    role: 'generator',
    status,
    source: 'ollama',
    filename: normalizedModelName,
    ollamaModelName: normalizedModelName,
    ollamaBaseUrl: baseUrl,
    sizeBytes: modelInfo?.size,
    contextLength: modelInfo?.details?.contextLength,
    parameters: modelInfo?.details?.parameterSize,
    quant: modelInfo?.details?.quantizationLevel,
    type: modelInfo?.details?.family ? `Ollama ${modelInfo.details.family}` : 'Ollama chat',
    description: [
      'Local chat model served by Ollama',
      'Used for answers after TokenSmith retrieves enabled source excerpts',
      'Works with enabled PDFs prepared by TokenSmith'
    ],
    download,
    addedAt: new Date().toISOString()
  }
}

function createOllamaEmbedderModel(
  modelName = recommendedOllamaEmbeddingModel,
  baseUrl = defaultOllamaBaseUrl,
  modelInfo?: OllamaModelInfo,
  status: LocalModel['status'] = 'ready',
  download?: ModelDownloadProgress
): LocalModel {
  const normalizedModelName = modelName.trim() || recommendedOllamaEmbeddingModel

  return {
    id: ollamaEmbedderModelId(normalizedModelName),
    name: `Ollama ${normalizedModelName}`,
    engine: 'ollama',
    role: 'embedder',
    status,
    source: 'ollama',
    filename: normalizedModelName,
    ollamaModelName: normalizedModelName,
    ollamaBaseUrl: baseUrl,
    sizeBytes: modelInfo?.size,
    contextLength: modelInfo?.details?.contextLength,
    parameters: modelInfo?.details?.parameterSize,
    quant: modelInfo?.details?.quantizationLevel,
    type: modelInfo?.details?.family ? `Ollama ${modelInfo.details.family}` : 'Ollama embedder',
    description: [
      'Local embedding model served by Ollama',
      'Prepares PDFs for search',
      'Embeds questions before source retrieval',
      'Folders record the embedder used during indexing'
    ],
    download,
    addedAt: new Date().toISOString()
  }
}

function createOllamaModelForRole(
  modelName: string,
  role: LocalModelRole,
  baseUrl = defaultOllamaBaseUrl,
  modelInfo?: OllamaModelInfo,
  status: LocalModel['status'] = 'ready',
  download?: ModelDownloadProgress
) {
  return role === 'embedder'
    ? createOllamaEmbedderModel(modelName, baseUrl, modelInfo, status, download)
    : createOllamaChatModel(modelName, baseUrl, modelInfo, status, download)
}

function inferOllamaModelRole(modelInfo: OllamaModelInfo): LocalModelRole {
  const label = `${modelInfo.name} ${modelInfo.model ?? ''} ${modelInfo.details?.family ?? ''}`.toLowerCase()
  return /\b(embed|embedding|nomic|bge|minilm|e5)\b/.test(label) ? 'embedder' : 'generator'
}

function ollamaProgressStatus(progress: OllamaPullProgress): ModelDownloadProgress['status'] {
  if (progress.status === 'complete') {
    return 'complete'
  }
  if (progress.status === 'incomplete') {
    return 'incomplete'
  }
  if (progress.status === 'removed') {
    return 'removed'
  }
  if (progress.status === 'error') {
    return 'error'
  }
  return 'downloading'
}

function ollamaProgressToModelDownloadProgress(
  progress: OllamaPullProgress,
  modelId = `ollama:${progress.model.trim().toLowerCase()}`
): ModelDownloadProgress {
  return {
    modelId,
    filename: progress.model,
    status: ollamaProgressStatus(progress),
    percent: Math.max(0, Math.min(100, Math.round(progress.percent ?? 0))),
    bytesReceived: progress.completed ?? 0,
    bytesTotal: progress.total,
    message: progress.message,
    error: progress.error
  }
}

function createDefaultAppState(appVersion = 'dev'): AppStateSnapshot {
  return {
    ...defaultAppState,
    appVersion,
    conversations: structuredClone(starterConversations),
    materials: structuredClone(defaultMaterials),
    models: structuredClone(defaultModels),
    settings: structuredClone(defaultSettings),
    updatedAt: new Date().toISOString()
  }
}

function normalizeMaterials(materials: CourseMaterial[] | undefined): CourseMaterial[] {
  if (!Array.isArray(materials)) {
    return structuredClone(defaultMaterials)
  }

  return materials.map((material, index) => {
    const normalizedMaterial: CourseMaterial = {
      id: material.id ?? createId('material'),
      title: material.title,
      detail: material.status === 'indexing' ? 'Parsing' : material.detail,
      status: material.status,
      kind: material.kind ?? (material.title.toLowerCase().includes('folder') ? 'folder' : 'document'),
      path: material.path,
      addedAt: material.addedAt ?? new Date(index).toISOString(),
      fileCount: material.fileCount,
      sizeBytes: material.sizeBytes,
      wordCount: material.wordCount,
      pageCount: material.pageCount,
      chunkCount: material.chunkCount,
      chunkSize: material.preparation ? material.chunkSize : defaultCollectionChunkSize,
      indexedAt: material.indexedAt,
      isActive: material.isActive ?? material.status === 'ready',
      embeddingModel: material.embeddingModel,
      embeddingModelId: material.embeddingModelId,
      embeddingModelName: material.embeddingModelName,
      cleaningProfileId: material.cleaningProfileId,
      cleaningProfileName: material.cleaningProfileName,
      cleaningProfileVersion: material.cleaningProfileVersion,
      cleaningRuleIds: normalizeCleaningRuleIds(material.cleaningRuleIds, material.cleaningProfileId),
      preparation: material.preparation,
      preparationModelName: material.preparationModelName,
      preparationIssueCount: material.preparationIssueCount,
      error: material.error,
      indexing: material.indexing
    }

    return {
      ...normalizedMaterial,
      indexing: normalizeIndexingProgress(normalizedMaterial)
    }
  })
}

function normalizeModels(models: LocalModel[] | undefined): LocalModel[] {
  if (!Array.isArray(models)) {
    return []
  }

  return models
    .filter((model) => {
      if (!model) {
        return false
      }

      return model.engine === 'ollama' || model.engine === 'remote'
    })
    .map((model, index) => {
      if (model.engine === 'ollama') {
        const ollamaModelName = model.ollamaModelName || model.name
        const role: LocalModelRole = model.role === 'embedder' || model.role === 'both' ? model.role : 'generator'
        return {
          id: model.id ?? `ollama:${ollamaModelName}`,
          name: model.name || `Ollama ${ollamaModelName}`,
          engine: 'ollama' as const,
          role,
          status: model.status ?? 'ready',
          source: 'ollama' as const,
          ollamaModelName,
          ollamaBaseUrl: model.ollamaBaseUrl ?? defaultOllamaBaseUrl,
          sizeBytes: model.sizeBytes,
          contextLength: normalizeOptionalContextLength(model.contextLength),
          inputTokenLimit: normalizeTokenLimit(model.inputTokenLimit),
          maxOutputTokens: normalizeTokenLimit(model.maxOutputTokens),
          parameters: model.parameters,
          quant: model.quant,
          type: model.type ?? (role === 'embedder' ? 'Ollama embedder' : 'Ollama chat'),
          description: model.description,
          addedAt: model.addedAt ?? new Date(index).toISOString()
        }
      }

      {
        const hasApiKey = Boolean(model.apiKey?.trim() || model.cloudCredentialStatus === 'connected')
        const role: LocalModelRole = model.role === 'embedder' || model.role === 'both' ? model.role : 'generator'
        return {
          id: model.id ?? createId('model'),
          name: model.name,
          engine: 'remote' as const,
          role,
          status: hasApiKey ? model.status ?? 'ready' : 'needsRuntime' as const,
          source: 'remote' as const,
          providerId: model.providerId,
          providerName: model.providerName,
          baseUrl: model.baseUrl,
          apiKey: model.apiKey,
          connectionId: model.connectionId,
          cloudCredentialStatus: model.cloudCredentialStatus,
          remoteModelName: model.remoteModelName,
          contextLength: normalizeOptionalContextLength(model.contextLength),
          inputTokenLimit: normalizeTokenLimit(model.inputTokenLimit),
          maxOutputTokens: normalizeTokenLimit(model.maxOutputTokens),
          type: model.type ?? 'OpenAI-compatible',
          description: model.description,
          addedAt: model.addedAt ?? new Date(index).toISOString()
        }
      }

    })
}

function clampNumber(value: unknown, defaultValue: number, min: number, max: number) {
  const numericValue = typeof value === 'number' ? value : Number(value)

  if (!Number.isFinite(numericValue)) {
    return defaultValue
  }

  return Math.max(min, Math.min(max, numericValue))
}

function normalizeOptionalContextLength(value: unknown): number | undefined {
  return normalizeTokenLimit(value)
}

function normalizeChoice<T extends string>(value: unknown, choices: readonly T[], defaultValue: T): T {
  return choices.includes(value as T) ? value as T : defaultValue
}

function normalizeSuggestionMode(value: unknown): SuggestionMode {
  return normalizeChoice(value, ['on', 'off'] as const, defaultApplicationSettings.suggestionMode)
}

function normalizeFollowUpSuggestionCount(value: unknown, suggestionMode: SuggestionMode): number {
  if (suggestionMode === 'off') {
    return 0
  }

  const count = Number(value)
  if (!Number.isFinite(count)) {
    return defaultApplicationSettings.followUpSuggestionCount
  }

  return count <= minFollowUpSuggestionCount ? minFollowUpSuggestionCount : defaultFollowUpSuggestionCount
}

function normalizeApplicationSettings(settings?: Partial<ApplicationSettings>): ApplicationSettings {
  const defaultModelId = typeof settings?.defaultModelId === 'string' ? settings.defaultModelId : ''
  const suggestionMode = normalizeSuggestionMode(settings?.suggestionMode)

  return {
    // Older "system" preferences used the gray palette; keep that fallback.
    theme: normalizeChoice(settings?.theme, ['light', 'sarah-and-duck'] as const, defaultApplicationSettings.theme),
    fontSize: normalizeFontSize(settings?.fontSize),
    defaultModelId,
    suggestionMode,
    searchMode: normalizeChoice(settings?.searchMode, ['vector', 'keyword', 'hybrid'] as const, defaultApplicationSettings.searchMode),
    explanationDepthEnabled: settings?.explanationDepthEnabled ?? defaultApplicationSettings.explanationDepthEnabled,
    explanationDepth: normalizeChoice(settings?.explanationDepth, ['simple', 'standard', 'detailed'] as const, defaultApplicationSettings.explanationDepth),
    followUpSuggestionCount: normalizeFollowUpSuggestionCount(settings?.followUpSuggestionCount, suggestionMode),
    showSources: settings?.showSources ?? defaultApplicationSettings.showSources,
    embeddingGpuEnabled: normalizeEmbeddingGpuEnabled(settings?.embeddingGpuEnabled)
  }
}

function normalizeModelRuntimeSettings(settings?: Partial<ModelRuntimeSettings>): ModelRuntimeSettings {
  const suggestedFollowUpPrompt = normalizeSuggestedFollowUpPrompt(settings?.suggestedFollowUpPrompt)
  const starterQuestionPrompt = normalizeStarterQuestionPrompt(settings?.starterQuestionPrompt)

  return {
    systemMessage: typeof settings?.systemMessage === 'string' ? settings.systemMessage : defaultModelRuntimeSettings.systemMessage,
    suggestedFollowUpPrompt,
    starterQuestionPrompt,
    contextLengthMode: migratedContextMode(settings),
    contextLength: Math.round(clampNumber(
      settings?.contextLength,
      defaultModelRuntimeSettings.contextLength,
      512,
      Number.MAX_SAFE_INTEGER
    )),
    maxLength: Math.round(clampNumber(
      settings?.maxLength,
      defaultModelRuntimeSettings.maxLength,
      64,
      8192
    )),
    thinking: settings?.thinking === true,
    reasoningMode: normalizeChoice(settings?.reasoningMode, ['auto', 'off', 'on'] as const,
      settings?.thinking === true ? 'on' : 'auto'),
    temperature: clampNumber(
      settings?.temperature,
      defaultModelRuntimeSettings.temperature,
      0,
      2
    ),
    topP: clampNumber(
      settings?.topP,
      defaultModelRuntimeSettings.topP,
      0,
      1
    ),
    topK: Math.round(clampNumber(settings?.topK, defaultModelRuntimeSettings.topK, 0, 1000)),
    minP: clampNumber(settings?.minP, defaultModelRuntimeSettings.minP, 0, 1),
    repeatPenalty: clampNumber(
      settings?.repeatPenalty,
      defaultModelRuntimeSettings.repeatPenalty,
      1,
      3
    ),
  }
}

function normalizeSettings(settings: Partial<TokenSmithSettings> | undefined): TokenSmithSettings {
  const incomingSettings = settings ?? {}
  const modelDefaults = normalizeModelRuntimeSettings(incomingSettings.modelDefaults)
  const modelSettingsById = Object.fromEntries(
    Object.entries(incomingSettings.modelSettingsById ?? {}).map(([modelId, modelSettings]) => [
      modelId,
      normalizeModelRuntimeSettings({
        ...modelDefaults,
        ...modelSettings,
        contextLengthMode: modelSettings.contextLengthMode ?? (modelSettings.contextLength !== undefined
          ? migratedContextMode(modelSettings) : modelDefaults.contextLengthMode)
      })
    ])
  )

  return {
    maxSources: Math.round(clampNumber(incomingSettings.maxSources, defaultSettings.maxSources, 1, 10)),
    application: normalizeApplicationSettings(incomingSettings.application),
    modelDefaults,
    modelSettingsById
  }
}

function modelSettingsFor(settings: TokenSmithSettings, modelId: string): ModelRuntimeSettings {
  return normalizeModelRuntimeSettings({
    ...settings.modelDefaults,
    ...(settings.modelSettingsById[modelId] ?? {})
  })
}

function isFreshConversation(conversation: Conversation) {
  return conversation.messages.length === 0 && conversation.title === 'New Study Chat'
}

function startWithFreshConversation(conversations: Conversation[], defaultModelId: string) {
  const savedConversations = conversations.length > 0 ? conversations : [createFreshConversation(defaultModelId)]
  const existingFresh = savedConversations.find(isFreshConversation)

  if (existingFresh) {
    return {
      activeConversationId: existingFresh.id,
      conversations: [
        existingFresh,
        ...savedConversations.filter((conversation) => conversation.id !== existingFresh.id)
      ]
    }
  }

  const freshConversation = createFreshConversation(defaultModelId)

  return {
    activeConversationId: freshConversation.id,
    conversations: [freshConversation, ...savedConversations]
  }
}

function mergeSavedState(savedState: AppStateSnapshot | null, appVersion = 'dev'): AppStateSnapshot {
  if (!savedState) {
    return createDefaultAppState(appVersion)
  }

  const { selectedModelId: legacyModelId, ...saved } = savedState as AppStateSnapshot & { selectedModelId?: string }
  const models = normalizeModels(saved.models)
  const settings = normalizeSettings(saved.settings)
  if (saved.settings?.application?.defaultModelId === undefined && typeof legacyModelId === 'string') {
    settings.application.defaultModelId = legacyModelId
  }
  const practice = normalizePracticeState(saved.practice)
  if (practice.mode === 'practice' && practice.draftModelId === undefined) {
    practice.draftModelId = settings.application.defaultModelId
  }
  const conversations = Array.isArray(savedState.conversations)
    ? savedState.conversations.map(({ id, modelId, title, period, messages, practiceReference, practiceSuggestions, documentScope }) => ({
        id, modelId, title, period, practiceReference, practiceSuggestions, documentScope,
        messages: messages.filter(message => !('quiz' in message) && !('kind' in message && String(message.kind).startsWith('quiz')))
      }))
    : structuredClone(starterConversations)
  const freshChatState = startWithFreshConversation(migrateConversationModels(conversations,
    typeof legacyModelId === 'string' ? legacyModelId : ''), settings.application.defaultModelId)
  const materials = normalizeMaterials(savedState.materials).map((material) =>
    material.status === 'indexing' && material.preparation
      ? { ...material, status: 'paused' as const, detail: 'Preparation paused. Resume to continue from saved progress.' }
      : material)
  const selectedEmbeddingModelId = (savedState as Partial<AppStateSnapshot>).selectedEmbeddingModelId

  return {
    ...createDefaultAppState(appVersion),
    ...saved,
    appVersion,
    activeScreen: 'chat',
    activeConversationId: freshChatState.activeConversationId,
    conversations: freshChatState.conversations,
    practice,
    materials,
    models,
    selectedEmbeddingModelId: typeof selectedEmbeddingModelId === 'string' ? selectedEmbeddingModelId : '',
    settings,
    version: 1
  }
}

function mergeIndexedMaterialsWithPending(currentMaterials: CourseMaterial[], indexedMaterials: CourseMaterial[]) {
  const normalizedIndexed = normalizeMaterials(indexedMaterials)
  const currentByKey = new Map(currentMaterials.map((material) => [materialIdentityKey(material), material]))
  const currentByLocation = new Map(currentMaterials.map((material) => [material.path ?? material.id, material]))
  const indexedWithDisplayMetadata = normalizedIndexed.map((material) => {
    const existing =
      currentByKey.get(materialIdentityKey(material)) ??
      currentByLocation.get(material.path ?? material.id)

    if (existing && (existing.status === 'indexing' || existing.status === 'paused' || existing.status === 'needsReview') && existing.preparation) {
      return { ...material, ...existing }
    }

    if (existing?.status === 'paused' && material.status === 'indexing') {
      return {
        ...material,
        ...existing,
        fileCount: existing.fileCount ?? material.fileCount,
        wordCount: existing.wordCount ?? material.wordCount,
        pageCount: existing.pageCount ?? material.pageCount,
        chunkCount: existing.chunkCount ?? material.chunkCount,
        indexedAt: existing.indexedAt ?? material.indexedAt,
        status: 'paused' as const,
        isActive: false,
        indexing: existing.indexing ?? material.indexing
      }
    }

    return {
      ...material,
      embeddingModelId: material.embeddingModelId ?? existing?.embeddingModelId,
      embeddingModelName: material.embeddingModelName ?? existing?.embeddingModelName,
      cleaningProfileId: material.cleaningProfileId ?? existing?.cleaningProfileId,
      cleaningProfileName: material.cleaningProfileName ?? existing?.cleaningProfileName,
      cleaningProfileVersion: material.cleaningProfileVersion ?? existing?.cleaningProfileVersion,
      cleaningRuleIds: material.cleaningRuleIds ?? existing?.cleaningRuleIds
    }
  })
  const indexedKeys = new Set(normalizedIndexed.map((material) => material.path ?? material.id))
  const pendingMaterials = currentMaterials.filter((material) => {
    if (material.status !== 'indexing' && material.status !== 'paused') {
      return false
    }

    return !indexedKeys.has(material.path ?? material.id)
  })

  return [...pendingMaterials, ...indexedWithDisplayMetadata]
}

function loadPreviewState(): AppStateSnapshot | null {
  try {
    const storedState = window.localStorage.getItem(stateStorageKey)
    if (!storedState) {
      return null
    }

    const state = JSON.parse(storedState) as AppStateSnapshot
    const safeState = redactStateSecretsForStorage(state)
    if (JSON.stringify(safeState) !== storedState) {
      window.localStorage.setItem(stateStorageKey, JSON.stringify(safeState))
    }

    return safeState
  } catch {
    return null
  }
}

function redactModelSecretsForStorage(model: LocalModel): LocalModel {
  if (model.engine !== 'remote' && model.source !== 'remote') {
    return model
  }

  const { apiKey: _apiKey, ...safeModel } = model
  return safeModel
}

function redactStateSecretsForStorage(state: AppStateSnapshot): AppStateSnapshot {
  return {
    ...state,
    models: state.models.map(redactModelSecretsForStorage)
  }
}

function savePreviewState(state: AppStateSnapshot) {
  window.localStorage.setItem(stateStorageKey, JSON.stringify(redactStateSecretsForStorage(state)))
}

function getConversationTitle(prompt: string) {
  const cleaned = prompt.replace(/\s+/g, ' ').trim()

  if (cleaned.length <= 28) {
    return cleaned || 'New Study Chat'
  }

  return `${cleaned.slice(0, 28).trim()}...`
}

function readableErrorMessage(error: unknown, fallback: string) {
  const message = error instanceof Error ? error.message : fallback
  return message.replace(/^Error invoking remote method '[^']+': Error:\s*/i, '').trim() || fallback
}

function readableOllamaError(error: unknown, fallback = 'Could not reach Ollama.') {
  const message = readableErrorMessage(error, fallback)
  if (/fetch failed|failed to fetch|econnrefused|connection refused|connect.*11434/i.test(message)) {
    return 'Ollama is not running yet.'
  }
  return message
}

function compactModelToken(label?: string) {
  const normalized = modelNameFromPath(label) ?? label?.trim()
  const token = normalized
    ?.split(/[-_\s:/.]+/)
    .map((part) => part.trim())
    .find((part) => /[a-z0-9]/i.test(part))

  return token ? titleCaseModelToken(token) : undefined
}

function isOpaqueEmbeddingModelKey(label?: string) {
  return Boolean(label?.startsWith('remote-openai:'))
}

function materialEmbedderName(material: CourseMaterial) {
  if (material.embeddingModel && !isOpaqueEmbeddingModelKey(material.embeddingModel)) {
    return compactModelToken(material.embeddingModel)
  }

  return compactModelToken(material.embeddingModelName) ?? compactModelToken(material.embeddingModel)
}

function materialEmbedderLabel(material: CourseMaterial) {
  const embedderName = materialEmbedderName(material)
  const collectionName = cleanMaterialTitle(material.title).split(/\s+/).find(Boolean)

  if (!embedderName || !collectionName) {
    return embedderName
  }

  return `${collectionName}-${embedderName}`
}

function sourceDisplayText(source: ChatSource) {
  return isMarkdownSource(source) ? source.context || source.excerpt : source.excerpt
}

function compactChunkSizeLabel(chunkSize?: number) {
  if (!chunkSize) {
    return undefined
  }

  return chunkSize >= 1000 ? `${chunkSize / 1000}K` : `${chunkSize}`
}

function searchTermForSource(source: ChatSource) {
  const excerpt = source.excerpt.replace(/\s+/g, ' ').replace(/^\.\.\./, '').replace(/\.\.\.$/, '').trim()
  if (!excerpt) {
    return ''
  }

  const sentence = excerpt.split(/(?<=[.!?])\s+/).find((part) => part.length >= 24) ?? excerpt
  return sentence.slice(0, 120).trim()
}

async function readSourceDocument(source: ChatSource): Promise<AnswerSourceDocument> {
  if (isMarkdownSource(source) && window.tokensmith?.getMarkdownForSource) {
    const document = await window.tokensmith.getMarkdownForSource(source)
    return { kind: 'markdown', document: { ...document, title: cleanMaterialTitle(document.title) || document.title } }
  }
  if (isPdfSource(source) && window.tokensmith?.getPdfForSource) {
    return { kind: 'pdf', document: { ...await window.tokensmith.getPdfForSource(source), searchTerm: searchTermForSource(source) } }
  }
  throw new Error('This source cannot be opened in the reader.')
}

function SourceDocumentReader({ state, navigation, onClose }: {
  state: SourceReaderState; navigation?: SourceNavigationProps; onClose: () => void
}) {
  const key = sourceEntryKey(state.sources[state.index])
  const viewer = state.document.kind === 'pdf'
    ? <PdfSourceViewer key={key} viewer={state.document.document} automatic={state.inline} navigation={navigation} onClose={onClose} />
    : <MarkdownSourceViewer key={key} viewer={state.document.document} automatic={state.inline} navigation={navigation} onClose={onClose} />
  return state.inline ? <div className="automatic-source-preview">{viewer}</div> : viewer
}

function modelStatusFromDownload(progress: ModelDownloadProgress): LocalModel['status'] {
  if (progress.status === 'complete') {
    return 'ready'
  }

  if (progress.status === 'incomplete') {
    return 'incomplete'
  }

  if (progress.status === 'error') {
    return 'downloadError'
  }

  return 'downloading'
}

function mergeModelDownloadProgress(model: LocalModel, progress: ModelDownloadProgress): LocalModel {
  return {
    ...model,
    status: modelStatusFromDownload(progress),
    path: progress.path ?? model.path,
    sizeBytes: progress.bytesTotal ?? model.sizeBytes,
    download: progress
  }
}

function modelMatchesDownload(model: LocalModel, progress: ModelDownloadProgress) {
  return (
    model.id === progress.modelId ||
    normalizeModelFilename(modelFilename(model)) === normalizeModelFilename(progress.filename) ||
    (model.engine === 'ollama' &&
      typeof model.ollamaModelName === 'string' &&
      ollamaModelNameMatches(model.ollamaModelName, progress.filename))
  )
}

export function App() {
  const generation = useGenerationProgress()
  const [appState, setAppState] = useState<AppStateSnapshot>(() => createDefaultAppState())
  const practiceState = appState.practice ?? emptyPracticeState()
  const [historyOpen, setHistoryOpen] = useState(() => window.innerWidth > 800)
  const [materialsOpen, setMaterialsOpen] = useState(() => window.innerWidth > 1200)
  const [collectionExpansion, setCollectionExpansion] = useState<Record<string, boolean>>({})
  const studyPanels: StudyPanels = { historyOpen, onHistoryOpenChange: setHistoryOpen, materialsOpen, onMaterialsOpenChange: setMaterialsOpen,
    collectionExpansion, onCollectionExpansionChange: (id, expanded) => setCollectionExpansion(current =>
      (current[id] ?? true) === expanded ? current : { ...current, [id]: expanded }) }
  const [isLogCardOpen, setIsLogCardOpen] = useState(false)
  const [logFile, setLogFile] = useState<TokenSmithLogFile | null>(null)
  const [logStatus, setLogStatus] = useState<'idle' | 'loading' | 'error'>('idle')
  const [logError, setLogError] = useState<string | null>(null)
  const logEntries = useMemo(() => parseTokenSmithLog(logFile?.text ?? ''), [logFile?.text])
  async function refreshLogFile() {
    if (!window.tokensmith?.getLogFile) {
      setLogStatus('error')
      setLogError('TokenSmith log is not available in this build.')
      return
    }
    setLogStatus('loading')
    setLogError(null)
    try {
      setLogFile(await window.tokensmith.getLogFile())
      setLogStatus('idle')
    } catch (error) {
      setLogStatus('error')
      setLogError(readableErrorMessage(error, 'Could not read the TokenSmith log.'))
    }
  }
  function openLog() { setIsLogCardOpen(true); void refreshLogFile() }
  const practiceReader = useSourceReader(readSourceDocument)
  const activePracticeSession = practiceState.sessions.find(item => item.id === practiceState.activeSessionId)
  const activePracticeQuestionId = activePracticeSession?.questions[activePracticeSession.currentIndex]?.id
  useEffect(() => { practiceReader.close() }, [practiceState.mode, practiceState.activeSessionId, activePracticeQuestionId, appState.activeScreen, practiceReader.close])
  function updatePractice(update: (current: PracticeState) => PracticeState) {
    updateAppState(current => ({ ...current, practice: update(current.practice ?? emptyPracticeState()) }))
  }
  const [practiceChatDraft, setPracticeChatDraft] = useState<{ conversationId: string; draft: ChatDraft }>()
  function chatAboutPractice(session: PracticeSession, question: PracticeQuestion, passage?: ChatSelectedPassage) {
    const conversation = practiceChatConversation(`chat-${crypto.randomUUID()}`, session, question)
    setPracticeChatDraft({ conversationId: conversation.id,
      draft: { text: question.helpDraft ?? '', selectedPassage: passage ?? question.helpPassage } })
    updateAppState(current => {
      const next = changeStudyScope(current, session.documents)
      const practice = next.practice ?? emptyPracticeState()
      return { ...next, activeScreen: 'chat', activeConversationId: conversation.id,
        conversations: [conversation, ...next.conversations],
        practice: { ...practice, mode: 'chat', activeSessionId: session.id,
          sessions: practice.sessions.map(item => item.id !== session.id ? item : { ...item,
            currentIndex: item.questions.findIndex(value => value.id === question.id),
            questions: item.questions.map(value => value.id === question.id ? { ...value, chatOpened: true } : value) }) } }
    })
  }
  function practiceTopic(topic: PracticeTopic) {
    updatePractice(current => ({ ...current, mode: 'practice', activeSessionId: undefined, draftTopic: topic,
      draftModelId: appState.settings.application.defaultModelId }))
  }
  function setStudyMode(mode: StudyMode) {
    updateAppState(current => {
      const practice = current.practice ?? emptyPracticeState()
      return { ...current, practice: { ...practice, mode,
        ...(mode === 'practice' && practice.draftModelId === undefined ? { draftModelId: current.settings.application.defaultModelId } : {}) } }
    })
  }
  function setStudyScope(scope: StudyDocumentRef[]) {
    updateAppState(current => changeStudyScope(current, scope))
  }
  const [appVersion, setAppVersion] = useState<string>('dev')
  const [engines, setEngines] = useState<EngineInfo[]>([
    {
      id: 'tokensmith',
      name: 'TokenSmith',
      status: 'needsSetup',
      detail: 'Local PDF preparation, source-backed chat, and configured model runtimes.'
    }
  ])
  const [, setSaveStatus] = useState<'loading' | 'saved' | 'saving' | 'local' | 'error'>('loading')
  const [hasLoadedState, setHasLoadedState] = useState(false)
  const metadataModelKey = appState.models.filter(modelCanGenerate).map(contextMetadataIdentity).join('\n')
  useEffect(() => {
    if (!hasLoadedState || !window.tokensmith?.getModelContextMetadata) return
    let cancelled = false
    for (const model of appState.models.filter(modelCanGenerate)) {
      if (model.status !== 'ready') continue
      const identity = contextMetadataIdentity(model)
      void window.tokensmith.getModelContextMetadata(model).then(metadata => {
        if (cancelled || Object.keys(metadata).length === 0) return
        setAppState(current => ({ ...current, models: current.models.map(saved =>
          contextMetadataIdentity(saved) === identity ? mergeModelContextMetadata(saved, metadata) : saved) }))
      }).catch(() => undefined)
    }
    return () => { cancelled = true }
  }, [hasLoadedState, metadataModelKey])
  const studyCatalog = useStudyDocuments(appState.materials, hasLoadedState)
  useEffect(() => {
    if (studyCatalog.status !== 'ready') return
    setAppState(current => initializeStudyScopes(current, studyCatalog.documents))
  }, [studyCatalog.status, studyCatalog.documents])
  const [libraryCreateRequest, setLibraryCreateRequest] = useState(0)
  const [isChatSetupCardDismissed, setChatSetupCardDismissed] = useState(false)
  const [cloudSetup, setCloudSetup] = useState<{ model?: LocalModel; target: ModelSelectionTarget } | null>(null)
  const [cloudNotice, setCloudNotice] = useState('')
  const hasLoadedStateRef = useRef(false)
  const activeIndexRequestsRef = useRef(new Set<string>())
  const cancelledIndexRequestsRef = useRef(new Set<string>())
  const pausedIndexRequestsRef = useRef(new Set<string>())
  const indexRequestSequenceRef = useRef(new Map<string, number>())
  const saveSequenceRef = useRef(0)

  useEffect(() => {
    if (!cloudNotice) return
    const timer = window.setTimeout(() => setCloudNotice(''), 5000)
    return () => window.clearTimeout(timer)
  }, [cloudNotice])

  function openCloudSetup(model?: LocalModel) { setCloudSetup({ model, target: studyModelTarget(appState) }) }

  function connectedCloudGenerator(model: LocalModel) {
    if (!cloudSetup) return
    const target = cloudSetup.target
    updateAppState(current => {
      const migrated = attachCloudProviderConnection(current.models, model)
      const models = mergeCloudConnectionModel(migrated, model)
      const connected = models.find(candidate => candidate.connectionId === model.connectionId &&
        candidate.remoteModelName === model.remoteModelName && (candidate.role ?? 'generator') === model.role)
      if (!connected) return current
      const modelId = connected.id
      return model.role === 'embedder'
        ? { ...current, models, selectedEmbeddingModelId: modelId }
        : selectStudyModel({ ...current, models }, modelId, target)
    })
    setCloudSetup(null)
    setCloudNotice(`Connected · ${model.remoteModelName}`)
  }

  function forgotCloudConnections(connectionIds: string[]) {
    updateAppState(current => ({
      ...current,
      models: disconnectCloudConnectionModels(current.models, connectionIds)
    }))
    setCloudNotice('API key removed · reconnect a model to continue')
  }

  useLayoutEffect(() => {
    document.documentElement.dataset.theme = appState.settings.application.theme
  }, [appState.settings.application.theme])

  useLayoutEffect(() => {
    document.documentElement.dataset.fontSize = appState.settings.application.fontSize
  }, [appState.settings.application.fontSize])

  useEffect(() => {
    const bridge = window.tokensmith

    if (!bridge) {
      setAppState(mergeSavedState(loadPreviewState(), 'dev'))
      setSaveStatus('local')
      hasLoadedStateRef.current = true
      setHasLoadedState(true)
      return undefined
    }

    let cancelled = false

    Promise.all([bridge.getAppVersion(), bridge.loadAppState()])
      .then(([version, savedState]) => {
        if (cancelled) {
          return
        }

        const mergedState = mergeSavedState(savedState, version)

        setAppVersion(version)
        setAppState(mergedState)
        setSaveStatus('saved')
        hasLoadedStateRef.current = true
        setHasLoadedState(true)

        void bridge
          .listEngines()
          .then((availableEngines) => {
            if (!cancelled) {
              setEngines(availableEngines)
            }
          })
          .catch(() => {
            if (!cancelled) {
              setEngines((current) =>
                current.map((engine) =>
                  engine.id === 'tokensmith'
                    ? {
                        ...engine,
                        status: 'unavailable',
                        detail: 'The local TokenSmith runtime is not available. PDF preparation and source-backed chat need it.'
                      }
                    : engine
                )
              )
            }
          })

        void bridge
          .listMaterials()
          .then((materials) => {
            if (!cancelled) {
              setAppState((current) => ({
                ...current,
                materials: mergeIndexedMaterialsWithPending(current.materials, materials)
              }))
            }
          })
          .catch(() => undefined)
      })
      .catch(() => {
        if (cancelled) {
          return
        }

        setSaveStatus('error')
        hasLoadedStateRef.current = true
        setHasLoadedState(true)
      })

    return () => {
      cancelled = true
    }
  }, [])

  useEffect(() => {
    if (!hasLoadedStateRef.current) {
      return undefined
    }

    const saveSequence = saveSequenceRef.current + 1
    const stateToSave = {
      ...appState,
      updatedAt: new Date().toISOString()
    }

    saveSequenceRef.current = saveSequence

    if (!window.tokensmith) {
      savePreviewState(stateToSave)
      setSaveStatus('local')
      return undefined
    }

    setSaveStatus('saving')
    window.tokensmith
      .saveAppState(stateToSave)
      .then(() => {
        if (saveSequenceRef.current === saveSequence) {
          setSaveStatus('saved')
        }
      })
      .catch(() => {
        if (saveSequenceRef.current === saveSequence) {
          setSaveStatus('error')
        }
      })

    return undefined
  }, [appState])

  useEffect(() => {
    if (!window.tokensmith) {
      return undefined
    }

    return window.tokensmith.onMaterialIndexProgress((progress) => {
      updateAppState((current) => {
        if (
          cancelledIndexRequestsRef.current.has(progress.materialId) ||
          pausedIndexRequestsRef.current.has(progress.materialId)
        ) {
          return current
        }

        return {
          ...current,
          materials: current.materials.map((material) => {
            if (material.id !== progress.materialId) {
              return material
            }

            if (progress.phase === 'complete') {
              return {
                ...material,
                status: 'ready',
                detail: 'Ready for chat',
                indexing: undefined,
                isActive: true
              }
            }

            const nextMaterial: CourseMaterial = {
              ...material,
              status: progress.phase === 'error' ? 'needsReview' : 'indexing',
              indexing: progress,
              isActive: material.indexedAt ? material.isActive : false
            }
            const normalizedProgress = normalizeIndexingProgress(nextMaterial)

            return {
              ...nextMaterial,
              detail: normalizedProgress?.message ?? material.detail,
              indexing: normalizedProgress
            }
          })
        }
      })
    })
  }, [])

  useEffect(() => {
    if (!window.tokensmith?.onOllamaPullProgress) {
      return undefined
    }

    return window.tokensmith.onOllamaPullProgress((progress) => {
      const download = ollamaProgressToModelDownloadProgress(progress)
      updateAppState((current) => {
        if (download.status === 'removed') {
          const remainingModels = current.models.filter((model) => !modelMatchesDownload(model, download))

          return {
            ...current,
            models: remainingModels
          }
        }

        let didUpdate = false
        const models = current.models.map((model) => {
          if (!modelMatchesDownload(model, download)) {
            return model
          }

          didUpdate = true
          return mergeModelDownloadProgress(model, {
            ...download,
            modelId: model.id,
            bytesTotal: download.bytesTotal ?? model.download?.bytesTotal ?? model.sizeBytes,
            bytesReceived: download.bytesReceived || model.download?.bytesReceived || 0
          })
        })

        return didUpdate ? { ...current, models } : current
      })
    })
  }, [])

  useEffect(() => {
    if (!hasLoadedState || !window.tokensmith) {
      return
    }

    for (const material of appState.materials) {
      if (material.status !== 'indexing' || !material.path || activeIndexRequestsRef.current.has(material.id)) {
        continue
      }

      startMaterialIndexing(material.id, material.path, undefined, {
        resume: true,
        title: material.title
      })
    }
  }, [appState.materials, hasLoadedState])

  function updateAppState(updater: (current: AppStateSnapshot) => AppStateSnapshot) {
    setAppState(current => {
      const next = updater(current)
      return { ...next, updatedAt: new Date().toISOString() }
    })
  }

  function updateChatState(
    updater: (current: Pick<AppStateSnapshot, 'activeConversationId' | 'conversations'>) => Pick<
      AppStateSnapshot,
      'activeConversationId' | 'conversations'
    >
  ) {
    updateAppState((current) => ({
      ...current,
      ...updater({
        activeConversationId: current.activeConversationId,
        conversations: current.conversations
      })
    }))
  }

  function updateSettings(settings: Partial<TokenSmithSettings>) {
    updateAppState((current) => {
      const nextSettings = normalizeSettings({
        ...current.settings,
        ...settings
      })

      return {
        ...current,
        settings: nextSettings
      }
    })
  }

  function addMaterials(materials: CourseMaterial[]) {
    if (materials.length === 0) {
      return
    }

    updateAppState((current) => {
      const existingMaterialKeys = new Set(
        current.materials.map(materialIdentityKey)
      )
      const newMaterials = materials
        .filter((material) => !existingMaterialKeys.has(materialIdentityKey(material)))
        .map((material) => ({
          ...material,
          isActive: material.isActive ?? material.status === 'ready'
        }))

      return {
        ...current,
        activeScreen: 'library',
        materials: [...newMaterials, ...current.materials]
      }
    })
  }

  function startMaterialIndexing(
    materialId: string,
    materialPath: string,
    embeddingModelOverride?: LocalModel,
    options: IndexMaterialOptions = {}
  ) {
    if (!window.tokensmith) {
      return
    }

    if (activeIndexRequestsRef.current.has(materialId)) {
      return
    }

    pausedIndexRequestsRef.current.delete(materialId)
    cancelledIndexRequestsRef.current.delete(materialId)
    const requestSequence = (indexRequestSequenceRef.current.get(materialId) ?? 0) + 1
    indexRequestSequenceRef.current.set(materialId, requestSequence)

    const indexingMaterial = appState.materials.find((material) => material.id === materialId)
    const embeddingModel = embeddingModelOverride ?? appState.models.find(model =>
      model.id === (indexingMaterial?.embeddingModelId || appState.selectedEmbeddingModelId))

    if (!embeddingModel || !modelCanEmbed(embeddingModel)) {
      updateAppState((current) => ({
        ...current,
        materials: current.materials.map((material) =>
          material.id === materialId
            ? {
                ...material,
                status: 'needsReview',
                detail: 'Needs embedder',
                error: 'Reconnect the selected search model or choose an available embedding model in Library settings.',
                indexing: undefined,
                isActive: material.indexedAt ? material.isActive : false
              }
            : material
        )
      }))
      return
    }

    const preparation = options.preparation ?? indexingMaterial?.preparation ?? defaultPreparation()
    const preparationModel = preparation.mode === 'ai'
      ? options.preparationModel ?? appState.models.find(model =>
        model.id === (preparation.modelId || appState.settings.application.defaultModelId) && model.role !== 'embedder')
      : undefined
    updateAppState(current => ({ ...current, materials: current.materials.map(item => item.id === materialId ? {
      ...item, preparation, status: 'indexing', error: undefined,
      isActive: item.indexedAt ? item.isActive : false,
      indexing: { materialId, phase: 'parsing', percent: 0, message: 'Queued for preparation' }
    } : item) }))

    activeIndexRequestsRef.current.add(materialId)

    void window.tokensmith
      .indexMaterial(materialId, materialPath, embeddingModel, {
        ...options,
        preparation,
        preparationModel,
        embeddingGpuEnabled: appState.settings.application.embeddingGpuEnabled,
        title: options.title ?? indexingMaterial?.title,
        cleaningProfileId: options.cleaningProfileId ?? indexingMaterial?.cleaningProfileId,
        cleaningRuleIds:
          options.cleaningRuleIds ??
          indexingMaterial?.cleaningRuleIds ??
          defaultCleaningRuleIdsForProfile(options.cleaningProfileId ?? indexingMaterial?.cleaningProfileId)
      })
      .then((indexedMaterial) => {
        if (indexRequestSequenceRef.current.get(materialId) !== requestSequence) {
          return
        }
        activeIndexRequestsRef.current.delete(materialId)
        if (cancelledIndexRequestsRef.current.delete(materialId)) {
          return
        }
        if (pausedIndexRequestsRef.current.has(materialId)) {
          return
        }

        updateAppState((current) => {
          let didReplace = false
          const materials = current.materials.map((material) => {
            if (material.id !== materialId) {
              return material
            }

            didReplace = true
            return {
              ...indexedMaterial,
              embeddingModelId: embeddingModel.id,
              embeddingModelName: displayModelName(embeddingModel),
              chunkSize: indexedMaterial.preparation ? undefined : defaultCollectionChunkSize,
              cleaningProfileId: indexedMaterial.cleaningProfileId ?? material.cleaningProfileId ?? options.cleaningProfileId,
              cleaningProfileName:
                indexedMaterial.cleaningProfileName ??
                material.cleaningProfileName ??
                cleaningProfileLabel(options.cleaningProfileId),
              cleaningProfileVersion: indexedMaterial.cleaningProfileVersion ?? material.cleaningProfileVersion,
              cleaningRuleIds:
                indexedMaterial.cleaningRuleIds ??
                material.cleaningRuleIds ??
                options.cleaningRuleIds,
              isActive: indexedMaterial.isActive ?? indexedMaterial.status === 'ready',
              indexing: undefined
            }
          })

          return didReplace
            ? {
                ...current,
                materials
              }
            : current
        })
      })
      .catch((error) => {
        if (indexRequestSequenceRef.current.get(materialId) !== requestSequence) {
          return
        }
        activeIndexRequestsRef.current.delete(materialId)
        if (cancelledIndexRequestsRef.current.delete(materialId)) {
          return
        }
        if (pausedIndexRequestsRef.current.has(materialId)) {
          return
        }

        updateAppState((current) => ({
          ...current,
          materials: current.materials.map((material) =>
            material.id === materialId
              ? {
                  ...material,
                  status: 'needsReview',
                  detail: 'Processing failed',
                  error: readableErrorMessage(error, 'Processing failed'),
                  indexing: {
                    materialId,
                    phase: 'error',
                    percent: material.indexing?.percent ?? 0,
                    message: 'Processing failed'
                  },
                  isActive: material.indexedAt ? material.isActive : false
                }
              : material
          )
        }))
      })
  }

  function addModel(model: LocalModel) {
    upsertModel(model, true)
  }

  function upsertModel(model: LocalModel, shouldSelect = false, nextScreen: ScreenId | null = 'models', target = studyModelTarget(appState)) {
    updateAppState((current) => {
      const modelFile = normalizeModelFilename(modelFilename(model))
      const isSameModel = (existingModel: LocalModel) => {
        if (existingModel.id === model.id) {
          return true
        }

        if (model.path && existingModel.path === model.path) {
          return true
        }

        if (modelFile && normalizeModelFilename(modelFilename(existingModel)) === modelFile) {
          return true
        }

        if (model.engine === 'remote' && existingModel.engine === 'remote') {
          return (
            existingModel.providerId === model.providerId &&
            existingModel.baseUrl === model.baseUrl &&
            existingModel.remoteModelName === model.remoteModelName &&
            modelRole(existingModel) === modelRole(model)
          )
        }

        if (model.engine === 'ollama' && existingModel.engine === 'ollama') {
          return (
            existingModel.ollamaModelName === model.ollamaModelName &&
            existingModel.ollamaBaseUrl === model.ollamaBaseUrl &&
            modelRole(existingModel) === modelRole(model)
          )
        }

        return false
      }
      const existingModels = current.models.filter((existingModel) => {
        return !isSameModel(existingModel)
      })

      const next = {
        ...current,
        activeScreen: nextScreen ?? current.activeScreen,
        models: [model, ...existingModels],
        selectedEmbeddingModelId: shouldSelect && model.status === 'ready' && modelCanEmbed(model) ? model.id : current.selectedEmbeddingModelId
      }
      return shouldSelect && model.status === 'ready' && isAnswerModel(model) ? selectStudyModel(next, model.id, target) : next
    })
  }

  function installOllamaChatModel(
    modelName = recommendedOllamaChatModel,
    baseUrl = defaultOllamaBaseUrl,
    modelInfo?: OllamaModelInfo
  ) {
    upsertModel(createOllamaChatModel(modelName, baseUrl, modelInfo), true, null)
  }

  function installOllamaEmbedderModel(
    modelName = recommendedOllamaEmbeddingModel,
    baseUrl = defaultOllamaBaseUrl,
    modelInfo?: OllamaModelInfo
  ) {
    upsertModel(createOllamaEmbedderModel(modelName, baseUrl, modelInfo), true, null)
  }

  async function latestOllamaModelInfo(modelName: string, baseUrl = defaultOllamaBaseUrl) {
    if (!window.tokensmith?.getOllamaStatus) {
      return undefined
    }

    try {
      const status = await window.tokensmith.getOllamaStatus()
      return status.models.find((model) => ollamaModelInfoMatches(model, modelName))
    } catch {
      return undefined
    }
  }

  function downloadOllamaModel(modelName: string, role: LocalModelRole, baseUrl = defaultOllamaBaseUrl) {
    const target = studyModelTarget(appState)
    const normalizedModelName = modelName.trim()
    if (!normalizedModelName || !window.tokensmith?.pullOllamaModel) {
      return
    }

    const existingModel = appState.models.find(
      (model) =>
        model.engine === 'ollama' &&
        typeof model.ollamaModelName === 'string' &&
        ollamaModelNameMatches(model.ollamaModelName, normalizedModelName) &&
        modelRole(model) === role
    )
    const modelId = existingModel?.id ?? (role === 'embedder' ? ollamaEmbedderModelId(normalizedModelName) : ollamaChatModelId(normalizedModelName))
    const startingProgress: ModelDownloadProgress = {
      modelId,
      filename: normalizedModelName,
      status: 'downloading',
      percent: existingModel?.download?.percent ?? 0,
      bytesReceived: existingModel?.download?.bytesReceived ?? 0,
      bytesTotal: existingModel?.download?.bytesTotal ?? existingModel?.sizeBytes,
      message: 'Starting download'
    }

    upsertModel(
      {
        ...createOllamaModelForRole(
          normalizedModelName,
          role,
          baseUrl,
          undefined,
          'downloading',
          startingProgress
        ),
        id: modelId,
        addedAt: existingModel?.addedAt ?? new Date().toISOString()
      },
      false
    )

    void window.tokensmith
      .pullOllamaModel(normalizedModelName, baseUrl)
      .then(async () => {
        const modelInfo = await latestOllamaModelInfo(normalizedModelName, baseUrl)
        if (!modelInfo) {
          updateAppState((current) => ({
            ...current,
            models: current.models.map((model) =>
              model.id === modelId
                ? {
                    ...model,
                    status: 'downloadError',
                    download: {
                      ...(model.download ?? startingProgress),
                      status: 'error',
                      error: 'Ollama finished the download, but TokenSmith could not verify the model.',
                      message: 'Verification failed'
                    }
                  }
                : model
            )
          }))
          return
        }
        const bytesTotal = modelInfo?.size ?? existingModel?.sizeBytes
        upsertModel(
          {
            ...createOllamaModelForRole(normalizedModelName, role, baseUrl, modelInfo, 'ready', {
              modelId,
              filename: normalizedModelName,
              status: 'complete',
              percent: 100,
              bytesReceived: bytesTotal ?? 0,
              bytesTotal,
              message: 'Downloaded'
            }),
            id: modelId,
            addedAt: existingModel?.addedAt ?? new Date().toISOString()
          },
          true, 'models', target
        )
      })
      .catch((error) => {
        if (error instanceof Error && /paused|cancelled|aborted/i.test(error.message)) {
          return
        }

        updateAppState((current) => ({
          ...current,
          models: current.models.map((model) =>
            model.id === modelId
              ? {
                  ...model,
                  status: 'downloadError',
                  download: {
                    ...(model.download ?? startingProgress),
                    status: 'error',
                    error: readableErrorMessage(error, 'Download failed'),
                    message: 'Download failed'
                  }
                }
              : model
          )
        }))
      })
  }

  function cancelOllamaModelDownload(modelName: string, baseUrl = defaultOllamaBaseUrl) {
    void window.tokensmith?.cancelOllamaPull(modelName, baseUrl).catch(() => {
      setSaveStatus('error')
    })
  }

  function selectModel(modelId: string) {
    updateAppState(current => selectStudyModel(current, modelId, studyModelTarget(current)))
  }

  function selectEmbeddingModel(modelId: string) {
    updateAppState((current) => {
      if (!current.models.some((model) => model.id === modelId && modelCanEmbed(model))) {
        return current
      }

      return {
        ...current,
        selectedEmbeddingModelId: modelId
      }
    })
  }

  function removeModel(model: LocalModel) {
    const shouldRestoreOnRemoveFailure = !(model.engine === 'ollama' && model.status !== 'ready')

    updateAppState((current) => {
      const remainingModels = current.models.filter((item) => item.id !== model.id)
      return {
        ...current,
        models: remainingModels
      }
    })

    void window.tokensmith?.removeModel(model).catch(() => {
      setSaveStatus('error')
      if (shouldRestoreOnRemoveFailure) {
        upsertModel(model, false, null)
      }
    })
  }

  function resumeMaterialIndexing(materialId: string) {
    const material = appState.materials.find((item) => item.id === materialId)

    if (!material?.path || material.status === 'indexing' || activeIndexRequestsRef.current.has(materialId)) {
      return
    }

    updateAppState((current) => ({
      ...current,
      materials: current.materials.map((item) =>
        item.id === materialId
          ? {
              ...item,
              status: 'indexing',
              detail: 'Resuming',
              error: undefined,
              isActive: item.indexedAt ? item.isActive : false,
              indexing: {
                materialId,
                phase: item.indexing?.phase ?? 'parsing',
                percent: Math.max(1, item.indexing?.percent ?? 1),
                processedFiles: item.indexing?.processedFiles ?? 0,
                totalFiles: item.indexing?.totalFiles ?? (item.kind === 'folder' ? item.fileCount ?? 0 : 1),
                processedEmbeddings: item.indexing?.processedEmbeddings ?? 0,
                totalEmbeddings: item.indexing?.totalEmbeddings ?? item.chunkCount ?? 0,
                message: 'Resuming'
              }
            }
          : item
      )
    }))

    const embeddingModel = appState.models.find(model =>
      model.id === (material.embeddingModelId || appState.selectedEmbeddingModelId))

    if (!embeddingModel || !modelCanEmbed(embeddingModel)) {
      updateAppState((current) => ({
        ...current,
        materials: current.materials.map((item) =>
          item.id === materialId
            ? {
                ...item,
                status: 'needsReview',
                detail: 'Needs embedder',
                error: 'Reconnect the selected search model or choose an available embedding model in Library settings.',
                indexing: undefined,
                isActive: item.indexedAt ? item.isActive : false
              }
            : item
        )
      }))
      return
    }

    startMaterialIndexing(materialId, material.path, embeddingModel, {
      resume: true,
      title: material.title,
      cleaningProfileId: material.cleaningProfileId,
      cleaningRuleIds: material.cleaningRuleIds
    })
  }

  function pauseMaterialIndexing(materialId: string) {
    const material = appState.materials.find((item) => item.id === materialId)

    if (!material || material.status !== 'indexing') {
      return
    }

    pausedIndexRequestsRef.current.add(materialId)
    cancelledIndexRequestsRef.current.add(materialId)
    activeIndexRequestsRef.current.delete(materialId)
    indexRequestSequenceRef.current.set(materialId, (indexRequestSequenceRef.current.get(materialId) ?? 0) + 1)

    void window.tokensmith?.cancelMaterialIndexing(materialId).catch(() => {
      setSaveStatus('error')
    })

    const progress = material.indexing ?? {
      materialId,
      phase: 'embedding' as const,
      percent: 0,
      processedFiles: material.fileCount ?? 0,
      totalFiles: material.fileCount ?? 0,
      processedEmbeddings: 0,
      totalEmbeddings: material.chunkCount ?? 0,
      message: 'Paused'
    }

    updateAppState((current) => ({
      ...current,
      materials: current.materials.map((item) =>
        item.id === materialId
          ? {
              ...item,
              status: 'paused',
              detail: 'Paused',
              error: undefined,
              indexing: {
                ...progress,
                message: 'Paused'
              },
              isActive: item.indexedAt ? item.isActive : false
            }
          : item
      )
    }))
  }

  function removeMaterial(materialId: string) {
    const material = appState.materials.find((item) => item.id === materialId)
    const wasIndexing = material?.status === 'indexing'

    if (!material) {
      return
    }

    if (wasIndexing) {
      cancelledIndexRequestsRef.current.add(materialId)
      activeIndexRequestsRef.current.delete(materialId)
      indexRequestSequenceRef.current.set(materialId, (indexRequestSequenceRef.current.get(materialId) ?? 0) + 1)
      void window.tokensmith?.cancelMaterialIndexing(materialId).catch(() => {
        setSaveStatus('error')
      })
    }

    updateAppState((current) => ({
      ...current,
      materials: current.materials.filter((item) => item.id !== materialId)
    }))

    void window.tokensmith?.removeMaterial(materialId, material.path).catch(() => {
      setSaveStatus('error')
    })
  }

  const activeScreen = appState.activeScreen
  const activeConversation = appState.conversations.find(chat => chat.id === appState.activeConversationId)
  const selectedModel = answerModel(appState.models, activeConversation?.modelId)
  const practiceModelId = practiceState.activeSessionId
    ? practiceState.sessions.find(session => session.id === practiceState.activeSessionId)?.modelId : practiceState.draftModelId
  const embeddingModels = appState.models.filter(modelCanEmbed)
  const activeTitle = useMemo(
    () => navItems.find((item) => item.id === activeScreen)?.label ?? 'Chat',
    [activeScreen]
  )

  return (
    <main className="desktop-app">
      <aside className="rail" aria-label="Primary navigation">
        <nav className="rail-nav">
          {navItems.map((item) => {
            const Icon = item.icon
            const isActive = activeScreen === item.id

            return (
              <button
                className={`rail-button ${isActive ? 'is-active' : ''}`}
                key={item.id}
                type="button"
                aria-current={isActive ? 'page' : undefined}
                aria-label={item.label}
                title={item.label}
                onClick={() =>
                  updateAppState((current) => ({
                    ...current,
                    activeScreen: item.id
                  }))
                }
              >
                <Icon size={27} strokeWidth={2} aria-hidden="true" />
                <span>{item.label}</span>
              </button>
            )
          })}
        </nav>
        <div className="rail-footer">
          <img className="rail-wordmark" src={tokensmithRailWordmark} alt="TokenSmith" />
          <small>v{appVersion}</small>
        </div>
      </aside>

      <section className="workspace" aria-label={`${activeTitle} screen`}>
        {!hasLoadedState && <LoadingScreen />}
        {hasLoadedState && (
          <ChatScreen
            {...studyPanels} onViewLog={openLog}
            progress={generation.api}
            isActive={activeScreen === 'chat' && practiceState.mode === 'chat'}
            onModeChange={setStudyMode} onPracticeTopic={practiceTopic} practiceChatDraft={practiceChatDraft}
            catalog={studyCatalog} scope={appState.studyScope ?? []} onScopeChange={setStudyScope}
            activeConversationId={appState.activeConversationId}
            conversations={appState.conversations}
            materials={appState.materials}
            models={appState.models}
            embeddingModels={embeddingModels}
            isSetupCardDismissed={isChatSetupCardDismissed}
            selectedModel={selectedModel}
            settings={appState.settings}
            onChatStateChange={updateChatState}
            onDismissSetupCard={() => setChatSetupCardDismissed(true)}
            onInstallOllamaChatModel={installOllamaChatModel}
            onInstallOllamaEmbedderModel={installOllamaEmbedderModel}
            onOpenLibrary={() => {
              setLibraryCreateRequest((request) => request + 1)
              updateAppState((current) => ({
                ...current,
                activeScreen: 'library'
              }))
            }}
            onRemoveModel={removeModel}
            onSelectModel={selectModel}
            onConnectCloud={openCloudSetup}
            onManageModels={() => updateAppState(current => ({ ...current, activeScreen: 'models' }))}
            onExplanationDepthChange={(explanationDepth) =>
              updateSettings({ application: { ...appState.settings.application, explanationDepth } })
            }
          />
        )}
        {hasLoadedState && <PracticeScreen isActive={activeScreen === 'chat' && practiceState.mode === 'practice'}
          {...studyPanels} onViewLog={openLog}
          state={practiceState} onChange={updatePractice} onModeChange={setStudyMode} onChat={chatAboutPractice}
          materials={appState.materials} embeddingModels={embeddingModels}
          catalog={studyCatalog} scope={appState.studyScope ?? []} onScopeChange={setStudyScope} models={appState.models} selectedModel={answerModel(appState.models, practiceModelId)} settings={appState.settings}
          modelSettingsFor={id => modelSettingsFor(appState.settings, id)}
          onSelectModel={selectModel} onConnectCloud={openCloudSetup}
          onManageModels={() => updateAppState(current => ({ ...current, activeScreen: 'models' }))}
          onOpenLibrary={() => updateAppState(current => ({ ...current, activeScreen: 'library' }))}
          onOpenSource={(source, sources) => void practiceReader.open(source, sources)}
          sourceError={practiceReader.error} progress={generation.api} />}
        {activeScreen === 'chat' && practiceState.mode === 'practice' && practiceReader.state &&
          <SourceDocumentReader state={practiceReader.state} navigation={practiceReader.navigation} onClose={practiceReader.close} />}
        {hasLoadedState && activeScreen === 'library' && (
          <LibraryScreen
            models={appState.models}
            selectedModelId={appState.settings.application.defaultModelId}
            createRequest={libraryCreateRequest}
            materials={appState.materials}
            selectedEmbeddingModelId={appState.selectedEmbeddingModelId}
            onAddMaterials={addMaterials}
            onRemoveMaterial={removeMaterial}
            onResumeMaterialIndexing={resumeMaterialIndexing}
            onPauseMaterialIndexing={pauseMaterialIndexing}
            onStartMaterialIndexing={startMaterialIndexing}
          />
        )}
        {hasLoadedState && activeScreen === 'models' && (
          <ModelsScreen
            engines={engines}
            models={appState.models}
            onAddModel={addModel}
            onCancelOllamaModelDownload={cancelOllamaModelDownload}
            onDownloadOllamaModel={downloadOllamaModel}
            onRemoveModel={removeModel}
            onSelectEmbeddingModel={selectEmbeddingModel}
            onSelectModel={selectModel}
            selectedEmbeddingModelId={appState.selectedEmbeddingModelId}
            selectedModelId={(practiceState.mode === 'practice' ? practiceModelId : activeConversation?.modelId) ?? ''}
            modelSelectionLocked={practiceState.mode === 'practice' && Boolean(practiceState.activeSessionId)}
            onConnectCloud={openCloudSetup}
          />
        )}
        {hasLoadedState && activeScreen === 'settings' && (
          <SettingsScreen models={appState.models} settings={appState.settings} onSettingsChange={updateSettings} isIndexing={appState.materials.some(material => material.status === 'indexing')} />
        )}
      </section>
      {isLogCardOpen && <TokenSmithLogCard entries={logEntries} error={logError} logFile={logFile} status={logStatus}
        onClose={() => setIsLogCardOpen(false)} onRefresh={refreshLogFile} />}
      <GenerationStatus progress={generation.current} onOpen={conversationId => updateAppState(current => {
        if (conversationId === 'practice-topics') return { ...current, activeScreen: 'chat',
          practice: { ...(current.practice ?? emptyPracticeState()), mode: 'practice', activeSessionId: undefined } }
        const session = current.practice?.sessions.find(item => item.id === conversationId)
        const next = session ? changeStudyScope(current, session.documents) : current
        return { ...next, activeScreen: 'chat',
          activeConversationId: next.conversations.some(c => c.id === conversationId) ? conversationId : next.activeConversationId,
          practice: { ...(next.practice ?? emptyPracticeState()), mode: session ? 'practice' : 'chat',
            activeSessionId: session?.id ?? next.practice?.activeSessionId } }
      })} />
      {cloudSetup && <CloudGeneratorDialog model={cloudSetup.model}
        localSearch={embeddingModels.length > 0 && embeddingModels.every(model => model.engine !== 'remote')}
        configuredModels={appState.models}
        onClose={() => setCloudSetup(null)} onConnected={connectedCloudGenerator} onForgotten={forgotCloudConnections} />}
      {cloudNotice && <div className="cloud-connected-notice" role="status">{cloudNotice}</div>}
    </main>
  )
}

function LoadingScreen() {
  return (
    <div className="view-frame loading-frame" aria-label="Loading TokenSmith">
      <div className="loading-mark" aria-hidden="true">
        <Sparkles size={30} />
      </div>
      <h1>Preparing your library</h1>
      <p>Loading chats, PDFs, and the local study engine.</p>
    </div>
  )
}

function ChatScreen({
  historyOpen: isSidebarOpen,
  onHistoryOpenChange: setIsSidebarOpen,
  materialsOpen,
  onMaterialsOpenChange: setMaterialsOpen,
  collectionExpansion,
  onCollectionExpansionChange,
  onViewLog,
  progress,
  isActive,
  onModeChange,
  onPracticeTopic,
  practiceChatDraft,
  catalog,
  scope: documentScope,
  onScopeChange,
  activeConversationId,
  conversations,
  embeddingModels,
  isSetupCardDismissed,
  materials,
  models,
  selectedModel,
  settings,
  onChatStateChange,
  onDismissSetupCard,
  onInstallOllamaChatModel,
  onInstallOllamaEmbedderModel,
  onOpenLibrary,
  onRemoveModel,
  onSelectModel,
  onConnectCloud,
  onManageModels,
  onExplanationDepthChange
}: {
  progress: GenerationProgressApi
  isActive: boolean
  onViewLog: () => void
  onModeChange: (mode: StudyMode) => void
  onPracticeTopic: (topic: PracticeTopic) => void
  practiceChatDraft?: { conversationId: string; draft: ChatDraft }
  catalog: StudyCatalog
  scope: StudyDocumentRef[]
  onScopeChange: (scope: StudyDocumentRef[]) => void
  activeConversationId: string
  conversations: Conversation[]
  embeddingModels: LocalModel[]
  isSetupCardDismissed: boolean
  materials: CourseMaterial[]
  models: LocalModel[]
  selectedModel?: LocalModel
  settings: TokenSmithSettings
  onDismissSetupCard: () => void
  onInstallOllamaChatModel: (modelName?: string, baseUrl?: string, modelInfo?: OllamaModelInfo) => void
  onInstallOllamaEmbedderModel: (modelName?: string, baseUrl?: string, modelInfo?: OllamaModelInfo) => void
  onOpenLibrary: () => void
  onRemoveModel: (model: LocalModel) => void
  onSelectModel: (modelId: string) => void
  onConnectCloud: (model?: LocalModel) => void
  onManageModels: () => void
  onExplanationDepthChange: (depth: ExplanationDepth) => void
  onChatStateChange: (
    updater: (current: Pick<AppStateSnapshot, 'activeConversationId' | 'conversations'>) => Pick<
      AppStateSnapshot,
      'activeConversationId' | 'conversations'
    >
  ) => void
} & StudyPanels) {
  const [drafts, setDrafts] = useState<Record<string, ChatDraft>>({})
  const [editingQuestionId, setEditingQuestionId] = useState<string | null>(null)
  const [pendingConversationId, setPendingConversationId] = useState<string | null>(null)
  const [isPreparingFollowUps, setIsPreparingFollowUps] = useState(false)
  const foregroundRequestRef = useRef<string | null>(null)
  const stopStarterRef = useRef<(() => void) | null>(null)
  function setPendingStatusText(text: string | null) {
    if (foregroundRequestRef.current) progress.stage(foregroundRequestRef.current,
      text ? text.replace(/\s*\.\.\.$/, '').replace(/^./, c => c.toUpperCase()) : 'Writing answer')
  }
  function cancelForeground() {
    const id = foregroundRequestRef.current
    if (id) {
      sourcePreview.finish(id)
      progress.cancel(id)
      void window.tokensmith?.cancelChatRequest?.(id).catch(() => {})
      foregroundRequestRef.current = null
    }
    setIsPreparingFollowUps(false)
  }
  function beginForeground(kind: GenerationKind, label: string, inputChars = 0, runtimeSettings = selectedModelSettings) {
    cancelForeground()
    stopStarterRef.current?.()
    const id = createId('generation')
    foregroundRequestRef.current = id
    reader.close()
    if (kind === 'answer' || kind === 'simplify') sourcePreview.begin(id)
    if (selectedModel) progress.begin({ id, kind, label, conversationId: activeConversation.id, model: selectedModel,
      settings: runtimeSettings, inputChars, contextual: activeConversation.messages.length > 0,
      depth: settings.application.explanationDepth, count: settings.application.followUpSuggestionCount,
      onStop: handleStopResponse })
    return id
  }
  const [pendingExplanationId, setPendingExplanationId] = useState<string | null>(null)
  const [explanationError, setExplanationError] = useState<{ answerId: string; text: string; view: AnswerView } | null>(null)
  const explanationInFlightRef = useRef(false)
  const [pendingSources, setPendingSources] = useState<ChatSource[]>([])
  const [chatError, setChatError] = useState<string | null>(null)
  const [expandedMessageId, setExpandedMessageId] = useState<string | null>(null)
  const [focusedAnswerId, setFocusedAnswerId] = useState<string | null>(null)
  const reader = useSourceReader(readSourceDocument)
  const [automaticSource, setAutomaticSource] = useState<SourceReaderState | null>(null)
  const sourcePreview = useMemo(() => createAnswerSourcePreview<SourceReaderState>(setAutomaticSource), [])
  useLayoutEffect(() => {
    sourcePreview.dismiss()
    reader.close()
  }, [isActive, activeConversationId, sourcePreview, reader.close])
  const [sourceTrayError, setSourceTrayError] = useState<string | null>(null)
  const [ollamaStatus, setOllamaStatus] = useState<OllamaStatus | null>(null)
  const [ollamaSetupPhase, setOllamaSetupPhase] = useState<'idle' | 'checking' | 'opening' | 'starting' | 'error'>('idle')
  const [ollamaPullingModel, setOllamaPullingModel] = useState<'chat' | 'embedder' | null>(null)
  const [ollamaPullProgress, setOllamaPullProgress] = useState<OllamaPullProgress | null>(null)
  const [ollamaSetupError, setOllamaSetupError] = useState<string | null>(null)
  const [starterQuestionSuggestions, setStarterQuestionSuggestions] = useState<string[]>([])
  const [starterQuestionStatus, setStarterQuestionStatus] = useState<'idle' | 'loading' | 'error'>('idle')
  const [starterQuestionError, setStarterQuestionError] = useState<string | null>(null)
  const requestSequenceRef = useRef(0)
  const starterQuestionRequestSequenceRef = useRef(0)
  const composerInputRef = useRef<HTMLTextAreaElement | null>(null)

  const activeConversation =
    conversations.find((conversation) => conversation.id === activeConversationId) ??
    conversations[0] ??
    starterConversations[0]
  const isPending = pendingConversationId === activeConversation.id
  const discussionDraft: ChatDraft = activeConversation.practiceReference && activeConversation.messages.length === 0
    ? { text: 'Can you help me understand this?', selectedPassage: practiceDiscussionPassage(activeConversation.practiceReference) } : { text: '' }
  const { text: draft, selectedPassage } = drafts[activeConversation.id] ?? discussionDraft
  function updateDraft(update: Partial<ChatDraft>) {
    setDrafts((current) => ({ ...current, [activeConversation.id]: { ...(current[activeConversation.id] ?? discussionDraft), ...update } }))
  }
  function clearDraft() {
    setDrafts((current) => ({ ...current, [activeConversation.id]: { text: '' } }))
  }
  useEffect(() => {
    if (!practiceChatDraft) return
    setDrafts(current => ({ ...current, [practiceChatDraft.conversationId]: practiceChatDraft.draft }))
    composerInputRef.current?.focus()
  }, [practiceChatDraft])
  const readyDocuments = new Set(catalog.documents.map(item => `${item.materialId}:${item.documentId}`))
  const practiceScopeReady = catalog.status === 'ready' && documentScope.length > 0 &&
    documentScope.every(item => readyDocuments.has(`${item.materialId}:${item.documentId}`))
  function practiceAnswer(message: ChatMessage) {
    if (!practiceScopeReady) return
    const index = activeConversation.messages.findIndex(item => item.id === message.id)
    const question = activeConversation.messages.slice(0, index).reverse().find(item => item.role === 'user')
    if (!question) return
    const answer = answerForDisplay(message)
    onPracticeTopic({ title: question.text.slice(0, 120), question: question.text, answer: answer.text,
      conversationId: activeConversation.id, answerId: message.id, sources: sourcesInScope(answer.sources ?? [], documentScope) })
  }
  const scopeMaterialIds = new Set(documentScope.map(item => item.materialId))
  const activeMaterials = materials.filter(material => scopeMaterialIds.has(material.id)).map(material => ({ ...material, isActive: true }))
  function changeDocumentScope(documentScope: StudyDocumentRef[]) {
    handleStopResponse()
    stopStarterRef.current?.()
    onScopeChange(documentScope)
  }
  const selectedModelLabel = selectedModel ? displayModelName(selectedModel) : 'Choose a model'
  const sourceMessages = activeConversation.messages.filter(
    (message) => message.role === 'assistant' && (message.sources?.length ?? 0) > 0
  )
  const selectedSourceMessage =
    sourceMessages.find((message) => message.id === (focusedAnswerId ?? expandedMessageId)) ?? sourceMessages[sourceMessages.length - 1]
  const selectedSources = useMemo(() => {
    if (!settings.application.showSources) {
      return []
    }
    const sources = isPending ? pendingSources : selectedSourceMessage ? answerForDisplay(selectedSourceMessage).sources ?? [] : []
    return sources.slice(0, maxSourceTrayCards)
  }, [isPending, pendingSources, selectedSourceMessage, settings.application.showSources])
  const activeMaterialKey = studyScopeKey(documentScope)
  const previousScopeKey = useRef(activeMaterialKey)
  useLayoutEffect(() => {
    if (previousScopeKey.current === activeMaterialKey) return
    previousScopeKey.current = activeMaterialKey
    handleStopResponse()
    sourcePreview.dismiss()
    reader.close()
    setChatError(null)
  }, [activeMaterialKey])
  const selectedModelSettings = selectedModel ? modelSettingsFor(settings, selectedModel.id) : settings.modelDefaults
  const canSuggestStarterQuestions =
    isActive &&
    !activeConversation.practiceReference &&
    activeConversation.messages.length === 0 &&
    !pendingConversationId &&
    Boolean(selectedModel) &&
    selectedModel?.status === 'ready' &&
    documentScope.length > 0 && catalog.status === 'ready' &&
    settings.application.suggestionMode !== 'off'
  const readyChatModel = models.find((model) => model.status === 'ready' && modelCanGenerate(model))
  const readyEmbeddingModel = models.find((model) => model.status === 'ready' && modelCanEmbed(model))
  const hasReadyMaterials = materials.some((material) => material.status === 'ready' || Boolean(material.indexedAt))
  const hasIndexingMaterials = materials.some((material) => material.status === 'indexing' || material.status === 'paused')
  const needsChatModelSetup = !readyChatModel
  const needsEmbeddingModelSetup = !readyEmbeddingModel
  const needsDocumentSetup = Boolean(readyEmbeddingModel) && !hasReadyMaterials && !hasIndexingMaterials
  const needsModelSetup = needsChatModelSetup || needsEmbeddingModelSetup
  const needsSetupCard = needsModelSetup || needsDocumentSetup
  const shouldShowSetupCard = needsSetupCard && !isSetupCardDismissed
  const {
    capabilities: setupDeviceCapabilities,
    state: setupDeviceCapabilityState
  } = useDeviceCapabilities(isActive && needsChatModelSetup)
  const setupRecommendation = useMemo<ModelRecommendation | null>(
    () =>
      setupDeviceCapabilities
        ? recommendModel(setupDeviceCapabilities, localModelCatalog, defaultDeviceTierPolicy)
        : null,
    [setupDeviceCapabilities]
  )
  const recommendsCloudChat = setupRecommendation?.kind === 'cloud'
  const setupChatModelName = recommendsCloudChat
    ? null
    : setupRecommendation?.recommendedModelId ?? recommendedOllamaChatModel
  const setupChatModelLabel = recommendsCloudChat
    ? setupRecommendation?.recommendedModelName ?? 'a cloud model'
    : setupRecommendation?.recommendedModelName ?? 'Gemma 4 E4B'
  const composerPlaceholder = selectedModel
      ? 'Ask about your documents...'
      : needsDocumentSetup
        ? 'Add documents to Library, then choose a chat model...'
        : 'Choose a recommended model to ask questions...'

  useEffect(() => {
    setEditingQuestionId(null)
    setFocusedAnswerId(null)
  }, [activeConversation.id])

  useEffect(() => {
    if (!isActive || (!needsModelSetup && !needsDocumentSetup) || !window.tokensmith?.getOllamaStatus) {
      return
    }

    let cancelled = false

    async function refresh() {
      setOllamaSetupPhase('checking')
      setOllamaSetupError(null)

      try {
        const status = await window.tokensmith?.getOllamaStatus()
        if (!cancelled && status) {
          setOllamaStatus(status)
          setOllamaSetupPhase('idle')
        }
      } catch (error) {
        if (!cancelled) {
          setOllamaSetupPhase('error')
          setOllamaSetupError(readableOllamaError(error, 'Could not check Ollama.'))
        }
      }
    }

    void refresh()

    return () => {
      cancelled = true
    }
  }, [isActive, needsModelSetup, needsDocumentSetup])

  useEffect(() => {
    if (!window.tokensmith?.onOllamaPullProgress) {
      return undefined
    }

    return window.tokensmith.onOllamaPullProgress((progress) => {
      if (
        !(setupChatModelName && ollamaModelNameMatches(progress.model, setupChatModelName)) &&
        !ollamaModelNameMatches(progress.model, recommendedOllamaEmbeddingModel)
      ) {
        return
      }

      setOllamaPullProgress(progress)
      const isTerminalPullStatus = progress.status === 'complete' || progress.status === 'error' || progress.status === 'incomplete' || progress.status === 'removed'
      if (setupChatModelName && ollamaModelNameMatches(progress.model, setupChatModelName)) {
        setOllamaPullingModel(isTerminalPullStatus ? null : 'chat')
      }
      if (ollamaModelNameMatches(progress.model, recommendedOllamaEmbeddingModel)) {
        setOllamaPullingModel(isTerminalPullStatus ? null : 'embedder')
      }
    })
  }, [setupChatModelName])

  useEffect(() => {
    if (!canSuggestStarterQuestions || !selectedModel || !window.tokensmith?.suggestChatQuestions) {
      setStarterQuestionSuggestions([])
      setStarterQuestionStatus('idle')
      setStarterQuestionError(null)
      return
    }

    const tokensmith = window.tokensmith
    let cancelled = false
    const requestId = `${activeConversation.id}:${++starterQuestionRequestSequenceRef.current}`
    setStarterQuestionSuggestions([])
    setStarterQuestionStatus('loading')
    setStarterQuestionError(null)

    const stop = () => {
      cancelled = true
      window.clearTimeout(startTimer)
      progress.cancel(requestId)
      void tokensmith.cancelChatQuestionSuggestions(requestId).catch(() => {})
      setStarterQuestionStatus('idle')
      if (stopStarterRef.current === stop) stopStarterRef.current = null
    }
    const startTimer = window.setTimeout(() => {
      progress.begin({ id: requestId, kind: 'starter', label: 'Preparing questions', conversationId: activeConversation.id,
        model: selectedModel, settings: selectedModelSettings, count: settings.application.followUpSuggestionCount, onStop: stop })
      void tokensmith
        .suggestChatQuestions(requestId, {
          messages: activeConversation.messages,
          materials: activeMaterials,
          documents: documentScope,
          model: selectedModel,
          settings,
          applicationSettings: settings.application,
          modelSettings: selectedModelSettings,
          retrievedSources: []
        })
        .then((response) => {
          if (cancelled) return
          progress.finish(requestId, 'Questions ready', response.suggestions.length > 0)
          setStarterQuestionSuggestions(response.suggestions)
          setStarterQuestionStatus('idle')
        })
        .catch((error) => {
          if (cancelled) return
          progress.cancel(requestId)
          setStarterQuestionSuggestions([])
          setStarterQuestionStatus('error')
          setStarterQuestionError(readableErrorMessage(error, 'Could not prepare suggested questions from the selected documents.'))
        })
    }, starterQuestionDebounceMs)

    stopStarterRef.current = stop
    return stop
  }, [
    activeConversation.id,
    activeConversation.messages.length,
    activeMaterialKey,
    canSuggestStarterQuestions,
    selectedModel?.id,
    selectedModelSettings.starterQuestionPrompt,
    settings.application.followUpSuggestionCount,
    settings.application.suggestionMode
  ])

  function handleNewChat() {
    if (window.innerWidth <= 800) setIsSidebarOpen(false)
    cancelForeground()
    requestSequenceRef.current += 1

    const newConversation: Conversation = {
      id: createId('conversation'),
      modelId: settings.application.defaultModelId,
      title: 'New Study Chat',
      period: 'Today',
      messages: [],
      documentScope: scopeRefs(documentScope)
    }

    onChatStateChange((current) => ({
      conversations: [newConversation, ...current.conversations],
      activeConversationId: newConversation.id
    }))
    setPendingConversationId(null)
    setPendingExplanationId(null)
    explanationInFlightRef.current = false
    setPendingStatusText(null)
    setPendingSources([])
    setChatError(null)
    setExpandedMessageId(null)
    setFocusedAnswerId(null)
    reader.close()
    setSourceTrayError(null)
  }

  function handleStopResponse() {
    cancelForeground()
    requestSequenceRef.current += 1
    setPendingConversationId(null)
    setPendingExplanationId(null)
    explanationInFlightRef.current = false
    setPendingStatusText(null)
    setPendingSources([])
  }

  function handleSelectConversation(conversationId: string) {
    if (window.innerWidth <= 800) setIsSidebarOpen(false)
    reader.close()
    setSourceTrayError(null)
    setPendingSources([])
    setChatError(null)
    onChatStateChange((current) => ({
      ...current,
      activeConversationId: conversationId
    }))
  }

  function handleRenameConversation(conversationId: string, title: string) {
    onChatStateChange((current) => ({
      ...current,
      conversations: current.conversations.map((item) =>
        item.id === conversationId ? { ...item, title } : item
      )
    }))
  }

  function handleDeleteConversation(conversationId: string) {
    const conversation = conversations.find((item) => item.id === conversationId)
    if (!conversation) {
      return
    }

    if (!window.confirm(`Delete "${conversation.title}"?`)) {
      return
    }

    if (pendingConversationId === conversationId) cancelForeground()
    requestSequenceRef.current += pendingConversationId === conversationId ? 1 : 0
    setDrafts((current) => {
      const next = { ...current }
      delete next[conversationId]
      return next
    })

    onChatStateChange((current) => {
      const remainingConversations = current.conversations.filter((item) => item.id !== conversationId)
      const nextConversations = remainingConversations.length > 0 ? remainingConversations : [{ ...createFreshConversation(settings.application.defaultModelId), documentScope: scopeRefs(documentScope) }]
      const nextActiveConversationId =
        current.activeConversationId === conversationId ? nextConversations[0].id : current.activeConversationId

      return {
        conversations: nextConversations,
        activeConversationId: nextActiveConversationId
      }
    })

    if (pendingConversationId === conversationId) {
      setPendingConversationId(null)
      setPendingExplanationId(null)
      explanationInFlightRef.current = false
      setPendingStatusText(null)
      setPendingSources([])
    }
    setExpandedMessageId(null)
    setFocusedAnswerId(null)
    reader.close()
    setSourceTrayError(null)
    setChatError(null)
  }

  function handleClearConversations() {
    if (!window.confirm('Clear all chats and start a new chat?')) {
      return
    }

    const freshConversation = { ...createFreshConversation(settings.application.defaultModelId), documentScope: scopeRefs(documentScope) }
    cancelForeground()
    requestSequenceRef.current += 1

    onChatStateChange(() => ({
      conversations: [freshConversation],
      activeConversationId: freshConversation.id
    }))

    setDrafts({})
    setPendingConversationId(null)
    setPendingExplanationId(null)
    explanationInFlightRef.current = false
    setPendingStatusText(null)
    setPendingSources([])
    setExpandedMessageId(null)
    setFocusedAnswerId(null)
    reader.close()
    setSourceTrayError(null)
    setChatError(null)
  }

  function handleCopyQuestion(text: string) {
    if (!navigator.clipboard) {
      return
    }

    void navigator.clipboard.writeText(text).catch(() => undefined)
  }

  function handleQuoteSelection(passage: ChatSelectedPassage) {
    updateDraft({ selectedPassage: passage })
    requestAnimationFrame(() => {
      const input = composerInputRef.current
      input?.focus()
      input?.setSelectionRange(input.value.length, input.value.length)
    })
  }

  function handleUseFollowUpSuggestion(suggestion: string) {
    void submitPrompt(suggestion)
  }

  function handleUseStarterQuestion(question: string) {
    void submitPrompt(question)
  }

  async function refreshOllamaStatus() {
    if (!window.tokensmith?.getOllamaStatus) {
      setOllamaSetupPhase('error')
      setOllamaSetupError('Ollama setup is not available in this build.')
      return
    }

    setOllamaSetupPhase('checking')
    setOllamaPullingModel(null)
    setOllamaPullProgress(null)
    setOllamaSetupError(null)

    try {
      const status = await window.tokensmith.getOllamaStatus()
      setOllamaStatus(status)
      setOllamaSetupPhase('idle')
    } catch (error) {
      setOllamaSetupPhase('error')
      setOllamaSetupError(readableOllamaError(error, 'Could not check Ollama.'))
    }
  }

  async function handleOpenOllamaDownloadPage() {
    if (!window.tokensmith?.openOllamaDownloadPage) {
      setOllamaSetupError('Ollama setup is not available in this build.')
      return
    }

    try {
      await window.tokensmith.openOllamaDownloadPage()
    } catch (error) {
      setOllamaSetupPhase('error')
      setOllamaSetupError(readableErrorMessage(error, 'Could not open the Ollama download page.'))
    }
  }

  async function handleOpenOllamaApp() {
    if (!window.tokensmith?.openOllamaApp) {
      setOllamaSetupError('Ollama setup is not available in this build.')
      return
    }

    setOllamaSetupPhase('opening')
    setOllamaSetupError(null)

    try {
      const result = await window.tokensmith.openOllamaApp()
      if (!result.opened) {
        setOllamaSetupPhase('error')
        setOllamaSetupError(result.message ?? 'Could not open Ollama.')
        return
      }

      await new Promise((resolve) => window.setTimeout(resolve, 1500))
      await refreshOllamaStatus()
    } catch (error) {
      setOllamaSetupPhase('error')
      setOllamaSetupError(readableErrorMessage(error, 'Could not open Ollama.'))
    }
  }

  async function handleStartOllamaService() {
    if (!window.tokensmith?.startOllamaService) {
      setOllamaSetupError('Ollama setup is not available in this build.')
      return
    }

    setOllamaSetupPhase('starting')
    setOllamaPullingModel(null)
    setOllamaPullProgress(null)
    setOllamaSetupError(null)

    try {
      const result = await window.tokensmith.startOllamaService()
      if (!result.opened) {
        setOllamaSetupPhase('error')
        setOllamaSetupError(result.message ?? 'Could not start Ollama.')
        return
      }

      await refreshOllamaStatus()
    } catch (error) {
      setOllamaSetupPhase('error')
      setOllamaSetupError(readableErrorMessage(error, 'Could not start Ollama.'))
    }
  }

  function pausedOllamaProgress(modelName: string, current: OllamaPullProgress | null): OllamaPullProgress {
    return current && ollamaModelNameMatches(current.model, modelName)
      ? { ...current, status: 'incomplete', message: 'Download paused' }
      : {
          model: modelName,
          status: 'incomplete',
          percent: 0,
          message: 'Download paused'
        }
  }

  async function handlePullOllamaChatModel() {
    if (!setupChatModelName || !window.tokensmith?.pullOllamaModel) {
      setOllamaSetupError('Ollama setup is not available in this build.')
      return
    }

    setOllamaPullingModel('chat')
    setOllamaPullProgress({
      model: setupChatModelName,
      status: 'starting',
      percent: 0,
      message: 'Starting download'
    })
    setOllamaSetupError(null)

    try {
      await window.tokensmith.pullOllamaModel(setupChatModelName)
      const status = await window.tokensmith.getOllamaStatus()
      setOllamaStatus(status)
      const modelInfo = status.models.find((model) => ollamaModelInfoMatches(model, setupChatModelName))
      if (!modelInfo) {
        const message = `Ollama finished the download, but TokenSmith could not verify ${setupChatModelLabel}.`
        setOllamaSetupPhase('error')
        setOllamaPullingModel(null)
        setOllamaPullProgress((current) =>
          current && ollamaModelNameMatches(current.model, setupChatModelName)
            ? { ...current, status: 'error', message, error: message }
            : { model: setupChatModelName, status: 'error', percent: 0, message, error: message }
        )
        setOllamaSetupError(message)
        return
      }
      onInstallOllamaChatModel(setupChatModelName, status.baseUrl, modelInfo)
      setOllamaSetupPhase('idle')
      setOllamaPullingModel(null)
      setOllamaPullProgress(null)
    } catch (error) {
      if (error instanceof Error && /paused|cancelled|aborted/i.test(error.message)) {
        setOllamaSetupPhase('idle')
        setOllamaPullingModel(null)
        setOllamaPullProgress((current) => pausedOllamaProgress(setupChatModelName, current))
        return
      }
      setOllamaSetupPhase('error')
      setOllamaPullingModel(null)
      setOllamaSetupError(readableErrorMessage(error, `Could not download ${setupChatModelLabel} with Ollama.`))
    }
  }

  async function handlePullOllamaEmbedderModel() {
    if (!window.tokensmith?.pullOllamaModel) {
      setOllamaSetupError('Ollama setup is not available in this build.')
      return
    }

    setOllamaPullingModel('embedder')
    setOllamaPullProgress({
      model: recommendedOllamaEmbeddingModel,
      status: 'starting',
      percent: 0,
      message: 'Starting download'
    })
    setOllamaSetupError(null)

    try {
      await window.tokensmith.pullOllamaModel(recommendedOllamaEmbeddingModel)
      const status = await window.tokensmith.getOllamaStatus()
      setOllamaStatus(status)
      const modelInfo = status.models.find((model) => ollamaModelInfoMatches(model, recommendedOllamaEmbeddingModel))
      if (!modelInfo) {
        const message = 'Ollama finished the download, but TokenSmith could not verify the Nomic Embedder Model.'
        setOllamaSetupPhase('error')
        setOllamaPullingModel(null)
        setOllamaPullProgress((current) =>
          current && ollamaModelNameMatches(current.model, recommendedOllamaEmbeddingModel)
            ? { ...current, status: 'error', message, error: message }
            : { model: recommendedOllamaEmbeddingModel, status: 'error', percent: 0, message, error: message }
        )
        setOllamaSetupError(message)
        return
      }
      onInstallOllamaEmbedderModel(recommendedOllamaEmbeddingModel, status.baseUrl, modelInfo)
      setOllamaSetupPhase('idle')
      setOllamaPullingModel(null)
      setOllamaPullProgress(null)
    } catch (error) {
      if (error instanceof Error && /paused|cancelled|aborted/i.test(error.message)) {
        setOllamaSetupPhase('idle')
        setOllamaPullingModel(null)
        setOllamaPullProgress((current) => pausedOllamaProgress(recommendedOllamaEmbeddingModel, current))
        return
      }
      setOllamaSetupPhase('error')
      setOllamaPullingModel(null)
      setOllamaSetupError(readableErrorMessage(error, 'Could not download the Nomic Embedder Model.'))
    }
  }

  function setupModelNameForRole(role: 'chat' | 'embedder') {
    const modelName = role === 'chat' ? setupChatModelName : recommendedOllamaEmbeddingModel
    if (!modelName) {
      throw new Error('A cloud recommendation does not have a local chat model.')
    }
    return modelName
  }

  function setupReadyModelForRole(role: 'chat' | 'embedder') {
    const expectedModel = setupModelNameForRole(role)
    if (!expectedModel) {
      return undefined
    }
    const roleMatches = role === 'chat' ? modelCanGenerate : modelCanEmbed
    const recommendedModel = models.find(
      (model) =>
        model.status === 'ready' &&
        roleMatches(model) &&
        (model.engine === 'ollama' || model.source === 'ollama') &&
        ollamaModelNameMatches(model.ollamaModelName ?? modelFilename(model) ?? model.name, expectedModel)
    )

    return recommendedModel ?? (role === 'chat' ? readyChatModel : readyEmbeddingModel)
  }

  function setupRemovableModelForRole(role: 'chat' | 'embedder') {
    const model = setupReadyModelForRole(role)
    return model && (model.engine === 'ollama' || model.source === 'ollama') ? model : undefined
  }

  async function handlePauseOllamaPull(role: 'chat' | 'embedder') {
    if (!window.tokensmith?.cancelOllamaPull) {
      setOllamaSetupError('Ollama setup is not available in this build.')
      return
    }

    const modelName = setupModelNameForRole(role)
    setOllamaSetupError(null)

    try {
      await window.tokensmith.cancelOllamaPull(modelName, ollamaStatus?.baseUrl)
      setOllamaPullingModel(null)
      setOllamaPullProgress((current) => pausedOllamaProgress(modelName, current))
    } catch (error) {
      setOllamaSetupPhase('error')
      setOllamaSetupError(readableErrorMessage(error, 'Could not pause the download.'))
    }
  }

  async function handleDeleteOllamaSetupModel(role: 'chat' | 'embedder') {
    const modelName = setupModelNameForRole(role)
    const readyModel = setupRemovableModelForRole(role)
    setOllamaSetupError(null)

    if (ollamaPullingModel === role) {
      setOllamaPullingModel(null)
    }
    setOllamaPullProgress((current) => (current && ollamaModelNameMatches(current.model, modelName) ? null : current))

    if (readyModel) {
      onRemoveModel(readyModel)
    } else if (window.tokensmith?.deleteOllamaModel) {
      try {
        await window.tokensmith.deleteOllamaModel(modelName, ollamaStatus?.baseUrl)
      } catch (error) {
        setOllamaSetupPhase('error')
        setOllamaSetupError(readableErrorMessage(error, `Could not delete ${role === 'chat' ? setupChatModelLabel : 'Nomic'}.`))
        return
      }
    } else {
      setOllamaSetupError('Ollama setup is not available in this build.')
      return
    }

    if (window.tokensmith?.getOllamaStatus) {
      try {
        const status = await window.tokensmith.getOllamaStatus()
        setOllamaStatus(status)
        setOllamaSetupPhase('idle')
      } catch {
        setOllamaSetupPhase('idle')
      }
    }
  }

  function ollamaPullProgressFor(role: 'chat' | 'embedder') {
    if (!ollamaPullProgress) {
      return null
    }

    const expectedModel = role === 'chat' ? setupChatModelName : recommendedOllamaEmbeddingModel
    if (!expectedModel) {
      return null
    }
    return ollamaModelNameMatches(ollamaPullProgress.model, expectedModel) ? ollamaPullProgress : null
  }

  function ollamaPullProgressLine(progress: OllamaPullProgress) {
    if (progress.error) {
      return 'Download failed'
    }

    const rawMessage = progress.message ?? 'Downloading'
    const message = rawMessage.startsWith('pulling ')
      ? rawMessage === 'pulling manifest'
        ? 'Checking model files...'
        : 'Downloading model files...'
      : rawMessage.startsWith('verifying ')
        ? 'Verifying download...'
        : rawMessage.startsWith('writing ')
          ? 'Finishing download...'
          : rawMessage === 'success'
            ? 'Downloaded'
            : rawMessage
    if (!progress.total) {
      return message
    }

    const completed = formatBytes(progress.completed)
    const total = formatBytes(progress.total)
    return `${progress.percent}% · ${completed} of ${total} · ${message}`
  }

  function renderOllamaPullProgress(role: 'chat' | 'embedder') {
    const progress = ollamaPullProgressFor(role)
    if (!progress) {
      return null
    }

    const width = progress.total ? progress.percent : progress.status === 'downloading' ? 8 : 0

    return (
      <>
        <div className="model-download-progress" aria-label={`Download progress ${progress.percent}%`}>
          <span style={{ width: `${Math.max(0, Math.min(100, width))}%` }} />
        </div>
        <small>{ollamaPullProgressLine(progress)}</small>
        {progress.error && <small className="model-download-error">{progress.error}</small>}
      </>
    )
  }

  function documentSetupLabel() {
    if (hasReadyMaterials) {
      return 'PDFs are ready for chat.'
    }

    if (hasIndexingMaterials) {
      return 'Preparing your PDFs...'
    }

    if (!readyEmbeddingModel) {
      return 'Download models first so TokenSmith can prepare your PDFs.'
    }

    return 'Choose the folder with your PDFs.'
  }

  function renderDocumentSetupAction() {
    if (hasReadyMaterials) {
      return (
        <span className="chat-setup-status is-ready">
          <Check size={16} aria-hidden="true" />
          <span>Ready</span>
        </span>
      )
    }

    if (hasIndexingMaterials) {
      return (
        <span className="chat-setup-status">
          <Loader2 size={16} aria-hidden="true" />
          <span>Preparing...</span>
        </span>
      )
    }

    if (!readyEmbeddingModel) {
      return (
        <span className="chat-setup-status">
          <span>Download models first</span>
        </span>
      )
    }

    return (
      <button className="model-action-button" type="button" onClick={onOpenLibrary}>
        <span>Choose Folder</span>
      </button>
    )
  }

  function renderSetupStepNumber(step: number, isReady: boolean) {
    return <div className={`chat-setup-step-number ${isReady ? 'is-ready' : ''}`}>{step}</div>
  }

  function ollamaInstallSetupLabel() {
    if (ollamaStatus?.running) {
      return 'Ollama is ready.'
    }

    if (ollamaSetupPhase === 'checking') {
      return 'Checking for Ollama...'
    }

    if (ollamaSetupPhase === 'starting') {
      return 'Starting Ollama...'
    }

    if (ollamaSetupPhase === 'opening') {
      return 'Opening Ollama...'
    }

    return ollamaStatus?.installedApp
      ? 'Start Ollama to prepare PDFs and use local models.'
      : 'Install Ollama to prepare PDFs and use local models.'
  }

  function renderOllamaInstallAction() {
    const isBusy = ollamaSetupPhase === 'checking' || ollamaSetupPhase === 'opening' || ollamaSetupPhase === 'starting'
    const busyLabel =
      ollamaSetupPhase === 'starting'
        ? 'Starting...'
        : ollamaSetupPhase === 'opening'
          ? 'Opening...'
          : 'Checking...'

    if (ollamaStatus?.running) {
      return (
        <span className="chat-setup-status is-ready">
          <Check size={16} aria-hidden="true" />
          <span>Ready</span>
        </span>
      )
    }

    if (isBusy) {
      return (
        <div className="chat-setup-action">
          <button className="model-action-button" type="button" disabled>
            <Loader2 size={16} aria-hidden="true" />
            <span>{busyLabel}</span>
          </button>
        </div>
      )
    }

    return (
      <div className="chat-setup-action">
        <button
          className="model-action-button"
          type="button"
          onClick={ollamaStatus?.installedApp ? handleStartOllamaService : handleOpenOllamaDownloadPage}
        >
          <span>{ollamaStatus?.installedApp ? 'Start Ollama' : 'Install Ollama'}</span>
        </button>
        {ollamaStatus?.installedApp && (
          <button className="model-action-button is-muted" type="button" onClick={handleOpenOllamaApp}>
            <span>Open App</span>
          </button>
        )}
        {ollamaStatus && (
          <button className="model-action-button is-muted" type="button" onClick={refreshOllamaStatus}>
            <span>Recheck</span>
          </button>
        )}
        {(ollamaSetupError || ollamaStatus?.error) && (
          <small className="model-download-error">{ollamaSetupError ?? readableOllamaError(new Error(ollamaStatus?.error), 'Ollama is not running yet.')}</small>
        )}
      </div>
    )
  }

  function downloadModelsSetupLabel() {
    if (setupDeviceCapabilityState === 'checking') {
      return 'Checking which model fits this device...'
    }

    if (!ollamaStatus?.running) {
      return recommendsCloudChat
        ? 'Start Ollama to set up PDF search, then connect the cloud chat model.'
        : 'Start Ollama, then download the recommended models.'
    }

    if (ollamaPullingModel === 'embedder') {
      return 'Downloading Nomic Embedder Model...'
    }

    if (ollamaPullingModel === 'chat') {
      return `Downloading ${setupChatModelLabel}...`
    }

    if (readyEmbeddingModel && readyChatModel) {
      return 'Both models are ready.'
    }

    if (recommendsCloudChat) {
      return readyEmbeddingModel
        ? `PDF search is ready. Connect ${setupChatModelLabel} for chat.`
        : `Download Nomic for PDF search, then connect ${setupChatModelLabel} for chat.`
    }

    if (readyEmbeddingModel) {
      return 'Nomic Embedder Model is ready.'
    }

    if (readyChatModel) {
      return `${setupChatModelLabel} is ready.`
    }

    return `Download Nomic and the recommended ${setupChatModelLabel}.`
  }

  function renderDownloadModelControl(role: 'embedder' | 'chat') {
    const isEmbedder = role === 'embedder'
    const readyModel = setupReadyModelForRole(role)
    const isReady = Boolean(readyModel)
    const removableModel = setupRemovableModelForRole(role)
    const shortLabel = isEmbedder ? 'Nomic Embedder Model' : setupChatModelLabel
    const downloadLabel = isEmbedder ? 'Download Nomic' : `Download ${setupChatModelLabel}`
    const busyLabel = isEmbedder ? 'Downloading Nomic...' : `Downloading ${setupChatModelLabel}...`
    const isPulling = ollamaPullingModel === role
    const progress = ollamaPullProgressFor(role)
    const canResume = progress?.status === 'incomplete' || progress?.status === 'error'
    const otherModelIsPulling = Boolean(ollamaPullingModel && ollamaPullingModel !== role)
    const handleDownload = isEmbedder ? handlePullOllamaEmbedderModel : handlePullOllamaChatModel

    if (isReady) {
      return (
        <div className="chat-setup-model-download">
          <span className="chat-setup-model-pill is-ready">
            <Check size={15} aria-hidden="true" />
            <span>{shortLabel} ready</span>
          </span>
          {removableModel && (
            <button className="model-action-button is-danger" type="button" onClick={() => handleDeleteOllamaSetupModel(role)}>
              <Trash2 size={16} aria-hidden="true" />
              <span>Delete</span>
            </button>
          )}
        </div>
      )
    }

    if (isPulling) {
      return (
        <div className="chat-setup-model-download">
          <div className="chat-setup-model-actions">
            <button className="model-action-button" type="button" onClick={() => handlePauseOllamaPull(role)}>
              <Pause size={16} aria-hidden="true" />
              <span>Pause</span>
            </button>
            <button className="model-action-button is-danger" type="button" onClick={() => handleDeleteOllamaSetupModel(role)}>
              <Trash2 size={16} aria-hidden="true" />
              <span>Delete</span>
            </button>
          </div>
          {renderOllamaPullProgress(role)}
        </div>
      )
    }

    if (canResume) {
      return (
        <div className="chat-setup-model-download">
          <div className="chat-setup-model-actions">
            <button
              className="model-action-button"
              type="button"
              disabled={!ollamaStatus?.running || otherModelIsPulling || (!isEmbedder && setupDeviceCapabilityState === 'checking')}
              onClick={handleDownload}
            >
              <span>{progress?.status === 'incomplete' ? 'Resume' : downloadLabel}</span>
            </button>
            <button className="model-action-button is-danger" type="button" onClick={() => handleDeleteOllamaSetupModel(role)}>
              <Trash2 size={16} aria-hidden="true" />
              <span>Delete</span>
            </button>
          </div>
          {renderOllamaPullProgress(role)}
        </div>
      )
    }

    return (
      <button
        className="model-action-button"
        type="button"
        disabled={!ollamaStatus?.running || otherModelIsPulling || (!isEmbedder && setupDeviceCapabilityState === 'checking')}
        onClick={handleDownload}
      >
        <span>{downloadLabel}</span>
      </button>
    )
  }

  function renderDownloadModelsAction() {
    if (!ollamaStatus?.running) {
      return (
        <span className="chat-setup-status">
          <span>Start Ollama first</span>
        </span>
      )
    }

    if (readyEmbeddingModel && readyChatModel) {
      return (
        <span className="chat-setup-status is-ready">
          <Check size={16} aria-hidden="true" />
          <span>Ready</span>
        </span>
      )
    }

    return (
      <div className="chat-setup-action is-wide">
        {renderDownloadModelControl('embedder')}
        {recommendsCloudChat ? (
          <button className="model-action-button" type="button" onClick={() => onConnectCloud()}>
            <span>Connect {setupChatModelLabel}</span>
          </button>
        ) : (
          renderDownloadModelControl('chat')
        )}
        {ollamaSetupError && <small className="model-download-error">{ollamaSetupError}</small>}
      </div>
    )
  }

  function renderChatSetupCard() {
    if (!shouldShowSetupCard) {
      return null
    }

    return (
      <section className="chat-setup-card" aria-label="Model setup">
        <div className="chat-setup-header">
          <div className="chat-setup-heading">
            <p className="section-kicker">FIRST-TIME TOKENSMITH SETUP</p>
            <h2>Set up TokenSmith</h2>
            <p>Set up PDF search on this device, then choose a local or cloud chat model.</p>
          </div>
          <button className="chat-setup-dismiss" type="button" onClick={onDismissSetupCard} aria-label="Hide setup guide" title="Hide setup guide">
            <X size={18} aria-hidden="true" />
          </button>
        </div>
        <div className="chat-setup-steps">
          <div className="chat-setup-step">
            {renderSetupStepNumber(1, Boolean(ollamaStatus?.running))}
          <div className="chat-setup-copy">
              <strong>Set up Ollama</strong>
              <span>{ollamaInstallSetupLabel()}</span>
            </div>
            {renderOllamaInstallAction()}
          </div>

          <div className="chat-setup-step">
            {renderSetupStepNumber(2, Boolean(readyEmbeddingModel && readyChatModel))}
            <div className="chat-setup-copy">
              <strong>Download models</strong>
              <span>{downloadModelsSetupLabel()}</span>
            </div>
            {renderDownloadModelsAction()}
          </div>

          <div className="chat-setup-step">
            {renderSetupStepNumber(3, hasReadyMaterials)}
            <div className="chat-setup-copy">
              <strong>Add PDFs</strong>
              <span>{documentSetupLabel()}</span>
            </div>
            {renderDocumentSetupAction()}
          </div>
        </div>
        {setupRecommendation && (
          <div className="cloud-generator-entry">
            <div>
              <strong>Recommended for this device: {setupChatModelLabel}</strong>
            </div>
          </div>
        )}
        {needsChatModelSetup && !recommendsCloudChat && <div className="cloud-generator-entry">
          <div><strong>Prefer a cloud chat model?</strong><p>Connect your API key and keep PDF search on this device.</p></div>
          <button type="button" className="secondary-action" onClick={() => onConnectCloud()}>Connect a cloud model</button>
        </div>}
      </section>
    )
  }

  function renderStarterQuestions() {
    if (starterQuestionStatus !== 'loading' && starterQuestionStatus !== 'error' && starterQuestionSuggestions.length === 0) {
      return null
    }

    return (
      <section className="starter-question-panel" aria-label="Suggested first questions">
        <div className="follow-up-title">
          <Sparkles size={18} aria-hidden="true" />
          <span>Suggested questions</span>
        </div>
        {starterQuestionStatus === 'loading' ? null : starterQuestionStatus === 'error' ? (
          <p className="starter-question-status is-error">{starterQuestionError}</p>
        ) : (
          <div className="follow-up-list">
            {starterQuestionSuggestions.map((question) => (
              <button
                className="follow-up-row"
                key={question}
                type="button"
                onClick={() => handleUseStarterQuestion(question)}
              >
                <MessageText text={question} inline />
                <Plus size={18} aria-hidden="true" />
              </button>
            ))}
          </div>
        )}
      </section>
    )
  }

  function previewFirstSource(requestId: string, sources: ChatSource[]) {
    const source = sources[0]
    if (!settings.application.showSources || !source || !canOpenSource(source)) return
    void sourcePreview.open(requestId, async () => ({
      document: await readSourceDocument(source), ...sourceNavigationList(source, sources), inline: true
    }))
  }

  function handleOpenSource(source: ChatSource) {
    sourcePreview.dismiss()
    setSourceTrayError(null)
    const sources = isPending ? pendingSources : selectedSourceMessage ? answerForDisplay(selectedSourceMessage).sources ?? [] : []
    void reader.open(source, sources)
  }

  function moveAutomaticSource(direction: number) {
    if (!automaticSource) return
    const source = automaticSource.sources[automaticSource.index + direction]
    if (!source) return
    // Choosing another source makes this a manual reading session. Keep the
    // existing page visible while loading, and leave it open when the answer arrives.
    sourcePreview.dismiss()
    void reader.open(source, automaticSource.sources, { inline: true, initial: automaticSource })
  }

  function selectExplanationView(answerId: string, view: AnswerView) {
    onChatStateChange(current => ({ ...current, conversations: current.conversations.map(conversation =>
      conversation.id === activeConversation.id ? { ...conversation, messages: conversation.messages.map(message =>
        message.id === answerId ? { ...message, explanationView: view } : message
      ) } : conversation
    ) }))
    setFocusedAnswerId(answerId)
  }

  async function improveAnswer(answerId: string, view: 'simple' | 'reasoning') {
    if (pendingConversationId || explanationInFlightRef.current || !selectedModel || selectedModel.status !== 'ready' || editingQuestionId || !window.tokensmith) return
    const buildRequest = view === 'reasoning' ? reasoningRetryRequest : simplerExplanationRequest
    const request = buildRequest(activeConversation.messages, answerId, {
      materials: activeMaterials, model: selectedModel, settings,
      modelSettings: modelSettingsFor(settings, selectedModel.id)
    })
    if (!request) return
    explanationInFlightRef.current = true
    const targetConversationId = activeConversation.id
    const responseStartedAt = performance.now()
    const progressId = beginForeground(view === 'reasoning' ? 'answer' : 'simplify',
      view === 'reasoning' ? 'Answering with reasoning' : 'Simplifying answer',
      request.prompt.length + (request.retrievedSources ?? []).reduce((n, source) => n + source.excerpt.length, 0), request.modelSettings)
    const requestSequence = ++requestSequenceRef.current
    setPendingConversationId(targetConversationId)
    setPendingExplanationId(answerId)
    setPendingSources(request.retrievedSources ?? [])
    setFocusedAnswerId(answerId)
    setExplanationError(null)
    setChatError(null)
    previewFirstSource(progressId, request.retrievedSources ?? [])
    try {
      let responseDurationMs: number | undefined
      const publish = (reply: EngineChatResponse) => {
        if (requestSequenceRef.current !== requestSequence || !reply.text.trim()) return
        responseDurationMs ??= Math.max(0, Math.round(performance.now() - responseStartedAt))
        sourcePreview.finish(progressId)
        const latencyTrace = combineLatencyTrace(progressId, undefined, reply.latencyTrace)
        const variant = { text: reply.text, sources: reply.sources, responseDurationMs,
          reasoning: reply.reasoning, followUpSuggestions: reply.followUpSuggestions, followUpError: reply.followUpError,
          ...(latencyTrace ? { latencyTrace } : {}) }
        onChatStateChange(current => ({ ...current, conversations: current.conversations.map(conversation =>
          conversation.id === targetConversationId ? { ...conversation, messages: conversation.messages.map(message =>
            message.id === answerId ? { ...message, explanationView: view,
              ...(view === 'reasoning' ? { reasoningAnswer: variant } : { simplerExplanation: variant }) } : message
          ) } : conversation
        ) }))
      }
      const reply = await window.tokensmith.sendChatMessage({ ...request, requestId: progressId }, (answer, hasFollowUps) => {
        if (requestSequenceRef.current !== requestSequence) return
        publish(answer)
        if (hasFollowUps) {
          setIsPreparingFollowUps(true)
          progress.next(progressId, 'Preparing follow-ups', { kind: 'follow-ups', inputChars: answer.text.length, contextual: false })
        } else progress.finish(progressId, 'Answer ready')
      })
      if (requestSequenceRef.current !== requestSequence) return
      if (!reply.text.trim()) throw new Error('The model returned an empty answer. Your original answer is still available.')
      publish(reply)
      progress.finish(progressId, view === 'reasoning' ? 'Answer ready' : 'Explanation ready')
    } catch (error) {
      progress.cancel(progressId)
      if (requestSequenceRef.current === requestSequence) {
        setExplanationError({ answerId, view, text: readableErrorMessage(error,
          view === 'reasoning' ? 'Could not answer with reasoning. Your original answer is still available.'
            : 'Could not simplify this answer. Your original answer is still available.') })
      }
    } finally {
      sourcePreview.finish(progressId)
      if (requestSequenceRef.current === requestSequence) {
        foregroundRequestRef.current = null
        explanationInFlightRef.current = false
        setIsPreparingFollowUps(false)
        setPendingConversationId(null)
        setPendingExplanationId(null)
        setPendingStatusText(null)
        setPendingSources([])
      }
    }
  }

  async function submitPrompt(rawPrompt: string, editMessageId?: string, passage?: ChatSelectedPassage) {
    const prompt = rawPrompt.trim()
    if (!prompt || catalog.status === 'loading' || (pendingConversationId && !isPreparingFollowUps) || !selectedModel || selectedModel.status !== 'ready' || (editingQuestionId && !editMessageId)) {
      return
    }

    if (isPreparingFollowUps) handleStopResponse()

    const editedMessages = editMessageId ? replaceQuestion(activeConversation.messages, editMessageId, prompt, passage) : undefined
    if (editMessageId && !editedMessages) return
    const history = editedMessages ? editedMessages.slice(0, -1) : activeConversation.messages
    const targetConversationId = activeConversation.id
    const responseStartedAt = performance.now()
    const progressId = beginForeground('answer', 'Preparing answer', prompt.length + history.reduce((n, m) => n + m.text.length, 0))
    const userMessage: ChatMessage = {
      id: editMessageId ?? createId('user'),
      role: 'user',
      text: prompt,
      ...(passage ? { selectedPassage: passage } : {})
    }

    onChatStateChange((current) => ({
      ...current,
      conversations: current.conversations.map((conversation) => {
        if (conversation.id !== targetConversationId) {
          return conversation
        }

        return {
          ...conversation,
          title: history.length === 0 ? getConversationTitle(prompt) : conversation.title,
          documentScope: scopeRefs(documentScope),
          messages: editedMessages ?? [...conversation.messages, userMessage],
        }
      })
    }))
    if (!editMessageId) clearDraft()
    setEditingQuestionId(null)
    setExpandedMessageId(null)
    setFocusedAnswerId(null)
    reader.close()
    setSourceTrayError(null)
    setChatError(null)
    setPendingConversationId(targetConversationId)
    setPendingSources([])
    const requestSequence = requestSequenceRef.current + 1
    requestSequenceRef.current = requestSequence
    const activeModelSettings = modelSettingsFor(settings, selectedModel.id)
    const searchEmbeddingModels = embeddingModels
    const retrievalSourceLimit = modelAwareRetrievalLimit(settings.maxSources, selectedModel, activeModelSettings)
    let retrievedSources: ChatSource[] | undefined = activeMaterials.length === 0 ? [] : undefined
    let conversationContextMode: ChatMessage['conversationContextMode'] = passage ? 'contextual' : 'standalone'
    let rewrittenRequest: EngineChatRequest | undefined
    let clarificationReply: EngineChatResponse | undefined
    let clientLatencySpans: LatencySpan[] = []

    try {
      if (window.tokensmith && (documentScope.length > 0 || activeConversation.practiceReference)) {
        const tokensmith = window.tokensmith
        const searchStartedAt = performance.now()
        setPendingStatusText('understanding question ...')
        const prepared = await prepareRewrittenStudyChat({
          requestId: progressId,
          prompt,
          selectedPassage: passage,
          messages: history,
          materials: activeMaterials,
          model: selectedModel,
          settings,
          applicationSettings: settings.application,
          modelSettings: activeModelSettings
        }, {
          resolve: (request) => tokensmith.resolveChatQuestion(request),
          search: (query) => {
            const reference = activeConversation.practiceReference
            if (history.length === 0 && reference && passage?.messageId === reference.questionId) {
              return Promise.resolve({ sources: sourcesInScope(reference.sources, documentScope) })
            }
            if (requestSequenceRef.current !== requestSequence) throw new Error('Response stopped.')
            setPendingStatusText('searching Library ...')
            return tokensmith.searchLibrary(query, activeMaterials, retrievalSourceLimit,
              searchEmbeddingModels, settings.application.searchMode, { embeddingGpuEnabled: settings.application.embeddingGpuEnabled, documents: documentScope })
          }
        })
        clientLatencySpans = prepared.latencySpans
        conversationContextMode = prepared.resolution.mode
        rewrittenRequest = prepared.request
        retrievedSources = prepared.request?.retrievedSources ?? []
        if (prepared.resolution.mode === 'clarify') {
          clarificationReply = {
            engineId: 'tokensmith', modelName: selectedModel.name,
            text: prepared.resolution.clarification, sources: [], followUpSuggestions: []
          }
        }

        if (requestSequenceRef.current !== requestSequence) {
          return
        }

        setPendingSources(settings.application.showSources ? retrievedSources : [])
        previewFirstSource(progressId, retrievedSources ?? [])
        progress.stage(progressId, 'Preparing answer', { inputChars: prompt.length + (retrievedSources ?? []).reduce((n, source) => n + source.excerpt.length, 0), contextual: conversationContextMode === 'contextual' })
        const searchDisplayMs = performance.now() - searchStartedAt
        if (searchDisplayMs < 450) {
          await new Promise((resolve) => window.setTimeout(resolve, 450 - searchDisplayMs))
        }

        if (requestSequenceRef.current !== requestSequence) {
          return
        }

        setPendingStatusText(null)
      }

      if (!window.tokensmith) {
        throw new Error('TokenSmith engine bridge is not available.')
      }

      progress.stage(progressId, 'Writing answer')
      const assistantId = createId('assistant')
      let answerDurationMs: number | undefined
      function publishAnswer(reply: EngineChatResponse) {
        if (requestSequenceRef.current !== requestSequence) return
        sourcePreview.finish(progressId)
        const latencyTrace = combineLatencyTrace(progressId, clientLatencySpans, reply.latencyTrace)
        const assistantMessage: ChatMessage = {
          id: assistantId,
          role: 'assistant',
          text: reply.text,
          // Kept whatever the display setting is, so a retelling can reuse this evidence.
          sources: reply.sources,
          reasoning: reply.reasoning,
          conversationContextMode,
          answerContext: {
            selectedPassage: passage,
            prompt, answerPrompt: rewrittenRequest?.answerPrompt,
            retrievalQuery: rewrittenRequest?.retrievalQuery,
            conversationContextMode: conversationContextMode === 'clarify' ? undefined : conversationContextMode,
            referenceExchange: rewrittenRequest?.referenceExchange,
            modelSettings: activeModelSettings,
            applicationSettings: settings.application
          },
          explanationDepth: settings.application.explanationDepthEnabled ? settings.application.explanationDepth : 'standard',
          responseDurationMs: answerDurationMs,
          followUpSuggestions: reply.followUpSuggestions ?? [],
          followUpError: reply.followUpError,
          ...(latencyTrace ? { latencyTrace } : {})
        }

        onChatStateChange((current) => ({
          ...current,
          conversations: current.conversations.map((conversation) => {
            if (conversation.id !== targetConversationId) {
              return conversation
            }

            return {
              ...conversation,
              messages: conversation.messages.some(m => m.id === assistantId)
                ? conversation.messages.map(m => m.id === assistantId ? assistantMessage : m)
                : [...conversation.messages, assistantMessage]
            }
          })
        }))
      }
      const reply = clarificationReply ?? await window.tokensmith.sendChatMessage({ ...(rewrittenRequest ?? {
        prompt,
        selectedPassage: passage,
        conversationContextMode: conversationContextMode === 'clarify' ? undefined : conversationContextMode,
        messages: history,
        materials: activeMaterials,
        model: selectedModel,
        settings,
        applicationSettings: settings.application,
        modelSettings: activeModelSettings,
        retrievedSources
      }), requestId: progressId }, (answer, hasFollowUps) => {
        if (requestSequenceRef.current !== requestSequence) return
        answerDurationMs = Math.max(0, Math.round(performance.now() - responseStartedAt))
        publishAnswer(answer)
        if (hasFollowUps) {
          setIsPreparingFollowUps(true)
          progress.next(progressId, 'Preparing follow-ups', { kind: 'follow-ups', inputChars: answer.text.length, contextual: false })
        } else {
          progress.finish(progressId, 'Answer ready')
        }
      })

      if (requestSequenceRef.current !== requestSequence) return
      answerDurationMs ??= Math.max(0, Math.round(performance.now() - responseStartedAt))
      publishAnswer(reply)
      progress.finish(progressId, 'Answer ready', !clarificationReply && !reply.followUpError)
    } catch (error) {
      progress.cancel(progressId)
      if (requestSequenceRef.current !== requestSequence) {
        return
      }

      setChatError(readableErrorMessage(error, 'Chat request failed.'))
    } finally {
      sourcePreview.finish(progressId)
      if (requestSequenceRef.current === requestSequence) {
        foregroundRequestRef.current = null
        setIsPreparingFollowUps(false)
        setPendingConversationId(null)
        setPendingExplanationId(null)
        explanationInFlightRef.current = false
        setPendingStatusText(null)
        setPendingSources([])
      }
    }
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    await submitPrompt(draft, undefined, selectedPassage)
  }

  return (
    <div hidden={!isActive} className={`view-frame study-frame chat-frame ${isSidebarOpen ? 'has-history' : ''} ${materialsOpen ? 'has-materials' : ''}`}>
      {isSidebarOpen && <StudySidebar mode="chat" activeId={activeConversation.id}
        groups={['Today', 'This week'].map(label => ({ label, items: conversations.filter(chat => chat.period === label) }))}
        onNew={handleNewChat} onSelect={handleSelectConversation} onRename={handleRenameConversation}
        onDelete={handleDeleteConversation} onClear={handleClearConversations} onViewLog={onViewLog} />}

      <section className="chat-main" aria-label="Study chat">
        <StudyHeader mode="chat" onModeChange={onModeChange} historyOpen={isSidebarOpen} onHistoryToggle={() => setIsSidebarOpen(!isSidebarOpen)}
          materialsOpen={materialsOpen} onMaterialsToggle={() => setMaterialsOpen(!materialsOpen)} count={documentScope.length}>
          <ChatModelPicker models={models} selectedModel={selectedModel} selectedModelId={activeConversation.modelId} disabled={isPending}
            onSelect={onSelectModel} onConnect={onConnectCloud} onManage={onManageModels} />
          {settings.application.explanationDepthEnabled && <ChatDepthPicker depth={settings.application.explanationDepth} disabled={isPending} onSelect={onExplanationDepthChange} />}
        </StudyHeader>

        <div className="chat-body">
          <ConversationViewport key={activeConversation.id} messages={activeConversation.messages} pending={isPending} isActive={isActive}
            onQuote={handleQuoteSelection} canQuote={!editingQuestionId && !pendingConversationId && Boolean(selectedModel)}>
            {activeConversation.messages.length === 0 && !isPending ? (
              <>
                {renderChatSetupCard()}
                {renderStarterQuestions()}
              </>
            ) : (
              activeConversation.messages.map((message) =>
                message.role === 'user' ? (
                  <UserMessage
                    key={message.id}
                    message={message}
                    editing={editingQuestionId === message.id}
                    editDisabled={Boolean(pendingConversationId) || !selectedModel}
                    laterQuestions={activeConversation.messages.slice(activeConversation.messages.indexOf(message) + 1).filter((item) => item.role === 'user').length}
                    onCopy={() => handleCopyQuestion(message.selectedPassage
                      ? `Selected passage:\n${message.selectedPassage.text}\n\n${message.text}` : message.text)}
                    onEdit={() => setEditingQuestionId(message.id)}
                    onCancelEdit={() => setEditingQuestionId(null)}
                    onSaveEdit={(text, passage) => void submitPrompt(text, message.id, passage)}
                  />
                ) : (
                  <AssistantMessage
                    expanded={expandedMessageId === message.id}
                    key={message.id}
                    message={message}
                    showSources={settings.application.showSources}
                    suggestionsDisabled={Boolean(pendingConversationId) || Boolean(editingQuestionId) || selectedModel?.status !== 'ready'}
                    onExplainSimpler={
                      canExplainSimpler(message) && message.explanationView !== 'reasoning'
                        ? () => void improveAnswer(message.id, 'simple')
                        : undefined
                    }
                    onPractice={message.text.trim() && message.conversationContextMode !== 'clarify' ? () => practiceAnswer(message) : undefined}
                    practiceDisabled={!practiceScopeReady}
                    onTryReasoning={canRetryWithReasoning(message, selectedModel)
                      ? () => void improveAnswer(message.id, 'reasoning') : undefined}
                    explanationPending={isPending && pendingExplanationId === message.id}
                    explanationError={explanationError?.answerId === message.id ? explanationError.text : undefined}
                    explanationErrorView={explanationError?.answerId === message.id ? explanationError.view : undefined}
                    onSelectExplanationView={(view) => selectExplanationView(message.id, view)}
                    onSelectFollowUp={handleUseFollowUpSuggestion}
                    onToggleSources={() => {
                      setFocusedAnswerId(message.id)
                      setExpandedMessageId((current) => (current === message.id ? null : message.id))
                    }}
                  />
                )
              )
            )}
            {activeConversation.practiceReference && activeConversation.messages.filter(message => message.role === 'user').length === 1 &&
              <PracticeChatSuggestions key={activeConversation.id} conversation={activeConversation} model={selectedModel}
                settings={settings} modelSettings={selectedModelSettings} isActive={isActive}
                disabled={Boolean(pendingConversationId) || Boolean(editingQuestionId)} progress={progress}
                onUse={handleUseFollowUpSuggestion} onSave={questions => onChatStateChange(current => ({ ...current,
                  conversations: current.conversations.map(conversation => conversation.id === activeConversation.id
                    ? { ...conversation, practiceSuggestions: questions } : conversation) }))} />}
            {chatError && <div className="chat-error-banner"><p>{chatError}</p>
              {selectedModel && isCloudGenerator(selectedModel) && <button className="secondary-action" type="button" onClick={() => onConnectCloud(selectedModel)}>Check cloud connection</button>}
            </div>}
            {selectedModel && isCloudGenerator(selectedModel) && selectedModel.status !== 'ready' &&
              <div className="chat-error-banner"><p>Enter an API key for {selectedModel.providerName || 'your online service'} to continue with this model.</p><button className="secondary-action" type="button" onClick={() => onConnectCloud(selectedModel)}>Enter API key</button></div>}
            {activeConversation.modelId && (!selectedModel || (selectedModel.engine === 'ollama' && selectedModel.status !== 'ready')) &&
              <div className="chat-error-banner" role="status"><p>This chat's model is unavailable. Choose another model or restore it in Models.</p>
                <button className="secondary-action" type="button" onClick={onManageModels}>Manage models</button></div>}
          </ConversationViewport>

          <form className="composer-area" aria-label="Message composer" onSubmit={handleSubmit}>
            {selectedModel && !isPending && !isCloudGenerator(selectedModel) && selectedModel.status !== 'ready' && (
              <span className="reload-chip"><RefreshCw size={15} aria-hidden="true" />Load - {selectedModelLabel}</span>
            )}
            <MessageComposer value={draft} onChange={text => updateDraft({ text })} inputRef={composerInputRef}
              label="Message" placeholder={selectedPassage ? 'Ask about this...' : composerPlaceholder} submitLabel="Send message" isActive={isActive}
              disabled={Boolean(editingQuestionId) || (Boolean(pendingConversationId) && !isPreparingFollowUps) || !selectedModel}
              submitDisabled={!draft.trim() || catalog.status === 'loading' || Boolean(editingQuestionId) || (Boolean(pendingConversationId) && !isPreparingFollowUps) || !selectedModel || selectedModel.status !== 'ready'}>
              {selectedPassage && <SelectedPassagePreview passage={selectedPassage}
                disabled={Boolean(editingQuestionId) || Boolean(pendingConversationId)}
                onRemove={() => { updateDraft({ selectedPassage: undefined }); composerInputRef.current?.focus() }} />}
            </MessageComposer>
          </form>
        </div>
      </section>

      <StudyMaterialsPane catalog={catalog} scope={documentScope} open={materialsOpen}
        collectionExpansion={collectionExpansion} onCollectionExpansionChange={onCollectionExpansionChange}
        disabled={Boolean(editingQuestionId)} onChange={changeDocumentScope}
        onClose={() => setMaterialsOpen(false)} onManage={onOpenLibrary}>
        {selectedSources.length > 0 && <SourceTray sources={selectedSources} error={reader.error || sourceTrayError} onOpenSource={handleOpenSource} />}
      </StudyMaterialsPane>
      {isActive && automaticSource && <SourceDocumentReader state={automaticSource} onClose={sourcePreview.dismiss}
        navigation={{ index: automaticSource.index, count: automaticSource.sources.length,
          onPrevious: () => moveAutomaticSource(-1), onNext: () => moveAutomaticSource(1) }} />}
      {isActive && reader.state && <SourceDocumentReader state={reader.state} navigation={reader.navigation} onClose={reader.close} />}
    </div>
  )
}

function TokenSmithLogCard({
  entries,
  error,
  logFile,
  onClose,
  onRefresh,
  status
}: {
  entries: ParsedLogEntry[]
  error: string | null
  logFile: TokenSmithLogFile | null
  onClose: () => void
  onRefresh: () => void
  status: 'idle' | 'loading' | 'error'
}) {
  return (
    <section className="log-card" aria-label="TokenSmith log">
      <header className="log-card-header">
        <div>
          <p>TokenSmith Log</p>
          <h2>Log</h2>
          <small>{logFile?.path ?? 'Loading log path...'}</small>
        </div>
        <div className="log-card-actions">
          <button
            className="icon-button subtle"
            type="button"
            aria-label="Refresh log"
            title="Refresh log"
            onClick={onRefresh}
            disabled={status === 'loading'}
          >
            <RefreshCw className={status === 'loading' ? 'spin' : undefined} size={17} aria-hidden="true" />
          </button>
          <button className="icon-button subtle" type="button" aria-label="Close log" title="Close log" onClick={onClose}>
            <X size={18} aria-hidden="true" />
          </button>
        </div>
      </header>

      {logFile && (
        <div className="log-card-meta">
          <span>{formatBytes(logFile.sizeBytes)}</span>
          {logFile.truncated && <span>Showing latest {formatBytes(logFile.maxBytes)}</span>}
        </div>
      )}

      {error && <p className="log-card-error">{error}</p>}
      {status === 'loading' && entries.length === 0 ? (
        <p className="log-card-empty">Loading log...</p>
      ) : entries.length === 0 ? (
        <p className="log-card-empty">No log entries yet.</p>
      ) : (
        <div className="log-entry-list">
          {entries.slice().reverse().map((entry) => (
            <article className="log-entry" key={entry.id}>
              <div className="log-entry-heading">
                <strong>{prettyLogEventName(entry.event)}</strong>
                {entry.time && <time>{entry.time}</time>}
              </div>
              <LogEntryBody entry={entry} />
            </article>
          ))}
        </div>
      )}
    </section>
  )
}

function LogEntryBody({ entry }: { entry: ParsedLogEntry }) {
  if (!entry.data) {
    return <pre className="log-raw-text">{entry.raw ?? ''}</pre>
  }

  const modelLabel = logModelLabel(entry.data.model)
  const prompt = logString(entry.data.prompt)
  const query = logString(entry.data.query) ?? logString(entry.data.retrievalQuery)
  const answerPrompt = logString(entry.data.answerPrompt)
  const conversationContextMode = logString(entry.data.conversationContextMode)
  const answer = logString(entry.data.text)
  const systemPrompt = logString(entry.data.systemPrompt)
  const sources = logRecords(entry.data.sources)
  const modelMessages = logRecords(entry.data.modelMessages)
  const materials = logRecords(entry.data.materials)
  const embeddingModels = logRecords(entry.data.embeddingModels)
  const sourceCount = typeof entry.data.sourceCount === 'number' ? entry.data.sourceCount : sources.length

  return (
    <div className="log-entry-body">
      <div className="log-entry-tags">
        {modelLabel && <span>{modelLabel}</span>}
        {conversationContextMode && <span>{conversationContextMode}</span>}
        {sourceCount > 0 && <span>{sourceCount} {sourceCount === 1 ? 'source' : 'sources'}</span>}
        {typeof entry.data.limit === 'number' && <span>limit {entry.data.limit}</span>}
      </div>

      {prompt && <LogTextPanel title="Current Question" text={prompt} tone="question" />}
      {query && <LogTextPanel title="Search Query" text={query} tone="question" />}
      {answerPrompt && answerPrompt !== prompt && <LogTextPanel title="Resolved Question" text={answerPrompt} tone="question" />}
      {answer && <LogTextPanel title="Answer" text={answer} />}
      {systemPrompt && <LogTextPanel title="System Prompt" text={systemPrompt} tone="system" />}
      {modelMessages.length > 0 && <LogMessages messages={modelMessages} />}
      {sources.length > 0 && <LogSources sources={sources} />}
      {materials.length > 0 && <LogObjectList title="Library folders" items={materials} />}
      {embeddingModels.length > 0 && <LogObjectList title="Embedding Models" items={embeddingModels} />}

      <details className="log-raw-details">
        <summary>Raw JSON</summary>
        <pre>{logJson(entry.data)}</pre>
      </details>
    </div>
  )
}

function LogTextPanel({
  text,
  title,
  tone
}: {
  text: string
  title: string
  tone?: 'question' | 'system'
}) {
  return (
    <section className={`log-panel ${tone ?? ''}`}>
      <h3>{title}</h3>
      <pre>{text}</pre>
    </section>
  )
}

function LogMessages({ messages }: { messages: Record<string, unknown>[] }) {
  let finalUserIndex = -1
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    if (logString(messages[index].role) === 'user') {
      finalUserIndex = index
      break
    }
  }

  return (
    <section className="log-panel">
      <h3>Ordered Messages Sent To Model</h3>
      <p className="log-panel-note">
        This is the exact message order. The last user message contains retrieved source context plus the question sent to the model.
      </p>
      <div className="log-message-list">
        {messages.map((message, index) => {
          const role = logString(message.role) ?? 'message'
          const content = logString(message.content) ?? logJson(message)
          const isFinalUserMessage = index === finalUserIndex

          return (
            <article className={`log-message ${isFinalUserMessage ? 'is-final' : ''}`} key={`${role}-${index}`}>
              <div className="log-message-heading">
                <span className="log-message-role">{role}</span>
                <span className="log-message-index">
                  Message {index + 1} of {messages.length}{isFinalUserMessage ? ' · final user message' : ''}
                </span>
              </div>
              <pre>{content}</pre>
            </article>
          )
        })}
      </div>
    </section>
  )
}

function LogSources({ sources }: { sources: Record<string, unknown>[] }) {
  return (
    <section className="log-panel">
      <h3>Sources</h3>
      <div className="log-source-list">
        {sources.map((source, index) => {
          const title = logSourceTitle(source, index)
          const locator = logSourceLocator(source)
          const score = logSourceScore(source)
          const path = logString(source.path)
          const excerpt = logString(source.excerpt)

          return (
            <details className="log-source" key={`${title}-${index}`} open={index === 0}>
              <summary>
                <strong>{title}</strong>
                <span>{[locator, score ? `score ${score}` : undefined].filter(Boolean).join(' · ')}</span>
              </summary>
              {path && <p>{path}</p>}
              {excerpt && <pre>{excerpt}</pre>}
            </details>
          )
        })}
      </div>
    </section>
  )
}

function LogObjectList({ items, title }: { items: Record<string, unknown>[]; title: string }) {
  return (
    <section className="log-panel">
      <h3>{title}</h3>
      <div className="log-object-list">
        {items.map((item, index) => (
          <pre key={`${title}-${index}`}>{logJson(item)}</pre>
        ))}
      </div>
    </section>
  )
}

function contextModeLabel(mode?: ChatMessage['conversationContextMode']): string | undefined {
  if (mode === 'clarify') {
    return 'Clarification'
  }
  if (mode === 'contextual') {
    return 'Contextual'
  }
  if (mode === 'standalone') {
    return 'Standalone'
  }
  return undefined
}

function responseDurationLabel(durationMs?: number): string | undefined {
  if (!Number.isFinite(durationMs) || durationMs === undefined || durationMs < 0) {
    return undefined
  }

  const seconds = durationMs / 1000
  return `${seconds < 10 ? seconds.toFixed(1) : Math.round(seconds).toString()}s`
}

function UserMessage({ message, editing, editDisabled, laterQuestions, onCopy, onEdit, onCancelEdit, onSaveEdit }: {
  message: ChatMessage
  editing: boolean
  editDisabled: boolean
  laterQuestions: number
  onCopy: () => void
  onEdit: () => void
  onCancelEdit: () => void
  onSaveEdit: (text: string, selectedPassage?: ChatSelectedPassage) => void
}) {

  return (
    <article className="message-row" data-question-id={message.id} tabIndex={-1} aria-label="Your question">
      <MessageAvatar role="user" />
      <div>
        <h3>
          You
        </h3>
        {editing ? <QuestionEditor key={message.id} text={message.text} selectedPassage={message.selectedPassage} laterQuestions={laterQuestions} disabled={editDisabled} onCancel={onCancelEdit} onSave={onSaveEdit} /> : (
          <>
            {message.selectedPassage && <SelectedPassagePreview passage={message.selectedPassage} />}
            <div data-chat-selectable data-chat-message-id={message.id}><MessageText text={message.text} /></div>
          </>
        )}
        {!editing && <div className="message-actions" aria-label="Question actions">
          <button className="message-action" type="button" aria-label="Edit question" title="Edit question" disabled={editDisabled} onClick={onEdit}>
            <Pencil size={18} aria-hidden="true" />
          </button>
          <button className="message-action" type="button" aria-label="Copy question" title="Copy question" onClick={onCopy}>
            <Copy size={18} aria-hidden="true" />
          </button>
        </div>}
      </div>
    </article>
  )
}

function AssistantMessage({
  expanded,
  message,
  showSources,
  suggestionsDisabled,
  onExplainSimpler,
  onPractice,
  practiceDisabled,
  onTryReasoning,
  explanationPending,
  explanationError,
  explanationErrorView,
  onSelectExplanationView,
  onSelectFollowUp,
  onToggleSources
}: {
  expanded: boolean
  message: ChatMessage
  showSources: boolean
  suggestionsDisabled: boolean
  onExplainSimpler?: () => void
  onPractice?: () => void
  practiceDisabled: boolean
  onTryReasoning?: () => void
  explanationPending: boolean
  explanationError?: string
  explanationErrorView?: AnswerView
  onSelectExplanationView: (view: AnswerView) => void
  onSelectFollowUp: (suggestion: string) => void
  onToggleSources: () => void
}) {
  const displayed = answerForDisplay(message)
  const sources = showSources ? displayed.sources ?? [] : []
  const suggestionOwner = message.explanationView === 'reasoning' && message.reasoningAnswer ? message.reasoningAnswer : message
  const suggestions = suggestionOwner.followUpSuggestions ?? []
  const modeLabel = contextModeLabel(message.conversationContextMode)
  const durationLabel = responseDurationLabel(displayed.responseDurationMs)

  return (
    <article className="message-row">
      <MessageAvatar role="assistant" />
      <div>
        <h3>
          TokenSmith
        </h3>
        {(modeLabel || durationLabel || displayed.reasoning?.used) && (
          <div className="message-meta-labels" aria-label="Response details">
            {modeLabel && <span>{modeLabel}</span>}
            {durationLabel && <span>{durationLabel}</span>}
            {displayed.reasoning?.used && <span>Reasoning used</span>}
          </div>
        )}
        <AnswerExplanation message={message} pending={explanationPending} error={explanationError}
          errorView={explanationErrorView} disabled={suggestionsDisabled} onSimplify={onExplainSimpler}
          onReasoning={onTryReasoning} onPractice={onPractice} practiceDisabled={practiceDisabled} onSelectView={onSelectExplanationView} />
        {suggestions.length > 0 && (
          <section className="follow-up-section" aria-label="Suggested follow-up questions">
            <div className="follow-up-title">
              <Library size={18} aria-hidden="true" />
              <span>Suggested follow-ups</span>
            </div>
            <div className="follow-up-list">
              {suggestions.map((suggestion) => (
                <button
                  className="follow-up-row"
                  key={suggestion}
                  type="button"
                  disabled={suggestionsDisabled}
                  onClick={() => onSelectFollowUp(suggestion)}
                >
                  <MessageText text={suggestion} inline />
                  <Plus size={18} aria-hidden="true" />
                </button>
              ))}
            </div>
          </section>
        )}
        {suggestionOwner.followUpError && <p className="follow-up-error">{suggestionOwner.followUpError}</p>}
        {sources.length > 0 && (
          <>
            <button
              className="source-chip"
              type="button"
              aria-expanded={expanded}
              onClick={onToggleSources}
            >
              <Database size={16} aria-hidden="true" />
              <span>
                {sources.length} {sources.length === 1 ? 'Source' : 'Sources'}
              </span>
              <ChevronDown size={14} aria-hidden="true" />
            </button>
            {expanded && (
              <div className="source-list">
                {sources.map((source) => (
                  <div className="source-card" key={`${source.title}-${source.locator}`}>
                    <strong>{source.title}</strong>
                    <span>{source.locator}</span>
                    {isMarkdownSource(source) ? (
                      <pre>{sourceDisplayText(source)}</pre>
                    ) : (
                      <p>{source.excerpt}</p>
                    )}
                  </div>
                ))}
              </div>
            )}
          </>
        )}
      </div>
    </article>
  )
}


function PdfSourceViewer({ viewer, onClose, automatic = false, navigation }: { viewer: PdfViewerState; onClose: () => void; automatic?: boolean; navigation?: SourceNavigationProps }) {
  const [pdfDocument, setPdfDocument] = useState<PDFDocumentProxy | null>(null)
  const [currentPage, setCurrentPage] = useState(viewer.page ?? 1)
  const [renderedPage, setRenderedPage] = useState<PdfRenderedPage | null>(null)
  const [searchQuery, setSearchQuery] = useState(viewer.searchTerm ?? '')
  const [searchMatches, setSearchMatches] = useState<PdfSearchMatch[]>([])
  const [activeSearchIndex, setActiveSearchIndex] = useState(0)
  const [isSearching, setIsSearching] = useState(false)
  const [hasSearched, setHasSearched] = useState(false)
  const [loadError, setLoadError] = useState('')

  useEffect(() => {
    let isCancelled = false
    const loadingTask = pdfjsLib.getDocument({ data: dataUrlToPdfBytes(viewer.dataUrl) })

    setPdfDocument(null)
    setRenderedPage(null)
    setSearchMatches([])
    setActiveSearchIndex(0)
    setLoadError('')

    void loadingTask.promise
      .then((document) => {
        if (isCancelled) {
          void document.destroy()
          return
        }
        setPdfDocument(document)
        setCurrentPage(clampPage(viewer.page ?? 1, document.numPages))
      })
      .catch((error: unknown) => {
        if (!isCancelled) {
          setLoadError(error instanceof Error ? error.message : 'The PDF could not be opened.')
        }
      })

    return () => {
      isCancelled = true
      void loadingTask.destroy()
    }
  }, [viewer.dataUrl, viewer.page])

  useEffect(() => {
    setSearchQuery(viewer.searchTerm ?? '')
    setSearchMatches([])
    setActiveSearchIndex(0)
    setHasSearched(false)
  }, [viewer.dataUrl, viewer.searchTerm])

  useEffect(() => {
    return () => {
      void pdfDocument?.destroy()
    }
  }, [pdfDocument])

  useEffect(() => {
    if (!pdfDocument) {
      return undefined
    }

    let isCancelled = false
    setRenderedPage(null)

    void renderPdfPage(pdfDocument, currentPage)
      .then((page) => {
        if (!isCancelled) {
          setRenderedPage(page)
        }
      })
      .catch((error: unknown) => {
        if (!isCancelled) {
          setLoadError(error instanceof Error ? error.message : 'The PDF page could not be rendered.')
        }
      })

    return () => {
      isCancelled = true
    }
  }, [pdfDocument, currentPage])

  async function runSearch() {
    const query = searchQuery.trim()
    if (!query || !pdfDocument) {
      setSearchMatches([])
      setActiveSearchIndex(0)
      setHasSearched(false)
      return
    }

    setIsSearching(true)
    try {
      const matches = await collectPdfSearchMatches(pdfDocument, query)
      setSearchMatches(matches)
      setActiveSearchIndex(0)
      if (matches[0]) {
        setCurrentPage(matches[0].pageNumber)
      }
    } finally {
      setHasSearched(true)
      setIsSearching(false)
    }
  }

  const searchStatus =
    searchMatches.length === 0
      ? ''
      : `${activeSearchIndex + 1} of ${searchMatches.length}`

  const activeMatch = searchMatches[activeSearchIndex]

  function goToSearchMatch(direction: 1 | -1) {
    if (searchMatches.length === 0) {
      return
    }

    const nextIndex = (activeSearchIndex + direction + searchMatches.length) % searchMatches.length
    setActiveSearchIndex(nextIndex)
    setCurrentPage(searchMatches[nextIndex].pageNumber)
  }

  function goToPage(direction: 1 | -1) {
    if (!pdfDocument) {
      return
    }
    setCurrentPage((page) => clampPage(page + direction, pdfDocument.numPages))
  }

  useEffect(() => {
    if (!pdfDocument) {
      return undefined
    }

    const pageCount = pdfDocument.numPages

    function handleKeyDown(event: KeyboardEvent) {
      const target = event.target
      if (target instanceof HTMLElement) {
        const tagName = target.tagName.toLowerCase()
        if (target.isContentEditable || tagName === 'input' || tagName === 'textarea' || tagName === 'select') {
          return
        }
      }

      if (event.key === 'ArrowLeft') {
        event.preventDefault()
        setCurrentPage((page) => clampPage(page - 1, pageCount))
      }

      if (event.key === 'ArrowRight') {
        event.preventDefault()
        setCurrentPage((page) => clampPage(page + 1, pageCount))
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [pdfDocument])

  return (
    <div className="pdf-viewer-backdrop" role="dialog" aria-modal={!automatic} aria-label="Source PDF viewer" onKeyDown={event => { if (event.key === 'Escape') { event.preventDefault(); onClose() } }}>
      <section className="pdf-viewer-panel">
        <header className="pdf-viewer-header">
          <div>
            <strong>{cleanMaterialTitle(viewer.title) || viewer.title}</strong>
            <span>Page {currentPage}{pdfDocument ? ` of ${pdfDocument.numPages}` : ''}</span>
          </div>
          <SourceNavigation navigation={navigation} />
          <button className="icon-button subtle" type="button" aria-label="Close PDF viewer" autoFocus={automatic} onClick={onClose}>
            <X size={18} aria-hidden="true" />
          </button>
        </header>
        <form
          className="pdf-viewer-search"
          onSubmit={(event) => {
            event.preventDefault()
            runSearch()
          }}
        >
          <Search size={16} aria-hidden="true" />
          <input
            aria-label="Search this PDF"
            placeholder="Search this PDF"
            value={searchQuery}
            onChange={(event) => {
              setSearchQuery(event.target.value)
              setSearchMatches([])
              setActiveSearchIndex(0)
              setHasSearched(false)
            }}
          />
          <span className="pdf-viewer-search-status" aria-live="polite">
            {isSearching ? 'Searching...' : hasSearched && searchMatches.length === 0 ? 'No matches' : searchStatus}
          </span>
          <button className="pdf-viewer-search-button" type="submit" disabled={!searchQuery.trim() || isSearching || !pdfDocument}>
            Find
          </button>
          <button
            className="pdf-viewer-search-button"
            type="button"
            disabled={isSearching || searchMatches.length === 0}
            onClick={() => goToSearchMatch(-1)}
          >
            Previous
          </button>
          <button
            className="pdf-viewer-search-button"
            type="button"
            disabled={isSearching || searchMatches.length === 0}
            onClick={() => goToSearchMatch(1)}
          >
            Next
          </button>
          <button className="pdf-viewer-search-button" type="button" disabled={!pdfDocument || currentPage <= 1} onClick={() => goToPage(-1)}>
            Prev page
          </button>
          <button className="pdf-viewer-search-button" type="button" disabled={!pdfDocument || currentPage >= pdfDocument.numPages} onClick={() => goToPage(1)}>
            Next page
          </button>
        </form>
        <div className="pdf-js-stage">
          {loadError && <p className="pdf-viewer-error">{loadError}</p>}
          {!loadError && !renderedPage && <p className="pdf-viewer-loading">Loading PDF...</p>}
          {renderedPage && (
            <div
              className="pdf-js-page"
              style={{ width: renderedPage.width, height: renderedPage.height }}
            >
              <img src={renderedPage.imageDataUrl} alt={`${viewer.title} page ${renderedPage.pageNumber}`} />
              {renderedPage.textItems.map((item) => {
                const isMatch = searchMatches.some(
                  (match) => match.pageNumber === renderedPage.pageNumber && match.itemIndex === item.index
                )
                if (!isMatch) {
                  return null
                }
                const isActive =
                  activeMatch?.pageNumber === renderedPage.pageNumber && activeMatch.itemIndex === item.index

                return (
                  <span
                    key={item.index}
                    className={`pdf-page-highlight${isActive ? ' is-active' : ''}`}
                    style={{
                      left: item.left,
                      top: item.top,
                      width: item.width,
                      height: item.height
                    }}
                    title={item.text}
                  />
                )
              })}
            </div>
          )}
        </div>
      </section>
    </div>
  )
}

function dataUrlToPdfBytes(dataUrl: string) {
  const [, payload = ''] = dataUrl.split(',', 2)
  if (dataUrl.slice(0, dataUrl.indexOf(',')).includes(';base64')) {
    const binary = window.atob(payload)
    const bytes = new Uint8Array(binary.length)
    for (let index = 0; index < binary.length; index += 1) {
      bytes[index] = binary.charCodeAt(index)
    }
    return bytes
  }
  return new TextEncoder().encode(decodeURIComponent(payload))
}

async function renderPdfPage(pdfDocument: PDFDocumentProxy, pageNumber: number): Promise<PdfRenderedPage> {
  const page = await pdfDocument.getPage(pageNumber)
  const viewport = page.getViewport({ scale: pdfViewerRenderScale })
  const canvas = document.createElement('canvas')
  canvas.width = Math.ceil(viewport.width)
  canvas.height = Math.ceil(viewport.height)

  const canvasContext = canvas.getContext('2d')
  if (!canvasContext) {
    throw new Error('Could not create a PDF canvas.')
  }

  await page.render({ canvas, canvasContext, viewport }).promise
  const textContent = await page.getTextContent()
  const textItems = textContent.items
    .filter(isPdfTextItem)
    .map((item, index) => pdfTextItemBox(item, viewport.transform as number[], viewport.scale, index))
    .filter((item) => item.text.trim())

  return {
    pageNumber,
    width: viewport.width,
    height: viewport.height,
    imageDataUrl: canvas.toDataURL('image/png'),
    textItems
  }
}

async function collectPdfSearchMatches(pdfDocument: PDFDocumentProxy, query: string): Promise<PdfSearchMatch[]> {
  const normalizedQuery = normalizePdfSearchText(query)
  if (!normalizedQuery) {
    return []
  }

  const matches: PdfSearchMatch[] = []
  for (let pageNumber = 1; pageNumber <= pdfDocument.numPages; pageNumber += 1) {
    const page = await pdfDocument.getPage(pageNumber)
    const textContent = await page.getTextContent()
    matchedPdfTextItemIndexes(textContent.items.filter(isPdfTextItem), normalizedQuery).forEach((itemIndex) => {
      matches.push({ pageNumber, itemIndex })
    })
  }
  return matches
}

function matchedPdfTextItemIndexes(items: TextItem[], normalizedQuery: string) {
  const directMatches = items
    .map((item, itemIndex) => ({ itemIndex, text: normalizePdfSearchText(item.str) }))
    .filter((item) => item.text.includes(normalizedQuery))
    .map((item) => item.itemIndex)

  if (directMatches.length > 0) {
    return directMatches
  }

  const ranges: Array<{ itemIndex: number; start: number; end: number }> = []
  let pageText = ''

  items.forEach((item, itemIndex) => {
    const text = normalizePdfSearchText(item.str)
    if (!text) {
      return
    }

    if (pageText) {
      pageText += ' '
    }

    const start = pageText.length
    pageText += text
    ranges.push({ itemIndex, start, end: pageText.length })
  })

  const matchedIndexes = new Set<number>()
  let searchFrom = 0
  let matchStart = pageText.indexOf(normalizedQuery, searchFrom)

  while (matchStart >= 0) {
    const matchEnd = matchStart + normalizedQuery.length
    ranges.forEach((range) => {
      if (range.start < matchEnd && range.end > matchStart) {
        matchedIndexes.add(range.itemIndex)
      }
    })
    searchFrom = matchEnd
    matchStart = pageText.indexOf(normalizedQuery, searchFrom)
  }

  return [...matchedIndexes]
}

function isPdfTextItem(item: unknown): item is TextItem {
  return typeof item === 'object' && item !== null && 'str' in item && typeof (item as TextItem).str === 'string'
}

function pdfTextItemBox(item: TextItem, viewportTransform: number[], viewportScale: number, index: number): PdfRenderedTextItem {
  const transform = multiplyPdfMatrices(viewportTransform, item.transform as number[])
  const fontHeight = Math.hypot(transform[2], transform[3]) || item.height * viewportScale || 10

  return {
    index,
    text: item.str,
    left: transform[4],
    top: transform[5] - fontHeight,
    width: Math.max(item.width * viewportScale, fontHeight * 0.45),
    height: Math.max(fontHeight, 6)
  }
}

function multiplyPdfMatrices(first: number[], second: number[]) {
  return [
    first[0] * second[0] + first[2] * second[1],
    first[1] * second[0] + first[3] * second[1],
    first[0] * second[2] + first[2] * second[3],
    first[1] * second[2] + first[3] * second[3],
    first[0] * second[4] + first[2] * second[5] + first[4],
    first[1] * second[4] + first[3] * second[5] + first[5]
  ]
}

function normalizePdfSearchText(text: string) {
  return text.toLocaleLowerCase().replace(/\s+/g, ' ').trim()
}

function clampPage(page: number, pageCount: number) {
  return Math.max(1, Math.min(page, pageCount))
}

function createIndexingCollection(
  title: string,
  collectionPath: string,
  embeddingModel?: LocalModel,
  cleaningProfileId: CleaningProfileId = defaultCleaningProfileId,
  cleaningRuleIds: CleaningRuleId[] = defaultCleaningRuleIdsForProfile(cleaningProfileId)
): CourseMaterial {
  const materialId = createId('material')

  return {
    id: materialId,
    title,
    detail: 'Parsing',
    status: 'indexing',
    kind: 'folder',
    path: collectionPath,
    addedAt: new Date().toISOString(),
    fileCount: 0,
    wordCount: 0,
    chunkCount: 0,
    chunkSize: defaultCollectionChunkSize,
    embeddingModelId: embeddingModel?.id,
    embeddingModelName: embeddingModel ? displayModelName(embeddingModel) : undefined,
    cleaningProfileId,
    cleaningProfileName: cleaningProfileLabel(cleaningProfileId),
    cleaningRuleIds,
    isActive: false,
    indexing: {
      materialId,
      phase: 'parsing',
      percent: 1,
      processedFiles: 0,
      totalFiles: 0,
      processedEmbeddings: 0,
      totalEmbeddings: 0,
      message: 'Parsing'
    }
  }
}

function LibraryScreen(props: Omit<React.ComponentProps<typeof LibraryWorkspace>, 'onOpenSource'>) {
  const reader = useSourceReader(readSourceDocument)
  return <>
    <LibraryWorkspace {...props} onOpenSource={(source, sources) => { void reader.open(source, sources) }} />
    {reader.error && !reader.state && <p className="inline-error" role="alert">{reader.error}</p>}
    {reader.state && <SourceDocumentReader state={reader.state} navigation={reader.navigation} onClose={reader.close} />}
  </>
}

function CollectionCard({
  item,
  onRemove,
  onPause,
  onResume,
  onToggle
}: {
  item: CourseMaterial
  onRemove: () => void
  onPause: () => void
  onResume: () => void
  onToggle: () => void
}) {
  const progress = item.indexing
  const isIndexing = item.status === 'indexing'
  const isPaused = item.status === 'paused'
  const percent = item.status === 'ready' ? 100 : Math.max(0, Math.min(100, Math.round(progress?.percent ?? 0)))
  const filesLabel =
    item.fileCount && item.fileCount > 0
      ? `${item.fileCount} ${item.fileCount === 1 ? 'file' : 'files'}`
      : progress?.totalFiles
        ? `${progress.processedFiles ?? 0} of ${progress.totalFiles} files`
        : 'Preparing files'
  const wordsLabel = item.wordCount ? `${formatNumber(item.wordCount)} words` : undefined
  const chunkSizeLabel = item.chunkSize ? `${formatNumber(item.chunkSize)} chars/chunk` : undefined
  const modelLabel = item.embeddingModelName ?? item.embeddingModel
  const cleaningLabel = item.cleaningProfileName ? `${item.cleaningProfileName} cleaning` : undefined
  const embeddingLabel =
    item.status === 'indexing' && (progress?.phase === 'embedding' || (progress?.totalEmbeddings ?? 0) > 0)
      ? `${progress?.processedEmbeddings ?? 0} of ${progress?.totalEmbeddings ?? 0} embeddings`
      : item.chunkCount
        ? `${formatNumber(item.chunkCount)} searchable chunks`
        : undefined
  const statusLabel =
    item.status === 'ready'
      ? 'READY'
      : item.status === 'needsReview'
        ? 'REVIEW'
        : item.status === 'paused'
          ? 'PAUSED'
          : progress?.phase === 'parsing'
            ? 'PARSING'
            : progress?.phase === 'chunking'
              ? 'CHUNKING'
              : progress?.phase === 'embedding'
                ? 'EMBEDDING'
                : 'UPDATING'
  const progressCopy =
    item.status === 'ready'
      ? 'Ready for chat'
      : item.status === 'needsReview'
        ? item.error ?? 'Needs review'
        : item.status === 'paused'
          ? `Paused at ${percent}%`
          : `${progress?.message ?? 'Parsing'} ${percent}%`

  return (
    <article className={`collection-card ${item.status}`}>
      <div className="collection-card-main">
        <h2>{item.title}</h2>
        {item.path && <p className="collection-path">{item.path}</p>}
        <p className="collection-meta">
          {[filesLabel, wordsLabel, embeddingLabel, chunkSizeLabel, cleaningLabel, modelLabel].filter(Boolean).join(' - ')}
        </p>
      </div>

      <div className="collection-card-status">
        <span className={`collection-badge ${item.status} ${progress?.phase ?? ''}`}>{statusLabel}</span>
        <span>{progressCopy}</span>
      </div>

      {(item.status === 'indexing' || item.status === 'paused') && (
        <div className="collection-progress" style={{ '--progress': `${percent}%` } as CSSProperties}>
          <span />
        </div>
      )}

      <div className="collection-card-actions">
        <div className="collection-primary-actions">
          {isIndexing ? (
            <button className="collection-action-button collection-pause" type="button" onClick={onPause}>
              <Pause size={14} aria-hidden="true" />
              <span>Pause</span>
            </button>
          ) : (
            <button
              className="collection-action-button collection-resume"
              type="button"
              onClick={onResume}
              disabled={!item.path}
            >
              <RefreshCw size={15} aria-hidden="true" />
              <span>{isPaused || item.status === 'needsReview' ? 'Resume' : 'Update'}</span>
            </button>
          )}
          <button
            className="collection-action-button collection-remove"
            type="button"
            onClick={onRemove}
          >
            <Trash2 size={15} aria-hidden="true" />
            <span>Remove</span>
          </button>
        </div>
        <button
          className="collection-enable"
          type="button"
          onClick={onToggle}
          disabled={item.status !== 'ready'}
          aria-pressed={item.isActive !== false}
        >
          {item.isActive === false ? 'Disabled' : 'Enabled'}
        </button>
      </div>
    </article>
  )
}

function MaterialRow({
  compact = false,
  item,
  onClick,
  onRemove
}: {
  compact?: boolean
  item: CourseMaterial
  onClick?: () => void
  onRemove?: () => void
}) {
  const isActive = (item.status === 'ready' || Boolean(item.indexedAt)) && item.isActive !== false
  const indexingPercent = Math.max(0, Math.min(100, Math.round(item.indexing?.percent ?? 0)))
  const embedderLabel = materialEmbedderLabel(item)
  const subline =
    item.status === 'indexing'
      ? `${item.indexing?.message ?? 'Parsing'} ${indexingPercent}%`
      : item.status === 'paused'
        ? `Paused at ${indexingPercent}%`
      : item.chunkCount && item.chunkCount > 0
      ? `${item.detail}${item.pageCount ? ` - ${item.pageCount} pages` : ''}`
      : item.error
        ? `${item.detail} - ${item.error}`
        : item.detail

  return (
    <div
      className={`material-row ${compact ? 'is-compact' : ''} ${onRemove ? 'has-remove' : ''} ${isActive ? '' : 'is-disabled'}`}
    >
      <button
        className="material-toggle"
        type="button"
        onClick={onClick}
        aria-pressed={item.status === 'ready' ? isActive : undefined}
        title={item.status === 'ready' ? (isActive ? 'Use in answers' : 'Do not use in answers') : undefined}
      >
        <span className={`status-box ${item.status} ${isActive ? '' : 'is-inactive'}`} aria-hidden="true">
          {isActive && <Check size={15} />}
          {item.status === 'indexing' && <RefreshCw size={15} />}
          {item.status === 'paused' && <Pause size={15} />}
          {item.status === 'needsReview' && <FileText size={15} />}
        </span>
        <span className="material-copy">
          <span className="material-title-line">
            <strong>{item.title}</strong>
            {embedderLabel && <em>{embedderLabel}</em>}
          </span>
          <small>{subline}</small>
        </span>
      </button>
      {onRemove && (
        <button
          className="material-remove"
          type="button"
          onClick={onRemove}
          aria-label={`Remove ${item.title}`}
          title="Remove from Library"
        >
          <Trash2 size={17} aria-hidden="true" />
        </button>
      )}
    </div>
  )
}

function ModelsScreen({
  engines,
  models,
  onAddModel,
  onCancelOllamaModelDownload,
  onDownloadOllamaModel,
  onRemoveModel,
  onSelectEmbeddingModel,
  onSelectModel,
  selectedEmbeddingModelId,
  selectedModelId,
  modelSelectionLocked,
  onConnectCloud
}: {
  engines: EngineInfo[]
  models: LocalModel[]
  onAddModel: (model: LocalModel) => void
  onCancelOllamaModelDownload: (modelName: string, baseUrl?: string) => void
  onDownloadOllamaModel: (modelName: string, role: LocalModelRole, baseUrl?: string) => void
  onRemoveModel: (model: LocalModel) => void
  onSelectEmbeddingModel: (modelId: string) => void
  onSelectModel: (modelId: string) => void
  selectedEmbeddingModelId: string
  selectedModelId: string
  modelSelectionLocked: boolean
  onConnectCloud: (model?: LocalModel) => void
}) {
  const [mode, setMode] = useState<'installed' | 'explore'>(models.length === 0 ? 'explore' : 'installed')
  const [exploreTab, setExploreTab] = useState<'ollama' | 'remote'>('ollama')
  const [ollamaStatus, setOllamaStatus] = useState<OllamaStatus | null>(null)
  const [ollamaStatusState, setOllamaStatusState] = useState<'idle' | 'checking' | 'error'>('idle')
  const [ollamaError, setOllamaError] = useState<string | null>(null)
  const [ollamaDraftName, setOllamaDraftName] = useState('')
  const [ollamaDraftRole, setOllamaDraftRole] = useState<LocalModelRole>('generator')
  const [ollamaSearchResults, setOllamaSearchResults] = useState<OllamaSearchResult[]>([])
  const [ollamaSearchStatus, setOllamaSearchStatus] = useState<'idle' | 'searching' | 'error'>('idle')
  const [ollamaSearchError, setOllamaSearchError] = useState<string | null>(null)
  const {
    capabilities: deviceCapabilities,
    state: deviceCapabilityState,
    error: deviceCapabilityError
  } = useDeviceCapabilities(mode === 'explore')
  const runtimeDetail = engines.find((engine) => engine.id === 'tokensmith')?.detail ?? 'TokenSmith runtime'
  const deviceRecommendation = useMemo<ModelRecommendation | null>(
    () =>
      deviceCapabilities
        ? recommendModel(deviceCapabilities, localModelCatalog, defaultDeviceTierPolicy)
        : null,
    [deviceCapabilities]
  )

  useEffect(() => {
    if (models.length === 0) {
      setMode('explore')
      setExploreTab('ollama')
    }
  }, [models.length])

  useEffect(() => {
    const shouldRefreshOllamaStatus =
      (mode === 'explore' && exploreTab === 'ollama') ||
      models.some((model) => model.engine === 'ollama' || model.source === 'ollama')
    if (!shouldRefreshOllamaStatus || !window.tokensmith?.getOllamaStatus) {
      return
    }

    let cancelled = false
    setOllamaStatusState('checking')
    setOllamaError(null)

    window.tokensmith
      .getOllamaStatus()
      .then((status) => {
        if (cancelled) {
          return
        }
        setOllamaStatus(status)
        setOllamaStatusState('idle')
        setOllamaError(status.error ? readableOllamaError(new Error(status.error), 'Ollama is not running yet.') : null)
      })
      .catch((error) => {
        if (cancelled) {
          return
        }
        setOllamaStatusState('error')
        setOllamaError(readableOllamaError(error, 'Could not check Ollama.'))
      })

    return () => {
      cancelled = true
    }
  }, [exploreTab, mode, models])

  function handleOllamaDraftNameChange(value: string) {
    setOllamaDraftName(value)
    setOllamaSearchResults([])
    setOllamaSearchStatus('idle')
    setOllamaSearchError(null)
  }

  function handleOllamaDraftRoleChange(role: LocalModelRole) {
    setOllamaDraftRole(role)
    setOllamaSearchResults([])
    setOllamaSearchStatus('idle')
    setOllamaSearchError(null)
  }

  function handleRemoveModel(model: LocalModel) {
    if (model.engine === 'ollama' && model.ollamaModelName) {
      if (ollamaModelNameMatches(ollamaDraftName, model.ollamaModelName)) {
        setOllamaDraftName('')
      }
    }
    onRemoveModel(model)
  }

  function modelByOllamaName(modelName: string, role: LocalModelRole) {
    return models.find(
      (model) =>
        model.engine === 'ollama' &&
        typeof model.ollamaModelName === 'string' &&
        ollamaModelNameMatches(model.ollamaModelName, modelName) &&
        modelRole(model) === role
    )
  }

  function ollamaInfoForModel(modelName: string) {
    return ollamaStatus?.models.find((model) => ollamaModelInfoMatches(model, modelName))
  }

  function ollamaInfoForLocalModel(model: LocalModel) {
    if (model.engine !== 'ollama' && model.source !== 'ollama') {
      return undefined
    }

    const modelName = model.ollamaModelName ?? modelFilename(model) ?? model.name
    return ollamaInfoForModel(modelName)
  }

  function withOllamaInfo(model: LocalModel, modelInfo?: OllamaModelInfo): LocalModel {
    if (!modelInfo) {
      return model
    }

    return {
      ...model,
      filename: model.filename ?? modelInfo.name,
      ollamaModelName: model.ollamaModelName ?? modelInfo.name,
      sizeBytes: model.sizeBytes ?? modelInfo.size,
      contextLength: model.contextLength ?? modelInfo.details?.contextLength,
      parameters: model.parameters ?? modelInfo.details?.parameterSize,
      quant: model.quant ?? modelInfo.details?.quantizationLevel,
      type: model.type ?? (modelInfo.details?.family ? `Ollama ${modelInfo.details.family}` : undefined)
    }
  }

  async function refreshOllamaModels() {
    if (!window.tokensmith?.getOllamaStatus) {
      setOllamaStatusState('error')
      setOllamaError('Ollama model management is only available in the desktop app.')
      return
    }

    setOllamaStatusState('checking')
    setOllamaError(null)

    try {
      const status = await window.tokensmith.getOllamaStatus()
      setOllamaStatus(status)
      setOllamaStatusState('idle')
      setOllamaError(status.error ? readableOllamaError(new Error(status.error), 'Ollama is not running yet.') : null)
    } catch (error) {
      setOllamaStatusState('error')
      setOllamaError(readableOllamaError(error, 'Could not check Ollama.'))
    }
  }

  async function searchOllamaModels(event?: FormEvent<HTMLFormElement>) {
    event?.preventDefault()

    const query = ollamaDraftName.trim()
    if (!query) {
      setOllamaSearchResults([])
      setOllamaSearchStatus('idle')
      setOllamaSearchError(null)
      return
    }

    if (!window.tokensmith?.searchOllamaModels) {
      setOllamaSearchStatus('error')
      setOllamaSearchError('Ollama model search is only available in the desktop app.')
      return
    }

    setOllamaSearchStatus('searching')
    setOllamaSearchError(null)

    try {
      const results = await window.tokensmith.searchOllamaModels(query, ollamaDraftRole, 20)
      setOllamaSearchResults(results)
      setOllamaSearchStatus('idle')
      setOllamaSearchError(results.length === 0 ? 'No Ollama models found for this search.' : null)
    } catch (error) {
      setOllamaSearchResults([])
      setOllamaSearchStatus('error')
      setOllamaSearchError(error instanceof Error ? error.message : 'Ollama model search failed.')
    }
  }

  function modelStats(model: LocalModel) {
    if (model.engine === 'ollama') {
      const isOllamaEmbedder = modelCanEmbed(model) && !modelCanGenerate(model)
      const stats = [
        { label: 'Role', value: isOllamaEmbedder ? 'Embedder Model' : 'Chat Model' },
        { label: 'Provider', value: 'Ollama' },
        { label: 'Model', value: model.ollamaModelName ?? model.name },
        { label: 'Endpoint', value: (model.ollamaBaseUrl ?? defaultOllamaBaseUrl).replace(/^https?:\/\//, '').replace(/\/+$/, '') }
      ]
      if (model.sizeBytes) {
        stats.push({ label: 'Size', value: formatBytes(model.sizeBytes) })
      }
      if (model.parameters) {
        stats.push({ label: 'Parameters', value: model.parameters })
      }
      if (model.quant) {
        stats.push({ label: 'Quant', value: model.quant })
      }
      return stats
    }

    if (model.engine === 'remote') {
      const isRemoteEmbedder = modelCanEmbed(model) && !modelCanGenerate(model)
      return [
        { label: 'Role', value: isRemoteEmbedder ? 'Remote Embedder Model' : 'Remote Chat Model' },
        { label: 'Provider', value: model.providerName ?? 'Remote' },
        { label: 'Model', value: model.remoteModelName ?? model.name },
        { label: 'Endpoint', value: model.baseUrl?.replace(/^https?:\/\//, '').replace(/\/+$/, '') ?? 'OpenAI-compatible' }
      ]
    }

    return []
  }

  function ollamaSearchStats(result: OllamaSearchResult, role: LocalModelRole) {
    return [
      { label: 'Role', value: modelRoleLabelForRole(role) },
      { label: 'Model', value: result.name },
      result.pulls ? { label: 'Pulls', value: result.pulls } : undefined,
      result.tagCount ? { label: 'Tags', value: String(result.tagCount) } : undefined,
      result.sizes.length > 0 ? { label: 'Sizes', value: result.sizes.join(', ') } : undefined,
      result.capabilities.length > 0 ? { label: 'Capabilities', value: result.capabilities.join(', ') } : undefined,
      result.updated ? { label: 'Updated', value: result.updated } : undefined
    ].filter((stat): stat is { label: string; value: string } => Boolean(stat))
  }

  function renderStats(stats: Array<{ label: string; value: string }>) {
    return (
      <div className="model-stats">
        {stats.map((stat) => (
          <div key={stat.label}>
            <span>{stat.label}</span>
            <strong>{stat.value}</strong>
          </div>
        ))}
      </div>
    )
  }

  function progressLine(progress?: ModelDownloadProgress) {
    if (!progress) {
      return undefined
    }

    if (progress.status === 'incomplete') {
      return progress.bytesTotal
        ? `Paused · ${formatBytes(progress.bytesReceived)} of ${formatBytes(progress.bytesTotal)}`
        : 'Paused'
    }

    const received = formatBytes(progress.bytesReceived)
    const total = progress.bytesTotal ? formatBytes(progress.bytesTotal) : 'unknown'
    const speed = progress.speedBytesPerSecond ? ` · ${formatBytes(progress.speedBytesPerSecond)}/s` : ''
    const message = progress.message ? ` · ${progress.message}` : ''
    return `${progress.percent}% · ${received} of ${total}${speed}${message}`
  }

  function renderDownloadProgress(progress?: ModelDownloadProgress) {
    if (!progress) {
      return null
    }

    return (
      <>
        <div className="model-download-progress" aria-label={`Download progress ${progress.percent}%`}>
          <span style={{ width: `${Math.max(0, Math.min(100, progress.percent))}%` }} />
        </div>
        <small>{progressLine(progress)}</small>
        {progress.error && <small className="model-download-error">{progress.error}</small>}
      </>
    )
  }

  function modelRoleLabel(model: LocalModel) {
    if (model.engine === 'ollama') {
      return modelCanEmbed(model) && !modelCanGenerate(model) ? 'Embedder Model' : 'Chat Model'
    }

    if (model.engine === 'remote') {
      return modelCanEmbed(model) && !modelCanGenerate(model) ? 'Remote Embedder Model' : 'Remote Chat Model'
    }

    if (modelCanEmbed(model) && !modelCanGenerate(model)) {
      return 'Embedder Model'
    }

    if (modelCanEmbed(model) && modelCanGenerate(model)) {
      return 'Chat + Embedder Model'
    }

    return 'Chat Model'
  }

  function modelRoleLabelForRole(role: LocalModelRole) {
    if (role === 'embedder') {
      return 'Embedder Model'
    }

    if (role === 'both') {
      return 'Chat + Embedder Model'
    }

    return 'Chat Model'
  }

  function canRemoveModel(model: LocalModel) {
    return (
      model.engine === 'ollama' ||
      model.engine === 'remote' ||
      model.source === 'ollama' ||
      model.source === 'remote' ||
      model.source === 'downloaded' ||
      model.source === 'local' ||
      Boolean(model.filename || model.path)
    )
  }

  const modelExploreTabs: Array<{ id: 'ollama' | 'remote'; label: string }> = [
    { id: 'ollama', label: 'Ollama' },
    { id: 'remote', label: 'Providers' }
  ]

  function renderActionBox(model?: LocalModel) {
    const filename = model ? modelFilename(model) : undefined
    const isEmbedderOnly = model ? modelCanEmbed(model) && !modelCanGenerate(model) : false
    const isSelected = model?.id === (isEmbedderOnly ? selectedEmbeddingModelId : selectedModelId)
    const progress = model?.download
    const isOllama = model?.engine === 'ollama'

    if (model?.status === 'downloading' && filename && isOllama) {
      return (
        <aside className="model-action-box">
          <button
            className="model-action-button"
            type="button"
            onClick={() => onCancelOllamaModelDownload(model.ollamaModelName ?? filename, model.ollamaBaseUrl)}
          >
            <span>Pause</span>
          </button>
          {renderDownloadProgress(progress)}
        </aside>
      )
    }

    if ((model?.status === 'incomplete' || model?.status === 'downloadError') && isOllama) {
      return (
        <aside className="model-action-box">
          <button
            className="model-action-button"
            type="button"
            onClick={() => onDownloadOllamaModel(model.ollamaModelName ?? model.name, modelRole(model), model.ollamaBaseUrl)}
          >
            <span>Resume</span>
          </button>
          {model && canRemoveModel(model) && (
            <button className="model-action-button is-danger" type="button" onClick={() => handleRemoveModel(model)}>
              <Trash2 size={16} aria-hidden="true" />
              <span>Remove</span>
            </button>
          )}
          {renderDownloadProgress(progress)}
        </aside>
      )
    }

    if (model?.status === 'ready') {
      const roleLabel = modelRoleLabel(model)
      const selectedLabel = isSelected ? (model.engine === 'ollama' ? 'Active Model' : `Active ${roleLabel}`) : `Use ${roleLabel}`

      return (
        <aside className="model-action-box">
          <button
            className={`model-action-button ${isSelected ? 'is-selected' : ''}`}
            type="button"
            onClick={() => isEmbedderOnly ? onSelectEmbeddingModel(model.id) : onSelectModel(model.id)}
            disabled={isSelected || (!isEmbedderOnly && modelSelectionLocked)}
          >
            <Check size={17} aria-hidden="true" />
            <span>{selectedLabel}</span>
          </button>
          {canRemoveModel(model) && (
            <button className="model-action-button is-danger" type="button" onClick={() => handleRemoveModel(model)}>
              <Trash2 size={16} aria-hidden="true" />
              <span>Remove</span>
            </button>
          )}
          <small>
            {model.engine === 'remote'
              ? isEmbedderOnly
                ? 'Remote embedding model'
                : 'Remote chat model'
              : model.engine === 'ollama'
                ? isEmbedderOnly
                  ? 'Used for PDF search'
                  : 'Used for local chat answers'
              : isEmbedderOnly
                ? 'Used for document and question embeddings'
                : runtimeDetail}
          </small>
        </aside>
      )
    }

    return (
      <aside className="model-action-box">
        <button className="model-action-button is-muted" type="button" disabled>
          <span>Unavailable</span>
        </button>
        <small>{runtimeDetail}</small>
      </aside>
    )
  }

  function renderInstalledModelCard(model: LocalModel) {
    if (model.engine === 'remote') {
      const isEmbedder = modelRole(model) === 'embedder'
      const selected = isEmbedder ? model.id === selectedEmbeddingModelId : model.id === selectedModelId
      return <section className="cloud-generator-entry" key={model.id}>
        <div><h2>{model.remoteModelName || model.name}</h2><p>{model.providerName || 'Cloud'} · {isEmbedder ? 'Embedding model' : 'Chat model'} · {model.status === 'ready' ? 'Connected' : 'Reconnect required'}</p></div>
        <div className="model-header-actions">
          {model.status === 'ready' && <button type="button" className="secondary-action" disabled={selected || (!isEmbedder && modelSelectionLocked)} onClick={() => isEmbedder ? onSelectEmbeddingModel(model.id) : onSelectModel(model.id)}>{selected ? 'Selected' : isEmbedder ? 'Use for search' : 'Use in chat'}</button>}
          <button type="button" className="secondary-action" onClick={() => onConnectCloud(model)}>{model.status === 'ready' ? 'Connection settings' : 'Reconnect'}</button>
          <button type="button" className="icon-button" aria-label={`Remove ${model.remoteModelName || model.name}`} onClick={() => handleRemoveModel(model)}><Trash2 size={16} /></button>
        </div>
      </section>
    }
    const displayModel = withOllamaInfo(model, ollamaInfoForLocalModel(model))
    const filename = modelFilename(displayModel)
    const title = displayModelName(displayModel)
    const isEmbedderOnly = modelCanEmbed(displayModel) && !modelCanGenerate(displayModel)
    const bullets =
      displayModel.description ??
      (isEmbedderOnly
          ? [
              'Local embedding model served by Ollama',
              'Prepares PDFs for search',
              'Embeds questions before source retrieval',
              'Removing this model deletes it from Ollama'
            ]
          : [
              'Local chat model served by Ollama',
              'Used for answers after TokenSmith retrieves enabled source excerpts',
              'Works with PDFs prepared by TokenSmith',
              'Removing this model deletes it from Ollama'
            ])

    return (
      <section className="model-card" key={model.id}>
        <div className="model-card-main">
          <div className="model-title-row">
            <div className="model-title-heading">
              <h2>{title}</h2>
              <span className={`model-role-badge ${isEmbedderOnly ? 'is-embedder' : 'is-chat'}`}>
                {modelRoleLabel(displayModel)}
              </span>
            </div>
            {filename && <p>{filename}</p>}
          </div>
          <ul className="model-description-list">
            {bullets.map((bullet) => (
              <li key={bullet}>{bullet}</li>
            ))}
          </ul>
          {renderStats(modelStats(displayModel))}
        </div>
        {renderActionBox(displayModel)}
      </section>
    )
  }

  const installedModels = models.filter((model) => model.engine === 'remote' || (model.status !== 'needsRuntime' && model.status !== 'missing'))

  function renderOllamaModelAction(model: LocalModel, isInstalledInOllama: boolean) {
    const isEmbedderOnly = modelCanEmbed(model) && !modelCanGenerate(model)
    const roleLabel = modelRoleLabel(model)

    if (model.status !== 'missing' && models.some((item) => item.id === model.id)) {
      return renderActionBox(model)
    }

    if (isInstalledInOllama) {
      return (
        <aside className="model-action-box">
          <button className="model-action-button" type="button" onClick={() => onAddModel(model)}>
            <Check size={17} aria-hidden="true" />
            <span>{`Use ${roleLabel}`}</span>
          </button>
          <small>{isEmbedderOnly ? 'Available in Ollama for PDF preparation' : 'Available in Ollama for local chat'}</small>
        </aside>
      )
    }

    return (
      <aside className="model-action-box">
        <button
          className="model-action-button is-download"
          type="button"
          disabled={!ollamaStatus?.running}
          onClick={() => onDownloadOllamaModel(model.ollamaModelName ?? model.name, modelRole(model), model.ollamaBaseUrl)}
        >
          <DownloadIcon size={17} aria-hidden="true" />
          <span>Download</span>
        </button>
        <small>{ollamaStatus?.running ? 'Download with Ollama' : 'Start Ollama first'}</small>
      </aside>
    )
  }

  function renderOllamaModelCard({
    description,
    isRecommended = false,
    modelInfo,
    modelName,
    role,
    searchResult,
    title
  }: {
    description?: string[]
    isRecommended?: boolean
    modelInfo?: OllamaModelInfo
    modelName: string
    role: LocalModelRole
    searchResult?: OllamaSearchResult
    title?: string
  }) {
    const existingModel = modelByOllamaName(modelName, role)
    const isInstalledInOllama = Boolean(modelInfo)
    const model =
      existingModel ??
      createOllamaModelForRole(
        modelName,
        role,
        ollamaStatus?.baseUrl ?? defaultOllamaBaseUrl,
        modelInfo,
        isInstalledInOllama ? 'ready' : 'missing'
      )
    const displayModel = withOllamaInfo(model, modelInfo)
    const isUninstalledDefaultModel = !existingModel && !isInstalledInOllama && !searchResult
    const bullets =
      description ??
      (isUninstalledDefaultModel
        ? []
        : (searchResult?.description ? [searchResult.description] : undefined) ??
          displayModel.description ??
          [])
    const roleClassName = modelCanEmbed(model) && !modelCanGenerate(model) ? 'is-embedder' : 'is-chat'
    const stats =
      searchResult && !existingModel && !isInstalledInOllama
        ? ollamaSearchStats(searchResult, role)
        : modelStats(displayModel)

    return (
      <section className={`model-card ${isRecommended ? 'is-device-recommended' : ''}`} key={`${role}-${modelName}`}>
        <div className="model-card-main">
          <div className="model-title-row">
            <div className="model-title-heading">
              <h2>{title ?? searchResult?.name ?? (isUninstalledDefaultModel ? modelName : displayModelName(displayModel))}</h2>
              <span className={`model-role-badge ${roleClassName}`}>{modelRoleLabel(displayModel)}</span>
              {isRecommended && (
                <span className="model-recommendation-badge">
                  <Sparkles size={13} aria-hidden="true" />
                  Recommended for this device
                </span>
              )}
            </div>
            <p>{isUninstalledDefaultModel ? 'Not installed in Ollama' : displayModel.ollamaModelName ?? modelName}</p>
          </div>
          {bullets.length > 0 && (
            <ul className="model-description-list">
              {bullets.map((bullet) => (
                <li key={bullet}>{bullet}</li>
              ))}
            </ul>
          )}
          {!isUninstalledDefaultModel && renderStats(stats)}
        </div>
        {renderOllamaModelAction(displayModel, isInstalledInOllama)}
      </section>
    )
  }

  function renderOllamaExplore() {
    const modelCards = [
      {
        modelName: recommendedOllamaEmbeddingModel,
        role: 'embedder' as LocalModelRole,
        title: 'Nomic Embedder Model',
        description: [
          'Local embedding model served by Ollama',
          'Prepares PDFs for search',
          'Download this first if you only want to add PDFs'
        ]
      },
      ...localModelCatalog.map((profile) => ({
        modelName: profile.id,
        role: 'generator' as LocalModelRole,
        title: profile.displayName,
        description: [
          `${profile.parameterLabel} parameters · ${profile.quantizationLabel} quantization · ${formatModelArtifactSize(profile.artifactSizeBytes)} download`
        ]
      }))
    ]
    const policyModelNames = new Set(modelCards.map((card) => card.modelName.toLowerCase()))
    const otherInstalledModels = (ollamaStatus?.models ?? []).filter(
      (model) => !policyModelNames.has(model.name.toLowerCase().replace(/:latest$/, ''))
    )
    const customModelName = ollamaDraftName.trim()
    const hasSearchResults = ollamaSearchResults.length > 0

    return (
      <>
        <p className="model-explore-copy">
          Download or connect Ollama models for local chat and PDF search.
        </p>

        <DeviceRecommendationPanel
          capabilities={deviceCapabilities}
          error={deviceCapabilityError}
          recommendation={deviceRecommendation}
          state={deviceCapabilityState}
        />

        <div className="ollama-model-toolbar">
          <button className="model-action-button" type="button" onClick={refreshOllamaModels} disabled={ollamaStatusState === 'checking'}>
            <RefreshCw size={16} aria-hidden="true" />
            <span>{ollamaStatusState === 'checking' ? 'Checking...' : 'Refresh Ollama'}</span>
          </button>
          {ollamaStatus?.running ? (
            <span className="chat-setup-model-pill is-ready">
              <Check size={15} aria-hidden="true" />
              <span>Ollama ready</span>
            </span>
          ) : (
            <span className="chat-setup-status">
              <span>Start Ollama first</span>
            </span>
          )}
        </div>

        {ollamaError && <p className="inline-error">{ollamaError}</p>}

        <form
          className="ollama-model-form"
          onSubmit={searchOllamaModels}
        >
          <input
            type="search"
            value={ollamaDraftName}
            onChange={(event) => handleOllamaDraftNameChange(event.target.value)}
            placeholder="Search Ollama models..."
            aria-label="Search Ollama models"
            list="ollama-installed-models"
          />
          <datalist id="ollama-installed-models">
            {(ollamaStatus?.models ?? []).map((model) => (
              <option key={model.name} value={model.name} />
            ))}
          </datalist>
          <select
            aria-label="Ollama model type"
            value={ollamaDraftRole}
            onChange={(event) => handleOllamaDraftRoleChange(event.target.value as LocalModelRole)}
          >
            <option value="generator">Chat Model</option>
            <option value="embedder">Embedder Model</option>
          </select>
          <button type="submit" disabled={!customModelName || ollamaSearchStatus === 'searching'}>
            {ollamaSearchStatus === 'searching' ? (
              <Loader2 size={17} aria-hidden="true" className="spin" />
            ) : (
              <Search size={17} aria-hidden="true" />
            )}
            <span>{ollamaSearchStatus === 'searching' ? 'Searching...' : 'Search'}</span>
          </button>
        </form>

        {ollamaSearchError && <p className={ollamaSearchStatus === 'error' ? 'inline-error' : 'model-explore-copy'}>{ollamaSearchError}</p>}

        <div className="model-list">
          {hasSearchResults
            ? ollamaSearchResults.map((result) =>
                renderOllamaModelCard({
                  modelName: result.name,
                  role: ollamaDraftRole,
                  modelInfo: ollamaInfoForModel(result.name),
                  searchResult: result
                })
              )
            : (
                <>
                  {modelCards.map((card) =>
                    renderOllamaModelCard({
                      ...card,
                      isRecommended:
                        card.role === 'generator' &&
                        deviceRecommendation?.kind === 'local' &&
                        deviceRecommendation.recommendedModelId === card.modelName,
                      modelInfo: ollamaInfoForModel(card.modelName)
                    })
                  )}
                  {otherInstalledModels.map((modelInfo) =>
                    renderOllamaModelCard({
                      modelInfo,
                      modelName: modelInfo.name,
                      role: inferOllamaModelRole(modelInfo)
                    })
                  )}
                </>
              )}
        </div>
      </>
    )
  }

  if (mode === 'explore') {
    return (
      <div className="view-frame standard-frame">
        <header className="screen-header">
          <div>
            <button className="secondary-action compact-action" type="button" onClick={() => setMode('installed')}>
              <span>← Existing Models</span>
            </button>
            <p className="section-kicker">Models</p>
            <h1>Explore Models</h1>
          </div>
        </header>

        <div className="explore-tabs" role="tablist" aria-label="Model sources">
          {modelExploreTabs.map(({ id, label }) => (
            <button
              className={exploreTab === id ? 'is-active' : ''}
              key={id}
              type="button"
              onClick={() => setExploreTab(id)}
            >
              {label}
            </button>
          ))}
        </div>

        {exploreTab === 'ollama' && renderOllamaExplore()}

        {exploreTab === 'remote' && (
          <>
            <p className="model-explore-copy">
              Connect a provider once, then add any compatible chat or embedding model with that same API key.
            </p>
            <div className="cloud-generator-entry"><div><h2>Online providers</h2><p>Connect once, choose a model, and get back to studying.</p></div><button type="button" className="primary-action" onClick={() => onConnectCloud()}>Connect a provider</button></div>
          </>
        )}

      </div>
    )
  }

  return (
    <div className="view-frame standard-frame">
      <header className="screen-header">
        <div>
          <p className="section-kicker">Models</p>
          <h1>Your models</h1>
          <p>Chat models answer questions. Embedder models build and search document vectors.</p>
        </div>
        <div className="model-header-actions">
          <button className="primary-action" type="button" onClick={() => onConnectCloud()}><Plus size={18} /><span>Connect cloud model</span></button>
          <button className="secondary-action" type="button" onClick={() => setMode('explore')}>
            <Plus size={18} aria-hidden="true" />
            <span>Add local model</span>
          </button>
        </div>
      </header>

      <div className="model-list">
        {installedModels.map((model) => renderInstalledModelCard(model))}
      </div>

      {installedModels.length === 0 && (
        <div className="model-empty-state">
          <h2>No Models Installed</h2>
          <p>Add or connect a chat model for answers and an embedder model for Library search.</p>
        </div>
      )}
    </div>
  )
}

function SettingsScreen({
  models,
  onSettingsChange,
  settings,
  isIndexing
}: {
  models: LocalModel[]
  onSettingsChange: (settings: Partial<TokenSmithSettings>) => void
  settings: TokenSmithSettings
  isIndexing: boolean
}) {
  const [activePane, setActivePane] = useState<'application' | 'model'>('application')
  const generatorModels = useMemo(() => models.filter(isAnswerModel), [models])
  const [selectedModelId, setSelectedModelId] = useState(settings.application.defaultModelId)
  const selectedModel = answerModel(generatorModels, selectedModelId)
  const activeModelSettings = selectedModel ? modelSettingsFor(settings, selectedModel.id) : settings.modelDefaults
  const modelOptions = [{ label: 'Choose a model', value: '' }, ...generatorModels.map(model => ({
    label: `${displayModelName(model)}${model.status === 'ready' ? '' : ' (unavailable)'}`, value: model.id }))]

  function updateApplicationSettings(application: Partial<ApplicationSettings>, root: Partial<TokenSmithSettings> = {}) {
    onSettingsChange({
      ...root,
      application: {
        ...settings.application,
        ...application
      }
    })
  }

  function updateModelSettings(modelSettings: Partial<ModelRuntimeSettings>) {
    if (!selectedModel) {
      return
    }

    onSettingsChange({
      modelSettingsById: {
        ...settings.modelSettingsById,
        [selectedModel.id]: {
          ...activeModelSettings,
          ...modelSettings
        }
      }
    })
  }

  function restoreApplicationDefaults() {
    updateApplicationSettings(
      {
        ...defaultApplicationSettings,
        defaultModelId: settings.application.defaultModelId
      },
      { maxSources: defaultSettings.maxSources }
    )
  }

  function restoreModelDefaults() {
    if (!selectedModel) {
      return
    }

    const nextSettings = { ...settings.modelSettingsById }
    delete nextSettings[selectedModel.id]
    onSettingsChange({ modelSettingsById: nextSettings })
  }

  return (
    <div className="view-frame settings-frame">
      <aside className="settings-nav" aria-label="Settings sections">
        {[
          ['application', 'Application'],
          ['model', 'Model']
        ].map(([id, label]) => (
          <button
            className={activePane === id ? 'is-active' : ''}
            key={id}
            type="button"
            onClick={() => setActivePane(id as 'application' | 'model')}
          >
            {label}
          </button>
        ))}
      </aside>

      <section className="settings-content">
        {activePane === 'application' ? (
          <>
            <SettingsPageHeader title="Application Settings" />

            <SettingsGroup title="Appearance">
              <SettingsRow label="Font size" description="Adjust text throughout the app. Changes apply immediately.">
                <SelectField
                  ariaLabel="Font size"
                  value={settings.application.fontSize}
                  onChange={(fontSize) =>
                    updateApplicationSettings({ fontSize: fontSize as ApplicationSettings['fontSize'] })
                  }
                  options={fontSizeOptions}
                />
              </SettingsRow>
              <ThemePicker
                value={settings.application.theme}
                onChange={(theme) => updateApplicationSettings({ theme })}
              />
            </SettingsGroup>

            <SettingsGroup title="General">
              <SettingsRow label="Default Model" description="The model for new chats and practice sessions. Existing sessions keep their model.">
                <SelectField
                  ariaLabel="Default model"
                  value={settings.application.defaultModelId}
                  onChange={(defaultModelId) => updateApplicationSettings({ defaultModelId })}
                  disabled={generatorModels.length === 0}
                  options={settings.application.defaultModelId && !answerModel(generatorModels, settings.application.defaultModelId)
                    ? [{ label: 'Selected model unavailable', value: settings.application.defaultModelId }, ...modelOptions] : modelOptions}
                />
              </SettingsRow>
              <SettingsRow
                label="Suggestion Mode"
                description="Generate suggested follow-up questions at the end of responses."
              >
                <CheckboxField
                  ariaLabel="Generate suggested follow-up questions"
                  checked={settings.application.suggestionMode === 'on'}
                  onChange={(checked) =>
                    updateApplicationSettings({
                      suggestionMode: checked ? 'on' : 'off',
                      followUpSuggestionCount: checked
                        ? settings.application.followUpSuggestionCount || defaultApplicationSettings.followUpSuggestionCount
                        : 0
                    })
                  }
                />
              </SettingsRow>
              <SettingsRow label="Follow-Up Count" description="Number of suggested follow-up questions to generate.">
                <SelectField
                  ariaLabel="Follow-up count"
                  disabled={settings.application.suggestionMode === 'off'}
                  value={String(
                    settings.application.followUpSuggestionCount === minFollowUpSuggestionCount
                      ? minFollowUpSuggestionCount
                      : defaultFollowUpSuggestionCount
                  )}
                  onChange={(followUpSuggestionCount) =>
                    updateApplicationSettings({
                      suggestionMode: 'on',
                      followUpSuggestionCount: Number(followUpSuggestionCount)
                    })
                  }
                  options={followUpSuggestionCountOptions.map((count) => ({
                    label: `${count} questions`,
                    value: String(count)
                  }))}
                />
              </SettingsRow>
              <SettingsRow label="Show Sources" description="Display the sources used for each response.">
                <CheckboxField
                  ariaLabel="Show sources"
                  checked={settings.application.showSources}
                  onChange={(showSources) => updateApplicationSettings({ showSources })}
                />
              </SettingsRow>
            </SettingsGroup>

            <SettingsGroup title="Advanced">
              <SettingsRow
                label="Search Mode"
                description="How library sources are retrieved: meaning (vector) matches on topic, keyword (BM25) matches on exact words, and hybrid blends both."
              >
                <SelectField
                  ariaLabel="Search mode"
                  value={settings.application.searchMode}
                  onChange={(searchMode) =>
                    updateApplicationSettings({ searchMode: searchMode as ApplicationSettings['searchMode'] })
                  }
                  options={[
                    { label: 'Meaning', value: 'vector' },
                    { label: 'Keyword', value: 'keyword' },
                    { label: 'Hybrid', value: 'hybrid' }
                  ]}
                />
              </SettingsRow>
              <SettingsRow
                label="Explanation Depth"
                description="Show a depth picker in the chat bar so you can ask for a simpler or more in-depth explanation question by question."
              >
                <CheckboxField
                  ariaLabel="Explanation depth"
                  checked={settings.application.explanationDepthEnabled}
                  onChange={(explanationDepthEnabled) => updateApplicationSettings({ explanationDepthEnabled })}
                />
              </SettingsRow>
              <SettingsRow label="GPU Acceleration for Document Search" description={isIndexing ? 'Available after document preparation finishes.' : 'Use available GPU acceleration for Ollama embeddings. Turn off to use the CPU for indexing and search. Chat models are unchanged.'}>
                <CheckboxField
                  ariaLabel="GPU acceleration for document search"
                  checked={settings.application.embeddingGpuEnabled}
                  disabled={isIndexing}
                  onChange={(embeddingGpuEnabled) => updateApplicationSettings({ embeddingGpuEnabled })}
                />
              </SettingsRow>
              <SettingsRow label="Maximum Sources" description="Maximum retrieved sources to include in answers.">
                <NumberField
                  ariaLabel="Maximum sources"
                  min={1}
                  max={10}
                  step={1}
                  value={settings.maxSources}
                  onChange={(maxSources) => onSettingsChange({ maxSources })}
                />
              </SettingsRow>
            </SettingsGroup>

            <button className="settings-restore" type="button" onClick={restoreApplicationDefaults}>
              Restore Defaults
            </button>
          </>
        ) : (
          <>
            <SettingsPageHeader title="Model Settings" />

            <div className="settings-toolbar">
              <SelectField
                ariaLabel="Model"
                value={selectedModel?.id ?? ''}
                onChange={setSelectedModelId}
                disabled={generatorModels.length === 0}
                options={modelOptions}
              />
            </div>

            {selectedModel ? (
              <>
                <SettingsGroup title="Identity">
                  <SettingsRow label="Name" description="The display name of the model.">
                    <TextField ariaLabel="Model name" value={displayModelName(selectedModel)} readOnly />
                  </SettingsRow>
                  <SettingsRow label="Model Identifier" description="The configured provider model name or local file for this model.">
                    <TextField
                      ariaLabel="Model file"
                      value={
                        modelFilename(selectedModel) ??
                        selectedModel.path ??
                        selectedModel.ollamaModelName ??
                        selectedModel.remoteModelName ??
                        'Remote model'
                      }
                      readOnly
                    />
                  </SettingsRow>
                </SettingsGroup>

                <SettingsGroup title="Prompts">
                  <SettingsRow
                    label="System Message"
                    description="A message to set the context or guide the behavior of the model. Leave blank for none."
                    wide
                  >
                    <TextAreaField
                      ariaLabel="System message"
                      value={activeModelSettings.systemMessage}
                      onChange={(systemMessage) => updateModelSettings({ systemMessage })}
                    />
                  </SettingsRow>
                  <SettingsRow label="Initial Question Prompt" description="" wide>
                    <TextAreaField
                      ariaLabel="Initial question prompt"
                      value={activeModelSettings.starterQuestionPrompt ?? defaultStarterQuestionPrompt}
                      onChange={(starterQuestionPrompt) => updateModelSettings({ starterQuestionPrompt })}
                    />
                  </SettingsRow>
                  <SettingsRow
                    label="Suggested Follow-Up Prompt"
                    description="Prompt used to generate suggested follow-up questions."
                    wide
                  >
                    <TextAreaField
                      ariaLabel="Suggested follow-up prompt"
                      value={activeModelSettings.suggestedFollowUpPrompt}
                      onChange={(suggestedFollowUpPrompt) => updateModelSettings({ suggestedFollowUpPrompt })}
                    />
                  </SettingsRow>
                </SettingsGroup>

                <SettingsGroup title="Generation">
                  {selectedModel.engine === 'ollama' && (
                    <div className="settings-row">
                      <span><strong>Reasoning</strong><small>Auto saves reasoning for calculations and problems that need several steps.</small></span>
                      <fieldset className="reasoning-mode" aria-label="Reasoning">
                        {(['auto', 'off', 'on'] as const).map(mode => (
                          <label key={mode}>
                            <input type="radio" name="reasoning-mode" value={mode}
                              checked={activeModelSettings.reasoningMode === mode}
                              onChange={() => updateModelSettings({ reasoningMode: mode, thinking: mode === 'on' })} />
                            <span>{mode === 'auto' ? 'Auto' : mode === 'off' ? 'Off' : 'On'}</span>
                          </label>
                        ))}
                      </fieldset>
                    </div>
                  )}
                  <div className="settings-two-column">
                    <SettingsRow label="Context Length" description={selectedModel.inputTokenLimit
                      ? `Input limit: ${selectedModel.inputTokenLimit.toLocaleString()} tokens.`
                      : selectedModel.contextLength
                        ? `Model limit: ${selectedModel.contextLength.toLocaleString()} tokens.`
                        : `Model limit unavailable. Auto uses ${fallbackContextLength.toLocaleString()} tokens.`}>
                      <div>
                        <SelectField ariaLabel="Context size" value={usesAutomaticContext(activeModelSettings) ? 'auto' : 'manual'}
                          options={[{ label: 'Auto', value: 'auto' }, { label: 'Custom', value: 'manual' }]}
                          onChange={mode => updateModelSettings({ contextLengthMode: mode as 'auto' | 'manual' })} />
                        {!usesAutomaticContext(activeModelSettings) && <NumberField
                          ariaLabel="Context length"
                          min={512}
                          max={reportedContextLimit(selectedModel, activeModelSettings) ?? Number.MAX_SAFE_INTEGER}
                          step={256}
                          value={Math.min(activeModelSettings.contextLength, reportedContextLimit(selectedModel, activeModelSettings) ?? Infinity)}
                          onChange={(contextLength) => updateModelSettings({ contextLengthMode: 'manual', contextLength: Math.min(
                            Math.max(512, contextLength), reportedContextLimit(selectedModel, activeModelSettings) ?? Number.MAX_SAFE_INTEGER) })}
                        />}
                      </div>
                    </SettingsRow>
                    <SettingsRow label="Max Length" description={selectedModel.engine === 'ollama'
                      ? 'Output token allowance. Reasoning reserves at least 4,096 tokens.' : 'Maximum generated tokens.'}>
                      <NumberField
                        ariaLabel="Max length"
                        min={64}
                        max={8192}
                        step={64}
                        value={activeModelSettings.maxLength}
                        onChange={(maxLength) => updateModelSettings({ maxLength })}
                      />
                    </SettingsRow>
                    <SettingsRow label="Temperature" description="Randomness of model output. Higher means more variation.">
                      <NumberField
                        ariaLabel="Temperature"
                        min={0}
                        max={2}
                        step={0.05}
                        value={activeModelSettings.temperature}
                        onChange={(temperature) => updateModelSettings({ temperature })}
                      />
                    </SettingsRow>
                    <SettingsRow label="Top-P" description="Nucleus sampling factor. Lower means more predictable.">
                      <NumberField
                        ariaLabel="Top-P"
                        min={0}
                        max={1}
                        step={0.01}
                        value={activeModelSettings.topP}
                        onChange={(topP) => updateModelSettings({ topP })}
                      />
                    </SettingsRow>
                    <SettingsRow label="Top-K" description="Size of selection pool for tokens.">
                      <NumberField
                        ariaLabel="Top-K"
                        min={0}
                        max={1000}
                        step={1}
                        value={activeModelSettings.topK}
                        onChange={(topK) => updateModelSettings({ topK })}
                      />
                    </SettingsRow>
                    <SettingsRow label="Min-P" description="Minimum token probability. Higher means more predictable.">
                      <NumberField
                        ariaLabel="Min-P"
                        min={0}
                        max={1}
                        step={0.01}
                        value={activeModelSettings.minP}
                        onChange={(minP) => updateModelSettings({ minP })}
                      />
                    </SettingsRow>
                    <SettingsRow label="Repeat Penalty" description="Repetition penalty factor. Set to 1 to disable.">
                      <NumberField
                        ariaLabel="Repeat penalty"
                        min={1}
                        max={3}
                        step={0.01}
                        value={activeModelSettings.repeatPenalty}
                        onChange={(repeatPenalty) => updateModelSettings({ repeatPenalty })}
                      />
                    </SettingsRow>
                  </div>
                </SettingsGroup>

                <button className="settings-restore" type="button" onClick={restoreModelDefaults}>
                  Restore Defaults
                </button>
              </>
            ) : (
              <div className="model-empty-state">
                <h2>No Chat Model Configured</h2>
                <p>Add a chat model before changing model-specific prompts and generation settings.</p>
              </div>
            )}
          </>
        )}
      </section>
    </div>
  )
}

function SettingsPageHeader({ title }: { title: string }) {
  return (
    <header className="settings-page-header">
      <h1>{title}</h1>
    </header>
  )
}

function SettingsGroup({ children, title }: { children: ReactNode; title: string }) {
  return (
    <section className="settings-group">
      <h2>{title}</h2>
      <div className="settings-group-rule" />
      <div className="settings-group-body">{children}</div>
    </section>
  )
}

function SettingsRow({
  children,
  description,
  label,
  wide = false
}: {
  children: ReactNode
  description: string
  label: string
  wide?: boolean
}) {
  return (
    <label className={`settings-row ${wide ? 'is-wide' : ''}`}>
      <span>
        <strong>{label}</strong>
        <small>{description}</small>
      </span>
      {children}
    </label>
  )
}

function SelectField({
  ariaLabel,
  disabled = false,
  onChange,
  options,
  value
}: {
  ariaLabel: string
  disabled?: boolean
  onChange: (value: string) => void
  options: Array<{ label: string; value: string }>
  value: string
}) {
  return (
    <select
      aria-label={ariaLabel}
      className="settings-control"
      disabled={disabled}
      value={value}
      onChange={(event) => onChange(event.currentTarget.value)}
    >
      {options.map((option) => (
        <option key={option.value} value={option.value}>
          {option.label}
        </option>
      ))}
    </select>
  )
}

function TextField({
  ariaLabel,
  onChange,
  readOnly = false,
  value
}: {
  ariaLabel: string
  onChange?: (value: string) => void
  readOnly?: boolean
  value: string
}) {
  return (
    <input
      aria-label={ariaLabel}
      className="settings-control"
      onChange={(event) => onChange?.(event.currentTarget.value)}
      readOnly={readOnly}
      value={value}
    />
  )
}

function NumberField({
  ariaLabel,
  max,
  min,
  onChange,
  step,
  value
}: {
  ariaLabel: string
  max: number
  min: number
  onChange: (value: number) => void
  step: number
  value: number
}) {
  return (
    <input
      aria-label={ariaLabel}
      className="settings-control is-number"
      max={max}
      min={min}
      onChange={(event) => onChange(Number(event.currentTarget.value))}
      step={step}
      type="number"
      value={value}
    />
  )
}

function CheckboxField({
  ariaLabel,
  checked,
  disabled,
  onChange
}: {
  ariaLabel: string
  checked: boolean
  disabled?: boolean
  onChange: (checked: boolean) => void
}) {
  return (
    <input
      aria-label={ariaLabel}
      checked={checked}
      disabled={disabled}
      className="settings-checkbox"
      onChange={(event) => onChange(event.currentTarget.checked)}
      type="checkbox"
    />
  )
}

function TextAreaField({
  ariaLabel,
  minRows = 4,
  mono = false,
  onChange,
  value
}: {
  ariaLabel: string
  minRows?: number
  mono?: boolean
  onChange: (value: string) => void
  value: string
}) {
  return (
    <textarea
      aria-label={ariaLabel}
      className={`settings-textarea ${mono ? 'is-mono' : ''}`}
      onChange={(event) => onChange(event.currentTarget.value)}
      rows={minRows}
      value={value}
    />
  )
}

function PanelIcon() {
  return (
    <svg width="25" height="21" viewBox="0 0 25 21" fill="none" aria-hidden="true">
      <rect x="1.5" y="2.5" width="22" height="16" rx="2.5" stroke="currentColor" strokeWidth="2" />
      <path d="M9 3V18" stroke="currentColor" strokeWidth="2" />
    </svg>
  )
}
