import { useEffect, useMemo, useRef, useState } from 'react'
import type { CSSProperties, FormEvent, ReactNode } from 'react'
import * as pdfjsLib from 'pdfjs-dist'
import pdfWorkerUrl from 'pdfjs-dist/build/pdf.worker.mjs?url'
import type { PDFDocumentProxy, TextItem } from 'pdfjs-dist/types/src/display/api'
import type {
  ApplicationSettings,
  AppStateSnapshot,
  ChatMessage,
  ChatSource,
  ComputeDevice,
  Conversation,
  CourseMaterial,
  LocalModel,
  LocalModelRole,
  MaterialIndexProgress,
  ModelDownloadProgress,
  ModelRuntimeSettings,
  ScreenId,
  SuggestionMode,
  TokenSmithSettings
} from '@shared/app-state'
import type { CleaningPreviewResult, EngineInfo, PdfSourceDocument, PdfSourceThumbnail } from '@shared/engine'
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
  catalogItemForFilename,
  catalogItemToLocalModel,
  firstRunRecommendedModelIds,
  normalizeModelFilename,
  tokenSmithTunedModels,
  type ModelCatalogItem
} from '@shared/model-catalog'
import {
  remoteProviderCatalog,
  type HuggingFaceSearchOptions,
  type HuggingFaceSort,
  type HuggingFaceSortDirection,
  type RemoteProviderCatalogItem,
  type RemoteProviderId
} from '@shared/model-providers'
import {
  defaultFollowUpSuggestionCount,
  defaultSuggestedFollowUpPrompt,
  followUpSuggestionCountOptions,
  minFollowUpSuggestionCount
} from '@shared/model-defaults'
import tokensmithAssistantMark from './assets/tokensmith-assistant-mark.png'
import tokensmithRailWordmark from './assets/tokensmith-rail-wordmark.png'
import {
  BookOpen,
  Check,
  ChevronDown,
  CircleUserRound,
  Copy,
  Database,
  Download as DownloadIcon,
  FileText,
  Library,
  Loader2,
  MessageSquare,
  Pencil,
  Plus,
  RefreshCw,
  Search,
  SendHorizonal,
  Settings,
  Sparkles,
  Square,
  Trash2,
  X
} from 'lucide-react'
import providerCustomLogo from './assets/provider-custom.svg'
import providerGeminiLogo from './assets/provider-gemini.svg'
import providerGroqLogo from './assets/provider-groq.svg'
import providerMistralLogo from './assets/provider-mistral.svg'
import providerOpenAiLogo from './assets/provider-openai.svg'

pdfjsLib.GlobalWorkerOptions.workerSrc = pdfWorkerUrl

interface NavItem {
  id: ScreenId
  label: string
  icon: typeof MessageSquare
}

type PdfViewerState = PdfSourceDocument & { searchTerm?: string }
type SourceThumbnailState = Record<string, PdfSourceThumbnail>
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

const navItems: NavItem[] = [
  { id: 'chat', label: 'Chat', icon: MessageSquare },
  { id: 'library', label: 'Library', icon: Library },
  { id: 'models', label: 'Models', icon: Database },
  { id: 'settings', label: 'Settings', icon: Settings }
]

const stateStorageKey = 'tokensmith-app-state-v1'
const maxSourceTrayCards = 5
const defaultCollectionChunkSize = 1000
const pdfViewerRenderScale = 1.45

const defaultMaterials: CourseMaterial[] = []

const remoteProviderLogoSources: Record<RemoteProviderId, string> = {
  groq: providerGroqLogo,
  openai: providerOpenAiLogo,
  gemini: providerGeminiLogo,
  mistral: providerMistralLogo,
  custom: providerCustomLogo
}

const defaultModelRuntimeSettings: ModelRuntimeSettings = {
  systemMessage: '',
  chatTemplate: '',
  suggestedFollowUpPrompt: defaultSuggestedFollowUpPrompt,
  contextLength: 2048,
  maxLength: 4096,
  promptBatchSize: 128,
  temperature: 0.7,
  topP: 0.4,
  topK: 40,
  minP: 0,
  repeatPenaltyTokens: 64,
  repeatPenalty: 1.18,
  gpuLayers: -1,
  device: 'applicationDefault'
}

const defaultApplicationSettings: ApplicationSettings = {
  theme: 'light',
  fontSize: 'small',
  defaultModelId: '',
  suggestionMode: 'on',
  followUpSuggestionCount: defaultFollowUpSuggestionCount,
  showSources: true,
  cpuThreads: 4
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
    title: 'New Study Chat',
    period: 'Today',
    messages: []
  }
]

const defaultAppState: AppStateSnapshot = {
  version: 1,
  appVersion: 'dev',
  activeScreen: 'models',
  activeConversationId: starterConversations[0].id,
  conversations: starterConversations,
  materials: defaultMaterials,
  models: defaultModels,
  selectedModelId: '',
  selectedEmbeddingModelId: '',
  settings: defaultSettings,
  updatedAt: new Date(0).toISOString()
}

function createFreshConversation(): Conversation {
  return {
    id: createId('conversation'),
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

function modelFilename(model: LocalModel) {
  return model.filename ?? pathLeaf(model.path)
}

function catalogForModel(model: LocalModel) {
  if (model.catalogId) {
    return tokenSmithTunedModels.find((item) => item.id === model.catalogId)
  }

  return catalogItemForFilename(modelFilename(model))
}

function modelQuantFromFilename(filename?: string) {
  const match = filename?.match(/(?:^|[-_. ])(q\d(?:[-_][a-z0-9]+)*|f16|fp16|bf16|int8|int4)(?:\.gguf)?$/i)
  return match ? match[1] : 'GGUF'
}

function modelTypeFromFilename(filename?: string) {
  const lowerFilename = filename?.toLowerCase() ?? ''

  if (lowerFilename.includes('llama-3') || lowerFilename.includes('llama3')) {
    return 'LLaMA3'
  }

  if (lowerFilename.includes('qwen')) {
    return 'qwen2'
  }

  if (lowerFilename.includes('deepseek')) {
    return 'deepseek'
  }

  return 'GGUF'
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
  if (material.status !== 'indexing') {
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

function firstGeneratorModel(models: LocalModel[]) {
  return models.find(modelCanGenerate)
}

function firstEmbeddingModel(models: LocalModel[]) {
  return models.find(modelCanEmbed)
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
      chunkSize: defaultCollectionChunkSize,
      indexedAt: material.indexedAt,
      isActive: material.isActive ?? material.status === 'ready',
      embeddingModel: material.embeddingModel,
      embeddingModelId: material.embeddingModelId,
      embeddingModelName: material.embeddingModelName,
      cleaningProfileId: material.cleaningProfileId,
      cleaningProfileName: material.cleaningProfileName,
      cleaningProfileVersion: material.cleaningProfileVersion,
      cleaningRuleIds: normalizeCleaningRuleIds(material.cleaningRuleIds, material.cleaningProfileId),
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

      return model.engine === 'python' || model.engine === 'remote'
    })
    .map((model, index) => {
      if (model.engine === 'remote') {
        const hasApiKey = Boolean(model.apiKey?.trim())
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
          remoteModelName: model.remoteModelName,
          embeddingPath: model.embeddingPath,
          type: model.type ?? 'OpenAI-compatible',
          description: model.description,
          addedAt: model.addedAt ?? new Date(index).toISOString()
        }
      }

      const filename = model.filename ?? pathLeaf(model.path)
      const catalog = model.catalogId
        ? tokenSmithTunedModels.find((item) => item.id === model.catalogId)
        : catalogItemForFilename(filename)
      const modelId = model.id ?? createId('model')
      const savedStatus: LocalModel['status'] = model.status === 'downloading'
        ? 'incomplete'
        : model.status ?? (model.path ? 'ready' : 'missing')
      const download = model.download && filename
        ? {
            ...model.download,
            modelId,
            catalogId: model.catalogId ?? catalog?.id,
            filename,
            status: model.download.status === 'downloading' ? 'incomplete' as const : model.download.status
          }
        : undefined

      return {
        id: modelId,
        name: model.name,
        engine: 'python' as const,
        role: model.role ?? catalog?.role ?? 'generator',
        status: savedStatus,
        source: model.source,
        catalogId: model.catalogId ?? catalog?.id,
        filename,
        path: model.path,
        embeddingPath: model.embeddingPath,
        url: model.url ?? catalog?.url,
        sizeBytes: model.sizeBytes ?? catalog?.sizeBytes,
        ramRequiredGb: model.ramRequiredGb ?? catalog?.ramRequiredGb,
        parameters: model.parameters ?? catalog?.parameters,
        quant: model.quant ?? catalog?.quant,
        type: model.type ?? catalog?.type,
        description: model.description ?? catalog?.description,
        download,
        addedAt: model.addedAt ?? new Date(index).toISOString()
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

function normalizeApplicationSettings(settings?: Partial<ApplicationSettings>, models: LocalModel[] = defaultModels): ApplicationSettings {
  const generatorModels = models.filter(modelCanGenerate)
  const firstAvailableModel = firstGeneratorModel(models)
  const defaultModelId =
    typeof settings?.defaultModelId === 'string' && generatorModels.some((model) => model.id === settings.defaultModelId)
      ? settings.defaultModelId
      : firstAvailableModel?.id ?? ''
  const suggestionMode = normalizeSuggestionMode(settings?.suggestionMode)

  return {
    theme: normalizeChoice(settings?.theme, ['light', 'system'] as const, defaultApplicationSettings.theme),
    fontSize: normalizeChoice(settings?.fontSize, ['small', 'medium', 'large'] as const, defaultApplicationSettings.fontSize),
    defaultModelId,
    suggestionMode,
    followUpSuggestionCount: normalizeFollowUpSuggestionCount(settings?.followUpSuggestionCount, suggestionMode),
    showSources: settings?.showSources ?? defaultApplicationSettings.showSources,
    cpuThreads: Math.round(clampNumber(settings?.cpuThreads, defaultApplicationSettings.cpuThreads, 1, 64))
  }
}

function normalizeModelRuntimeSettings(settings?: Partial<ModelRuntimeSettings>): ModelRuntimeSettings {
  const chatTemplate = typeof settings?.chatTemplate === 'string'
    ? settings.chatTemplate
    : defaultModelRuntimeSettings.chatTemplate
  const suggestedFollowUpPrompt = typeof settings?.suggestedFollowUpPrompt === 'string'
    ? settings.suggestedFollowUpPrompt
    : defaultModelRuntimeSettings.suggestedFollowUpPrompt

  return {
    systemMessage: typeof settings?.systemMessage === 'string' ? settings.systemMessage : defaultModelRuntimeSettings.systemMessage,
    chatTemplate,
    suggestedFollowUpPrompt,
    contextLength: Math.round(clampNumber(
      settings?.contextLength,
      defaultModelRuntimeSettings.contextLength,
      512,
      32768
    )),
    maxLength: Math.round(clampNumber(
      settings?.maxLength,
      defaultModelRuntimeSettings.maxLength,
      64,
      8192
    )),
    promptBatchSize: Math.round(clampNumber(settings?.promptBatchSize, defaultModelRuntimeSettings.promptBatchSize, 1, 4096)),
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
    repeatPenaltyTokens: Math.round(
      clampNumber(settings?.repeatPenaltyTokens, defaultModelRuntimeSettings.repeatPenaltyTokens, 0, 4096)
    ),
    repeatPenalty: clampNumber(
      settings?.repeatPenalty,
      defaultModelRuntimeSettings.repeatPenalty,
      1,
      3
    ),
    gpuLayers: Math.round(clampNumber(settings?.gpuLayers, defaultModelRuntimeSettings.gpuLayers, -1, 999)),
    device: normalizeChoice(
      settings?.device,
      ['applicationDefault', 'cpu', 'gpu'] as const,
      defaultModelRuntimeSettings.device
    )
  }
}

function normalizeSettings(settings: Partial<TokenSmithSettings> | undefined, models: LocalModel[]): TokenSmithSettings {
  const incomingSettings = settings ?? {}
  const modelDefaults = normalizeModelRuntimeSettings(incomingSettings.modelDefaults)
  const modelSettingsById = Object.fromEntries(
    Object.entries(incomingSettings.modelSettingsById ?? {}).map(([modelId, modelSettings]) => [
      modelId,
      normalizeModelRuntimeSettings({
        ...modelDefaults,
        ...modelSettings
      })
    ])
  )

  return {
    maxSources: Math.round(clampNumber(incomingSettings.maxSources, defaultSettings.maxSources, 1, 10)),
    application: normalizeApplicationSettings(incomingSettings.application, models),
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

function startWithFreshConversation(conversations: Conversation[]) {
  const savedConversations = conversations.length > 0 ? conversations : structuredClone(starterConversations)
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

  const freshConversation = createFreshConversation()

  return {
    activeConversationId: freshConversation.id,
    conversations: [freshConversation, ...savedConversations]
  }
}

function mergeSavedState(savedState: AppStateSnapshot | null, appVersion = 'dev'): AppStateSnapshot {
  if (!savedState) {
    return createDefaultAppState(appVersion)
  }

  const shouldResetConversations = savedState.appVersion !== appVersion
  const conversations = !shouldResetConversations && Array.isArray(savedState.conversations)
    ? savedState.conversations
    : structuredClone(starterConversations)
  const freshChatState = startWithFreshConversation(conversations)
  const materials = normalizeMaterials(savedState.materials)
  const models = normalizeModels(savedState.models)
  const selectedModelExists = models.some((model) => model.id === savedState.selectedModelId && modelCanGenerate(model))
  const selectedEmbeddingModelId = (savedState as Partial<AppStateSnapshot>).selectedEmbeddingModelId
  const selectedEmbeddingModelExists =
    typeof selectedEmbeddingModelId === 'string' &&
    models.some((model) => model.id === selectedEmbeddingModelId && modelCanEmbed(model))

  return {
    ...createDefaultAppState(appVersion),
    ...savedState,
    appVersion,
    activeScreen: models.length === 0 ? 'models' : 'chat',
    activeConversationId: freshChatState.activeConversationId,
    conversations: freshChatState.conversations,
    materials,
    models,
    selectedModelId: selectedModelExists ? savedState.selectedModelId : firstGeneratorModel(models)?.id ?? '',
    selectedEmbeddingModelId: selectedEmbeddingModelExists ? selectedEmbeddingModelId : firstEmbeddingModel(models)?.id ?? '',
    settings: normalizeSettings(savedState.settings, models),
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
    if (material.status !== 'indexing') {
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

function cleanMaterialTitle(title?: string) {
  return (title ?? '')
    .replace(/\.(pdf|md|markdown|txt)$/i, '')
    .replace(/\s+/g, ' ')
    .trim()
}

function getLibraryTitle(materials: CourseMaterial[]) {
  const activeMaterials = compatibleActiveMaterials(materials)
  const visibleMaterials = activeMaterials.length > 0 ? activeMaterials : materials

  if (visibleMaterials.length === 1) {
    return cleanMaterialTitle(visibleMaterials[0].title) || 'Library'
  }

  return 'Library'
}

function materialEmbeddingIdentity(material: Pick<CourseMaterial, 'embeddingModel' | 'embeddingModelId'>) {
  return material.embeddingModel ?? material.embeddingModelId ?? ''
}

function compatibleActiveMaterials(materials: CourseMaterial[]) {
  const activeMaterials = materials.filter((material) => material.status === 'ready' && material.isActive !== false)
  const firstEmbeddingKey = activeMaterials.map(materialEmbeddingIdentity).find(Boolean)

  if (!firstEmbeddingKey) {
    return activeMaterials
  }

  return activeMaterials.filter((material) => materialEmbeddingIdentity(material) === firstEmbeddingKey)
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
  return Boolean(label?.startsWith('llama-cpp:') || label?.startsWith('remote-openai:'))
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

function embeddingModelsForMaterials(materials: CourseMaterial[], embeddingModels: LocalModel[]) {
  const embeddingModelId = materials.map((material) => material.embeddingModelId).find(Boolean)

  if (!embeddingModelId) {
    return embeddingModels
  }

  const model = embeddingModels.find((item) => item.id === embeddingModelId)
  return model ? [model] : embeddingModels
}

function isPdfSource(source: ChatSource) {
  return Boolean(source.path?.toLowerCase().endsWith('.pdf'))
}

function sourcePageLabel(source: ChatSource) {
  if (source.pageStart && source.pageEnd && source.pageEnd > source.pageStart) {
    return `Pages ${source.pageStart}-${source.pageEnd}`
  }

  if (source.pageStart) {
    return `Page ${source.pageStart}`
  }

  return source.locator
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

function sourceDocumentTitle(source: ChatSource) {
  const rawTitle = source.documentTitle || source.title
  return cleanMaterialTitle(rawTitle) || 'Source'
}

function sourceTrayKey(source: ChatSource, index: number) {
  return `${source.chunkId ?? source.chunkRowid ?? source.path ?? source.title}-${index}`
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
  return model.id === progress.modelId || normalizeModelFilename(modelFilename(model)) === normalizeModelFilename(progress.filename)
}

export function App() {
  const [appState, setAppState] = useState<AppStateSnapshot>(() => createDefaultAppState())
  const [appVersion, setAppVersion] = useState<string>('dev')
  const [engines, setEngines] = useState<EngineInfo[]>([
    {
      id: 'tokensmith',
      name: 'TokenSmith',
      status: 'needsSetup',
      detail: 'Local indexing, source-backed chat, and configured model runtimes.'
    }
  ])
  const [, setSaveStatus] = useState<'loading' | 'saved' | 'saving' | 'local' | 'error'>('loading')
  const [hasLoadedState, setHasLoadedState] = useState(false)
  const hasLoadedStateRef = useRef(false)
  const activeIndexRequestsRef = useRef(new Set<string>())
  const cancelledIndexRequestsRef = useRef(new Set<string>())
  const saveSequenceRef = useRef(0)

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
                        detail: 'The local TokenSmith runtime is not available. Material indexing and source-backed chat need it.'
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
        if (cancelledIndexRequestsRef.current.has(progress.materialId)) {
          return current
        }

        return {
          ...current,
          materials: current.materials.map((material) => {
            if (material.id !== progress.materialId) {
              return material
            }

            const nextMaterial: CourseMaterial = {
              ...material,
              status: progress.phase === 'error' ? 'needsReview' : 'indexing',
              indexing: progress,
              isActive: false
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
    if (!window.tokensmith) {
      return undefined
    }

    return window.tokensmith.onModelDownloadProgress((progress) => {
      updateAppState((current) => {
        if (progress.status === 'removed') {
          const selectedModel = current.models.find((model) => model.id === current.selectedModelId)
          const selectedEmbeddingModel = current.models.find((model) => model.id === current.selectedEmbeddingModelId)
          const remainingModels = current.models.filter((model) => !modelMatchesDownload(model, progress))

          return {
            ...current,
            models: remainingModels,
            selectedModelId: selectedModel && modelMatchesDownload(selectedModel, progress)
              ? firstGeneratorModel(remainingModels)?.id ?? ''
              : current.selectedModelId,
            selectedEmbeddingModelId: selectedEmbeddingModel && modelMatchesDownload(selectedEmbeddingModel, progress)
              ? firstEmbeddingModel(remainingModels)?.id ?? ''
              : current.selectedEmbeddingModelId
          }
        }

        let didUpdate = false
        const models = current.models.map((model) => {
          if (!modelMatchesDownload(model, progress)) {
            return model
          }

          didUpdate = true
          return mergeModelDownloadProgress(model, progress)
        })

        if (didUpdate) {
          return {
            ...current,
            models
          }
        }

        const catalogItem = catalogItemForFilename(progress.filename)
        if (!catalogItem) {
          return current
        }

        return {
          ...current,
          models: [
            mergeModelDownloadProgress(
              catalogItemToLocalModel(
                catalogItem,
                progress.modelId,
                progress.path,
                new Date().toISOString(),
                modelStatusFromDownload(progress)
              ),
              progress
            ),
            ...current.models
          ]
        }
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
    setAppState((current) => ({
      ...updater(current),
      updatedAt: new Date().toISOString()
    }))
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
      }, current.models)
      const requestedDefaultModelId = settings.application?.defaultModelId
      const selectedModelId =
        requestedDefaultModelId && current.models.some((model) => model.id === requestedDefaultModelId && modelCanGenerate(model))
          ? requestedDefaultModelId
          : current.selectedModelId

      return {
        ...current,
        selectedModelId,
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
    options: {
      resume?: boolean
      title?: string
      cleaningProfileId?: CleaningProfileId
      cleaningRuleIds?: CleaningRuleId[]
    } = {}
  ) {
    if (!window.tokensmith) {
      return
    }

    if (activeIndexRequestsRef.current.has(materialId)) {
      return
    }

    const indexingMaterial = appState.materials.find((material) => material.id === materialId)
    const embeddingModel =
      (embeddingModelOverride && modelCanEmbed(embeddingModelOverride) ? embeddingModelOverride : undefined) ??
      appState.models.find((model) => model.id === indexingMaterial?.embeddingModelId && modelCanEmbed(model)) ??
      appState.models.find((model) => model.id === appState.selectedEmbeddingModelId && modelCanEmbed(model)) ??
      firstEmbeddingModel(appState.models)

    if (!embeddingModel) {
      updateAppState((current) => ({
        ...current,
        materials: current.materials.map((material) =>
          material.id === materialId
            ? {
                ...material,
                status: 'needsReview',
                detail: 'Needs embedder',
                error: 'Download or select an embedder model before indexing this collection.',
                indexing: undefined,
                isActive: false
              }
            : material
        )
      }))
      return
    }

    activeIndexRequestsRef.current.add(materialId)

    void window.tokensmith
      .indexMaterial(materialId, materialPath, embeddingModel, {
        ...options,
        title: options.title ?? indexingMaterial?.title,
        cleaningProfileId: options.cleaningProfileId ?? indexingMaterial?.cleaningProfileId,
        cleaningRuleIds:
          options.cleaningRuleIds ??
          indexingMaterial?.cleaningRuleIds ??
          defaultCleaningRuleIdsForProfile(options.cleaningProfileId ?? indexingMaterial?.cleaningProfileId)
      })
      .then((indexedMaterial) => {
        activeIndexRequestsRef.current.delete(materialId)
        if (cancelledIndexRequestsRef.current.delete(materialId)) {
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
              chunkSize: defaultCollectionChunkSize,
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
        activeIndexRequestsRef.current.delete(materialId)
        if (cancelledIndexRequestsRef.current.delete(materialId)) {
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
                  error: error instanceof Error ? error.message : 'Processing failed',
                  indexing: {
                    materialId,
                    phase: 'error',
                    percent: material.indexing?.percent ?? 0,
                    message: 'Processing failed'
                  },
                  isActive: false
                }
              : material
          )
        }))
      })
  }

  function addModel(model: LocalModel) {
    upsertModel(model, true)
  }

  function upsertModel(model: LocalModel, shouldSelect = false) {
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

        return false
      }
      const existingModels = current.models.filter((existingModel) => {
        return !isSameModel(existingModel)
      })

      return {
        ...current,
        activeScreen: 'models',
        models: [model, ...existingModels],
        selectedModelId: shouldSelect && modelCanGenerate(model) ? model.id : current.selectedModelId,
        selectedEmbeddingModelId: shouldSelect && modelCanEmbed(model) ? model.id : current.selectedEmbeddingModelId
      }
    })
  }

  function selectModel(modelId: string) {
    updateAppState((current) => {
      if (!current.models.some((model) => model.id === modelId && modelCanGenerate(model))) {
        return current
      }

      return {
        ...current,
        selectedModelId: modelId
      }
    })
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

  function downloadModel(model: ModelCatalogItem) {
    const existingModel = appState.models.find(
      (item) => normalizeModelFilename(modelFilename(item)) === normalizeModelFilename(model.filename)
    )
    const modelId = existingModel?.id ?? createId('model')
    const startedAt = new Date().toISOString()
    const optimisticModel = {
      ...catalogItemToLocalModel(model, modelId, existingModel?.path, existingModel?.addedAt ?? startedAt, 'downloading'),
      download: {
        modelId,
        catalogId: model.id,
        filename: model.filename,
        status: 'downloading' as const,
        percent: existingModel?.download?.percent ?? 0,
        bytesReceived: existingModel?.download?.bytesReceived ?? 0,
        bytesTotal: model.sizeBytes,
        message: 'Starting download'
      }
    }

    upsertModel(optimisticModel, false)

    if (!window.tokensmith) {
      return
    }

    void window.tokensmith
      .downloadModel(model, modelId)
      .then((downloadedModel) => {
        upsertModel({
          ...downloadedModel,
          download: {
            modelId,
            catalogId: model.id,
            filename: model.filename,
            status: 'complete',
            percent: 100,
            bytesReceived: downloadedModel.sizeBytes ?? model.sizeBytes,
            bytesTotal: downloadedModel.sizeBytes ?? model.sizeBytes,
            path: downloadedModel.path,
            message: 'Downloaded'
          }
        }, true)
      })
      .catch((error) => {
        if (error instanceof Error && /cancelled/i.test(error.message)) {
          return
        }

        updateAppState((current) => ({
          ...current,
          models: current.models.map((item) =>
            item.id === modelId
              ? {
                  ...item,
                  status: 'downloadError',
                  download: {
                    ...(item.download ?? optimisticModel.download),
                    status: 'error',
                    error: error instanceof Error ? error.message : 'Download failed',
                    message: 'Download failed'
                  }
                }
              : item
          )
        }))
      })
  }

  function cancelModelDownload(filename: string) {
    void window.tokensmith?.cancelModelDownload(filename).catch(() => {
      setSaveStatus('error')
    })
  }

  function removeModel(model: LocalModel) {
    const selectedModelWillBeRemoved = appState.selectedModelId === model.id
    const selectedEmbeddingModelWillBeRemoved = appState.selectedEmbeddingModelId === model.id

    updateAppState((current) => {
      const remainingModels = current.models.filter((item) => item.id !== model.id)
      return {
        ...current,
        models: remainingModels,
        selectedModelId: selectedModelWillBeRemoved ? firstGeneratorModel(remainingModels)?.id ?? '' : current.selectedModelId,
        selectedEmbeddingModelId: selectedEmbeddingModelWillBeRemoved
          ? firstEmbeddingModel(remainingModels)?.id ?? ''
          : current.selectedEmbeddingModelId
      }
    })

    void window.tokensmith?.removeModel(model).catch(() => {
      setSaveStatus('error')
      upsertModel(model, selectedModelWillBeRemoved || selectedEmbeddingModelWillBeRemoved)
    })
  }

  function toggleMaterialActive(materialId: string) {
    const material = appState.materials.find((item) => item.id === materialId)
    if (!material || material.status !== 'ready') {
      return
    }

    const nextActive = material.isActive === false
    const targetEmbeddingKey = materialEmbeddingIdentity(material)
    const incompatibleActiveMaterialIds = nextActive
      ? appState.materials
          .filter(
            (item) =>
              item.id !== material.id &&
              item.status === 'ready' &&
              item.isActive !== false &&
              materialEmbeddingIdentity(item) !== targetEmbeddingKey
          )
          .map((item) => item.id)
      : []
    const nextSelectedEmbeddingModelId =
      nextActive &&
      material.embeddingModelId &&
      appState.models.some((model) => model.id === material.embeddingModelId && modelCanEmbed(model))
        ? material.embeddingModelId
        : appState.selectedEmbeddingModelId

    updateAppState((current) => ({
      ...current,
      selectedEmbeddingModelId: nextSelectedEmbeddingModelId,
      materials: current.materials.map((material) => {
        if (material.status !== 'ready') {
          return material
        }

        if (material.id === materialId) {
          return {
            ...material,
            isActive: nextActive
          }
        }

        if (incompatibleActiveMaterialIds.includes(material.id)) {
          return {
            ...material,
            isActive: false
          }
        }

        return material
      })
    }))

    void window.tokensmith?.setMaterialEnabled(materialId, nextActive).catch(() => {
      setSaveStatus('error')
    })

    incompatibleActiveMaterialIds.forEach((id) => {
      void window.tokensmith?.setMaterialEnabled(id, false).catch(() => {
        setSaveStatus('error')
      })
    })
  }

  function rebuildMaterial(materialId: string) {
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
              detail: 'Parsing',
              error: undefined,
              isActive: false,
              indexing: {
                materialId,
                phase: 'parsing',
                percent: 1,
                processedFiles: 0,
                totalFiles: item.kind === 'folder' ? item.fileCount ?? 0 : 1,
                processedEmbeddings: 0,
                totalEmbeddings: 0,
                message: 'Parsing'
              }
            }
          : item
      )
    }))

    const embeddingModel =
      appState.models.find((model) => model.id === material.embeddingModelId && modelCanEmbed(model)) ??
      appState.models.find((model) => model.id === appState.selectedEmbeddingModelId && modelCanEmbed(model)) ??
      firstEmbeddingModel(appState.models)

    if (!embeddingModel) {
      updateAppState((current) => ({
        ...current,
        materials: current.materials.map((item) =>
          item.id === materialId
            ? {
                ...item,
                status: 'needsReview',
                detail: 'Needs embedder',
                error: 'Download or select an embedder model before rebuilding this collection.',
                indexing: undefined,
                isActive: false
              }
            : item
        )
      }))
      return
    }

    startMaterialIndexing(materialId, material.path, embeddingModel, {
      title: material.title
    })
  }

  function stopMaterialIndexing(materialId: string) {
    const material = appState.materials.find((item) => item.id === materialId)

    if (!material || material.status !== 'indexing') {
      return
    }

    cancelledIndexRequestsRef.current.add(materialId)
    activeIndexRequestsRef.current.delete(materialId)

    void window.tokensmith?.cancelMaterialIndexing(materialId).catch(() => {
      setSaveStatus('error')
    })

    const hasExistingIndex = (material.chunkCount ?? 0) > 0
    const nextStatus: CourseMaterial['status'] = hasExistingIndex ? 'ready' : 'needsReview'

    updateAppState((current) => ({
      ...current,
      materials: current.materials.map((item) =>
        item.id === materialId
          ? {
              ...item,
              status: nextStatus,
              detail: hasExistingIndex ? 'Ready for chat' : 'Indexing stopped',
              error: hasExistingIndex ? undefined : 'Indexing was stopped before any chunks were saved.',
              indexing: undefined,
              isActive: hasExistingIndex
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
  const selectedModel =
    appState.models.find((model) => model.id === appState.selectedModelId && modelCanGenerate(model)) ??
    firstGeneratorModel(appState.models)
  const embeddingModels = appState.models.filter(modelCanEmbed)
  const activeTitle = useMemo(
    () => navItems.find((item) => item.id === activeScreen)?.label ?? 'Chat',
    [activeScreen]
  )

  return (
    <main className={`desktop-app font-size-${appState.settings.application.fontSize}`}>
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
        {hasLoadedState && activeScreen === 'chat' && (
          <ChatScreen
            activeConversationId={appState.activeConversationId}
            conversations={appState.conversations}
            materials={appState.materials}
            models={appState.models}
            embeddingModels={embeddingModels}
            selectedModel={selectedModel}
            settings={appState.settings}
            onChatStateChange={updateChatState}
            onOpenLibrary={() =>
              updateAppState((current) => ({
                ...current,
                activeScreen: 'library'
              }))
            }
            onSelectModel={selectModel}
            onToggleMaterialActive={toggleMaterialActive}
          />
        )}
        {hasLoadedState && activeScreen === 'library' && (
          <LibraryScreen
            embeddingModels={embeddingModels}
            materials={appState.materials}
            selectedEmbeddingModelId={appState.selectedEmbeddingModelId}
            onAddMaterials={addMaterials}
            onRemoveMaterial={removeMaterial}
            onRebuildMaterial={rebuildMaterial}
            onSelectEmbeddingModel={selectEmbeddingModel}
            onStopMaterialIndexing={stopMaterialIndexing}
            onStartMaterialIndexing={startMaterialIndexing}
            onToggleMaterialActive={toggleMaterialActive}
          />
        )}
        {hasLoadedState && activeScreen === 'models' && (
          <ModelsScreen
            engines={engines}
            models={appState.models}
            onAddModel={addModel}
            onCancelModelDownload={cancelModelDownload}
            onDownloadModel={downloadModel}
            onRemoveModel={removeModel}
            onSelectEmbeddingModel={selectEmbeddingModel}
            onSelectModel={selectModel}
            selectedEmbeddingModelId={appState.selectedEmbeddingModelId}
            selectedModelId={appState.selectedModelId}
          />
        )}
        {hasLoadedState && activeScreen === 'settings' && (
          <SettingsScreen models={appState.models} settings={appState.settings} onSettingsChange={updateSettings} />
        )}
      </section>
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
      <p>Loading chats, indexed materials, and the local study engine.</p>
    </div>
  )
}

function ChatScreen({
  activeConversationId,
  conversations,
  embeddingModels,
  materials,
  models,
  selectedModel,
  settings,
  onChatStateChange,
  onOpenLibrary,
  onSelectModel,
  onToggleMaterialActive
}: {
  activeConversationId: string
  conversations: Conversation[]
  embeddingModels: LocalModel[]
  materials: CourseMaterial[]
  models: LocalModel[]
  selectedModel?: LocalModel
  settings: TokenSmithSettings
  onOpenLibrary: () => void
  onSelectModel: (modelId: string) => void
  onToggleMaterialActive: (materialId: string) => void
  onChatStateChange: (
    updater: (current: Pick<AppStateSnapshot, 'activeConversationId' | 'conversations'>) => Pick<
      AppStateSnapshot,
      'activeConversationId' | 'conversations'
    >
  ) => void
}) {
  const [draft, setDraft] = useState('')
  const [isSidebarOpen, setIsSidebarOpen] = useState(true)
  const [pendingConversationId, setPendingConversationId] = useState<string | null>(null)
  const [pendingStatusText, setPendingStatusText] = useState<string | null>(null)
  const [expandedMessageId, setExpandedMessageId] = useState<string | null>(null)
  const [pdfViewer, setPdfViewer] = useState<PdfViewerState | null>(null)
  const [sourceTrayError, setSourceTrayError] = useState<string | null>(null)
  const [renamingConversationId, setRenamingConversationId] = useState<string | null>(null)
  const [renameDraft, setRenameDraft] = useState('')
  const requestSequenceRef = useRef(0)
  const messagesEndRef = useRef<HTMLDivElement | null>(null)
  const composerInputRef = useRef<HTMLInputElement | null>(null)
  const hasMountedMessagesRef = useRef(false)

  const activeConversation =
    conversations.find((conversation) => conversation.id === activeConversationId) ??
    conversations[0] ??
    starterConversations[0]
  const isPending = pendingConversationId === activeConversation.id
  const activeMaterials = materials.filter((material) => material.status === 'ready' && material.isActive !== false)
  const libraryTitle = getLibraryTitle(materials)
  const selectedModelLabel = selectedModel ? displayModelName(selectedModel) : 'Choose a model'
  const sourceMessages = activeConversation.messages.filter(
    (message) => message.role === 'assistant' && (message.sources?.length ?? 0) > 0
  )
  const selectedSourceMessage =
    sourceMessages.find((message) => message.id === expandedMessageId) ?? sourceMessages[sourceMessages.length - 1]
  const selectedSources = useMemo(
    () => selectedSourceMessage?.sources?.slice(0, maxSourceTrayCards) ?? [],
    [selectedSourceMessage]
  )
  const selectableModels = useMemo(() => {
    const readyModels = models.filter((model) => model.status === 'ready' && modelCanGenerate(model))
    if (!selectedModel) {
      return readyModels
    }
    return readyModels.some((model) => model.id === selectedModel.id)
      ? readyModels
      : [selectedModel, ...readyModels]
  }, [models, selectedModel])

  useEffect(() => {
    if (!hasMountedMessagesRef.current) {
      hasMountedMessagesRef.current = true
      return
    }

    messagesEndRef.current?.scrollIntoView({ block: 'end' })
  }, [activeConversation.id, activeConversation.messages.length, isPending])

  function handleNewChat() {
    requestSequenceRef.current += 1

    const newConversation: Conversation = {
      id: createId('conversation'),
      title: 'New Study Chat',
      period: 'Today',
      messages: []
    }

    onChatStateChange((current) => ({
      conversations: [newConversation, ...current.conversations],
      activeConversationId: newConversation.id
    }))
    setDraft('')
    setPendingConversationId(null)
    setPendingStatusText(null)
    setExpandedMessageId(null)
    setPdfViewer(null)
    setSourceTrayError(null)
    setRenamingConversationId(null)
    setRenameDraft('')
  }

  function handleStopResponse() {
    requestSequenceRef.current += 1
    setPendingConversationId(null)
    setPendingStatusText(null)
  }

  function handleSelectConversation(conversationId: string) {
    setRenamingConversationId(null)
    setRenameDraft('')
    setPdfViewer(null)
    setSourceTrayError(null)
    onChatStateChange((current) => ({
      ...current,
      activeConversationId: conversationId
    }))
  }

  function handleStartRenameConversation(conversationId: string) {
    const conversation = conversations.find((item) => item.id === conversationId)
    if (!conversation) {
      return
    }

    setRenamingConversationId(conversationId)
    setRenameDraft(conversation.title)
  }

  function handleCommitRenameConversation() {
    if (!renamingConversationId) {
      return
    }

    const conversation = conversations.find((item) => item.id === renamingConversationId)
    const nextTitle = renameDraft.trim()
    const conversationId = renamingConversationId

    setRenamingConversationId(null)
    setRenameDraft('')

    if (!conversation) {
      return
    }

    if (!nextTitle || nextTitle === conversation.title) {
      return
    }

    onChatStateChange((current) => ({
      ...current,
      conversations: current.conversations.map((item) =>
        item.id === conversationId ? { ...item, title: nextTitle } : item
      )
    }))
  }

  function handleCancelRenameConversation() {
    setRenamingConversationId(null)
    setRenameDraft('')
  }

  function handleDeleteConversation(conversationId: string) {
    const conversation = conversations.find((item) => item.id === conversationId)
    if (!conversation) {
      return
    }

    if (!window.confirm(`Delete "${conversation.title}"?`)) {
      return
    }

    requestSequenceRef.current += pendingConversationId === conversationId ? 1 : 0

    onChatStateChange((current) => {
      const remainingConversations = current.conversations.filter((item) => item.id !== conversationId)
      const nextConversations = remainingConversations.length > 0 ? remainingConversations : [createFreshConversation()]
      const nextActiveConversationId =
        current.activeConversationId === conversationId ? nextConversations[0].id : current.activeConversationId

      return {
        conversations: nextConversations,
        activeConversationId: nextActiveConversationId
      }
    })

    if (pendingConversationId === conversationId) {
      setPendingConversationId(null)
      setPendingStatusText(null)
    }
    if (renamingConversationId === conversationId) {
      setRenamingConversationId(null)
      setRenameDraft('')
    }
    setExpandedMessageId(null)
    setPdfViewer(null)
    setSourceTrayError(null)
  }

  function handleClearConversations() {
    if (!window.confirm('Clear all chats and start a new chat?')) {
      return
    }

    const freshConversation = createFreshConversation()
    requestSequenceRef.current += 1

    onChatStateChange(() => ({
      conversations: [freshConversation],
      activeConversationId: freshConversation.id
    }))

    setDraft('')
    setPendingConversationId(null)
    setPendingStatusText(null)
    setExpandedMessageId(null)
    setPdfViewer(null)
    setSourceTrayError(null)
    setRenamingConversationId(null)
    setRenameDraft('')
  }

  function handleCopyQuestion(text: string) {
    if (!navigator.clipboard) {
      return
    }

    void navigator.clipboard.writeText(text).catch(() => undefined)
  }

  function handleEditQuestion(messageId: string, text: string) {
    requestSequenceRef.current += 1
    setDraft(text)
    setPendingConversationId(null)
    setPendingStatusText(null)
    setExpandedMessageId(null)

    onChatStateChange((current) => ({
      ...current,
      conversations: current.conversations.map((conversation) => {
        if (conversation.id !== activeConversation.id) {
          return conversation
        }

        const messageIndex = conversation.messages.findIndex((message) => message.id === messageId)
        if (messageIndex < 0) {
          return conversation
        }

        return {
          ...conversation,
          messages: conversation.messages.slice(0, messageIndex)
        }
      })
    }))
  }

  function handleUseFollowUpSuggestion(suggestion: string) {
    void submitPrompt(suggestion)
  }

  async function handleOpenSource(source: ChatSource) {
    setSourceTrayError(null)

    if (!window.tokensmith?.getPdfForSource) {
      setSourceTrayError('The PDF viewer is not available in this build.')
      return
    }

    try {
      const pdf = await window.tokensmith.getPdfForSource(source)
      setPdfViewer({ ...pdf, searchTerm: searchTermForSource(source) })
    } catch (error) {
      setSourceTrayError(error instanceof Error ? error.message : 'Could not open the source PDF.')
    }
  }

  async function submitPrompt(rawPrompt: string) {
    const prompt = rawPrompt.trim()
    if (!prompt || pendingConversationId || !selectedModel) {
      return
    }

    const targetConversationId = activeConversation.id
    const userMessage: ChatMessage = {
      id: createId('user'),
      role: 'user',
      text: prompt
    }

    onChatStateChange((current) => ({
      ...current,
      conversations: current.conversations.map((conversation) => {
        if (conversation.id !== targetConversationId) {
          return conversation
        }

        return {
          ...conversation,
          title: conversation.messages.length === 0 ? getConversationTitle(prompt) : conversation.title,
          messages: [...conversation.messages, userMessage]
        }
      })
    }))
    setDraft('')
    setExpandedMessageId(null)
    setPdfViewer(null)
    setSourceTrayError(null)
    setPendingConversationId(targetConversationId)
    const requestSequence = requestSequenceRef.current + 1
    requestSequenceRef.current = requestSequence
    const activeModelSettings = modelSettingsFor(settings, selectedModel.id)
    const searchEmbeddingModels = embeddingModelsForMaterials(activeMaterials, embeddingModels)
    let retrievedSources: ChatSource[] | undefined = activeMaterials.length === 0 ? [] : undefined

    try {
      if (window.tokensmith && activeMaterials.length > 0) {
        const searchLabels = activeMaterials
          .map(materialEmbedderLabel)
          .filter((label): label is string => Boolean(label))
        setPendingStatusText(`searching ${searchLabels.length ? searchLabels.join(', ') : 'Library'} ...`)
        const searchStartedAt = performance.now()
        try {
          retrievedSources = await window.tokensmith.searchLibrary(
            prompt,
            activeMaterials,
            settings.maxSources,
            searchEmbeddingModels
          )
        } catch {
          retrievedSources = []
        }

        if (requestSequenceRef.current !== requestSequence) {
          return
        }

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

      const reply = await window.tokensmith.sendChatMessage({
        prompt,
        messages: activeConversation.messages,
        materials: activeMaterials,
        model: selectedModel,
        settings,
        applicationSettings: settings.application,
        modelSettings: activeModelSettings,
        retrievedSources
      })

      if (requestSequenceRef.current !== requestSequence) {
        return
      }

      const assistantMessage: ChatMessage = {
        id: createId('assistant'),
        role: 'assistant',
        text: reply.text,
        sources: settings.application.showSources ? reply.sources : [],
        followUpSuggestions: reply.followUpSuggestions ?? []
      }

      onChatStateChange((current) => ({
        ...current,
        conversations: current.conversations.map((conversation) => {
          if (conversation.id !== targetConversationId) {
            return conversation
          }

          return {
            ...conversation,
            messages: [...conversation.messages, assistantMessage]
          }
        })
      }))
    } catch (error) {
      if (requestSequenceRef.current !== requestSequence) {
        return
      }

      const assistantMessage: ChatMessage = {
        id: createId('assistant'),
        role: 'assistant',
        text:
          'I could not reach the local engine service yet. The app kept your message, and you can try again after the engine is configured.',
        sources: []
      }

      onChatStateChange((current) => ({
        ...current,
        conversations: current.conversations.map((conversation) => {
          if (conversation.id !== targetConversationId) {
            return conversation
          }

          return {
            ...conversation,
            messages: [...conversation.messages, assistantMessage]
          }
        })
      }))
    } finally {
      if (requestSequenceRef.current === requestSequence) {
        setPendingConversationId(null)
        setPendingStatusText(null)
      }
    }
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    await submitPrompt(draft)
  }

  return (
    <div className={`view-frame chat-frame ${isSidebarOpen ? '' : 'is-sidebar-collapsed'}`}>
      {isSidebarOpen && (
        <aside className="chat-sidebar" aria-label="Conversation list">
          <button className="new-chat-button" type="button" onClick={handleNewChat}>
            <Plus size={17} aria-hidden="true" />
            <span>New Chat</span>
          </button>
          <div className="conversation-groups">
            <ConversationGroup
              activeConversationId={activeConversation.id}
              label="Today"
              items={conversations.filter((chat) => chat.period === 'Today')}
              renameDraft={renameDraft}
              renamingConversationId={renamingConversationId}
              onCancelRename={handleCancelRenameConversation}
              onCommitRename={handleCommitRenameConversation}
              onDelete={handleDeleteConversation}
              onRenameDraftChange={setRenameDraft}
              onRenameStart={handleStartRenameConversation}
              onSelect={handleSelectConversation}
            />
            <ConversationGroup
              activeConversationId={activeConversation.id}
              label="This week"
              items={conversations.filter((chat) => chat.period === 'This week')}
              renameDraft={renameDraft}
              renamingConversationId={renamingConversationId}
              onCancelRename={handleCancelRenameConversation}
              onCommitRename={handleCommitRenameConversation}
              onDelete={handleDeleteConversation}
              onRenameDraftChange={setRenameDraft}
              onRenameStart={handleStartRenameConversation}
              onSelect={handleSelectConversation}
            />
          </div>
          <div className="chat-sidebar-footer">
            <button className="clear-chats-button" type="button" onClick={handleClearConversations}>
              <Trash2 size={15} aria-hidden="true" />
              <span>Clear All Chats</span>
            </button>
          </div>
        </aside>
      )}

      <section className="chat-main" aria-label="Study chat">
        <header className="chat-topbar">
          <button
            className="icon-button subtle"
            type="button"
            aria-label={isSidebarOpen ? 'Hide conversations' : 'Show conversations'}
            aria-pressed={isSidebarOpen}
            title={isSidebarOpen ? 'Hide conversations' : 'Show conversations'}
            onClick={() => setIsSidebarOpen((current) => !current)}
          >
            <PanelIcon />
          </button>
          <div className="model-picker-shell">
            <select
              className="model-picker"
              aria-label="Choose chat model"
              title="Choose chat model"
              value={selectedModel?.id ?? ''}
              disabled={isPending || selectableModels.length === 0}
              onChange={(event) => onSelectModel(event.target.value)}
            >
              {selectableModels.length === 0 && <option value="">Choose a model</option>}
              {selectableModels.map((model) => (
                <option key={model.id} value={model.id}>
                  {displayModelName(model)}
                  {model.status === 'ready' ? '' : ' (not ready)'}
                </option>
              ))}
            </select>
            <ChevronDown className="model-picker-chevron" size={17} aria-hidden="true" />
          </div>
          <button
            className="library-pill"
            type="button"
            aria-label="Open Library"
            title="Open Library"
            onClick={onOpenLibrary}
          >
            <span className="count-badge">{activeMaterials.length}</span>
            <span>Library</span>
          </button>
        </header>

        <div className="chat-body">
          <div className="message-list">
            {activeConversation.messages.length === 0 && !isPending ? (
              null
            ) : (
              activeConversation.messages.map((message) =>
                message.role === 'user' ? (
                  <UserMessage
                    key={message.id}
                    text={message.text}
                    onCopy={() => handleCopyQuestion(message.text)}
                    onEdit={() => handleEditQuestion(message.id, message.text)}
                  />
                ) : (
                  <AssistantMessage
                    expanded={expandedMessageId === message.id}
                    key={message.id}
                    message={message}
                    onSelectFollowUp={handleUseFollowUpSuggestion}
                    onToggleSources={() =>
                      setExpandedMessageId((current) => (current === message.id ? null : message.id))
                    }
                  />
                )
              )
            )}
            {isPending && <ThinkingMessage modelName={selectedModelLabel} statusText={pendingStatusText} />}
            <div ref={messagesEndRef} />
          </div>

          <form className="composer-area" aria-label="Message composer" onSubmit={handleSubmit}>
            {selectedModel && (isPending || selectedModel.status !== 'ready') && (
              <button
                className={`reload-chip ${isPending ? 'is-stop' : ''}`}
                type="button"
                onClick={isPending ? handleStopResponse : undefined}
              >
                {isPending ? (
                  <>
                    <Square size={14} aria-hidden="true" />
                    <span>Stop response</span>
                  </>
                ) : (
                  <>
                    <RefreshCw size={15} aria-hidden="true" />
                    <span>Load - {selectedModelLabel}</span>
                  </>
                )}
              </button>
            )}
            <div className="composer">
              <input
                aria-label="Message"
                disabled={Boolean(pendingConversationId) || !selectedModel}
                ref={composerInputRef}
                onChange={(event) => setDraft(event.target.value)}
                placeholder={selectedModel ? 'Ask about your course materials...' : 'Download a chat model to start...'}
                value={draft}
              />
              <button
                className="send-button"
                disabled={!draft.trim() || Boolean(pendingConversationId) || !selectedModel}
                type="submit"
                aria-label="Send message"
                title="Send message"
              >
                <SendHorizonal size={21} aria-hidden="true" />
              </button>
            </div>
          </form>
        </div>
      </section>

      <aside className="context-panel" aria-label="Selected course materials">
        <div className="context-header">
          <p>Library</p>
          <strong>{libraryTitle}</strong>
        </div>
        <div className="mini-material-list">
          {materials.length === 0 ? (
            <div className="mini-empty">No course materials added</div>
          ) : (
            materials.slice(0, 2).map((item) => (
              <MaterialRow
                item={item}
                compact
                key={item.id}
                onClick={() => onToggleMaterialActive(item.id)}
              />
            ))
          )}
        </div>
        <button className="secondary-action" type="button" onClick={onOpenLibrary}>
          <Plus size={17} aria-hidden="true" />
          <span>Add Materials</span>
        </button>
        <SourceTray sources={selectedSources} error={sourceTrayError} onOpenSource={handleOpenSource} />
      </aside>
      {pdfViewer && <PdfSourceViewer viewer={pdfViewer} onClose={() => setPdfViewer(null)} />}
    </div>
  )
}

function ConversationGroup({
  activeConversationId,
  label,
  items,
  renameDraft,
  renamingConversationId,
  onCancelRename,
  onCommitRename,
  onDelete,
  onRenameDraftChange,
  onRenameStart,
  onSelect,
}: {
  activeConversationId: string
  label: string
  items: Conversation[]
  renameDraft: string
  renamingConversationId: string | null
  onCancelRename: () => void
  onCommitRename: () => void
  onDelete: (id: string) => void
  onRenameDraftChange: (value: string) => void
  onRenameStart: (id: string) => void
  onSelect: (id: string) => void
}) {
  if (items.length === 0) {
    return null
  }

  return (
    <section className="conversation-group" aria-label={label}>
      <h2>{label}</h2>
      <div className="conversation-list">
        {items.map((item) => {
          const isSelected = item.id === activeConversationId
          const isRenaming = item.id === renamingConversationId

          return (
            <div className={`conversation-item ${isSelected ? 'is-selected' : ''}`} key={item.id}>
              {isRenaming ? (
                <form
                  className="conversation-rename-form"
                  onSubmit={(event) => {
                    event.preventDefault()
                    onCommitRename()
                  }}
                >
                  <input
                    aria-label="Chat name"
                    autoFocus
                    value={renameDraft}
                    onBlur={onCommitRename}
                    onChange={(event) => onRenameDraftChange(event.target.value)}
                    onKeyDown={(event) => {
                      if (event.key === 'Escape') {
                        event.preventDefault()
                        onCancelRename()
                      }
                    }}
                  />
                </form>
              ) : (
                <button className="conversation-select" type="button" onClick={() => onSelect(item.id)}>
                  <span>{item.title}</span>
                </button>
              )}
              {isSelected && !isRenaming && (
                <div className="conversation-actions" aria-label={`${item.title} actions`}>
                  <button
                    className="conversation-action"
                    type="button"
                    aria-label={`Rename ${item.title}`}
                    title="Rename chat"
                    onClick={() => onRenameStart(item.id)}
                  >
                    <Pencil size={17} aria-hidden="true" />
                  </button>
                  <button
                    className="conversation-action"
                    type="button"
                    aria-label={`Delete ${item.title}`}
                    title="Delete chat"
                    onClick={() => onDelete(item.id)}
                  >
                    <Trash2 size={18} aria-hidden="true" />
                  </button>
                </div>
              )}
            </div>
          )
        })}
      </div>
    </section>
  )
}

function UserMessage({ text, onCopy, onEdit }: { text: string; onCopy: () => void; onEdit: () => void }) {
  return (
    <article className="message-row">
      <CircleUserRound className="message-avatar user" size={26} aria-hidden="true" />
      <div>
        <h3>You</h3>
        <p>{text}</p>
        <div className="message-actions" aria-label="Question actions">
          <button className="message-action" type="button" aria-label="Edit question" title="Edit question" onClick={onEdit}>
            <Pencil size={18} aria-hidden="true" />
          </button>
          <button className="message-action" type="button" aria-label="Copy question" title="Copy question" onClick={onCopy}>
            <Copy size={18} aria-hidden="true" />
          </button>
        </div>
      </div>
    </article>
  )
}

function AssistantMessage({
  expanded,
  message,
  onSelectFollowUp,
  onToggleSources
}: {
  expanded: boolean
  message: ChatMessage
  onSelectFollowUp: (suggestion: string) => void
  onToggleSources: () => void
}) {
  const sources = message.sources ?? []
  const suggestions = message.followUpSuggestions ?? []

  return (
    <article className="message-row">
      <img className="message-avatar assistant assistant-mark" src={tokensmithAssistantMark} alt="" aria-hidden="true" />
      <div>
        <h3>TokenSmith</h3>
        <MessageText text={message.text} />
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
                  onClick={() => onSelectFollowUp(suggestion)}
                >
                  <span>{suggestion}</span>
                  <Plus size={18} aria-hidden="true" />
                </button>
              ))}
            </div>
          </section>
        )}
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
                    <p>{source.excerpt}</p>
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

function SourceTray({
  error,
  sources,
  onOpenSource
}: {
  error: string | null
  sources: ChatSource[]
  onOpenSource: (source: ChatSource) => void
}) {
  const [thumbnails, setThumbnails] = useState<SourceThumbnailState>({})

  useEffect(() => {
    let canceled = false

    setThumbnails({})

    const bridge = window.tokensmith
    if (!bridge?.getPdfThumbnailForSource) {
      return () => {
        canceled = true
      }
    }

    sources.forEach((source, index) => {
      if (!isPdfSource(source)) {
        return
      }

      const key = sourceTrayKey(source, index)
      void bridge
        .getPdfThumbnailForSource(source)
        .then((thumbnail) => {
          if (canceled) {
            return
          }
          setThumbnails((current) => ({ ...current, [key]: thumbnail }))
        })
        .catch(() => {
          // Thumbnail rendering is best-effort; the source card still opens the PDF.
        })
    })

    return () => {
      canceled = true
    }
  }, [sources])

  if (sources.length === 0) {
    return null
  }

  return (
    <section className="source-tray" aria-label="Top answer sources">
      <div className="source-tray-header">
        <BookOpen size={15} aria-hidden="true" />
        <span>Top sources</span>
      </div>
      <div className="source-tray-list">
        {sources.map((source, index) => {
          const canOpen = isPdfSource(source)
          const pageLabel = sourcePageLabel(source)
          const sourceKey = sourceTrayKey(source, index)
          const thumbnail = thumbnails[sourceKey]
          const sourceContextLabel = [source.collectionName, source.sectionHeader].filter(Boolean).join(' · ')

          return (
            <button
              className={`source-tray-card ${canOpen ? '' : 'is-text-only'}`}
              disabled={!canOpen}
              key={sourceKey}
              type="button"
              onClick={() => onOpenSource(source)}
            >
              <span className="source-media" aria-hidden="true">
                <span className={`source-thumbnail ${thumbnail ? 'has-image' : ''}`}>
                  {thumbnail ? (
                    <img src={thumbnail.dataUrl} alt="" />
                  ) : (
                    <>
                      <FileText size={17} />
                      <strong>PDF</strong>
                    </>
                  )}
                </span>
              </span>
              <span className="source-tray-copy">
                <strong>{sourceDocumentTitle(source)}</strong>
                {sourceContextLabel && <small>{sourceContextLabel}</small>}
                <span>{source.excerpt}</span>
              </span>
              <span className="source-card-badges">
                <small className="source-page-label">{pageLabel}</small>
              </span>
            </button>
          )
        })}
      </div>
      {error && <p className="source-tray-error">{error}</p>}
    </section>
  )
}

function PdfSourceViewer({ viewer, onClose }: { viewer: PdfViewerState; onClose: () => void }) {
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
    <div className="pdf-viewer-backdrop" role="dialog" aria-modal="true" aria-label="Source PDF viewer">
      <section className="pdf-viewer-panel">
        <header className="pdf-viewer-header">
          <div>
            <strong>{cleanMaterialTitle(viewer.title) || viewer.title}</strong>
            <span>Page {currentPage}{pdfDocument ? ` of ${pdfDocument.numPages}` : ''}</span>
          </div>
          <button className="icon-button subtle" type="button" aria-label="Close PDF viewer" onClick={onClose}>
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

function normalizeMessageMarkdown(text: string) {
  return text
    .replace(/\r\n/g, '\n')
    .replace(/([.!?:;)])\s+([*-])\s+(?=\*\*|[A-Z0-9])/g, '$1\n$2 ')
    .replace(/(\*\*)\s+([*-])\s+(?=\*\*|[A-Z0-9])/g, '$1\n$2 ')
    .replace(/([.!?:;)])\s+(\d+\.)\s+(?=\*\*|[A-Z0-9])/g, '$1\n$2 ')
    .trim()
}

function renderInlineMarkdown(text: string): ReactNode[] {
  const parts: ReactNode[] = []
  const boldPattern = /\*\*([^*]+)\*\*/g
  let lastIndex = 0
  let match: RegExpExecArray | null

  while ((match = boldPattern.exec(text)) !== null) {
    if (match.index > lastIndex) {
      parts.push(text.slice(lastIndex, match.index))
    }

    parts.push(<strong key={`${match.index}-${match[1]}`}>{match[1]}</strong>)
    lastIndex = match.index + match[0].length
  }

  if (lastIndex < text.length) {
    parts.push(text.slice(lastIndex))
  }

  return parts
}

function MessageText({ text }: { text: string }) {
  const normalizedText = normalizeMessageMarkdown(text)
  const blocks: ReactNode[] = []
  let paragraphLines: string[] = []
  let bulletItems: string[] = []
  let numberedItems: string[] = []

  function flushParagraph() {
    if (paragraphLines.length === 0) {
      return
    }

    const paragraph = paragraphLines.join(' ').replace(/\s+/g, ' ').trim()
    if (paragraph) {
      blocks.push(<p key={`p-${blocks.length}`}>{renderInlineMarkdown(paragraph)}</p>)
    }
    paragraphLines = []
  }

  function flushBullets() {
    if (bulletItems.length === 0) {
      return
    }

    blocks.push(
      <ul key={`ul-${blocks.length}`}>
        {bulletItems.map((item) => (
          <li key={item}>{renderInlineMarkdown(item)}</li>
        ))}
      </ul>
    )
    bulletItems = []
  }

  function flushNumbered() {
    if (numberedItems.length === 0) {
      return
    }

    blocks.push(
      <ol key={`ol-${blocks.length}`}>
        {numberedItems.map((item) => (
          <li key={item}>{renderInlineMarkdown(item)}</li>
        ))}
      </ol>
    )
    numberedItems = []
  }

  for (const rawLine of normalizedText.split('\n')) {
    const line = rawLine.trim()

    if (!line) {
      flushParagraph()
      flushBullets()
      flushNumbered()
      continue
    }

    const bulletMatch = line.match(/^[-*]\s+(.+)$/)
    if (bulletMatch) {
      flushParagraph()
      flushNumbered()
      bulletItems.push(bulletMatch[1])
      continue
    }

    const numberedMatch = line.match(/^\d+\.\s+(.+)$/)
    if (numberedMatch) {
      flushParagraph()
      flushBullets()
      numberedItems.push(numberedMatch[1])
      continue
    }

    flushBullets()
    flushNumbered()
    paragraphLines.push(line)
  }

  flushParagraph()
  flushBullets()
  flushNumbered()

  return (
    <div className="message-text">
      {blocks.length > 0 ? blocks : <p>{text}</p>}
    </div>
  )
}

function ThinkingMessage({ modelName, statusText }: { modelName: string; statusText?: string | null }) {
  return (
    <article className="message-row thinking-row" aria-live="polite">
      <img className="message-avatar assistant assistant-mark" src={tokensmithAssistantMark} alt="" aria-hidden="true" />
      <div>
        <h3>
          TokenSmith <span>{modelName}</span>
          {statusText && <span className="thinking-inline-status">{statusText}</span>}
        </h3>
        {!statusText && (
          <div className="thinking-bubble">
            <span />
            <span />
            <span />
          </div>
        )}
      </div>
    </article>
  )
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

function LibraryScreen({
  embeddingModels,
  materials,
  selectedEmbeddingModelId,
  onAddMaterials,
  onRemoveMaterial,
  onRebuildMaterial,
  onSelectEmbeddingModel,
  onStopMaterialIndexing,
  onStartMaterialIndexing,
  onToggleMaterialActive
}: {
  embeddingModels: LocalModel[]
  materials: CourseMaterial[]
  selectedEmbeddingModelId: string
  onAddMaterials: (materials: CourseMaterial[]) => void
  onRemoveMaterial: (materialId: string) => void
  onRebuildMaterial: (materialId: string) => void
  onSelectEmbeddingModel: (modelId: string) => void
  onStopMaterialIndexing: (materialId: string) => void
  onStartMaterialIndexing: (
    materialId: string,
    materialPath: string,
    embeddingModel?: LocalModel,
    options?: {
      resume?: boolean
      title?: string
      cleaningProfileId?: CleaningProfileId
      cleaningRuleIds?: CleaningRuleId[]
    }
  ) => void
  onToggleMaterialActive: (materialId: string) => void
}) {
  const [mode, setMode] = useState<'empty' | 'list' | 'create'>(materials.length === 0 ? 'empty' : 'list')
  const [collectionName, setCollectionName] = useState('')
  const [collectionNameEdited, setCollectionNameEdited] = useState(false)
  const [folderPath, setFolderPath] = useState('')
  const [collectionEmbeddingModelId, setCollectionEmbeddingModelId] = useState(selectedEmbeddingModelId)
  const [collectionCleaningProfileId, setCollectionCleaningProfileId] =
    useState<CleaningProfileId>(defaultCleaningProfileId)
  const [collectionCleaningRuleIds, setCollectionCleaningRuleIds] = useState<CleaningRuleId[]>(
    defaultCleaningRuleIdsForProfile(defaultCleaningProfileId)
  )
  const [showCleaningRules, setShowCleaningRules] = useState(false)
  const [cleaningPreview, setCleaningPreview] = useState<CleaningPreviewResult | null>(null)
  const [previewStatus, setPreviewStatus] = useState<'idle' | 'loading' | 'error'>('idle')
  const [importStatus, setImportStatus] = useState<'idle' | 'picking' | 'error'>('idle')

  useEffect(() => {
    if (mode === 'create') {
      return
    }

    setMode(materials.length === 0 ? 'empty' : 'list')
  }, [materials.length, mode])

  useEffect(() => {
    if (embeddingModels.some((model) => model.id === collectionEmbeddingModelId)) {
      return
    }

    setCollectionEmbeddingModelId(embeddingModels[0]?.id ?? selectedEmbeddingModelId)
  }, [collectionEmbeddingModelId, embeddingModels, selectedEmbeddingModelId])

  function clearCleaningPreviewState() {
    setCleaningPreview(null)
    setPreviewStatus('idle')
    setShowCleaningRules(false)
  }

  function updateFolderPath(nextPath: string, suggestedTitle?: string) {
    setFolderPath(nextPath)

    if (!collectionNameEdited) {
      setCollectionName(suggestedTitle || pathLeaf(normalizeCollectionPath(nextPath)) || '')
    }

    clearCleaningPreviewState()
  }

  async function handleBrowseFolder() {
    setImportStatus('picking')

    try {
      if (!window.tokensmith) {
        setImportStatus('error')
        return
      }

      const result = await window.tokensmith.pickMaterialFolder()

      if (!result.canceled && result.path) {
        updateFolderPath(result.path, result.title || pathLeaf(result.path))
      }

      setImportStatus('idle')
    } catch {
      setImportStatus('error')
    }
  }

  async function handlePreviewCleaning() {
    const normalizedPath = normalizeCollectionPath(folderPath)
    if (!normalizedPath || !window.tokensmith) {
      setPreviewStatus('error')
      setImportStatus('error')
      return
    }

    setPreviewStatus('loading')
    setImportStatus('idle')
    setShowCleaningRules(true)

    try {
      const preview = await window.tokensmith.previewCleaning(normalizedPath, {
        cleaningProfileId: collectionCleaningProfileId,
        cleaningRuleIds: collectionCleaningRuleIds
      })
      setCleaningPreview(preview)
      setCollectionCleaningRuleIds(normalizeCleaningRuleIds(preview.cleaningRuleIds, preview.profile.id))
      setPreviewStatus('idle')
    } catch {
      setCleaningPreview(null)
      setPreviewStatus('error')
    }
  }

  function handleToggleCleaningRule(ruleId: CleaningRuleId, checked: boolean) {
    const rule = cleaningRules.find((candidate) => candidate.id === ruleId)
    if (rule?.locked) {
      return
    }

    const nextRuleIds = checked
      ? [...collectionCleaningRuleIds, ruleId]
      : collectionCleaningRuleIds.filter((candidate) => candidate !== ruleId)

    setCollectionCleaningRuleIds(normalizeCleaningRuleIds(nextRuleIds, collectionCleaningProfileId))
    setCleaningPreview(null)
    setPreviewStatus('idle')
  }

  function closeCleaningPreview() {
    setCleaningPreview(null)
    setPreviewStatus('idle')
    setShowCleaningRules(false)
  }

  function handleCreateCollection(event: FormEvent) {
    event.preventDefault()

    const normalizedPath = normalizeCollectionPath(folderPath)
    if (!normalizedPath) {
      setImportStatus('error')
      return
    }

    const title = collectionName.trim() || pathLeaf(normalizedPath) || 'Course materials'
    const embeddingModel =
      embeddingModels.find((model) => model.id === collectionEmbeddingModelId) ?? embeddingModels[0]

    if (!embeddingModel) {
      setImportStatus('error')
      return
    }

    const material = createIndexingCollection(
      title,
      normalizedPath,
      embeddingModel,
      collectionCleaningProfileId,
      collectionCleaningRuleIds
    )

    onSelectEmbeddingModel(embeddingModel.id)

    onAddMaterials([material])
    onStartMaterialIndexing(material.id, normalizedPath, embeddingModel, {
      title,
      cleaningProfileId: collectionCleaningProfileId,
      cleaningRuleIds: collectionCleaningRuleIds
    })
    setCollectionName('')
    setCollectionNameEdited(false)
    setFolderPath('')
    setCollectionCleaningProfileId(defaultCleaningProfileId)
    setCollectionCleaningRuleIds(defaultCleaningRuleIdsForProfile(defaultCleaningProfileId))
    setCleaningPreview(null)
    setShowCleaningRules(false)
    setPreviewStatus('idle')
    setImportStatus('idle')
    setMode('list')
  }

  if (mode === 'create') {
    return (
      <div className="view-frame collection-create-frame">
        <form className="collection-create-form" onSubmit={handleCreateCollection}>
          <div>
            <h1>Add Document Collection</h1>
            <p>Add a folder containing plain text files, PDFs, or Markdown.</p>
          </div>

          <label className="collection-field">
            <span>Folder</span>
            <input
              value={folderPath}
              onChange={(event) => {
                updateFolderPath(event.target.value)
              }}
              placeholder="Folder path..."
            />
            <button type="button" onClick={handleBrowseFolder}>
              {importStatus === 'picking' ? 'Choosing...' : 'Browse'}
            </button>
          </label>

          <label className="collection-field">
            <span>Name</span>
            <input
              value={collectionName}
              onChange={(event) => {
                setCollectionName(event.target.value)
                setCollectionNameEdited(event.target.value.trim().length > 0)
              }}
              placeholder="Collection name..."
            />
          </label>

          <label className="collection-field">
            <span>Embedder</span>
            <select
              value={collectionEmbeddingModelId}
              onChange={(event) => setCollectionEmbeddingModelId(event.currentTarget.value)}
              disabled={embeddingModels.length === 0}
            >
              {embeddingModels.map((model) => (
                <option key={model.id} value={model.id}>
                  {displayModelName(model)}
                </option>
              ))}
              {embeddingModels.length === 0 && <option value="">No embedder models installed</option>}
            </select>
          </label>

          <label className="collection-field">
            <span>Cleaning</span>
            <select
              value={collectionCleaningProfileId}
              onChange={(event) => {
                const nextProfileId = event.currentTarget.value as CleaningProfileId
                setCollectionCleaningProfileId(nextProfileId)
                setCollectionCleaningRuleIds(defaultCleaningRuleIdsForProfile(nextProfileId))
                setCleaningPreview(null)
                setPreviewStatus('idle')
                setShowCleaningRules(false)
              }}
            >
              {cleaningProfiles.map((profile) => (
                <option key={profile.id} value={profile.id}>
                  {profile.name}
                </option>
              ))}
            </select>
            <button type="button" onClick={handlePreviewCleaning} disabled={previewStatus === 'loading'}>
              {previewStatus === 'loading' ? 'Previewing...' : 'Preview'}
            </button>
            <small className="collection-field-help">
              {cleaningProfiles.find((profile) => profile.id === collectionCleaningProfileId)?.description}
            </small>
          </label>

          {showCleaningRules && (
            <section className="cleaning-rules-panel" aria-label="Cleaning rules">
              <header>
                <strong>Cleaning rules</strong>
                <span>Toggle rules, then preview again.</span>
              </header>
              <div className="cleaning-rules-list">
                {cleaningRules.map((rule) => {
                  const checked = collectionCleaningRuleIds.includes(rule.id)

                  return (
                    <label className="cleaning-rule-row" key={rule.id}>
                      <input
                        type="checkbox"
                        checked={checked}
                        disabled={rule.locked}
                        onChange={(event) => handleToggleCleaningRule(rule.id, event.currentTarget.checked)}
                      />
                      <span>
                        <strong>{rule.name}</strong>
                        <small>{rule.description}</small>
                      </span>
                      <em>{checked ? 'Enabled' : 'Disabled'}</em>
                    </label>
                  )
                })}
              </div>
            </section>
          )}

          {importStatus === 'error' && (
            <p className="inline-error">
              {embeddingModels.length === 0 ? 'Download an embedder model before creating a collection.' : 'Choose a readable folder first.'}
            </p>
          )}

          {previewStatus === 'error' && (
            <p className="inline-error">
              The preview could not be generated for the selected collection.
            </p>
          )}

          {cleaningPreview && (
            <section className="cleaning-preview" aria-label="Cleaning preview">
              <header>
                <div>
                  <strong>{cleaningPreview.document.title}</strong>
                  <span>
                    {cleaningPreview.profile.name} preview · first {cleaningPreview.rawPages.length}{' '}
                    {cleaningPreview.rawPages.length === 1 ? 'page' : 'pages'}
                  </span>
                </div>
                <div className="cleaning-preview-header-actions">
                  {cleaningPreview.document.pageCount && <span>{cleaningPreview.document.pageCount} pages total</span>}
                  <button type="button" onClick={closeCleaningPreview} aria-label="Close cleaning preview">
                    <X size={16} />
                  </button>
                </div>
              </header>

              <div className="cleaning-preview-grid">
                <article className="cleaning-preview-pane">
                  <h2>Raw</h2>
                  <pre>{cleaningPreview.rawPages.map((page) => page.text).join('\n\n')}</pre>
                </article>
                <article className="cleaning-preview-pane">
                  <h2>Cleaned</h2>
                  <pre>{cleaningPreview.cleanedPages.map((page) => page.text).join('\n\n')}</pre>
                </article>
              </div>

              {cleaningPreview.chunks.length > 0 && (
                <div className="cleaning-preview-chunks">
                  <h2>Sample chunks</h2>
                  {cleaningPreview.chunks.slice(0, 3).map((chunk, index) => (
                    <article key={`${chunk.pageStart ?? 'chunk'}-${index}`}>
                      <span>
                        Chunk {index + 1}
                        {chunk.chunkSize ? ` · ${compactChunkSizeLabel(chunk.chunkSize)}` : ''}
                        {chunk.pageStart ? ` · Page ${chunk.pageStart}` : ''}
                        {chunk.sectionHeader ? ` · Section: ${chunk.sectionHeader}` : ''}
                      </span>
                      <p>{chunk.text}</p>
                    </article>
                  ))}
                </div>
              )}
            </section>
          )}

          <div className="collection-create-actions">
            <button
              className="secondary-action"
              type="button"
              onClick={() => {
                setCollectionCleaningProfileId(defaultCleaningProfileId)
                setCollectionCleaningRuleIds(defaultCleaningRuleIdsForProfile(defaultCleaningProfileId))
                setCollectionName('')
                setCollectionNameEdited(false)
                setFolderPath('')
                clearCleaningPreviewState()
                setMode(materials.length ? 'list' : 'empty')
              }}
            >
              Cancel
            </button>
            <button className="primary-action" type="submit" disabled={embeddingModels.length === 0}>
              Create Collection
            </button>
          </div>
        </form>
      </div>
    )
  }

  if (materials.length === 0) {
    return (
      <div className="view-frame collection-empty-frame">
        <div className="collection-empty-state">
          <h1>No Collections Installed</h1>
          <p>Install a collection of local documents to get started using this feature</p>
          <button className="primary-action" type="button" onClick={() => setMode('create')}>
            <Plus size={17} aria-hidden="true" />
            <span>Add Collection</span>
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="view-frame standard-frame">
      <header className="screen-header">
        <div>
          <p className="section-kicker">Library</p>
          <h1>Document Collections</h1>
        </div>
        <button className="primary-action" type="button" onClick={() => setMode('create')}>
          <Plus size={17} aria-hidden="true" />
          <span>Add Collection</span>
        </button>
      </header>

      <div className="collection-card-list">
        {materials.map((material) => (
          <CollectionCard
            item={material}
            key={material.id}
            onRemove={() => onRemoveMaterial(material.id)}
            onRebuild={() => onRebuildMaterial(material.id)}
            onStop={() => onStopMaterialIndexing(material.id)}
            onToggle={() => onToggleMaterialActive(material.id)}
          />
        ))}
      </div>
    </div>
  )
}

function CollectionCard({
  item,
  onRemove,
  onRebuild,
  onStop,
  onToggle
}: {
  item: CourseMaterial
  onRemove: () => void
  onRebuild: () => void
  onStop: () => void
  onToggle: () => void
}) {
  const progress = item.indexing
  const isIndexing = item.status === 'indexing'
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

      {item.status === 'indexing' && (
        <div className="collection-progress" style={{ '--progress': `${percent}%` } as CSSProperties}>
          <span />
        </div>
      )}

      <div className="collection-card-actions">
        <div className="collection-primary-actions">
          {isIndexing ? (
            <button className="collection-action-button collection-stop" type="button" onClick={onStop}>
              <Square size={14} aria-hidden="true" />
              <span>Stop</span>
            </button>
          ) : (
            <button
              className="collection-action-button collection-rebuild"
              type="button"
              onClick={onRebuild}
              disabled={!item.path}
            >
              <RefreshCw size={15} aria-hidden="true" />
              <span>Rebuild</span>
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
  const isActive = item.status === 'ready' && item.isActive !== false
  const indexingPercent = Math.max(0, Math.min(100, Math.round(item.indexing?.percent ?? 0)))
  const embedderLabel = materialEmbedderLabel(item)
  const subline =
    item.status === 'indexing'
      ? `${item.indexing?.message ?? 'Parsing'} ${indexingPercent}%`
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
  onCancelModelDownload,
  onDownloadModel,
  onRemoveModel,
  onSelectEmbeddingModel,
  onSelectModel,
  selectedEmbeddingModelId,
  selectedModelId
}: {
  engines: EngineInfo[]
  models: LocalModel[]
  onAddModel: (model: LocalModel) => void
  onCancelModelDownload: (filename: string) => void
  onDownloadModel: (model: ModelCatalogItem) => void
  onRemoveModel: (model: LocalModel) => void
  onSelectEmbeddingModel: (modelId: string) => void
  onSelectModel: (modelId: string) => void
  selectedEmbeddingModelId: string
  selectedModelId: string
}) {
  type RemoteProviderDraft = {
    apiKey: string
    baseUrl: string
    role: LocalModelRole
    modelName: string
    models: string[]
    isLoading: boolean
    error?: string
  }

  const [mode, setMode] = useState<'installed' | 'explore'>(models.length === 0 ? 'explore' : 'installed')
  const [exploreTab, setExploreTab] = useState<'tokensmith' | 'remote' | 'huggingface'>('tokensmith')
  const [catalogFilter, setCatalogFilter] = useState<'all' | 'chat' | 'embedder'>('all')
  const [remoteDrafts, setRemoteDrafts] = useState<Record<RemoteProviderId, RemoteProviderDraft>>(() =>
    remoteProviderCatalog.reduce(
      (drafts, provider) => {
        drafts[provider.id] = {
          apiKey: '',
          baseUrl: provider.baseUrl ?? '',
          role: 'generator',
          modelName: '',
          models: [],
          isLoading: false
        }

        return drafts
      },
      {} as Record<RemoteProviderId, RemoteProviderDraft>
    )
  )
  const [hfQuery, setHfQuery] = useState('')
  const [hfResults, setHfResults] = useState<ModelCatalogItem[]>([])
  const [hfStatus, setHfStatus] = useState<'idle' | 'searching' | 'error'>('idle')
  const [hfError, setHfError] = useState<string | null>(null)
  const [hfShowTools, setHfShowTools] = useState(false)
  const [hfRole, setHfRole] = useState<LocalModelRole>('generator')
  const [hfSort, setHfSort] = useState<HuggingFaceSort>('likes')
  const [hfSortDirection, setHfSortDirection] = useState<HuggingFaceSortDirection>('desc')
  const [hfLimit, setHfLimit] = useState(20)
  const runtimeDetail = engines.find((engine) => engine.id === 'tokensmith')?.detail ?? 'TokenSmith runtime'
  const firstRunRecommendedModelIdSet = useMemo(() => new Set<string>(firstRunRecommendedModelIds), [])
  const filteredCatalogModels = tokenSmithTunedModels.filter((model) => {
    const role = model.role ?? 'generator'
    if (catalogFilter === 'all') {
      return true
    }
    if (catalogFilter === 'chat') {
      return role === 'generator' || role === 'both'
    }
    if (catalogFilter === 'embedder') {
      return role === 'embedder' || role === 'both'
    }
    return true
  })
  const visibleCatalogModels =
    models.length === 0
      ? tokenSmithTunedModels.filter((model) => firstRunRecommendedModelIdSet.has(model.id))
      : filteredCatalogModels

  useEffect(() => {
    if (models.length === 0) {
      setMode('explore')
      setExploreTab('tokensmith')
    }
  }, [models.length])

  function modelByCatalogItem(catalogModel: ModelCatalogItem) {
    return models.find((model) => normalizeModelFilename(modelFilename(model)) === normalizeModelFilename(catalogModel.filename))
  }

  function updateRemoteDraft(providerId: RemoteProviderId, draft: Partial<(typeof remoteDrafts)[RemoteProviderId]>) {
    setRemoteDrafts((current) => ({
      ...current,
      [providerId]: {
        ...current[providerId],
        ...draft
      }
    }))
  }

  async function loadRemoteProviderModels(provider: RemoteProviderCatalogItem) {
    const draft = remoteDrafts[provider.id]
    const baseUrl = draft.baseUrl.trim()
    const apiKey = draft.apiKey.trim()

    if (!window.tokensmith || !apiKey || !baseUrl) {
      return
    }

    updateRemoteDraft(provider.id, { isLoading: true, error: undefined })

    try {
      const visibleModels = await window.tokensmith.listRemoteProviderModels(apiKey, baseUrl, draft.role)

      updateRemoteDraft(provider.id, {
        isLoading: false,
        models: visibleModels,
        modelName: draft.modelName || visibleModels[0] || '',
        error:
          visibleModels.length === 0
            ? `No compatible ${draft.role === 'embedder' ? 'embedding' : 'chat'} models were returned by this provider.`
            : undefined
      })
    } catch (error) {
      updateRemoteDraft(provider.id, {
        isLoading: false,
        models: [],
        error: error instanceof Error ? error.message : 'Could not load provider models.'
      })
    }
  }

  function installRemoteProviderModel(provider: RemoteProviderCatalogItem) {
    const draft = remoteDrafts[provider.id]
    const apiKey = draft.apiKey.trim()
    const baseUrl = draft.baseUrl.trim()
    const remoteModelName = draft.modelName.trim()

    if (!apiKey || !baseUrl || !remoteModelName) {
      updateRemoteDraft(provider.id, { error: 'API key, base URL, and model name are required.' })
      return
    }

    const isEmbedder = draft.role === 'embedder'
    const model: LocalModel = {
      id: createId('model'),
      name: provider.isCustom ? remoteModelName : `${provider.name} ${remoteModelName}`,
      engine: 'remote',
      role: draft.role,
      status: 'ready',
      source: 'remote',
      providerId: provider.id,
      providerName: provider.name,
      baseUrl,
      apiKey,
      remoteModelName,
      type: isEmbedder ? 'OpenAI-compatible embeddings' : 'OpenAI-compatible chat',
      description: [
        `${provider.name} remote ${isEmbedder ? 'embedding' : 'chat'} model`,
        isEmbedder
          ? 'Uses an OpenAI-compatible embeddings endpoint'
          : 'Uses an OpenAI-compatible chat completions endpoint',
        isEmbedder
          ? 'Builds collection vectors and embeds questions for retrieval'
          : 'Uses TokenSmith retrieval before sending source-backed prompts',
        'API key is kept for this app session and redacted from saved state'
      ],
      addedAt: new Date().toISOString()
    }

    onAddModel(model)
    setMode('installed')
  }

  async function searchHuggingFace(event?: FormEvent<HTMLFormElement>) {
    event?.preventDefault()

    const query = hfQuery.trim()
    if (!query) {
      setHfResults([])
      setHfStatus('idle')
      setHfError(null)
      return
    }

    if (!window.tokensmith) {
      setHfStatus('error')
      setHfError('HuggingFace search is only available in the desktop app.')
      return
    }

    const options: HuggingFaceSearchOptions = {
      sort: hfSort,
      direction: hfSortDirection,
      limit: hfLimit
    }

    setHfStatus('searching')
    setHfError(null)

    try {
      const results = await window.tokensmith.searchHuggingFaceModels(query, options)
      setHfResults(results.map((result) => ({ ...result, role: hfRole })))
      setHfStatus('idle')
    } catch (error) {
      setHfStatus('error')
      setHfError(error instanceof Error ? error.message : 'HuggingFace search failed.')
    }
  }

  function modelStats(model: LocalModel, catalog?: ModelCatalogItem) {
    if (model.engine === 'remote') {
      const isRemoteEmbedder = modelCanEmbed(model) && !modelCanGenerate(model)
      return [
        { label: 'Role', value: isRemoteEmbedder ? 'Remote Embedder Model' : 'Remote Chat Model' },
        { label: 'Provider', value: model.providerName ?? 'Remote' },
        { label: 'Model', value: model.remoteModelName ?? model.name },
        { label: 'Endpoint', value: model.baseUrl?.replace(/^https?:\/\//, '').replace(/\/+$/, '') ?? 'OpenAI-compatible' }
      ]
    }

    if (modelCanEmbed(model) && !modelCanGenerate(model)) {
      return [
        { label: 'Role', value: 'Embedder Model' },
        { label: 'File size', value: formatBytes(model.sizeBytes) },
        { label: 'Quant', value: model.quant ?? modelQuantFromFilename(modelFilename(model)) },
        { label: 'Type', value: model.type ?? modelTypeFromFilename(modelFilename(model)) }
      ]
    }

    const filename = modelFilename(model)
    const ramRequiredGb = catalog?.ramRequiredGb ?? model.ramRequiredGb

    return [
      { label: 'Role', value: modelCanEmbed(model) ? 'Chat + Embedder Model' : 'Chat Model' },
      { label: 'File size', value: formatBytes(catalog?.sizeBytes ?? model.sizeBytes) },
      { label: 'RAM required', value: ramRequiredGb ? `${ramRequiredGb} GB` : 'Unknown' },
      { label: 'Parameters', value: catalog?.parameters ?? model.parameters ?? 'Unknown' },
      { label: 'Quant', value: catalog?.quant ?? model.quant ?? modelQuantFromFilename(filename) },
      { label: 'Type', value: catalog?.type ?? model.type ?? modelTypeFromFilename(filename) }
    ]
  }

  function catalogStats(model: ModelCatalogItem) {
    return [
      { label: 'Role', value: modelRoleLabelForRole(model.role ?? 'generator') },
      { label: 'File size', value: formatBytes(model.sizeBytes) },
      { label: 'RAM required', value: model.ramRequiredGb ? `${model.ramRequiredGb} GB` : 'Unknown' },
      { label: 'Parameters', value: model.parameters },
      { label: 'Quant', value: model.quant },
      { label: 'Type', value: model.type }
    ]
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

    const received = formatBytes(progress.bytesReceived)
    const total = progress.bytesTotal ? formatBytes(progress.bytesTotal) : 'unknown'
    const speed = progress.speedBytesPerSecond ? ` · ${formatBytes(progress.speedBytesPerSecond)}/s` : ''
    return `${progress.percent}% · ${received} of ${total}${speed}`
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
      model.engine === 'remote' ||
      model.source === 'remote' ||
      model.source === 'downloaded' ||
      model.source === 'local' ||
      Boolean(model.filename || model.path)
    )
  }

  const modelExploreTabs: Array<{ id: 'tokensmith' | 'remote' | 'huggingface'; label: string }> = [
    { id: 'tokensmith', label: 'TokenSmith' },
    { id: 'remote', label: 'Remote Providers' },
    { id: 'huggingface', label: 'HuggingFace' }
  ]

  const modelCatalogFilters: Array<{ id: 'all' | 'chat' | 'embedder'; label: string }> = [
    { id: 'all', label: 'All' },
    { id: 'chat', label: 'Chat' },
    { id: 'embedder', label: 'Embedder' }
  ]

  function renderActionBox(model?: LocalModel, catalog?: ModelCatalogItem) {
    const filename = model ? modelFilename(model) : catalog?.filename
    const isEmbedderOnly = model ? modelCanEmbed(model) && !modelCanGenerate(model) : false
    const isSelected = model?.id === (isEmbedderOnly ? selectedEmbeddingModelId : selectedModelId)
    const progress = model?.download

    if (model?.status === 'downloading' && filename) {
      return (
        <aside className="model-action-box">
          <button className="model-action-button" type="button" onClick={() => onCancelModelDownload(filename)}>
            <span>Cancel</span>
          </button>
          {renderDownloadProgress(progress)}
        </aside>
      )
    }

    if ((model?.status === 'incomplete' || model?.status === 'downloadError') && catalog) {
      return (
        <aside className="model-action-box">
          <button className="model-action-button" type="button" onClick={() => onDownloadModel(catalog)}>
            <span>Resume</span>
          </button>
          {model && canRemoveModel(model) && (
            <button className="model-action-button is-danger" type="button" onClick={() => onRemoveModel(model)}>
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

      return (
        <aside className="model-action-box">
          <button
            className={`model-action-button ${isSelected ? 'is-selected' : ''}`}
            type="button"
            onClick={() => isEmbedderOnly ? onSelectEmbeddingModel(model.id) : onSelectModel(model.id)}
            disabled={isSelected}
          >
            <Check size={17} aria-hidden="true" />
            <span>{isSelected ? `Active ${roleLabel}` : `Use ${roleLabel}`}</span>
          </button>
          {canRemoveModel(model) && (
            <button className="model-action-button is-danger" type="button" onClick={() => onRemoveModel(model)}>
              <Trash2 size={16} aria-hidden="true" />
              <span>Remove</span>
            </button>
          )}
          <small>
            {model.engine === 'remote'
              ? isEmbedderOnly
                ? 'Remote embedding model'
                : 'Remote chat model'
              : isEmbedderOnly
                ? 'Used for collection and question embeddings'
                : runtimeDetail}
          </small>
        </aside>
      )
    }

    if (catalog) {
      return (
        <aside className="model-action-box">
          <button className="model-action-button is-download" type="button" onClick={() => onDownloadModel(catalog)}>
            <DownloadIcon size={17} aria-hidden="true" />
            <span>Download</span>
          </button>
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
    const catalog = catalogForModel(model)
    const filename = modelFilename(model)
    const title = catalog?.name ?? displayModelName(model)
    const isEmbedderOnly = modelCanEmbed(model) && !modelCanGenerate(model)
    const bullets =
      model.description ??
      catalog?.description ??
      (isEmbedderOnly
        ? [
            'Local GGUF embedding model',
            'Builds vectors for document collections',
            'Used to embed questions before source retrieval',
            'Collections record the embedder used during indexing'
          ]
        : [
            'Local GGUF chat model',
            'Runs through the TokenSmith Python runtime',
            'Used for answers after document retrieval',
            'Works with enabled document collections'
          ])

    return (
      <section className="model-card" key={model.id}>
        <div className="model-card-main">
          <div className="model-title-row">
            <div className="model-title-heading">
              <h2>{title}</h2>
              <span className={`model-role-badge ${isEmbedderOnly ? 'is-embedder' : 'is-chat'}`}>
                {modelRoleLabel(model)}
              </span>
            </div>
            {filename && <p>{filename}</p>}
          </div>
          <ul className="model-description-list">
            {bullets.map((bullet) => (
              <li key={bullet}>{bullet}</li>
            ))}
          </ul>
          {renderStats(modelStats(model, catalog))}
        </div>
        {renderActionBox(model, catalog)}
      </section>
    )
  }

  const installedModels = models.filter((model) => model.status !== 'needsRuntime' && model.status !== 'missing')

  function renderCatalogModelCard(catalogModel: ModelCatalogItem) {
    const installedModel = modelByCatalogItem(catalogModel)
    const role = catalogModel.role ?? 'generator'
    const roleClassName = role === 'embedder' ? 'is-embedder' : 'is-chat'

    return (
      <section className="model-card" key={catalogModel.id}>
        <div className="model-card-main">
          <div className="model-title-row">
            <div className="model-title-heading">
              <h2>{catalogModel.name}</h2>
              <span className={`model-role-badge ${roleClassName}`}>{modelRoleLabelForRole(role)}</span>
            </div>
            <p>{catalogModel.sourceFilename ?? catalogModel.filename}</p>
          </div>
          <ul className="model-description-list">
            {catalogModel.description.map((bullet) => (
              <li key={bullet}>{bullet}</li>
            ))}
          </ul>
          {renderStats(catalogStats(catalogModel))}
        </div>
        {renderActionBox(installedModel, catalogModel)}
      </section>
    )
  }

  function renderRemoteProviderCard(provider: RemoteProviderCatalogItem) {
    const draft = remoteDrafts[provider.id]
    const canInstall = Boolean(draft.apiKey.trim() && draft.baseUrl.trim() && draft.modelName.trim())
    const modelListId = `remote-models-${provider.id}`

    return (
      <section className="remote-provider-card" key={provider.id}>
        <div className="remote-provider-main">
          <div className="remote-provider-heading">
            <img className="remote-provider-logo" src={remoteProviderLogoSources[provider.id]} alt={`${provider.name} logo`} />
            <h2>{provider.name}</h2>
          </div>
          <p>{provider.description}</p>
          {provider.apiKeyUrl && (
            <a href={provider.apiKeyUrl} rel="noreferrer" target="_blank">
              Get your API key
            </a>
          )}
        </div>

        <div className="remote-provider-form">
          <label>
            <span>API Key</span>
            <input
              type="password"
              autoComplete="off"
              placeholder="enter API key"
              value={draft.apiKey}
              onChange={(event) => updateRemoteDraft(provider.id, { apiKey: event.target.value, error: undefined })}
            />
          </label>

          {provider.isCustom && (
            <label>
              <span>Base URL</span>
              <input
                type="url"
                placeholder="https://host.example/v1"
                value={draft.baseUrl}
                onChange={(event) => updateRemoteDraft(provider.id, { baseUrl: event.target.value, error: undefined })}
              />
            </label>
          )}

          <label>
            <span>Model Type</span>
            <select
              value={draft.role}
              onChange={(event) => {
                const role = event.target.value as LocalModelRole
                updateRemoteDraft(provider.id, { role, modelName: '', models: [], error: undefined })
              }}
            >
              <option value="generator">Chat Model</option>
              <option value="embedder">Embedder Model</option>
            </select>
          </label>

          <label>
            <span>Model Name</span>
            <input
              list={provider.isCustom ? undefined : modelListId}
              placeholder={provider.isCustom ? 'enter model name' : 'load models or enter model name'}
              value={draft.modelName}
              onChange={(event) => updateRemoteDraft(provider.id, { modelName: event.target.value, error: undefined })}
            />
            {!provider.isCustom && (
              <datalist id={modelListId}>
                {draft.models.map((modelName) => (
                  <option key={modelName} value={modelName} />
                ))}
              </datalist>
            )}
          </label>

          {!provider.isCustom && (
            <button
              className="model-action-button"
              type="button"
              disabled={!draft.apiKey.trim() || draft.isLoading}
              onClick={() => loadRemoteProviderModels(provider)}
            >
              <RefreshCw size={16} aria-hidden="true" />
              <span>{draft.isLoading ? 'Loading...' : 'Load Models'}</span>
            </button>
          )}

          <button
            className="model-action-button is-download"
            type="button"
            disabled={!canInstall}
            onClick={() => installRemoteProviderModel(provider)}
          >
            <Plus size={16} aria-hidden="true" />
            <span>Install</span>
          </button>

          {draft.error && <small className="model-download-error">{draft.error}</small>}
        </div>
      </section>
    )
  }

  function renderHuggingFaceSearch() {
    return (
      <>
        <p className="model-explore-copy">
          Search HuggingFace for GGUF models. TokenSmith downloads the selected model locally; many community
          models may require extra configuration before they work well.
        </p>
        <form className="hf-search-form" onSubmit={searchHuggingFace}>
          <input
            type="search"
            value={hfQuery}
            onChange={(event) => setHfQuery(event.target.value)}
            placeholder="Discover and download models by keyword search..."
            readOnly={hfStatus === 'searching'}
            aria-label="Search HuggingFace models"
          />
          <button type="button" aria-label="Search options" onClick={() => setHfShowTools((current) => !current)}>
            <Settings size={17} aria-hidden="true" />
          </button>
          <button className="hf-search-submit" type="submit" disabled={hfStatus === 'searching'}>
            {hfStatus === 'searching' ? (
              <>
                <Loader2 size={16} aria-hidden="true" />
                <span>Searching</span>
              </>
            ) : (
              <>
                <SendHorizonal size={17} aria-hidden="true" />
                <span>Search</span>
              </>
            )}
          </button>
        </form>

        {hfShowTools && (
          <div className="hf-tools">
            <select
              aria-label="Sort HuggingFace results"
              value={hfSort}
              onChange={(event) => setHfSort(event.target.value as HuggingFaceSort)}
            >
              <option value="default">Sort by: Default</option>
              <option value="likes">Sort by: Likes</option>
              <option value="downloads">Sort by: Downloads</option>
              <option value="recent">Sort by: Recent</option>
            </select>
            <select
              aria-label="HuggingFace sort direction"
              value={hfSortDirection}
              onChange={(event) => setHfSortDirection(event.target.value as HuggingFaceSortDirection)}
            >
              <option value="desc">Sort dir: Desc</option>
              <option value="asc">Sort dir: Asc</option>
            </select>
            <select
              aria-label="HuggingFace result limit"
              value={hfLimit}
              onChange={(event) => setHfLimit(Number(event.target.value))}
            >
              {[5, 10, 20, 50, 100].map((limit) => (
                <option key={limit} value={limit}>
                  Limit: {limit}
                </option>
              ))}
            </select>
          </div>
        )}

        {hfError && <p className="inline-error">{hfError}</p>}

        <div className="hf-tools hf-role-tools">
          <select
            aria-label="Install HuggingFace results as"
            value={hfRole}
            onChange={(event) => {
              const role = event.target.value as LocalModelRole
              setHfRole(role)
              setHfResults((current) => current.map((model) => ({ ...model, role })))
            }}
          >
            <option value="generator">Install as: Chat Model</option>
            <option value="embedder">Install as: Embedder Model</option>
          </select>
        </div>

        <div className="model-list">
          {hfResults.map((model) => renderCatalogModelCard(model))}
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

        {exploreTab === 'tokensmith' && (
          <>
            <p className="model-explore-copy">
              {models.length === 0
                ? 'Download one chat model and one embedder model to start. Choose smaller models for a quicker first run.'
                : 'These local chat and embedder models have been selected for TokenSmith. Download only models that fit in available memory.'}
            </p>
            {models.length > 0 && (
              <div className="model-filter-row">
                {modelCatalogFilters.map(({ id, label }) => (
                  <button
                    className={catalogFilter === id ? 'is-active' : ''}
                    key={id}
                    type="button"
                    onClick={() => setCatalogFilter(id)}
                  >
                    {label}
                  </button>
                ))}
              </div>
            )}
            <div className="model-list">
              {visibleCatalogModels.map((model) => renderCatalogModelCard(model))}
            </div>
          </>
        )}

        {exploreTab === 'remote' && (
          <>
            <p className="model-explore-copy">
              Add OpenAI-compatible hosted models. TokenSmith still retrieves enabled collection sources locally before
              sending a chat request to the provider.
            </p>
            <div className="remote-provider-grid">
              {remoteProviderCatalog.map((provider) => renderRemoteProviderCard(provider))}
            </div>
          </>
        )}

        {exploreTab === 'huggingface' && renderHuggingFaceSearch()}
      </div>
    )
  }

  return (
    <div className="view-frame standard-frame">
      <header className="screen-header">
        <div>
          <p className="section-kicker">Models</p>
          <h1>Installed Models</h1>
          <p>Chat models answer questions. Embedder models build and search collection vectors.</p>
        </div>
        <div className="model-header-actions">
          <button className="primary-action" type="button" onClick={() => setMode('explore')}>
            <Plus size={18} aria-hidden="true" />
            <span>Add Model</span>
          </button>
        </div>
      </header>

      <div className="model-list">
        {installedModels.map((model) => renderInstalledModelCard(model))}
      </div>

      {installedModels.length === 0 && (
        <div className="model-empty-state">
          <h2>No Models Installed</h2>
          <p>Add a chat model to answer questions, or an embedder model to index collections.</p>
        </div>
      )}
    </div>
  )
}

function SettingsScreen({
  models,
  onSettingsChange,
  settings
}: {
  models: LocalModel[]
  onSettingsChange: (settings: Partial<TokenSmithSettings>) => void
  settings: TokenSmithSettings
}) {
  const [activePane, setActivePane] = useState<'application' | 'model'>('application')
  const generatorModels = useMemo(() => models.filter(modelCanGenerate), [models])
  const [selectedModelId, setSelectedModelId] = useState(settings.application.defaultModelId || generatorModels[0]?.id || '')
  const selectedModel =
    generatorModels.find((model) => model.id === selectedModelId) ?? firstGeneratorModel(generatorModels)
  const activeModelSettings = selectedModel ? modelSettingsFor(settings, selectedModel.id) : settings.modelDefaults

  useEffect(() => {
    if (!generatorModels.some((model) => model.id === selectedModelId)) {
      setSelectedModelId(firstGeneratorModel(generatorModels)?.id ?? '')
    }
  }, [generatorModels, selectedModelId])

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
        defaultModelId: firstGeneratorModel(generatorModels)?.id ?? ''
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

            <SettingsGroup title="General">
              <SettingsRow label="Theme" description="The application color scheme.">
                <SelectField
                  ariaLabel="Theme"
                  value={settings.application.theme}
                  onChange={(theme) => updateApplicationSettings({ theme: theme as ApplicationSettings['theme'] })}
                  options={[
                    { label: 'Light', value: 'light' },
                    { label: 'System', value: 'system' }
                  ]}
                />
              </SettingsRow>
              <SettingsRow label="Font Size" description="The size of text in the application.">
                <SelectField
                  ariaLabel="Font size"
                  value={settings.application.fontSize}
                  onChange={(fontSize) =>
                    updateApplicationSettings({ fontSize: fontSize as ApplicationSettings['fontSize'] })
                  }
                  options={[
                    { label: 'Small', value: 'small' },
                    { label: 'Medium', value: 'medium' },
                    { label: 'Large', value: 'large' }
                  ]}
                />
              </SettingsRow>
              <SettingsRow label="Default Model" description="The preferred model for new chats.">
                <SelectField
                  ariaLabel="Default model"
                  value={settings.application.defaultModelId}
                  onChange={(defaultModelId) => updateApplicationSettings({ defaultModelId })}
                  disabled={generatorModels.length === 0}
                  options={
                    generatorModels.length > 0
                      ? generatorModels.map((model) => ({ label: displayModelName(model), value: model.id }))
                      : [{ label: 'No chat models installed', value: '' }]
                  }
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
              <SettingsRow label="CPU Threads" description="The number of CPU threads used for inference.">
                <NumberField
                  ariaLabel="CPU threads"
                  min={1}
                  max={64}
                  step={1}
                  value={settings.application.cpuThreads}
                  onChange={(cpuThreads) => updateApplicationSettings({ cpuThreads })}
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
                options={
                  generatorModels.length > 0
                    ? generatorModels.map((model) => ({ label: displayModelName(model), value: model.id }))
                    : [{ label: 'No chat models installed', value: '' }]
                }
              />
            </div>

            {selectedModel ? (
              <>
                <SettingsGroup title="Identity">
                  <SettingsRow label="Name" description="The display name of the model.">
                    <TextField ariaLabel="Model name" value={displayModelName(selectedModel)} readOnly />
                  </SettingsRow>
                  <SettingsRow label="Model File" description="The local GGUF file used by this model.">
                    <TextField
                      ariaLabel="Model file"
                      value={
                        modelFilename(selectedModel) ??
                        selectedModel.path ??
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
                  <SettingsRow label="Chat Template" description="This Jinja template turns the chat into input for the model." wide>
                    <TextAreaField
                      ariaLabel="Chat template"
                      mono
                      minRows={7}
                      value={activeModelSettings.chatTemplate}
                      onChange={(chatTemplate) => updateModelSettings({ chatTemplate })}
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
                  <SettingsRow label="Device" description="The compute device used for text generation.">
                    <SelectField
                      ariaLabel="Device"
                      value={activeModelSettings.device}
                      onChange={(device) => updateModelSettings({ device: device as ComputeDevice })}
                      options={[
                        { label: 'Application default', value: 'applicationDefault' },
                        { label: 'CPU', value: 'cpu' },
                        { label: 'GPU', value: 'gpu' }
                      ]}
                    />
                  </SettingsRow>
                  <div className="settings-two-column">
                    <SettingsRow label="Context Length" description="Number of input and output tokens the model sees.">
                      <NumberField
                        ariaLabel="Context length"
                        min={512}
                        max={32768}
                        step={256}
                        value={activeModelSettings.contextLength}
                        onChange={(contextLength) => updateModelSettings({ contextLength })}
                      />
                    </SettingsRow>
                    <SettingsRow label="Max Length" description="Maximum response length, in tokens.">
                      <NumberField
                        ariaLabel="Max length"
                        min={64}
                        max={8192}
                        step={64}
                        value={activeModelSettings.maxLength}
                        onChange={(maxLength) => updateModelSettings({ maxLength })}
                      />
                    </SettingsRow>
                    <SettingsRow label="Prompt Batch Size" description="The batch size used for prompt processing.">
                      <NumberField
                        ariaLabel="Prompt batch size"
                        min={1}
                        max={4096}
                        step={1}
                        value={activeModelSettings.promptBatchSize}
                        onChange={(promptBatchSize) => updateModelSettings({ promptBatchSize })}
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
                    <SettingsRow label="Repeat Penalty Tokens" description="Number of previous tokens used for penalty.">
                      <NumberField
                        ariaLabel="Repeat penalty tokens"
                        min={0}
                        max={4096}
                        step={1}
                        value={activeModelSettings.repeatPenaltyTokens}
                        onChange={(repeatPenaltyTokens) => updateModelSettings({ repeatPenaltyTokens })}
                      />
                    </SettingsRow>
                    <SettingsRow label="GPU Layers" description="Number of model layers to load into VRAM. Use -1 for all.">
                      <NumberField
                        ariaLabel="GPU layers"
                        min={-1}
                        max={999}
                        step={1}
                        value={activeModelSettings.gpuLayers}
                        onChange={(gpuLayers) => updateModelSettings({ gpuLayers })}
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
                <h2>No Chat Model Installed</h2>
                <p>Download a chat model before changing model-specific prompts and generation settings.</p>
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
  onChange
}: {
  ariaLabel: string
  checked: boolean
  onChange: (checked: boolean) => void
}) {
  return (
    <input
      aria-label={ariaLabel}
      checked={checked}
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
