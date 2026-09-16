import { randomUUID } from 'node:crypto'
import { mkdir, readFile, rename, writeFile } from 'node:fs/promises'
import { dirname } from 'node:path'
import type { LocalModel } from '../../shared/app-state'
import { remoteProviderCatalog } from '../../shared/model-providers'
import { isCloudGenerator } from '../../shared/cloud-generators'
import type { CloudConnection, CloudConnectionInput, CloudConnectionStatus, CloudGeneratorInput, CloudResult, CloudSetupError } from '../../shared/cloud-generators'
import { remoteChatParameters } from './remote-chat-parameters'

interface Encryption {
  isEncryptionAvailable(): boolean
  getSelectedStorageBackend?(): string
  encryptString(value: string): Buffer
  decryptString(value: Buffer): string
}
interface StoredConnection extends Omit<CloudConnection, 'connected'> { encryptedKey?: string }

class SetupError extends Error {
  constructor(public code: CloudSetupError['code'], message: string) { super(message) }
}

export async function cloudResult<T>(work: () => Promise<T>): Promise<CloudResult<T>> {
  try { return { ok: true, value: await work() } }
  catch (error) {
    return { ok: false, error: error instanceof SetupError
      ? { code: error.code, message: error.message }
      : { code: 'network', message: 'Could not complete the connection. Please try again.' } }
  }
}

export function cloudChatModelIds(data: unknown, providerId: string): string[] {
  const rows = (data as { data?: Array<{ id?: string; capabilities?: { completion_chat?: boolean } }> })?.data
  if (!Array.isArray(rows)) return []
  return [...new Set(rows.filter(row => row.capabilities?.completion_chat !== false).map(row => {
    const id = typeof row.id === 'string' ? row.id.trim() : ''
    return providerId === 'gemini' ? id.replace(/^models\//, '') : id
  }).filter(id => id && !/(embed|moderation|whisper|transcri|\btts\b|dall-e|image|realtime|audio|video|sora|rerank)/i.test(id))
    // Gemini's catalog also contains agents, music, video and live-only models.
    // Keep ordinary Gemini/Gemma text models here; unusual endpoints can use manual entry.
    .filter(id => providerId !== 'gemini' || (/^(gemini-|gemma-)/.test(id) && !/(live|robotics|computer-use|customtools)/i.test(id))))]
    .sort((a, b) => a.localeCompare(b, undefined, { numeric: true }))
}

export class CloudGeneratorService {
  private records = new Map<string, StoredConnection>()
  private keys = new Map<string, string>()
  private unreadableStore = false
  private pendingWrite = Promise.resolve()
  private requests = new Map<string, AbortController>()

  cancel(requestId: string): void { this.requests.get(requestId)?.abort() }

  private async withRequest<T>(input: CloudConnectionInput, work: (signal: AbortSignal) => Promise<T>): Promise<T> {
    const id = input.requestId || randomUUID()
    const controller = new AbortController()
    this.requests.set(id, controller)
    try { return await work(controller.signal) }
    finally { this.requests.delete(id) }
  }

  constructor(private file: string, private encryption: Encryption, private request: typeof fetch = fetch) {}

  secureStorageAvailable(): boolean {
    try {
      return !this.unreadableStore && this.encryption.isEncryptionAvailable() &&
        this.encryption.getSelectedStorageBackend?.() !== 'basic_text'
    } catch { return false }
  }

  async initialize(): Promise<void> {
    try {
      const saved = JSON.parse(await readFile(this.file, 'utf8')) as { version: number; connections: StoredConnection[] }
      if (saved.version !== 1 || !Array.isArray(saved.connections)) throw new Error('Invalid store')
      for (const record of saved.connections) {
        this.records.set(record.id, record)
        try {
          if (record.encryptedKey && this.secureStorageAvailable()) {
            this.keys.set(record.id, this.encryption.decryptString(Buffer.from(record.encryptedKey, 'base64')))
          }
        } catch { /* Preserve the connection and offer repair if secure storage is locked. */ }
      }
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== 'ENOENT') this.unreadableStore = true
    }
  }

  status(): CloudConnectionStatus {
    return { secureStorageAvailable: this.secureStorageAvailable(), connections: [...this.records.values()].map(record => ({
      id: record.id, providerId: record.providerId, baseUrl: record.baseUrl,
      remembered: record.remembered, connected: this.keys.has(record.id)
    })) }
  }

  credentialFor(model: LocalModel): string | undefined {
    if (!isCloudGenerator(model) || !model.connectionId) return undefined
    const record = this.records.get(model.connectionId)
    if (!record || record.providerId !== model.providerId || record.baseUrl !== model.baseUrl?.replace(/\/+$/, '')) return undefined
    return this.keys.get(record.id)
  }

  describeModel(model: LocalModel): LocalModel {
    if (!isCloudGenerator(model) || !model.connectionId) return model
    const connected = Boolean(this.credentialFor(model))
    return { ...model, cloudCredentialStatus: connected ? 'connected' : 'reconnect', status: connected ? 'ready' : 'needsRuntime' }
  }

  private resolve(input: CloudConnectionInput) {
    const provider = remoteProviderCatalog.find(provider => provider.id === input.providerId)
    if (!provider) throw new SetupError('configuration', 'Choose a supported service.')
    const record = input.connectionId ? this.records.get(input.connectionId) : undefined
    const baseUrl = (provider.baseUrl ?? input.baseUrl ?? record?.baseUrl ?? '').trim().replace(/\/+$/, '')
    try {
      const url = new URL(baseUrl)
      const local = ['localhost', '127.0.0.1', '[::1]'].includes(url.hostname)
      if (url.username || url.password || url.search || url.hash ||
          (url.protocol !== 'https:' && !(local && url.protocol === 'http:'))) throw new Error('Invalid URL')
    } catch { throw new SetupError('configuration', 'Enter an HTTPS API address, or an HTTP address on this device.') }
    if (record && (record.providerId !== provider.id || record.baseUrl !== baseUrl)) {
      throw new SetupError('configuration', 'This saved connection belongs to another service. Enter a new API key.')
    }
    const apiKey = input.apiKey?.trim() || (record && this.keys.get(record.id))
    if (!apiKey) throw new SetupError('credentials', `Paste your ${provider.name} API key to continue.`)
    return { provider, baseUrl, apiKey, record }
  }

  private async json(url: string, apiKey: string, body?: object, signal?: AbortSignal): Promise<unknown> {
    const service = remoteProviderCatalog.find(p => p.baseUrl && url.startsWith(p.baseUrl))?.name || 'the service'
    const operation = body ? 'chat model' : 'model list'
    try {
      const response = await this.request(url, {
        method: body ? 'POST' : 'GET', redirect: 'error', signal: signal ? AbortSignal.any([signal, AbortSignal.timeout(30_000)]) : AbortSignal.timeout(30_000),
        headers: { Authorization: `Bearer ${apiKey}`, 'Content-Type': 'application/json', Accept: 'application/json' },
        ...(body ? { body: JSON.stringify(body) } : {})
      })
      if (!response.ok) {
        const detail = (await response.text()).slice(0, 8000)
        if (response.status === 401 || /API_KEY_INVALID|invalid_api_key/i.test(detail)) throw new SetupError('credentials', 'This API key was not accepted. Check it or create a new key.')
        if (response.status === 402 || /insufficient_quota|billing|credit balance|quota.*exceed|quota.*exhaust/i.test(detail)) throw new SetupError('quota', 'This account has no API usage available. Check billing or quota with your service, then retry.')
        if (response.status === 429) throw new SetupError('rate_limit', 'The service is receiving too many requests. Wait a moment and retry.')
        if (response.status === 403) throw new SetupError('access', 'This account does not have access. Check its permissions or choose another model.')
        if (response.status === 404) throw new SetupError('model', body
          ? 'This model is no longer available or is not available to your account. Choose another chat model.'
          : `Could not load the ${service} model list. Retry, or enter a model name manually.`)
        if (response.status === 400 || response.status === 422) throw new SetupError('model', `The service could not use this ${operation}. Check your choices and try again.`)
        throw new SetupError('network', 'The service is unavailable right now. Try again shortly.')
      }
      try { return await response.json() }
      catch { throw new SetupError('model', `The service returned an unreadable ${operation}. Retry or check the API address.`) }
    } catch (error) {
      if (error instanceof SetupError) throw error
      const failure = error as { name?: string; code?: string; cause?: { code?: string } }
      const code = failure.cause?.code || failure.code || ''
      const timeout = failure.name === 'TimeoutError' || /TIMEOUT|TIMEDOUT/.test(code)
      const certificate = /CERT|TLS|UNABLE_TO_VERIFY_LEAF_SIGNATURE/.test(code)
      throw new SetupError('network', timeout
        ? `${service} did not respond in time. Retry the connection.`
        : certificate
          ? `TokenSmith could not verify ${service}'s secure connection. Check your device's certificate or proxy settings, then retry.`
          : `Could not reach ${service}. Check your connection or proxy settings, then retry.`)
    }
  }

  async discover(input: CloudConnectionInput): Promise<string[]> {
    const { provider, baseUrl, apiKey } = this.resolve(input)
    return this.withRequest(input, async signal => cloudChatModelIds(await this.json(`${baseUrl}/models`, apiKey, undefined, signal), provider.id))
  }

  async connect(input: CloudGeneratorInput): Promise<LocalModel> {
    return this.withRequest(input, signal => this.connectRequest(input, signal))
  }

  private async connectRequest(input: CloudGeneratorInput, signal: AbortSignal): Promise<LocalModel> {
    const { provider, baseUrl, apiKey, record } = this.resolve(input)
    const modelName = input.modelName.trim().replace(provider.id === 'gemini' ? /^models\// : /$^/, '')
    if (!modelName || modelName.length > 240) throw new SetupError('model', 'Choose a chat model to continue.')
    if (input.remember && !this.secureStorageAvailable()) throw new SetupError('storage', 'Secure storage is unavailable. Unlock it or choose “This session only”.')
    const response = await this.json(`${baseUrl}/chat/completions`, apiKey, {
      model: modelName, messages: [{ role: 'user', content: 'Reply with the word OK.' }],
      ...remoteChatParameters(baseUrl, modelName, 512)
    }, signal) as { choices?: Array<{ message?: { content?: string }; text?: string }> }
    signal.throwIfAborted()
    const content = response.choices?.[0]?.message?.content ?? response.choices?.[0]?.text
    if (typeof content !== 'string' || !content.trim()) throw new SetupError('model', 'The model did not return a text answer. Choose another chat model or retry.')

    const id = record?.id ?? randomUUID()
    const operation = this.pendingWrite.then(async () => {
      signal.throwIfAborted()
      const next: StoredConnection = { id, providerId: provider.id, baseUrl, remembered: input.remember }
      try {
        if (input.remember) next.encryptedKey = this.encryption.encryptString(apiKey).toString('base64')
        const records = new Map(this.records).set(id, next)
        // Session-only credentials never reach disk. Preserve unrelated encrypted records.
        if (!this.unreadableStore && (input.remember || record?.remembered)) {
          await mkdir(dirname(this.file), { recursive: true })
          signal.throwIfAborted()
          const temp = `${this.file}.${randomUUID()}.tmp`
          await writeFile(temp, JSON.stringify({ version: 1, connections: [...records.values()].filter(r => r.remembered) }), { mode: 0o600 })
          await rename(temp, this.file)
        }
        this.records = records
        this.keys.set(id, apiKey)
      } catch { throw new SetupError('storage', 'Could not save the connection securely. Retry or choose “This session only”.') }
    })
    this.pendingWrite = operation.catch(() => {})
    await operation
    return { id: input.modelId || `cloud:${id}:${modelName}`, name: `${provider.name} ${modelName}`,
      engine: 'remote', source: 'remote', role: 'generator', status: 'ready',
      providerId: provider.id, providerName: provider.name, baseUrl, remoteModelName: modelName,
      connectionId: id, cloudCredentialStatus: 'connected', addedAt: new Date().toISOString() }
  }
}
