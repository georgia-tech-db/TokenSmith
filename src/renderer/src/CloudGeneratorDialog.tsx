import { useEffect, useRef, useState } from 'react'
import { ArrowLeft, Check, ChevronRight, Cloud, ExternalLink, Eye, EyeOff, KeyRound, Loader2, LockKeyhole, Search, X } from 'lucide-react'
import type { LocalModel } from '@shared/app-state'
import { remoteProviderCatalog, type RemoteProviderCatalogItem } from '@shared/model-providers'
import type { CloudConnection, CloudConnectionStatus, CloudSetupError } from '@shared/cloud-generators'
import geminiLogo from './assets/provider-gemini.svg'
import openaiLogo from './assets/provider-openai.svg'
import groqLogo from './assets/provider-groq.svg'
import mistralLogo from './assets/provider-mistral.svg'
import customLogo from './assets/provider-custom.svg'

const providerLabels: Record<string, string> = { gemini: 'Google Gemini', openai: 'OpenAI', groq: 'Groq', mistral: 'Mistral', custom: 'Custom connection' }
const providerCopy: Record<string, string> = { gemini: 'Connect with a Google AI Studio key', openai: 'Connect with an OpenAI API key', groq: 'Hosted open models with a Groq key', mistral: 'Connect with a Mistral API key', custom: 'An OpenAI-compatible service' }
const providerLogos: Record<string, string> = { gemini: geminiLogo, openai: openaiLogo, groq: groqLogo, mistral: mistralLogo, custom: customLogo }

function secureStorageSteps(platform: string): string[] {
  if (platform === 'darwin') {
    return [
      'Open Keychain Access and unlock your login keychain.',
      'Allow TokenSmith to use the keychain if macOS asks.',
      'Restart TokenSmith and open this connection again.'
    ]
  }
  if (platform === 'win32') {
    return [
      'Sign in with your normal Windows account; no additional password manager is required.',
      'Restart TokenSmith and open this connection again.',
      'If storage remains unavailable on a managed device, ask your administrator whether Windows credential protection is disabled.'
    ]
  }
  return [
    'Install GNOME Keyring with libsecret support, or KWallet, using your system software manager.',
    'Open and unlock the keyring or wallet, and configure it to unlock when you sign in.',
    'Restart TokenSmith and open this connection again.'
  ]
}

export function CloudGeneratorDialog({ model, localSearch, onClose, onConnected }: {
  model?: LocalModel; localSearch: boolean; onClose: () => void; onConnected: (model: LocalModel) => void
}) {
  const [provider, setProvider] = useState<RemoteProviderCatalogItem | undefined>(() => remoteProviderCatalog.find(p => p.id === model?.providerId))
  const [status, setStatus] = useState<CloudConnectionStatus | null>(null)
  const [connection, setConnection] = useState<CloudConnection | undefined>()
  const [apiKey, setApiKey] = useState('')
  const [baseUrl, setBaseUrl] = useState(model?.baseUrl || '')
  const [remember, setRemember] = useState(true)
  const [showKey, setShowKey] = useState(false)
  const [step, setStep] = useState<'key' | 'model'>('key')
  const [models, setModels] = useState<string[]>([])
  const [modelName, setModelName] = useState(model?.remoteModelName || '')
  const [search, setSearch] = useState('')
  const [manual, setManual] = useState(false)
  const [busy, setBusy] = useState<'discover' | 'verify' | null>(null)
  const [error, setError] = useState<CloudSetupError | null>(null)
  const dialog = useRef<HTMLDialogElement>(null)
  const keyField = useRef<HTMLInputElement>(null)
  const requestId = useRef<string | null>(null)
  const live = useRef(true)
  const connected = useRef(false)

  useEffect(() => {
    live.current = true
    const previous = document.activeElement as HTMLElement
    dialog.current?.showModal()
    const bridge = window.tokensmith
    if (bridge?.getCloudConnections) {
      void bridge.getCloudConnections().then(value => {
        if (!live.current) return
        setStatus(value)
        const saved = value.connections.find(c => c.id === model?.connectionId)
        setConnection(saved)
        setRemember(value.secureStorageAvailable && (saved?.remembered ?? true))
      }).catch(() => { if (live.current) setError({ code: 'storage', message: 'Could not read saved connections. Close this sheet and try again.' }) })
    } else setError({ code: 'configuration', message: 'Open the TokenSmith desktop app to connect a cloud model.' })
    return () => {
      live.current = false
      if (requestId.current) void window.tokensmith?.cancelCloudSetup?.(requestId.current)
      dialog.current?.close()
      if (connected.current) requestAnimationFrame(() => document.querySelector<HTMLElement>('textarea[aria-label="Message"]')?.focus())
      else previous?.focus()
    }
  }, [])

  useEffect(() => {
    if (!provider) return
    const field = step === 'key' ? keyField.current : dialog.current?.querySelector<HTMLInputElement>('.cloud-search input, input[aria-label="Cloud model name"]')
    field?.focus()
  }, [provider?.id, step, manual])

  function cancelRequest() {
    if (requestId.current) void window.tokensmith?.cancelCloudSetup?.(requestId.current)
    requestId.current = null
    setBusy(null)
  }
  function chooseProvider(next: RemoteProviderCatalogItem) {
    cancelRequest(); setProvider(next); setStep('key'); setApiKey(''); setShowKey(false); setError(null)
    setModels([]); setModelName(''); setManual(false); setSearch(''); setBaseUrl(next.baseUrl || '')
    const saved = status?.connections.find(c => c.providerId === next.id && c.connected)
    setConnection(saved); setRemember(Boolean(status?.secureStorageAvailable) && (saved?.remembered ?? true))
    if (saved) setBaseUrl(saved.baseUrl)
  }
  function back() {
    cancelRequest(); setError(null)
    if (step === 'model') setStep('key')
    else { setProvider(undefined); setApiKey(''); setShowKey(false); setConnection(undefined) }
  }
  function changeKey() { setConnection(current => current ? { ...current, connected: false } : undefined); setApiKey(''); setError(null); requestAnimationFrame(() => keyField.current?.focus()) }
  async function discover() {
    if (!provider || !window.tokensmith?.discoverCloudModels) return
    const id = crypto.randomUUID(); requestId.current = id; setBusy('discover'); setError(null)
    try {
      const result = await window.tokensmith.discoverCloudModels({ requestId: id, providerId: provider.id, connectionId: connection?.id,
        apiKey, baseUrl: provider.baseUrl || baseUrl })
      if (!live.current || requestId.current !== id) return
      if (!result.ok) { setError(result.error); return }
      setModels(result.value); setStep('model'); setManual(!result.value.length)
      // No arbitrary default: keep a prior choice only if the user already selected it.
      if (!result.value.includes(modelName)) setModelName('')
    } catch { if (live.current && requestId.current === id) setError({ code: 'network', message: 'Could not connect. Please try again.' }) }
    finally { if (live.current && requestId.current === id) { setBusy(null); requestId.current = null } }
  }
  async function connect() {
    if (!provider || !modelName.trim() || !window.tokensmith?.connectCloudGenerator) return
    const id = crypto.randomUUID(); requestId.current = id; setBusy('verify'); setError(null)
    try {
      const result = await window.tokensmith.connectCloudGenerator({ requestId: id, providerId: provider.id, connectionId: connection?.id,
        apiKey, baseUrl: provider.baseUrl || baseUrl, modelName, remember,
        modelId: model?.remoteModelName === modelName && model.providerId === provider.id ? model.id : undefined })
      if (!live.current || requestId.current !== id) return
      if (!result.ok) { setError(result.error); return }
      setApiKey(''); requestId.current = null; connected.current = true; onConnected(result.value)
    } catch { if (live.current && requestId.current === id) setError({ code: 'network', message: 'Could not finish setup. Your current model is still selected. Please retry.' }) }
    finally { if (live.current && requestId.current === id) { setBusy(null); requestId.current = null } }
  }
  const name = provider ? providerLabels[provider.id] : ''
  const canConnect = !!provider && !!status && (!!apiKey.trim() || !!connection?.connected) && (provider.id !== 'custom' || !!baseUrl.trim())
  const choices = models.filter(id => id.toLowerCase().includes(search.toLowerCase()))
  const saved = status?.connections.filter(c => c.providerId === provider?.id) || []

  return <dialog ref={dialog} className="cloud-generator-dialog" aria-labelledby="cloud-dialog-title" onCancel={event => { event.preventDefault(); onClose() }}>
    <header className="cloud-dialog-header">
      {provider ? <button className="cloud-icon-button" type="button" aria-label="Back" onClick={back}><ArrowLeft size={19} /></button> : <span className="cloud-dialog-symbol"><Cloud size={21} /></span>}
      <span className="cloud-eyebrow">Cloud chat</span>
      <button className="cloud-icon-button" type="button" aria-label="Close cloud setup" onClick={onClose}><X size={20} /></button>
    </header>
    <div className="cloud-dialog-body">
      <h2 id="cloud-dialog-title">{!provider ? 'Bring another model to your study session' : step === 'key' ? `Connect ${name}` : 'Choose your chat model'}</h2>
      <p className="cloud-intro">{!provider ? 'Choose a service you use. Connect once, then switch models right here in chat.' : step === 'key' ? 'Use your own API key to connect your account.' : `Choose a model available through your ${name} account.`}</p>
      {!provider ? <div className="cloud-provider-list">
        {['gemini', 'openai', 'groq', 'mistral', 'custom'].map(id => {
          const item = remoteProviderCatalog.find(p => p.id === id)!
          return <button type="button" key={id} className={`cloud-provider-choice provider-${id}`} onClick={() => chooseProvider(item)}>
            <span className="cloud-provider-mark" aria-hidden="true"><img src={providerLogos[id]} alt="" /></span>
            <span><strong>{providerLabels[id]}</strong><small>{providerCopy[id]}</small></span><ChevronRight size={17} />
          </button>
        })}
      </div> : <form onSubmit={event => { event.preventDefault(); void (step === 'key' ? discover() : connect()) }}>
        {step === 'key' ? <>
          {saved.length > 1 && <label className="cloud-field">Saved connection<select disabled={!!busy} value={connection?.id || ''} onChange={e => {
            const next = saved.find(c => c.id === e.target.value); setConnection(next); setApiKey(''); setShowKey(false); setBaseUrl(next?.baseUrl || provider.baseUrl || ''); setRemember(Boolean(status?.secureStorageAvailable) && (next?.remembered ?? true)); setError(null)
          }}><option value="">Use a new API key</option>{saved.map((c, i) => <option key={c.id} value={c.id}>{name} connection {i + 1}{c.connected ? '' : ' · Reconnect'}</option>)}</select></label>}
          {provider.isCustom && <label className="cloud-field">API address<input type="url" value={baseUrl} disabled={!!busy} onChange={e => { setBaseUrl(e.target.value); setConnection(undefined); setError(null) }} placeholder="https://your-service.example/v1" required /></label>}
          {connection?.connected ? <div className="cloud-saved-connection"><Check size={18} /><span>Using your saved connection</span><button type="button" disabled={!!busy} onClick={changeKey}>Change key</button></div> : <label className="cloud-field">
            <span className="cloud-field-heading">API key {provider.apiKeyUrl && <a href={provider.apiKeyUrl} target="_blank" rel="noreferrer">Get an API key <ExternalLink size={12} /></a>}</span>
            <span className="cloud-key-input"><KeyRound size={17} /><input ref={keyField} type={showKey ? 'text' : 'password'} value={apiKey} disabled={!!busy} autoComplete="off" spellCheck={false} autoCapitalize="none" aria-label={`${name} API key`} placeholder="Paste your API key" onChange={e => { setApiKey(e.target.value); setError(null) }} /><button className="cloud-icon-button" type="button" aria-label={showKey ? 'Hide API key' : 'Show API key'} onClick={() => setShowKey(!showKey)}>{showKey ? <EyeOff size={17} /> : <Eye size={17} />}</button></span>
          </label>}
          <label className="cloud-field">Save connection<select value={remember ? 'remember' : 'session'} disabled={!!busy} onChange={e => { setRemember(e.target.value === 'remember'); setError(null) }}>
            <option value="remember" disabled={!status?.secureStorageAvailable}>Remember securely on this device</option><option value="session">This session only</option>
          </select></label>
          {!status?.secureStorageAvailable && status && <div className="cloud-storage-help">
            <p>Secure storage is unavailable. You can connect for this session; the API key will be forgotten when TokenSmith closes.</p>
            <details>
              <summary>How to enable secure storage</summary>
              <ol>{secureStorageSteps(window.tokensmith?.platform || '').map(step => <li key={step}>{step}</li>)}</ol>
            </details>
          </div>}
          <p className="cloud-billing">API usage is billed by your service. A chat subscription may not include API credits.</p>
        </> : <>
          {!manual && <>
            <label className="cloud-search"><Search size={17} /><input aria-label="Search available cloud models" placeholder="Search available models…" value={search} disabled={!!busy} onChange={e => setSearch(e.target.value)} /></label>
            <div className="cloud-model-list" role="radiogroup" aria-label="Available chat models">
              {choices.map(id => <label key={id} className={`cloud-model-choice ${modelName === id ? 'is-selected' : ''}`}><input type="radio" name="cloud-model" value={id} checked={modelName === id} disabled={!!busy} onChange={() => { setModelName(id); setError(null) }} /><span>{id}</span>{modelName === id && <Check size={16} />}</label>)}
              {!choices.length && <p className="cloud-muted">No matching models.</p>}
            </div>
          </>}
          {manual && <label className="cloud-field">Model name<input aria-label="Cloud model name" value={modelName} disabled={!!busy} onChange={e => { setModelName(e.target.value); setError(null) }} placeholder="Exact model name from your service" /></label>}
          <button type="button" className="cloud-text-button" disabled={!!busy} onClick={() => { setManual(!manual); setModelName(''); setError(null) }}>{manual && models.length ? 'Choose from available models' : 'Enter a model name manually'}</button>
          <p className="cloud-billing">We’ll send a short connection test before switching models. This may use a small amount of API credit.</p>
        </>}
        {error && <div className="cloud-error" role="alert"><p>{error.message}</p>
          {step === 'key' && error.code === 'model' && <button type="button" onClick={() => { setStep('model'); setManual(true); setError(null) }}>Enter a model name instead</button>}
          {step === 'model' && error.code === 'credentials' && <button type="button" onClick={() => { setStep('key'); changeKey() }}>Update API key</button>}
          {error.code === 'storage' && remember && <button type="button" onClick={() => { setRemember(false); setError(null) }}>Use this session only</button>}
        </div>}
        <div className="cloud-disclosure"><LockKeyhole size={16} /><p>Chat messages and relevant document excerpts are sent to {name}.{localSearch ? ' PDF search stays on this device.' : ''}</p></div>
        <button className="cloud-primary-button" type="submit" disabled={!!busy || (step === 'key' ? !canConnect : !modelName.trim())}>
          {busy && <Loader2 size={17} className="spin" />}<span>{busy === 'discover' ? 'Connecting…' : busy === 'verify' ? 'Checking model…' : step === 'key' ? error?.code === 'network' ? 'Retry connection' : 'Connect and find models' : 'Use this model'}</span>{!busy && <ChevronRight size={17} />}
        </button>
      </form>}
      {!provider && error && <p className="cloud-error" role="alert">{error.message}</p>}
      {!provider && <p className="cloud-footer-note">Your current question and sources stay in place.</p>}
    </div>
  </dialog>
}
