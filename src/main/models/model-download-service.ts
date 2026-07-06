import { app, BrowserWindow } from 'electron'
import { createWriteStream, existsSync, mkdirSync, statSync, type WriteStream } from 'node:fs'
import { rename, rm } from 'node:fs/promises'
import http, { type ClientRequest, type IncomingMessage } from 'node:http'
import https from 'node:https'
import { join } from 'node:path'
import { URL } from 'node:url'
import type { LocalModel, ModelDownloadProgress } from '../../shared/app-state'
import { catalogItemToLocalModel, type ModelCatalogItem } from '../../shared/model-catalog'

interface ActiveModelDownload {
  cancelled: boolean
  item: ModelCatalogItem
  modelId: string
  reject: (error: Error) => void
  request?: ClientRequest
  stream?: WriteStream
}

const activeDownloads = new Map<string, ActiveModelDownload>()

function modelDirectory(): string {
  return join(app.getPath('userData'), 'models')
}

function assertSafeFilename(filename: string): void {
  if (!filename || filename.includes('/') || filename.includes('\\')) {
    throw new Error('Model filename is invalid.')
  }
}

function modelPath(filename: string): string {
  assertSafeFilename(filename)
  return join(modelDirectory(), filename)
}

function incompleteModelPath(filename: string): string {
  return `${modelPath(filename)}.part`
}

function emitProgress(progress: ModelDownloadProgress): void {
  for (const window of BrowserWindow.getAllWindows()) {
    window.webContents.send('models:download-progress', progress)
  }
}

function progressPercent(bytesReceived: number, bytesTotal?: number): number {
  if (!bytesTotal || bytesTotal <= 0) {
    return 0
  }

  return Math.max(0, Math.min(100, Math.round((bytesReceived / bytesTotal) * 100)))
}

function parseContentLength(response: IncomingMessage): number | undefined {
  const rawLength = response.headers['content-length']
  const contentLength = Number(Array.isArray(rawLength) ? rawLength[0] : rawLength)
  return Number.isFinite(contentLength) && contentLength > 0 ? contentLength : undefined
}

function parseContentRangeTotal(response: IncomingMessage): number | undefined {
  const rawRange = response.headers['content-range']
  const range = Array.isArray(rawRange) ? rawRange[0] : rawRange
  const match = range?.match(/\/(\d+)$/)
  const total = Number(match?.[1])
  return Number.isFinite(total) && total > 0 ? total : undefined
}

function createDownloadedModel(item: ModelCatalogItem, modelId: string, path: string): LocalModel {
  return catalogItemToLocalModel(item, modelId, path, new Date().toISOString(), 'ready')
}

function emitExistingProgress(
  item: ModelCatalogItem,
  modelId: string,
  status: ModelDownloadProgress['status'],
  bytesReceived: number,
  error?: string
): void {
  emitProgress({
    modelId,
    catalogId: item.id,
    filename: item.filename,
    status,
    percent: progressPercent(bytesReceived, item.sizeBytes),
    bytesReceived,
    bytesTotal: item.sizeBytes,
    path: status === 'complete' ? modelPath(item.filename) : undefined,
    message: status === 'complete' ? 'Downloaded' : status === 'incomplete' ? 'Download paused' : undefined,
    error
  })
}

export async function downloadModelFromCatalog(item: ModelCatalogItem, modelId: string): Promise<LocalModel> {
  assertSafeFilename(item.filename)
  mkdirSync(modelDirectory(), { recursive: true })

  const finalPath = modelPath(item.filename)
  const partialPath = incompleteModelPath(item.filename)

  if (existsSync(finalPath)) {
    const model = createDownloadedModel(item, modelId, finalPath)
    emitExistingProgress(item, modelId, 'complete', model.sizeBytes ?? item.sizeBytes)
    return model
  }

  if (activeDownloads.has(item.filename)) {
    throw new Error('This model is already downloading.')
  }

  let bytesReceived = existsSync(partialPath) ? statSync(partialPath).size : 0
  let bytesTotal = item.sizeBytes
  let lastSpeedAt = Date.now()
  let lastSpeedBytes = bytesReceived
  let settled = false

  return new Promise<LocalModel>((resolve, reject) => {
    const activeDownload: ActiveModelDownload = {
      cancelled: false,
      item,
      modelId,
      reject: (error) => {
        if (settled) {
          return
        }

        settled = true
        reject(error)
      }
    }

    activeDownloads.set(item.filename, activeDownload)

    const cleanup = () => {
      activeDownloads.delete(item.filename)
      activeDownload.stream = undefined
      activeDownload.request = undefined
    }

    const rejectDownload = (error: Error) => {
      cleanup()
      emitProgress({
        modelId,
        catalogId: item.id,
        filename: item.filename,
        status: activeDownload.cancelled ? 'incomplete' : 'error',
        percent: progressPercent(bytesReceived, bytesTotal),
        bytesReceived,
        bytesTotal,
        error: activeDownload.cancelled ? undefined : error.message,
        message: activeDownload.cancelled ? 'Download paused' : 'Download failed'
      })
      activeDownload.reject(error)
    }

    const resolveDownload = (model: LocalModel) => {
      cleanup()
      if (settled) {
        return
      }

      settled = true
      resolve(model)
    }

    const sendProgress = (force = false) => {
      const now = Date.now()
      if (!force && now - lastSpeedAt < 500) {
        return
      }

      const elapsedSeconds = Math.max((now - lastSpeedAt) / 1000, 0.001)
      const speedBytesPerSecond = Math.max(0, Math.round((bytesReceived - lastSpeedBytes) / elapsedSeconds))
      lastSpeedAt = now
      lastSpeedBytes = bytesReceived

      emitProgress({
        modelId,
        catalogId: item.id,
        filename: item.filename,
        status: 'downloading',
        percent: progressPercent(bytesReceived, bytesTotal),
        bytesReceived,
        bytesTotal,
        speedBytesPerSecond,
        message: bytesReceived > 0 ? 'Downloading' : 'Starting download'
      })
    }

    const finishDownload = () => {
      void rename(partialPath, finalPath)
        .then(() => {
          const model = createDownloadedModel(item, modelId, finalPath)
          emitProgress({
            modelId,
            catalogId: item.id,
            filename: item.filename,
            status: 'complete',
            percent: 100,
            bytesReceived: bytesTotal || bytesReceived,
            bytesTotal: bytesTotal || bytesReceived,
            path: finalPath,
            message: 'Downloaded'
          })
          resolveDownload(model)
        })
        .catch((error: unknown) => {
          rejectDownload(error instanceof Error ? error : new Error('Could not finish model download.'))
        })
    }

    const startRequest = (url: string, redirectCount = 0) => {
      if (redirectCount > 5) {
        rejectDownload(new Error('Too many redirects while downloading the model.'))
        return
      }

      const parsedUrl = new URL(url)
      const transport = parsedUrl.protocol === 'http:' ? http : https
      const headers: Record<string, string> = {
        'Accept-Encoding': 'identity'
      }

      if (bytesReceived > 0) {
        headers.Range = `bytes=${bytesReceived}-`
      }

      sendProgress(true)

      const request = transport.get(parsedUrl, { headers }, (response) => {
        const statusCode = response.statusCode ?? 0
        const location = response.headers.location

        if ([301, 302, 303, 307, 308].includes(statusCode) && location) {
          response.resume()
          startRequest(new URL(location, parsedUrl).toString(), redirectCount + 1)
          return
        }

        if (statusCode !== 200 && statusCode !== 206) {
          response.resume()
          rejectDownload(new Error(`Model download failed with HTTP ${statusCode}.`))
          return
        }

        const contentLength = parseContentLength(response)
        const contentRangeTotal = parseContentRangeTotal(response)
        const shouldAppend = statusCode === 206 && bytesReceived > 0

        if (!shouldAppend && bytesReceived > 0) {
          bytesReceived = 0
          lastSpeedBytes = 0
        }

        bytesTotal = contentRangeTotal ?? (contentLength ? bytesReceived + contentLength : item.sizeBytes)

        const stream = createWriteStream(partialPath, { flags: shouldAppend ? 'a' : 'w' })
        activeDownload.stream = stream

        response.on('data', (chunk: Buffer) => {
          bytesReceived += chunk.length
          sendProgress()
        })

        response.once('error', (error) => {
          rejectDownload(error)
        })

        stream.once('error', (error) => {
          rejectDownload(error)
        })

        stream.once('finish', () => {
          if (activeDownload.cancelled) {
            return
          }

          finishDownload()
        })

        response.pipe(stream)
      })

      request.once('error', (error) => {
        rejectDownload(activeDownload.cancelled ? new Error('Download cancelled.') : error)
      })

      activeDownload.request = request
    }

    startRequest(item.url)
  })
}

export async function cancelModelDownload(filename: string): Promise<void> {
  assertSafeFilename(filename)
  const activeDownload = activeDownloads.get(filename)

  if (!activeDownload) {
    const partialPath = incompleteModelPath(filename)
    const bytesReceived = existsSync(partialPath) ? statSync(partialPath).size : 0

    if (bytesReceived > 0) {
      emitExistingProgress(
        { id: filename, name: filename, filename, sizeBytes: bytesReceived, ramRequiredGb: 0, parameters: '', quant: '', type: '', description: [], url: '' },
        filename,
        'incomplete',
        bytesReceived
      )
    }
    return
  }

  activeDownload.cancelled = true
  activeDownload.request?.destroy(new Error('Download cancelled.'))
  activeDownload.stream?.destroy()
  activeDownloads.delete(filename)

  const partialPath = incompleteModelPath(filename)
  const bytesReceived = existsSync(partialPath) ? statSync(partialPath).size : 0
  emitExistingProgress(activeDownload.item, activeDownload.modelId, 'incomplete', bytesReceived)
  activeDownload.reject(new Error('Download cancelled.'))
}

export async function removeDownloadedModel(model: LocalModel): Promise<void> {
  if (model.engine === 'remote' || model.source === 'remote') {
    return
  }

  const filename = model.filename ?? model.path?.split(/[\\/]/).pop()

  if (!filename) {
    throw new Error('Model filename is missing.')
  }

  assertSafeFilename(filename)

  if (activeDownloads.has(filename)) {
    await cancelModelDownload(filename)
  }

  await Promise.all([
    rm(modelPath(filename), { force: true }),
    rm(incompleteModelPath(filename), { force: true })
  ])

  emitProgress({
    modelId: model.id,
    catalogId: model.catalogId,
    filename,
    status: 'removed',
    percent: 0,
    bytesReceived: 0,
    bytesTotal: model.sizeBytes,
    message: 'Removed'
  })
}
