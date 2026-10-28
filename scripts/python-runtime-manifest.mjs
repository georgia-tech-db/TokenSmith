// Pinned install_only_stripped archives published by Astral's
// python-build-standalone project. Update the filename and SHA-256 together.
const releaseTag = '20260901'
const releaseBaseUrl = `https://github.com/astral-sh/python-build-standalone/releases/download/${releaseTag}`

const runtimes = {
  'linux-x64': {
    filename: 'cpython-3.12.14+20260901-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz',
    sha256: '72748da13197c1fb161e3afeef20a6a385ff24f2165e6e2758e47008e7faba4c'
  },
  'darwin-arm64': {
    filename: 'cpython-3.12.14+20260901-aarch64-apple-darwin-install_only_stripped.tar.gz',
    sha256: '81a359f1cfadd4da11766534c5913791cea55f26e1bb902cacd2a531bb1e4b2b'
  },
  'win32-x64': {
    filename: 'cpython-3.12.14+20260901-x86_64-pc-windows-msvc-install_only_stripped.tar.gz',
    sha256: '7c45c9622400d578709a9b2cddbe8124cc21d382409d9f13406d706d28e31b14'
  }
}

export function runtimeKey(platform = process.platform, arch = process.arch) {
  return `${platform}-${arch}`
}

export function standaloneRuntime(platform = process.platform, arch = process.arch) {
  const key = runtimeKey(platform, arch)
  const runtime = runtimes[key]

  if (!runtime) {
    const supported = Object.keys(runtimes).join(', ')
    throw new Error(`TokenSmith does not provide a Python runtime for ${key}. Supported targets: ${supported}.`)
  }

  return {
    ...runtime,
    key,
    pythonVersion: '3.12.14',
    releaseTag,
    url: `${releaseBaseUrl}/${runtime.filename.replace('+', '%2B')}`
  }
}

// The NLI cross-encoder that checks whether retrieved passages entail an
// answer's claims. Pinned to a Hugging Face revision; update the revision and
// every SHA-256 together.
const nliModelRepo = 'Xenova/deberta-v3-base-tasksource-nli'
const nliModelRevision = 'b2f7405d6a3d25f3fcce84c5cdbd54eac6587578'

export const nliModel = {
  name: 'deberta-v3-base-tasksource-nli',
  files: [
    { path: 'config.json', sha256: 'a3b1f85299f8c6b1ecf57e73a58bd3d136af6e3b40a9f95917a0d7b9dfdf61dd' },
    { path: 'tokenizer.json', sha256: 'fa89bd3f10b58dfd80d4d673771fb24a40386d53053e9dbb4a08976c069a00f1' },
    { path: 'onnx/model_quantized.onnx', sha256: '442bf6ec019b56558029cf301fce36e50bda4cb3ce332a98dfe2748d6a4ad3e0' }
  ].map((file) => ({
    ...file,
    url: `https://huggingface.co/${nliModelRepo}/resolve/${nliModelRevision}/${file.path}`
  }))
}
