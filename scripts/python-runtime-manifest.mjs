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
