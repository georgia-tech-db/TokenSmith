import assert from 'node:assert/strict'
import { execFileSync } from 'node:child_process'
import { existsSync, readFileSync } from 'node:fs'
import { join } from 'node:path'
const app = join(process.cwd(), 'release', 'mas-unsigned', 'TokenSmith.app')
const resources = join(app, 'Contents', 'Resources', 'app')
const binaryArch = process.arch === 'x64' ? 'x86_64' : 'arm64'
for (const binary of [join(app, 'Contents', 'MacOS', 'TokenSmith'), join(resources, 'app_runtime', 'python', 'bin', 'python')]) {
  execFileSync('/usr/bin/lipo', [binary, '-verify_arch', binaryArch])
}
assert.equal(existsSync(join(app, 'Contents', 'Frameworks', 'Squirrel.framework')), false, 'Must use MAS Electron, not the direct-download runtime')
assert.ok(existsSync(join(resources, 'out', 'preload', 'index.js')))
assert.ok(readFileSync(join(resources, 'out', 'main', 'index.js'), 'utf8').includes('process.mas'))
const runtime = join(resources, 'app_runtime', 'python')
execFileSync(join(runtime, 'bin', 'python'), ['-c', 'import faiss, numpy, pypdfium2, PIL, onnxruntime, cv2; from rapidocr import RapidOCR; print("Packaged Python imports passed")'], {
  env: { ...process.env, PYTHONHOME: runtime, PYTHONDONTWRITEBYTECODE: '1', DYLD_FALLBACK_LIBRARY_PATH: join(runtime, 'lib') }, stdio: 'inherit'
})
console.log('MAS structure verified. This unsigned check does not validate sandbox execution or App Store eligibility.')
