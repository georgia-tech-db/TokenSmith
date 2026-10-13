import { createRequire } from 'node:module'
import { existsSync, mkdirSync, readFileSync, renameSync, writeFileSync } from 'node:fs'
import { dirname, join, relative, resolve } from 'node:path'
import ts from 'typescript'

const require = createRequire(import.meta.url)
const runtimeDir = join(process.cwd(), '.coverage-ts-runtime')

export function requireTranspiledTs(sourcePath) {
  mkdirSync(runtimeDir, { recursive: true })

  const staged = new Set()

  function stageTsFile(filePath) {
    const absolutePath = resolve(filePath)
    const outputPath = join(runtimeDir, relative(process.cwd(), absolutePath))

    if (staged.has(absolutePath)) {
      return outputPath
    }

    staged.add(absolutePath)

    const source = readFileSync(absolutePath, 'utf8')
    const output = ts.transpileModule(source, {
      compilerOptions: {
        esModuleInterop: true,
        inlineSourceMap: true,
        inlineSources: true,
        sourceRoot: `${dirname(absolutePath)}/`,
        module: ts.ModuleKind.CommonJS,
        target: ts.ScriptTarget.ES2022
      },
      fileName: absolutePath
    })

    mkdirSync(dirname(outputPath), { recursive: true })
    // Parallel test processes share this cache; readers must never see a partial module.
    const temporaryPath = `${outputPath}.${process.pid}.tmp`
    writeFileSync(temporaryPath, output.outputText, 'utf8')
    renameSync(temporaryPath, outputPath)

    for (const match of output.outputText.matchAll(/require\(["'](\.{1,2}\/[^"']+)["']\)/g)) {
      const dependencyPath = resolve(dirname(absolutePath), `${match[1]}.ts`)
      if (existsSync(dependencyPath)) {
        stageTsFile(dependencyPath)
      }
    }

    return outputPath
  }

  const outputPath = stageTsFile(sourcePath)

  const previousTsLoader = require.extensions['.ts']
  require.extensions['.ts'] = require.extensions['.js']

  try {
    return require(outputPath)
  } finally {
    if (previousTsLoader) {
      require.extensions['.ts'] = previousTsLoader
    } else {
      delete require.extensions['.ts']
    }
  }
}
