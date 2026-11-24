import { existsSync } from 'node:fs'
import { resolve } from 'node:path'

export function macDistribution(args = process.argv.slice(2), env = process.env) {
  const unknown = args.filter(arg => !['--mas', '--mas-unsigned'].includes(arg))
  if (unknown.length || (args.includes('--mas') && args.includes('--mas-unsigned'))) throw new Error('Use --mas or --mas-unsigned, not both.')
  const mas = args.length > 0
  const unsigned = args.includes('--mas-unsigned')
  const type = env.TOKENSMITH_MAS_TYPE || 'development'
  if (mas && !['development', 'distribution'].includes(type)) throw new Error('TOKENSMITH_MAS_TYPE must be development or distribution.')
  const bundleId = env.TOKENSMITH_MAS_BUNDLE_ID || 'ai.tokensmith.desktop'
  if (mas && !/^[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+$/.test(bundleId)) throw new Error('Invalid TOKENSMITH_MAS_BUNDLE_ID.')
  const identity = env.TOKENSMITH_MAS_IDENTITY
  const installerIdentity = env.TOKENSMITH_MAS_INSTALLER_IDENTITY
  const profile = env.TOKENSMITH_MAS_PROFILE && resolve(env.TOKENSMITH_MAS_PROFILE)
  if (mas && !unsigned) {
    if (env.TOKENSMITH_SKIP_CODESIGN === '1') throw new Error('Signed MAS builds cannot skip signing; use --mas-unsigned for packaging checks only.')
    if (!identity || identity === '-' || !profile || !existsSync(profile)) {
      throw new Error('MAS signing requires TOKENSMITH_MAS_IDENTITY and TOKENSMITH_MAS_PROFILE (an existing provisioning profile). Use --mas-unsigned only for packaging checks.')
    }
    if (type === 'distribution' && (!installerIdentity || installerIdentity === '-')) throw new Error('MAS distribution also requires TOKENSMITH_MAS_INSTALLER_IDENTITY for the upload package.')
  }
  return { mas, unsigned, type, identity, installerIdentity, profile, bundleId,
    channel: mas ? `mas-${unsigned ? 'unsigned' : type}` : 'mac' }
}
