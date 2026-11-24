# Mac App Store build

TokenSmith has a separate macOS App Store (MAS) target. The regular DMG remains a direct-download build. This work prepares the package and application behavior; it does not claim App Store approval or a tested Apple-signed sandbox build.

## Model connections

Cloud chat and cloud embeddings already work without Ollama. Book topic generation now uses the same model router as chat, including remembered cloud credentials. In the MAS runtime, setup offers cloud models directly and Ollama is an optional connection to an already-running local server. Users manage Ollama and its models independently.

MAS does not offer installation, launch, service start, model download, pause, or deletion. Main-process guards reject these operations even if invoked through IPC. Disconnecting an Ollama model removes its TokenSmith configuration without deleting it from Ollama. The DMG retains its existing management features. The channel comes from Electron's `process.mas`, not an environment variable or a saved setting.

## Assemble without Apple credentials

Use macOS with Node.js 22.12 or newer and the same architecture as the Python runtime:

```sh
npm ci
npm run setup:python-runtime
npm run package:mas:unsigned
node scripts/verify-mas-package.mjs
```

The packager downloads the MAS distribution for the exact installed Electron version using `@electron/get` (including its checksum verification). It does not replace the Electron runtime used for development or DMG builds. Outputs use `mas-unsigned` in their names and include a build-info JSON stating their limits. They are packaging artifacts, not runnable sandbox tests or upload-ready releases.

The Mac App Store Build workflow assembles ARM64 and x64 packages and checks the included Python libraries. It does not publish releases or submit to Apple.

## Sign for development or distribution

GT must first supply/approve the app identifier, Apple team access, certificate and matching provisioning profile. Do not commit signing materials. Developer ID certificates for the DMG are different from MAS certificates.

For a development build on a Mac registered in the profile:

```sh
export TOKENSMITH_MAS_IDENTITY='Apple Development: YOUR IDENTITY'
export TOKENSMITH_MAS_PROFILE='/absolute/path/development.provisionprofile'
export TOKENSMITH_MAS_BUNDLE_ID='ai.tokensmith.desktop' # replace with the GT-approved ID
npm run package:mas
```

For an App Store upload package, set `TOKENSMITH_MAS_TYPE=distribution`, use the matching Apple Distribution certificate/profile, and set `TOKENSMITH_MAS_INSTALLER_IDENTITY` to the Mac Installer Distribution identity. The packager signs nested code and the app with `@electron/osx-sign`, verifies the bundle, then creates and checks a signed `.pkg`. A distribution-signed app is not a local development test.

Signed MAS builds fail if credentials are missing; `TOKENSMITH_SKIP_CODESIGN` cannot silently turn them into unsigned builds. The root entitlement file allows sandboxing, outgoing network connections, user-selected read/write files, and persistent file bookmarks. Child code inherits the sandbox. No camera, microphone, broad home-directory, or server-listening entitlement is requested.

## File access and validation still needed

Open/save dialogs request security-scoped bookmarks in MAS. Grants are restored before libraries initialize and released at shutdown. Re-select a file or folder if its grant expires. Existing DMG libraries are not automatically migrated into the sandbox container.

Before submission, validate the development-signed build on physical Macs:

- Cloud-only first run: connect chat and embedding models, prepare documents including chapter topics, search, chat and practice with Ollama absent.
- Optional Ollama: connect an existing model; disconnect without deleting it; handle an unavailable server.
- Python and native extensions under the actual sandbox: OCR, indexing, search and process shutdown. In particular, verify access by both persistent and newly spawned Python workers to files selected after startup; unsigned import checks do not prove grant inheritance.
- Selected folders after relaunch, external libraries, import/export, source previews, and revoked/moved files. Validate macOS secure storage inside the sandbox.
- App Store Connect validation, helper identifiers, provisioning entitlements, privacy disclosures, dependency licenses, and GT's ownership/export review.

The project has no Apple signing identity configured on the preparation machine. Signed execution, TestFlight, final App Store validation, and submission remain dependent on GT onboarding. Developer ID signing/notarization for the DMG is a separate follow-up.

References: [Electron MAS guide](https://www.electronjs.org/docs/latest/tutorial/mac-app-store-submission-guide), [Electron file dialogs](https://www.electronjs.org/docs/latest/api/dialog), [Apple App Review Guidelines](https://developer.apple.com/app-store/review/guidelines/).
