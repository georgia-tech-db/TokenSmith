# Code signing policy

Free code signing provided by [SignPath.io](https://signpath.io), certificate by [SignPath Foundation](https://signpath.org).

## Scope and status

TokenSmith's Windows MSI signing integration uses SignPath. Publicly trusted release signing requires an issued release certificate and approval of the specific build. Test-signed installers are only for validating this integration and must not be distributed as trusted releases. Existing portable ZIP downloads are not covered by an installer signature. macOS Developer ID signing and notarization are separate from this Windows integration.

We sign the MSI built from this repository, including its embedded application payload. We preserve upstream Electron, Python, and library binaries instead of applying TokenSmith's certificate to them. Windows Installer handles installation and removal, so there is no separate TokenSmith uninstaller executable to sign.

## Team responsibilities

- Authors and reviewers: maintainers with write access to [georgia-tech-db/TokenSmith](https://github.com/georgia-tech-db/TokenSmith). Contributions from other authors require maintainer review.
- Release signing approver: [Joy Arulraj](https://github.com/jarulraj).
- Automated submitter: SignPath's **CI builds** account, used by the Windows Build workflow.

Signing team members must use multi-factor authentication for both GitHub and SignPath. Release signing requires a GitHub-hosted build, verification of this repository's origin, and one manual approval by Joy Arulraj. The CI account can submit requests but cannot approve releases.

## Maintainer setup

The SignPath project is `TokenSmith` in organization `b6c0eccc-553e-4e23-b765-80eeedda4ccf`. The artifact configuration is `windows-installer`, matching [the version-controlled XML](../build-resources/signpath/windows-installer.xml). It signs only `TokenSmith-*-win-x64.msi` inside the GitHub artifact ZIP.

Store the CI account's API token as the GitHub Actions repository secret `SIGNPATH_API_TOKEN`. A personal account is used to access the dashboard and approve releases; do not attempt to sign in as the CI account.

Run **Windows Build** manually with `test-signing` to verify the integration. A self-signed test certificate may report `NotTrusted`; test artifacts are explicitly labeled and never attached to releases. The workflow also installs the MSI on a disposable GitHub runner, checks the installed executable and Python runtime, and uninstalls it.

Run with `release-signing` only after the release certificate is valid, or push an approved `v*` release tag. Approve the matching request in SignPath after reviewing its source revision and build. The workflow waits up to 30 minutes for signing, requires a valid, timestamped SignPath Foundation signature, and only then attaches the MSI to the draft GitHub release. Missing credentials, pending certificates, rejected requests, or invalid signatures fail the build; there is no unsigned installer fallback for a release.

Use `none` for an unsigned packaging check. Manual builds upload artifacts for seven days. Unsigned signing inputs are retained for one day. The portable ZIP remains available separately and is not described as signed.

## Building locally on Windows

Run `npm ci`, `npm run setup:python-runtime`, and `npm run package:win`. With WiX Toolset 3.14 installed (`WIX` pointing to its installation directory), run `./scripts/package-win-installer.ps1` in PowerShell to create the MSI. The installer is per-user and does not require administrator access. User-created TokenSmith data is outside the package and is preserved on uninstall.

See [SignPath's GitHub integration documentation](https://docs.signpath.io/trusted-build-systems/github) for the signing request protocol.
