# Development

Looking for the app? See [downloads and getting started](../README.md#download).
These instructions are for working on TokenSmith's source code.

## Local Setup

Use Node.js 22 (the version used in CI) and npm. Run these commands from the
repository root.

Install dependencies:

```sh
npm install
```

Download the pinned, platform-specific Python runtime and install TokenSmith's
Python dependencies inside it:

```sh
npm run setup:python-runtime
```

This creates `app_runtime/python` inside the repository. It does not require,
modify, or install packages into your system Python. The download is selected
for Linux x64, Windows x64, macOS Apple Silicon (ARM64), or macOS Intel (x64),
and verified with SHA-256 before it is used. Intel Macs require macOS 15 or
newer because of the pinned native Python dependencies. Both Mac architectures
use the same setup commands and dependency versions.

The runtime comes from Astral's
[python-build-standalone](https://github.com/astral-sh/python-build-standalone)
`install_only_stripped` archives. TokenSmith pins the Python version, release,
target platform, and checksum rather than copying the developer's Python
installation.

Start the app locally:

```sh
npm run dev
```

For checks, benchmark model requirements, and coverage, see
[testing and benchmarks](testing.md).

## Packaging

Packaging requires `npm run setup:python-runtime` first. Release workflows run
that command automatically and include the private runtime in each application,
so students do not need to install Python.

Run the packaging command on its matching operating system.
On macOS, install Apple's Command Line Tools (`xcode-select --install`) first.

Create a macOS DMG:

```sh
npm run package:mac
```

Mac packaging uses the architecture of Node.js. Run the same command on an
Intel Mac to create
`release/TokenSmith-<version>-mac-x64.dmg`, or on Apple Silicon to create
`release/TokenSmith-<version>-mac-arm64.dmg`. The Intel application declares
macOS 15.0 as its minimum version.

Use Node.js, Electron, and the Python runtime for the same architecture.
Packaging checks the Electron and Python binaries before building the app;
after changing Node.js architectures, reinstall Node dependencies and rerun
`npm run setup:python-runtime`.

The macOS release workflow builds both architectures on native GitHub
runners, tests the native runtime, verifies packaged binary architectures and
vector search, and keeps their artifacts separate.
It runs manually or when a `v*` release tag is pushed.

Create a Windows portable ZIP:

```sh
npm run package:win
```

Create Linux packages:

```sh
npm run package:linux
```

When publishing a new release, update the version and installer links in the
[README download table](../README.md#download) after verifying the assets exist.
