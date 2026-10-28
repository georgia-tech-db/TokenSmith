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
for Linux x64, Windows x64, or macOS ARM64 and verified with SHA-256 before it
is used.

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

Create a macOS DMG:

```sh
npm run package:mac
```

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
