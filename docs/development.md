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

## Collections

The interface uses **Libraries** and **Documents**. **New library** asks for a
name and documents; storage is allocated automatically under the app's
`libraries/<uuid>` directory. **Add documents** accepts PDF, Markdown, and text
files, including multiple files from the same directory. Each file is indexed
independently so updating or removing it does not affect its siblings.

**Open library** opens an existing saved library without preparing documents or
calling a model. **Close library** cancels work using that library, waits for its
Python workers to exit, and removes it from the searchable catalog. It leaves the
library and original documents on disk. Closed libraries remain listed and can
be reopened with one click. Chat and Practice use the same **Documents** selector,
grouped by library without an intermediate import-folder level.

The storage format and internal API identifiers still use `collection` for
compatibility.
A collection folder contains:

```text
collection.json             Required manifest
tokensmith.sqlite           Required document/chunk/embedding index
tokensmith.faiss             Optional, rebuildable vector-search cache
topics.json                 Optional saved Practice topics
documents/                  Optional original documents
graph.json                  Optional future derived data; currently preserved, not queried
```

Other existing preparation caches and thumbnails stay inside the index folder.
The manifest format is:

```json
{
  "format": "tokensmith-collection",
  "version": 1,
  "id": "a-stable-collection-uuid",
  "name": "Database Systems",
  "indexRevision": "a-revision-uuid",
  "sourceBasePath": "/original/location/of/this/collection"
}
```

The ID stays fixed when moving or copying the folder. Two copies of the same ID
cannot be loaded simultaneously. The app accepts the current SQLite schema
(version 11); unsupported or damaged databases are rejected without upgrading
them. A failed load does not unload other collections. An unavailable collection
at startup appears as unloaded with an error.

SQLite document and chunk IDs are local. References crossing IPC use a qualified
material ID, `<collection-id>/<local-material-id>`, alongside the local document
and chunk IDs. This prevents collisions across loaded indexes. Each collection
has its own Python worker; results from an old mount are rejected even if that
collection has since been reloaded. Retrieval merges ranks across collections
instead of comparing raw scores from potentially different embedding models.

Original documents are optional for retrieval but required for opening the full
source. Relative source paths resolve inside the collection. `sourceBasePath`
allows existing absolute paths for embedded documents to follow a moved collection
folder. External originals remain external. **Locate original documents** in document details
records a replacement location in the app's registry; it does not edit the shared
index or manifest. For sharing a self-contained collection, put originals inside
the collection folder before adding them to the index.

Saved topics use local material/document IDs:

```json
{
  "version": 1,
  "indexRevision": "a-revision-uuid",
  "topics": [
    { "title": "B+ trees", "documents": [{ "materialId": "1", "documentId": 1 }] }
  ]
}
```

Practice reuses matching topics without an inference call. Topics are eligible
only when their revision matches and all their documents are selected. Index
updates invalidate the revision. Newly generated topics are saved for their
document selection when it belongs to one collection. Missing or malformed
optional data does not block retrieval; unknown files and metadata are preserved.
Graph-based retrieval is not implemented.

App-owned `collections.json` holds local paths, loaded flags, and source-location
overrides. Model settings and credentials remain in their existing app storage.
On first use of this format, old global-library references and study/chat history
are reset rather than migrated. The old index files are not deleted. New history
keeps its qualified references while a collection is unloaded and becomes usable
again when that collection is loaded.

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
