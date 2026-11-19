"""Portable .tokensmith libraries: snapshot indexes, original documents, and preparation data."""
import json
import os
from pathlib import Path, PurePosixPath
import sqlite3
import stat
import tempfile
import zipfile

try:
    from tokensmith_collections import validate
except ImportError:
    from python_engine.tokensmith_collections import validate

MAX_BYTES = 16 * 1024**3
MAX_FILES = 100_000


def plan(payload):
    root = Path(payload['userDataPath']).resolve()
    validate({'userDataPath': str(root)})
    with sqlite3.connect((root / 'tokensmith.sqlite').as_uri() + '?mode=ro', uri=True) as conn:
        documents = [{'path': row[0], 'materialId': str(row[1]), 'documentId': row[2]}
                     for row in conn.execute('SELECT d.document_path, ci.collection_id, d.id FROM documents d JOIN collection_items ci ON ci.folder_id=d.folder_id')]
        materials = [{'path': row[0], 'materialId': str(row[1]), 'kind': row[2]}
                     for row in conn.execute('SELECT import_path, collection_id, kind FROM tokensmith_collection_state') if row[0]]
    return {'documents': documents, 'materials': materials}


def export_library(payload):
    root = Path(payload['userDataPath']).resolve()
    destination = Path(payload['destination']).resolve()
    if destination.suffix.lower() != '.tokensmith':
        raise ValueError('Choose a .tokensmith filename for this library.')
    manifest = json.loads((root / 'collection.json').read_text())
    source_paths = payload['sources']
    listing = plan(payload)
    if not listing['documents']:
        raise ValueError('Add and prepare documents before exporting this library.')
    mappings, assets = {}, {}
    for document in listing['documents']:
        original = document['path']
        source = Path(source_paths[original]).resolve()
        if not source.is_file():
            raise ValueError(f'Original document is missing: {source.name}. Locate it before exporting.')
        name = f"documents/{document['materialId']}/{document['documentId']}/{source.name}"
        mappings[original] = name
        mappings[str(source)] = name
        assets[name] = source
    for material in listing['materials']:
        mappings.setdefault(material['path'], f"documents/{material['materialId']}")
    manifest['sourceMappings'] = mappings
    # A package contains library data only; never include the application profile or credentials.
    allowed = {'format', 'version', 'id', 'name', 'indexRevision', 'sourceMappings'}
    manifest = {key: value for key, value in manifest.items() if key in allowed}
    if destination == root / 'tokensmith.sqlite' or destination in assets.values():
        raise ValueError('Choose a new .tokensmith file, not an original document or index.')
    with tempfile.TemporaryDirectory(prefix='tokensmith-export-') as temporary:
        snapshot = Path(temporary) / 'tokensmith.sqlite'
        with sqlite3.connect((root / 'tokensmith.sqlite').as_uri() + '?mode=ro', uri=True) as source:
            with sqlite3.connect(snapshot) as target:
                source.backup(target)
                target.execute('DELETE FROM pdf_page_thumbnails')
        archive = Path(temporary) / 'library.tokensmith'
        with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, allowZip64=True) as out:
            out.writestr('collection.json', json.dumps(manifest))
            out.write(snapshot, 'tokensmith.sqlite')
            topics = root / 'topic-inventory.json'
            if topics.is_file():
                out.write(topics, 'topic-inventory.json')
            for name, source in assets.items():
                out.write(source, name)
        with zipfile.ZipFile(archive) as completed:
            if len(completed.infolist()) > MAX_FILES or sum(info.file_size for info in completed.infolist()) > MAX_BYTES:
                raise ValueError('This library is too large to export (limit: 16 GB and 100,000 files).')
        # Publish only the complete archive; a failed export must not replace a previous file.
        destination.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        fd, staging = tempfile.mkstemp(prefix='.tokensmith-export-', dir=destination.parent)
        try:
            with os.fdopen(fd, 'wb') as target, archive.open('rb') as source:
                shutil.copyfileobj(source, target)
            os.replace(staging, destination)
        finally:
            Path(staging).unlink(missing_ok=True)
    return {'ok': True}


def safe_name(name):
    path = PurePosixPath(name)
    if not name or '\\' in name or ':' in name or path.is_absolute() or '..' in path.parts or str(path) != name.rstrip('/'):
        raise ValueError('This library contains an unsafe file path.')
    allowed = name in ('collection.json', 'tokensmith.sqlite', 'topic-inventory.json') or path.parts[0] == 'documents'
    if not allowed:
        raise ValueError('This file contains unsupported library data.')
    return path


def import_library(payload):
    root = Path(payload['userDataPath']).resolve()
    if not root.is_dir() or any(root.iterdir()):
        raise ValueError('Import requires a new empty library folder.')
    try:
        with zipfile.ZipFile(payload['archive']) as archive:
            infos = archive.infolist()
            if len(infos) > MAX_FILES or sum(info.file_size for info in infos) > MAX_BYTES:
                raise ValueError('This library is too large to import (limit: 16 GB and 100,000 files).')
            seen = set()
            for info in infos:
                if info.filename.endswith('.json') and info.file_size > 20_000_000:
                    raise ValueError('This library contains oversized metadata.')
                safe_name(info.filename)
                key = info.filename.rstrip('/').casefold()
                if key in seen or info.flag_bits & 1 or stat.S_ISLNK(info.external_attr >> 16):
                    raise ValueError('This library contains duplicate, encrypted, or linked files.')
                seen.add(key)
            if not {'collection.json', 'tokensmith.sqlite'} <= seen:
                raise ValueError('Choose a valid .tokensmith library file.')
            for info in infos:
                target = root / info.filename
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    import shutil
                    with archive.open(info) as source, target.open('xb') as out:
                        shutil.copyfileobj(source, out)
        manifest = json.loads((root / 'collection.json').read_text())
        if manifest.get('format') != 'tokensmith-collection' or manifest.get('version') != 1:
            raise ValueError('This library format is not supported.')
        mappings = manifest.get('sourceMappings')
        if not isinstance(mappings, dict):
            raise ValueError('This library is missing its portable document paths.')
        for original, relative in mappings.items():
            if not isinstance(original, str) or not isinstance(relative, str) or safe_name(relative).parts[0] != 'documents':
                raise ValueError('This library has invalid document paths.')
            if not (root / relative).exists():
                raise ValueError('This library is missing an original document.')
        validate({'userDataPath': str(root)})
        for doc in plan({'userDataPath': str(root)})['documents']:
            if doc['path'] not in mappings or not (root / mappings[doc['path']]).is_file():
                raise ValueError('This library is missing an original document.')
        old_id, new_id = manifest['id'], payload['collectionId']
        relocated = {old: str(root / relative) for old, relative in mappings.items()}
        def rekey(value):
            if isinstance(value, dict):
                result = {}
                for key, item in value.items():
                    if key == 'thumbnailPath':
                        continue  # Thumbnails are regenerated, not exported.
                    if key == 'path' and isinstance(item, str) and item and item not in relocated:
                        raise ValueError('This library contains an invalid source path.')
                    result[relocated.get(key, key)] = rekey(item)
                return result
            if isinstance(value, list):
                return [rekey(item) for item in value]
            if isinstance(value, str):
                if value in relocated:
                    return relocated[value]
                if value == old_id:
                    return new_id
                if value.startswith(old_id + '/'):
                    return new_id + value[len(old_id):]
            return value
        with sqlite3.connect(root / 'tokensmith.sqlite') as conn:
            conn.row_factory = sqlite3.Row
            for row in conn.execute('SELECT collection_id, import_path, preparation_json FROM tokensmith_collection_state').fetchall():
                if row['import_path'] not in relocated:
                    raise ValueError('This library has an invalid import path.')
                conn.execute('UPDATE tokensmith_collection_state SET import_path=?, preparation_json=? WHERE collection_id=?',
                             (relocated[row['import_path']], json.dumps(rekey(json.loads(row['preparation_json'] or '{}'))), row['collection_id']))
            for row in conn.execute('SELECT id, document_path FROM documents').fetchall():
                conn.execute('UPDATE documents SET document_path=? WHERE id=?', (relocated[row['document_path']], row['id']))
            for row in conn.execute('SELECT id, path FROM folders').fetchall():
                # Folder identities remain local; their filesystem locations must be portable too.
                doc = conn.execute('SELECT document_path FROM documents WHERE folder_id=? LIMIT 1', (row['id'],)).fetchone()
                if doc:
                    conn.execute('UPDATE folders SET path=? WHERE id=?', (str(Path(doc[0]).parent), row['id']))
            conn.execute('DELETE FROM pdf_page_thumbnails')
        manifest.pop('sourceMappings', None)
        manifest['id'] = new_id
        manifest['sourceBasePath'] = str(root)
        (root / 'collection.json').write_text(json.dumps(manifest))
        topics = root / 'topic-inventory.json'
        if topics.is_file():
            topics.write_text(json.dumps(rekey(json.loads(topics.read_text()))))
        _restore_reports(root)
        return {'ok': True}
    except (zipfile.BadZipFile, KeyError, json.JSONDecodeError, sqlite3.DatabaseError) as error:
        raise ValueError('This .tokensmith file is damaged or incomplete.') from error


def _restore_reports(root):
    """Rebuild Details from the portable index, without carrying model caches or credentials."""
    try:
        from tokensmith_preparation_job import job_directory
        from tokensmith_preparation import digest, write_json
    except ImportError:
        from python_engine.tokensmith_preparation_job import job_directory
        from python_engine.tokensmith_preparation import digest, write_json
    with sqlite3.connect(root / 'tokensmith.sqlite') as conn:
        conn.row_factory = sqlite3.Row
        for material in conn.execute('SELECT collection_id, import_path FROM tokensmith_collection_state').fetchall():
            directory = job_directory(str(root), material['import_path'])
            docs = []
            for doc in conn.execute('SELECT d.* FROM documents d JOIN collection_items ci ON ci.folder_id=d.folder_id WHERE ci.collection_id=?', (material['collection_id'],)).fetchall():
                chunks = [{'text': c['chunk_text'], 'pageStart': c['page'], 'pageEnd': c['page'],
                           'lineFrom': c['line_from'], 'lineTo': c['line_to'], 'sectionHeader': c['section_header'],
                           'tokensmithChunkKind': c['chunk_kind']} for c in conn.execute('SELECT * FROM chunks WHERE document_id=? ORDER BY id', (doc['id'],))]
                name = 'documents/' + digest(doc['document_path']) + '.json'
                write_json(directory / name, chunks)
                docs.append({'path': doc['document_path'], 'title': Path(doc['document_path']).stem,
                             'status': 'ready', 'chunkCount': len(chunks), 'chunksFile': name})
            write_json(directory / 'report.json', {'documents': docs})
