"""Validate collection stores without creating, upgrading, or embedding anything."""
from pathlib import Path
import sqlite3

try:
    from tokensmith_store import SCHEMA_VERSION, init_db
except ImportError:
    from python_engine.tokensmith_store import SCHEMA_VERSION, init_db


def validate(payload):
    root = Path(payload['userDataPath']).resolve()
    database = root / 'tokensmith.sqlite'
    if not database.is_file():
        raise ValueError('The collection is missing its index database (tokensmith.sqlite).')
    if database.is_symlink() or (root / 'tokensmith.faiss').is_symlink():
        raise ValueError('The collection database must be inside its index folder.')
    with sqlite3.connect(database.as_uri() + '?mode=ro', uri=True) as conn:
        conn.execute('PRAGMA query_only=ON')
        row = conn.execute("SELECT value FROM schema_info WHERE key = 'version'").fetchone()
        if not row or str(row[0]) != str(SCHEMA_VERSION):
            raise ValueError('This collection uses an unsupported index version.')
        checks = conn.execute('PRAGMA quick_check').fetchall()
        if checks != [('ok',)]:
            raise ValueError('The collection index is damaged.')
        required = {
            'collections': {'id', 'name', 'embedding_model'},
            'documents': {'id', 'folder_id', 'document_path'},
            'chunks': {'id', 'document_id', 'chunk_text', 'stable_chunk_id'},
            'embeddings': {'model', 'folder_id', 'chunk_id', 'embedding'},
            'tokensmith_collection_state': {'collection_id', 'status', 'import_path'},
            'folders': {'id', 'path'}, 'collection_items': {'collection_id', 'folder_id'},
        }
        for table, columns in required.items():
            actual = {row[1] for row in conn.execute(f'PRAGMA table_info({table})')}
            if not columns <= actual:
                raise ValueError('The collection index is incomplete.')
        if conn.execute('SELECT model FROM embeddings GROUP BY model HAVING COUNT(DISTINCT length(embedding)) > 1 OR MIN(length(embedding)) = 0 OR MAX(length(embedding) % 4) != 0').fetchone():
            raise ValueError('The collection has inconsistent embedding dimensions.')
        if conn.execute('PRAGMA foreign_key_check').fetchone():
            raise ValueError('The collection contains broken document references.')
    return {'ok': True}


def create(payload):
    root = Path(payload['userDataPath'])
    if (root / 'tokensmith.sqlite').exists():
        raise ValueError('An index already exists in this folder.')
    init_db(str(root))
    return {'ok': True}
