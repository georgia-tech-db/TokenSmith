from __future__ import annotations

import json
import random
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover - optional until app runtime is installed
    faiss = None  # type: ignore


DB_NAME = "tokensmith.sqlite"
FAISS_NAME = "tokensmith.faiss"
SCHEMA_VERSION = 9


def db_path(user_data_path: str) -> Path:
    return Path(user_data_path) / DB_NAME


def faiss_path(user_data_path: str) -> Path:
    return Path(user_data_path) / FAISS_NAME


def connect(user_data_path: str) -> sqlite3.Connection:
    path = db_path(user_data_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def init_db(user_data_path: str) -> None:
    with connect(user_data_path) as conn:
        create_schema(conn)
        ensure_column(conn, "tokensmith_collection_state", "embedding_model_id", "TEXT")
        ensure_column(conn, "tokensmith_collection_state", "embedding_model_name", "TEXT")
        ensure_column(conn, "tokensmith_collection_state", "cleaning_profile_id", "TEXT")
        ensure_column(conn, "tokensmith_collection_state", "cleaning_profile_name", "TEXT")
        ensure_column(conn, "tokensmith_collection_state", "cleaning_profile_version", "INTEGER")
        ensure_column(conn, "tokensmith_collection_state", "cleaning_rule_ids_json", "TEXT")
        ensure_column(conn, "tokensmith_collection_state", "chunk_size", "INTEGER")
        ensure_column(conn, "chunks", "chunk_size", "INTEGER")
        ensure_column(conn, "chunks", "section_header", "TEXT")
        set_schema_value(conn, "version", str(SCHEMA_VERSION))


def ensure_column(conn: sqlite3.Connection, table_name: str, column_name: str, definition: str) -> None:
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    if any(row["name"] == column_name for row in rows):
        return
    conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {definition}")


def create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS schema_info (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS collections (
            id INTEGER PRIMARY KEY,
            name TEXT UNIQUE NOT NULL,
            start_update_time INTEGER,
            last_update_time INTEGER,
            embedding_model TEXT
        );

        CREATE TABLE IF NOT EXISTS folders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT UNIQUE NOT NULL
        );

        CREATE TABLE IF NOT EXISTS collection_items (
            collection_id INTEGER NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
            folder_id INTEGER NOT NULL REFERENCES folders(id) ON DELETE CASCADE,
            UNIQUE(collection_id, folder_id)
        );

        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY,
            folder_id INTEGER NOT NULL REFERENCES folders(id) ON DELETE CASCADE,
            document_time INTEGER NOT NULL,
            document_path TEXT UNIQUE NOT NULL
        );

        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            chunk_text TEXT NOT NULL,
            file TEXT NOT NULL,
            title TEXT,
            author TEXT,
            subject TEXT,
            keywords TEXT,
            page INTEGER,
            line_from INTEGER,
            line_to INTEGER,
            words INTEGER NOT NULL DEFAULT 0,
            tokens INTEGER NOT NULL DEFAULT 0,
            chunk_size INTEGER,
            section_header TEXT
        );

        CREATE TABLE IF NOT EXISTS embeddings (
            model TEXT NOT NULL,
            folder_id INTEGER NOT NULL REFERENCES folders(id) ON DELETE CASCADE,
            chunk_id INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
            embedding BLOB NOT NULL,
            PRIMARY KEY(model, folder_id, chunk_id),
            UNIQUE(model, chunk_id)
        );

        CREATE TABLE IF NOT EXISTS pdf_page_thumbnails (
            document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            page INTEGER NOT NULL,
            thumbnail_path TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY(document_id, page)
        );

        CREATE TABLE IF NOT EXISTS tokensmith_collection_state (
            collection_id INTEGER PRIMARY KEY REFERENCES collections(id) ON DELETE CASCADE,
            status TEXT NOT NULL DEFAULT 'ready',
            kind TEXT NOT NULL DEFAULT 'folder',
            import_path TEXT,
            embedding_model_id TEXT,
            embedding_model_name TEXT,
            cleaning_profile_id TEXT,
            cleaning_profile_name TEXT,
            cleaning_profile_version INTEGER,
            cleaning_rule_ids_json TEXT,
            detail TEXT,
            added_at TEXT NOT NULL,
            indexed_at TEXT,
            is_active INTEGER NOT NULL DEFAULT 1,
            file_count INTEGER DEFAULT 0,
            size_bytes INTEGER DEFAULT 0,
            word_count INTEGER DEFAULT 0,
            page_count INTEGER,
            chunk_count INTEGER DEFAULT 0,
            chunk_size INTEGER,
            error TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_collection_items_collection_id ON collection_items(collection_id);
        CREATE INDEX IF NOT EXISTS idx_collection_items_folder_id ON collection_items(folder_id);
        CREATE INDEX IF NOT EXISTS idx_documents_folder_id ON documents(folder_id);
        CREATE INDEX IF NOT EXISTS idx_embeddings_model ON embeddings(model);
        CREATE INDEX IF NOT EXISTS idx_embeddings_chunk_id ON embeddings(chunk_id);
        CREATE INDEX IF NOT EXISTS idx_pdf_page_thumbnails_document_id ON pdf_page_thumbnails(document_id);
        """
    )
    create_fts_schema(conn)


def create_fts_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
        USING fts5(
            id UNINDEXED,
            document_id UNINDEXED,
            chunk_text,
            file,
            title,
            author,
            subject,
            keywords,
            content='chunks',
            content_rowid='id',
            tokenize='porter'
        );

        CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
            INSERT INTO chunks_fts(rowid, id, document_id, chunk_text, file, title, author, subject, keywords)
            VALUES (new.id, new.id, new.document_id, new.chunk_text, new.file, new.title, new.author, new.subject, new.keywords);
        END;

        CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
            INSERT INTO chunks_fts(chunks_fts, rowid, id, document_id, chunk_text, file, title, author, subject, keywords)
            VALUES ('delete', old.id, old.id, old.document_id, old.chunk_text, old.file, old.title, old.author, old.subject, old.keywords);
        END;

        CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
            INSERT INTO chunks_fts(chunks_fts, rowid, id, document_id, chunk_text, file, title, author, subject, keywords)
            VALUES ('delete', old.id, old.id, old.document_id, old.chunk_text, old.file, old.title, old.author, old.subject, old.keywords);

            INSERT INTO chunks_fts(rowid, id, document_id, chunk_text, file, title, author, subject, keywords)
            VALUES (new.id, new.id, new.document_id, new.chunk_text, new.file, new.title, new.author, new.subject, new.keywords);
        END;
        """
    )


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def now_ms() -> int:
    return int(time.time() * 1000)


def set_schema_value(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        """
        INSERT INTO schema_info(key, value)
        VALUES (?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """,
        (key, value),
    )


def get_schema_value(user_data_path: str, key: str) -> Optional[str]:
    init_db(user_data_path)

    with connect(user_data_path) as conn:
        row = conn.execute("SELECT value FROM schema_info WHERE key = ?", (key,)).fetchone()

    return str(row["value"]) if row else None


def vector_to_blob(vector: Iterable[float]) -> Tuple[bytes, int]:
    arr = np.asarray(list(vector), dtype=np.float32)
    return arr.tobytes(), int(arr.shape[0])


def blob_to_vector(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def normalize_matrix(vectors: np.ndarray) -> np.ndarray:
    vectors = vectors.astype("float32")
    if vectors.ndim == 1:
        vectors = np.expand_dims(vectors, axis=0)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def collection_id_from_material_id(material_id: Any) -> Optional[int]:
    if material_id is None:
        return None
    try:
        value = int(str(material_id))
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def path_candidates(path: Any) -> List[str]:
    if not path:
        return []
    raw_path = str(path).replace("file://", "", 1)
    candidates = [raw_path]
    try:
        candidates.append(str(Path(raw_path).expanduser().resolve()))
    except Exception:
        pass
    return list(dict.fromkeys(candidate for candidate in candidates if candidate))


def folder_path_for_import(import_path: str, kind: str) -> str:
    path = Path(import_path)
    if kind == "folder":
        return str(path)
    return str(path.parent)


def upsert_folder(conn: sqlite3.Connection, folder_path: str) -> int:
    conn.execute("INSERT OR IGNORE INTO folders(path) VALUES (?)", (folder_path,))
    row = conn.execute("SELECT id FROM folders WHERE path = ?", (folder_path,)).fetchone()
    return int(row["id"])


def unique_collection_name(conn: sqlite3.Connection, desired_name: str, collection_id: Optional[int] = None) -> str:
    base = desired_name.strip() or "Course materials"
    candidate = base
    counter = 2

    while True:
        row = conn.execute("SELECT id FROM collections WHERE name = ?", (candidate,)).fetchone()
        if not row or (collection_id is not None and int(row["id"]) == collection_id):
            return candidate
        candidate = f"{base} ({counter})"
        counter += 1


def collection_ids_for_import_path(conn: sqlite3.Connection, import_path: str) -> List[int]:
    candidates = path_candidates(import_path)
    if not candidates:
        return []

    placeholders = ",".join("?" for _ in candidates)
    rows = conn.execute(
        f"""
        SELECT DISTINCT c.id
        FROM collections c
        LEFT JOIN tokensmith_collection_state s ON s.collection_id = c.id
        LEFT JOIN collection_items ci ON ci.collection_id = c.id
        LEFT JOIN folders f ON f.id = ci.folder_id
        LEFT JOIN documents d ON d.folder_id = f.id
        WHERE s.import_path IN ({placeholders})
           OR f.path IN ({placeholders})
           OR d.document_path IN ({placeholders})
        """,
        [*candidates, *candidates, *candidates],
    ).fetchall()
    return [int(row["id"]) for row in rows]


def embedding_models_for_collections(conn: sqlite3.Connection, collection_ids: Sequence[int]) -> List[str]:
    if not collection_ids:
        return []
    placeholders = ",".join("?" for _ in collection_ids)
    rows = conn.execute(
        f"""
        SELECT DISTINCT e.model
        FROM embeddings e
        JOIN folders f ON f.id = e.folder_id
        JOIN collection_items ci ON ci.folder_id = f.id
        WHERE ci.collection_id IN ({placeholders})
        """,
        list(collection_ids),
    ).fetchall()
    return [str(row["model"]) for row in rows if row["model"]]


def embedding_models_by_collection_ids(user_data_path: str, collection_ids: Sequence[str]) -> Dict[str, str]:
    init_db(user_data_path)
    numeric_ids = []
    for collection_id in collection_ids:
        try:
            numeric_ids.append(int(collection_id))
        except (TypeError, ValueError):
            continue

    if not numeric_ids:
        return {}

    placeholders = ",".join("?" for _ in numeric_ids)
    with connect(user_data_path) as conn:
        rows = conn.execute(
            f"""
            SELECT id, embedding_model
            FROM collections
            WHERE id IN ({placeholders})
              AND embedding_model IS NOT NULL
              AND embedding_model != ''
            """,
            numeric_ids,
        ).fetchall()

    return {str(row["id"]): str(row["embedding_model"]) for row in rows if row["embedding_model"]}


def embedded_chunk_signatures(
    user_data_path: str,
    import_path: str,
    embedding_model: str,
) -> Set[Tuple[str, str, Optional[int], Optional[int], Optional[int], Optional[int]]]:
    init_db(user_data_path)

    with connect(user_data_path) as conn:
        collection_ids = collection_ids_for_import_path(conn, import_path)
        if not collection_ids or not embedding_model:
            return set()

        placeholders = ",".join("?" for _ in collection_ids)
        rows = conn.execute(
            f"""
            SELECT d.document_path, c.chunk_text, c.page, c.line_from, c.line_to, c.chunk_size
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            JOIN folders f ON f.id = d.folder_id
            JOIN collection_items ci ON ci.folder_id = f.id
            JOIN embeddings e ON e.chunk_id = c.id
                             AND e.folder_id = f.id
                             AND e.model = ?
            WHERE ci.collection_id IN ({placeholders})
            """,
            [embedding_model, *collection_ids],
        ).fetchall()

    return {
        (
            str(row["document_path"]),
            str(row["chunk_text"]),
            row["page"],
            row["line_from"],
            row["line_to"],
            row["chunk_size"],
        )
        for row in rows
    }


def delete_collection_ids(conn: sqlite3.Connection, collection_ids: Sequence[int]) -> bool:
    if not collection_ids:
        return False

    unique_ids = list(dict.fromkeys(int(collection_id) for collection_id in collection_ids))
    placeholders = ",".join("?" for _ in unique_ids)
    folder_rows = conn.execute(
        f"""
        SELECT DISTINCT folder_id
        FROM collection_items
        WHERE collection_id IN ({placeholders})
        """,
        unique_ids,
    ).fetchall()
    folder_ids = [int(row["folder_id"]) for row in folder_rows]

    cursor = conn.execute(f"DELETE FROM collections WHERE id IN ({placeholders})", unique_ids)

    for folder_id in folder_ids:
        still_used = conn.execute(
            """
            SELECT 1
            FROM collection_items
            WHERE folder_id = ?
            LIMIT 1
            """,
            (folder_id,),
        ).fetchone()
        if still_used is None:
            conn.execute("DELETE FROM folders WHERE id = ?", (folder_id,))

    return cursor.rowcount > 0


def delete_material_by_import_path(conn: sqlite3.Connection, import_path: str) -> bool:
    return delete_collection_ids(conn, collection_ids_for_import_path(conn, import_path))


def delete_material(user_data_path: str, material_id: str, import_path: Optional[str] = None) -> bool:
    init_db(user_data_path)
    collection_id = collection_id_from_material_id(material_id)

    with connect(user_data_path) as conn:
        collection_ids = [collection_id] if collection_id is not None else []
        if import_path:
            collection_ids.extend(collection_ids_for_import_path(conn, import_path))
        collection_ids = list(dict.fromkeys(collection_ids))
        embedding_models = embedding_models_for_collections(conn, collection_ids)
        deleted = delete_collection_ids(conn, collection_ids)

    if deleted:
        for embedding_model in embedding_models:
            rebuild_faiss(user_data_path, embedding_model)

    return deleted


def find_material_id_by_import_path(user_data_path: str, import_path: str) -> Optional[str]:
    init_db(user_data_path)

    with connect(user_data_path) as conn:
        collection_ids = collection_ids_for_import_path(conn, import_path)

    return str(collection_ids[0]) if collection_ids else None


def upsert_collection_state(
    conn: sqlite3.Connection,
    collection_id: int,
    material: Dict[str, Any],
) -> None:
    conn.execute(
        """
        INSERT INTO tokensmith_collection_state (
            collection_id, status, kind, import_path, embedding_model_id, embedding_model_name,
            cleaning_profile_id, cleaning_profile_name, cleaning_profile_version, cleaning_rule_ids_json,
            detail, added_at, indexed_at, is_active, file_count, size_bytes, word_count, page_count,
            chunk_count, chunk_size, error
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(collection_id) DO UPDATE SET
            status = excluded.status,
            kind = excluded.kind,
            import_path = excluded.import_path,
            embedding_model_id = excluded.embedding_model_id,
            embedding_model_name = excluded.embedding_model_name,
            cleaning_profile_id = excluded.cleaning_profile_id,
            cleaning_profile_name = excluded.cleaning_profile_name,
            cleaning_profile_version = excluded.cleaning_profile_version,
            cleaning_rule_ids_json = excluded.cleaning_rule_ids_json,
            detail = excluded.detail,
            added_at = excluded.added_at,
            indexed_at = excluded.indexed_at,
            is_active = excluded.is_active,
            file_count = excluded.file_count,
            size_bytes = excluded.size_bytes,
            word_count = excluded.word_count,
            page_count = excluded.page_count,
            chunk_count = excluded.chunk_count,
            chunk_size = excluded.chunk_size,
            error = excluded.error
        """,
        (
            collection_id,
            material.get("status") or "ready",
            material.get("kind") or "folder",
            material.get("path"),
            material.get("embeddingModelId"),
            material.get("embeddingModelName"),
            material.get("cleaningProfileId"),
            material.get("cleaningProfileName"),
            material.get("cleaningProfileVersion"),
            json.dumps(material.get("cleaningRuleIds") or []),
            material.get("detail"),
            material.get("addedAt") or now_iso(),
            material.get("indexedAt"),
            1 if material.get("isActive", True) is not False else 0,
            material.get("fileCount") or 0,
            material.get("sizeBytes") or 0,
            material.get("wordCount") or 0,
            material.get("pageCount"),
            material.get("chunkCount") or 0,
            material.get("chunkSize"),
            material.get("error"),
        ),
    )


def insert_document(conn: sqlite3.Connection, folder_id: int, document_path: str) -> int:
    try:
        document_time = int(Path(document_path).stat().st_mtime)
    except Exception:
        document_time = int(time.time())

    conn.execute(
        """
        INSERT OR IGNORE INTO documents(folder_id, document_time, document_path)
        VALUES (?, ?, ?)
        """,
        (folder_id, document_time, document_path),
    )
    conn.execute(
        """
        UPDATE documents
        SET folder_id = ?, document_time = ?
        WHERE document_path = ?
        """,
        (folder_id, document_time, document_path),
    )
    row = conn.execute("SELECT id FROM documents WHERE document_path = ?", (document_path,)).fetchone()
    return int(row["id"])


def insert_chunk(conn: sqlite3.Connection, document_id: int, chunk: Dict[str, Any]) -> int:
    text = str(chunk.get("text") or "")
    word_count = int(chunk.get("wordCount") or len(text.split()))
    cursor = conn.execute(
        """
        INSERT INTO chunks (
            document_id, chunk_text, file, title, author, subject, keywords,
            page, line_from, line_to, words, tokens, chunk_size, section_header
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            document_id,
            text,
            str(chunk.get("path") or ""),
            chunk.get("documentTitle"),
            None,
            None,
            None,
            chunk.get("pageStart"),
            chunk.get("lineFrom"),
            chunk.get("lineTo"),
            word_count,
            int(chunk.get("tokens") or word_count),
            chunk.get("chunkSize"),
            chunk.get("sectionHeader"),
        ),
    )
    return int(cursor.lastrowid)


def find_existing_chunk_id(conn: sqlite3.Connection, document_id: int, chunk: Dict[str, Any]) -> Optional[int]:
    text = str(chunk.get("text") or "")
    row = conn.execute(
        """
        SELECT id
        FROM chunks
        WHERE document_id = ?
          AND chunk_text = ?
          AND COALESCE(page, -1) = COALESCE(?, -1)
          AND COALESCE(line_from, -1) = COALESCE(?, -1)
          AND COALESCE(line_to, -1) = COALESCE(?, -1)
          AND COALESCE(chunk_size, -1) = COALESCE(?, -1)
        LIMIT 1
        """,
        (
            document_id,
            text,
            chunk.get("pageStart"),
            chunk.get("lineFrom"),
            chunk.get("lineTo"),
            chunk.get("chunkSize"),
        ),
    ).fetchone()
    return int(row["id"]) if row else None


def upsert_chunk(conn: sqlite3.Connection, document_id: int, chunk: Dict[str, Any]) -> int:
    return find_existing_chunk_id(conn, document_id, chunk) or insert_chunk(conn, document_id, chunk)


def replace_document_thumbnails(conn: sqlite3.Connection, document_id: int, thumbnails: Sequence[Dict[str, Any]]) -> None:
    conn.execute("DELETE FROM pdf_page_thumbnails WHERE document_id = ?", (document_id,))

    for thumbnail in thumbnails:
        page = thumbnail.get("page")
        thumbnail_path = thumbnail.get("path")
        if not page or not thumbnail_path:
            continue
        conn.execute(
            """
            INSERT OR REPLACE INTO pdf_page_thumbnails(document_id, page, thumbnail_path, updated_at)
            VALUES (?, ?, ?, ?)
            """,
            (document_id, int(page), str(thumbnail_path), now_iso()),
        )


def material_folder_id(conn: sqlite3.Connection, collection_id: int) -> Optional[int]:
    row = conn.execute(
        """
        SELECT folder_id
        FROM collection_items
        WHERE collection_id = ?
        LIMIT 1
        """,
        (collection_id,),
    ).fetchone()
    return int(row["folder_id"]) if row else None


def upsert_material_header(
    conn: sqlite3.Connection,
    material: Dict[str, Any],
    documents: List[Dict[str, Any]],
    *,
    embedding_model: str,
    replace_existing: bool = True,
    replace_thumbnails: bool = True,
) -> Tuple[int, int, Dict[str, int], Dict[str, int], List[str]]:
    deleted_embedding_models: List[str] = []

    target_collection_id = collection_id_from_material_id(material.get("id"))
    if replace_existing:
        delete_ids = [target_collection_id] if target_collection_id is not None else collection_ids_for_import_path(
            conn,
            str(material.get("path") or ""),
        )
        deleted_embedding_models = embedding_models_for_collections(conn, [collection_id for collection_id in delete_ids if collection_id])
        delete_collection_ids(conn, [collection_id for collection_id in delete_ids if collection_id])
    elif target_collection_id is None:
        existing_ids = collection_ids_for_import_path(conn, str(material.get("path") or ""))
        target_collection_id = existing_ids[0] if existing_ids else None

    collection_name = unique_collection_name(conn, material["title"], target_collection_id)
    update_time = now_ms()
    if target_collection_id is not None:
        conn.execute(
            """
            INSERT INTO collections(id, name, start_update_time, last_update_time, embedding_model)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                name = excluded.name,
                last_update_time = excluded.last_update_time,
                embedding_model = excluded.embedding_model
            """,
            (target_collection_id, collection_name, update_time, update_time, embedding_model),
        )
        collection_id = target_collection_id
    else:
        cursor = conn.execute(
            """
            INSERT INTO collections(name, start_update_time, last_update_time, embedding_model)
            VALUES (?, ?, ?, ?)
            """,
            (collection_name, update_time, update_time, embedding_model),
        )
        collection_id = int(cursor.lastrowid)

    material["id"] = str(collection_id)
    material["title"] = collection_name
    folder_id = upsert_folder(conn, folder_path_for_import(str(material["path"]), str(material.get("kind") or "folder")))
    conn.execute(
        """
        INSERT OR IGNORE INTO collection_items(collection_id, folder_id)
        VALUES (?, ?)
        """,
        (collection_id, folder_id),
    )
    upsert_collection_state(conn, collection_id, material)

    document_id_by_source_id: Dict[str, int] = {}
    document_id_by_path: Dict[str, int] = {}
    for document in documents:
        document_path = str(document["path"])
        document_id = insert_document(conn, folder_id, document_path)
        if replace_thumbnails:
            replace_document_thumbnails(conn, document_id, document.get("thumbnails") or [])
        document_id_by_source_id[str(document["id"])] = document_id
        document_id_by_path[document_path] = document_id

    return collection_id, folder_id, document_id_by_source_id, document_id_by_path, deleted_embedding_models


def upsert_material(
    user_data_path: str,
    material: Dict[str, Any],
    documents: List[Dict[str, Any]],
    chunks: List[Dict[str, Any]],
    *,
    embedding_model: str,
    replace_existing: bool = True,
    rebuild_index: bool = True,
) -> None:
    init_db(user_data_path)
    deleted_embedding_models: List[str] = []

    with connect(user_data_path) as conn:
        _collection_id, folder_id, document_id_by_source_id, document_id_by_path, deleted_embedding_models = upsert_material_header(
            conn,
            material,
            documents,
            embedding_model=embedding_model,
            replace_existing=replace_existing,
            replace_thumbnails=True,
        )

        for chunk in chunks:
            document_id = document_id_by_source_id.get(str(chunk.get("documentId"))) or document_id_by_path.get(str(chunk["path"]))
            if not document_id:
                continue

            chunk_id = upsert_chunk(conn, document_id, chunk)
            embedding = chunk.get("embedding")
            if embedding:
                blob, _dim = vector_to_blob(embedding)
                chunk_embedding_model = chunk.get("embeddingModel") or embedding_model
                conn.execute(
                    """
                    INSERT OR REPLACE INTO embeddings(model, folder_id, chunk_id, embedding)
                    VALUES (?, ?, ?, ?)
                    """,
                    (chunk_embedding_model, folder_id, chunk_id, blob),
                )

    if rebuild_index:
        for deleted_embedding_model in deleted_embedding_models:
            if deleted_embedding_model != embedding_model:
                rebuild_faiss(user_data_path, deleted_embedding_model)
        rebuild_faiss(user_data_path, embedding_model)


def begin_material_index(
    user_data_path: str,
    material: Dict[str, Any],
    documents: List[Dict[str, Any]],
    *,
    embedding_model: str,
    replace_existing: bool = True,
) -> List[str]:
    init_db(user_data_path)

    with connect(user_data_path) as conn:
        _collection_id, _folder_id, _source_ids, _path_ids, deleted_embedding_models = upsert_material_header(
            conn,
            material,
            documents,
            embedding_model=embedding_model,
            replace_existing=replace_existing,
            replace_thumbnails=True,
        )

    return deleted_embedding_models


def append_material_chunks(
    user_data_path: str,
    material_id: str,
    chunks: List[Dict[str, Any]],
    *,
    embedding_model: str,
) -> None:
    if not chunks:
        return

    init_db(user_data_path)
    collection_id = collection_id_from_material_id(material_id)
    if collection_id is None:
        raise ValueError("A saved material id is required before chunks can be appended.")

    with connect(user_data_path) as conn:
        folder_id = material_folder_id(conn, collection_id)
        if folder_id is None:
            raise ValueError("The material folder was not initialized before chunks were appended.")

        document_paths = sorted({str(chunk.get("path") or "") for chunk in chunks if chunk.get("path")})
        document_id_by_path: Dict[str, int] = {}
        for document_path in document_paths:
            row = conn.execute("SELECT id FROM documents WHERE document_path = ?", (document_path,)).fetchone()
            document_id_by_path[document_path] = int(row["id"]) if row else insert_document(conn, folder_id, document_path)

        for chunk in chunks:
            document_id = document_id_by_path.get(str(chunk.get("path") or ""))
            if not document_id:
                continue

            chunk_id = upsert_chunk(conn, document_id, chunk)
            embedding = chunk.get("embedding")
            if not embedding:
                continue

            blob, _dim = vector_to_blob(embedding)
            chunk_embedding_model = chunk.get("embeddingModel") or embedding_model
            conn.execute(
                """
                INSERT OR REPLACE INTO embeddings(model, folder_id, chunk_id, embedding)
                VALUES (?, ?, ?, ?)
                """,
                (chunk_embedding_model, folder_id, chunk_id, blob),
            )


def update_material_index_state(
    user_data_path: str,
    material: Dict[str, Any],
    *,
    embedding_model: str,
    rebuild_index: bool = False,
    deleted_embedding_models: Optional[Sequence[str]] = None,
) -> None:
    init_db(user_data_path)
    collection_id = collection_id_from_material_id(material.get("id"))
    if collection_id is None:
        raise ValueError("A saved material id is required before material state can be updated.")

    with connect(user_data_path) as conn:
        conn.execute(
            """
            UPDATE collections
            SET last_update_time = ?, embedding_model = ?
            WHERE id = ?
            """,
            (now_ms(), embedding_model, collection_id),
        )
        upsert_collection_state(conn, collection_id, material)

    if rebuild_index:
        for deleted_embedding_model in deleted_embedding_models or []:
            if deleted_embedding_model != embedding_model:
                rebuild_faiss(user_data_path, deleted_embedding_model)
        rebuild_faiss(user_data_path, embedding_model)


def rebuild_faiss(user_data_path: str, embedding_model: str) -> None:
    init_db(user_data_path)
    path = faiss_path(user_data_path)

    if faiss is None:
        if path.exists():
            path.unlink()
        return

    with connect(user_data_path) as conn:
        rows = conn.execute(
            """
            SELECT chunk_id, embedding
            FROM embeddings
            WHERE model = ?
            ORDER BY chunk_id
            """,
            (embedding_model,),
        ).fetchall()

    if not rows:
        if path.exists():
            path.unlink()
        with connect(user_data_path) as conn:
            set_schema_value(conn, "faiss_embedding_model", "")
        return

    ids = np.asarray([int(row["chunk_id"]) for row in rows], dtype=np.int64)
    vectors = np.vstack([blob_to_vector(row["embedding"]) for row in rows]).astype("float32")
    vectors = normalize_matrix(vectors)

    index = faiss.IndexIDMap2(faiss.IndexFlatIP(vectors.shape[1]))
    index.add_with_ids(vectors, ids)
    faiss.write_index(index, str(path))

    with connect(user_data_path) as conn:
        set_schema_value(conn, "faiss_embedding_model", embedding_model)


def ensure_faiss(user_data_path: str, embedding_model: str):
    path = faiss_path(user_data_path)
    indexed_model = get_schema_value(user_data_path, "faiss_embedding_model")

    if indexed_model != embedding_model or not path.exists():
        rebuild_faiss(user_data_path, embedding_model)

    if faiss is None or not path.exists():
        return None

    return faiss.read_index(str(path))


def source_row_select() -> str:
    return """
        SELECT
            ch.id AS rowid,
            ch.id AS id,
            d.id AS document_id,
            CAST(col.id AS TEXT) AS material_id,
            col.name AS material_title,
            COALESCE(ch.title, '') AS document_title,
            d.document_path AS path,
            ch.chunk_text AS text,
            ch.words AS word_count,
            ch.page AS page_start,
            ch.page AS page_end,
            ch.id AS chunk_index,
            ch.chunk_size AS chunk_size,
            ch.section_header AS section_header,
            col.embedding_model AS embedding_model,
            pt.thumbnail_path AS thumbnail_path
        FROM chunks ch
        JOIN documents d ON d.id = ch.document_id
        JOIN folders f ON f.id = d.folder_id
        JOIN collection_items ci ON ci.folder_id = f.id
        JOIN collections col ON col.id = ci.collection_id
        JOIN tokensmith_collection_state s ON s.collection_id = col.id
        LEFT JOIN pdf_page_thumbnails pt ON pt.document_id = d.id AND pt.page = ch.page
    """


def get_chunks_by_rowids(
    user_data_path: str,
    rowids: List[int],
    active_material_ids: Sequence[str],
) -> List[Dict[str, Any]]:
    if not rowids or not active_material_ids:
        return []

    row_placeholders = ",".join("?" for _ in rowids)
    active_placeholders = ",".join("?" for _ in active_material_ids)

    with connect(user_data_path) as conn:
        rows = conn.execute(
            f"""
            {source_row_select()}
            WHERE ch.id IN ({row_placeholders})
              AND CAST(col.id AS TEXT) IN ({active_placeholders})
              AND s.status = 'ready'
              AND s.is_active = 1
            """,
            [*rowids, *active_material_ids],
        ).fetchall()

    by_id = {int(row["rowid"]): dict(row) for row in rows}
    return [by_id[rowid] for rowid in rowids if rowid in by_id]


def vector_search(
    user_data_path: str,
    query_embedding: Iterable[float],
    active_material_ids: Sequence[str],
    limit: int,
    embedding_model: str,
) -> List[Tuple[int, float]]:
    if not active_material_ids:
        return []

    index = ensure_faiss(user_data_path, embedding_model)
    if index is None:
        return []

    q = np.asarray([list(query_embedding)], dtype=np.float32)
    q = normalize_matrix(q)

    try:
        distances, labels = index.search(q, max(limit * 8, limit))
    except Exception:
        return []

    raw = [
        (int(label), float(score))
        for label, score in zip(labels[0], distances[0])
        if int(label) >= 0
    ]

    rowids = [rowid for rowid, _score in raw]
    allowed_chunks = get_chunks_by_rowids(user_data_path, rowids, active_material_ids)
    allowed = {int(chunk["rowid"]) for chunk in allowed_chunks}

    return [(rowid, score) for rowid, score in raw if rowid in allowed][:limit]


def fetch_sources(
    user_data_path: str,
    scored_rowids: List[Tuple[int, float]],
    active_material_ids: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    if not scored_rowids:
        return []

    rowids = [rowid for rowid, _score in scored_rowids]
    placeholders = ",".join("?" for _ in rowids)
    active_filter = ""
    params: List[Any] = list(rowids)
    if active_material_ids:
        active_placeholders = ",".join("?" for _ in active_material_ids)
        active_filter = f" AND CAST(col.id AS TEXT) IN ({active_placeholders})"
        params.extend(active_material_ids)

    with connect(user_data_path) as conn:
        rows = conn.execute(
            f"""
            {source_row_select()}
            WHERE ch.id IN ({placeholders})
              AND s.status = 'ready'
              {active_filter}
            """,
            params,
        ).fetchall()

    by_id = {int(row["rowid"]): dict(row) for row in rows}
    result: List[Dict[str, Any]] = []

    for rowid, score in scored_rowids:
        row = by_id.get(rowid)
        if not row:
            continue
        row["score"] = score
        if not row.get("document_title"):
            row["document_title"] = Path(str(row.get("path") or "")).stem
        result.append(row)

    return result


def starter_source_rows(
    user_data_path: str,
    active_material_ids: Sequence[str],
    limit: int = 4,
) -> List[Dict[str, Any]]:
    init_db(user_data_path)

    if limit <= 0 or not active_material_ids:
        return []

    material_ids = [str(material_id) for material_id in active_material_ids if str(material_id).strip()]
    if not material_ids:
        return []

    with connect(user_data_path) as conn:
        for material_id in material_ids:
            document = conn.execute(
                """
                SELECT d.id
                FROM documents d
                JOIN folders f ON f.id = d.folder_id
                JOIN collection_items ci ON ci.folder_id = f.id
                JOIN collections col ON col.id = ci.collection_id
                JOIN tokensmith_collection_state s ON s.collection_id = col.id
                WHERE CAST(col.id AS TEXT) = ?
                  AND s.status = 'ready'
                  AND s.is_active = 1
                  AND EXISTS (
                      SELECT 1
                      FROM chunks ch
                      WHERE ch.document_id = d.id
                  )
                ORDER BY d.document_path COLLATE NOCASE, d.id
                LIMIT 1
                """,
                (material_id,),
            ).fetchone()

            if not document:
                continue

            rows = conn.execute(
                f"""
                {source_row_select()}
                WHERE d.id = ?
                  AND CAST(col.id AS TEXT) = ?
                  AND s.status = 'ready'
                  AND s.is_active = 1
                ORDER BY COALESCE(ch.page, 0), ch.id
                """,
                (document["id"], material_id),
            ).fetchall()

            result: List[Dict[str, Any]] = []
            sampled_rows = rows if len(rows) <= limit else random.sample(list(rows), limit)
            sampled_rows = sorted(sampled_rows, key=lambda row: (row["page_start"] or 0, row["rowid"]))
            for row in sampled_rows:
                source_row = dict(row)
                source_row["score"] = 0.0
                if not source_row.get("document_title"):
                    source_row["document_title"] = Path(str(source_row.get("path") or "")).stem
                result.append(source_row)
            if result:
                return result

    return []


def source_document_for_source(user_data_path: str, source: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    init_db(user_data_path)

    chunk_id = None
    for key in ("chunkRowid", "chunkId"):
        value = source.get(key)
        try:
            if value is not None and str(value).strip():
                chunk_id = int(value)
                break
        except (TypeError, ValueError):
            continue

    document_id = None
    try:
        if source.get("documentId") is not None and str(source.get("documentId")).strip():
            document_id = int(source["documentId"])
    except (TypeError, ValueError):
        document_id = None

    source_path = source.get("path")
    document_path = str(Path(str(source_path)).expanduser().resolve()) if source_path else None
    page = None
    try:
        if source.get("pageStart") is not None:
            page = int(source["pageStart"])
    except (TypeError, ValueError):
        page = None

    where_clauses: List[str] = []
    params: List[Any] = []
    if chunk_id is not None:
        where_clauses.append("ch.id = ?")
        params.append(chunk_id)
    if document_id is not None:
        if page is not None:
            where_clauses.append("(d.id = ? AND ch.page = ?)")
            params.extend([document_id, page])
        else:
            where_clauses.append("d.id = ?")
            params.append(document_id)
    if document_path:
        if page is not None:
            where_clauses.append("(d.document_path = ? AND ch.page = ?)")
            params.extend([document_path, page])
        else:
            where_clauses.append("d.document_path = ?")
            params.append(document_path)

    if not where_clauses:
        return None

    with connect(user_data_path) as conn:
        row = conn.execute(
            f"""
            SELECT
                ch.id AS chunk_id,
                d.id AS document_id,
                d.document_path AS path,
                COALESCE(ch.title, '') AS title,
                ch.page AS page,
                col.name AS collection_name,
                pt.thumbnail_path AS thumbnail_path
            FROM chunks ch
            JOIN documents d ON d.id = ch.document_id
            JOIN folders f ON f.id = d.folder_id
            JOIN collection_items ci ON ci.folder_id = f.id
            JOIN collections col ON col.id = ci.collection_id
            LEFT JOIN pdf_page_thumbnails pt ON pt.document_id = d.id AND pt.page = ch.page
            WHERE ({' OR '.join(where_clauses)})
            ORDER BY CASE WHEN ch.id = ? THEN 0 ELSE 1 END, ch.id
            LIMIT 1
            """,
            [*params, chunk_id if chunk_id is not None else -1],
        ).fetchone()

    if not row:
        return None

    title = str(row["title"] or "") or Path(str(row["path"])).stem
    return {
        "chunkId": int(row["chunk_id"]),
        "documentId": int(row["document_id"]),
        "path": str(row["path"]),
        "title": title,
        "page": int(row["page"]) if row["page"] is not None else None,
        "collectionName": row["collection_name"],
        "thumbnailPath": row["thumbnail_path"],
    }


def enabled_material_ids(user_data_path: str, requested_material_ids: Sequence[str]) -> List[str]:
    init_db(user_data_path)

    if not requested_material_ids:
        return []

    unique_requested = list(dict.fromkeys(material_id for material_id in requested_material_ids if material_id))
    if not unique_requested:
        return []

    placeholders = ",".join("?" for _ in unique_requested)
    with connect(user_data_path) as conn:
        rows = conn.execute(
            f"""
            SELECT CAST(c.id AS TEXT) AS id
            FROM collections c
            JOIN tokensmith_collection_state s ON s.collection_id = c.id
            WHERE CAST(c.id AS TEXT) IN ({placeholders})
              AND s.status = 'ready'
              AND s.is_active = 1
            """,
            unique_requested,
        ).fetchall()

    allowed = {str(row["id"]) for row in rows}
    return [material_id for material_id in unique_requested if material_id in allowed]


def enabled_material_ids_for_requests(user_data_path: str, requested_materials: Sequence[Dict[str, Any]]) -> List[str]:
    requested_ids = [str(material.get("id")) for material in requested_materials if material.get("id")]
    return enabled_material_ids(user_data_path, requested_ids)


def parse_json_string_list(value: Any) -> List[str]:
    if not value:
        return []

    try:
        parsed = json.loads(str(value))
    except Exception:
        return []

    if not isinstance(parsed, list):
        return []

    return [str(item) for item in parsed if isinstance(item, str)]


def list_materials(user_data_path: str) -> List[Dict[str, Any]]:
    init_db(user_data_path)

    with connect(user_data_path) as conn:
        rows = conn.execute(
            """
            SELECT
                c.id,
                c.name,
                c.start_update_time,
                c.last_update_time,
                c.embedding_model,
                s.status,
                s.kind,
                s.import_path,
                s.embedding_model_id,
                s.embedding_model_name,
                s.cleaning_profile_id,
                s.cleaning_profile_name,
                s.cleaning_profile_version,
                s.cleaning_rule_ids_json,
                s.detail,
                s.added_at,
                s.indexed_at,
                s.is_active,
                s.file_count,
                s.size_bytes,
                s.word_count,
                s.page_count,
                s.chunk_count,
                s.chunk_size,
                s.error,
                MIN(f.path) AS folder_path
            FROM collections c
            LEFT JOIN tokensmith_collection_state s ON s.collection_id = c.id
            LEFT JOIN collection_items ci ON ci.collection_id = c.id
            LEFT JOIN folders f ON f.id = ci.folder_id
            GROUP BY c.id
            ORDER BY COALESCE(s.indexed_at, s.added_at, CAST(c.last_update_time AS TEXT)) DESC
            """
        ).fetchall()

    materials: List[Dict[str, Any]] = []
    for row in rows:
        title = str(row["name"])
        path_value = row["import_path"] or row["folder_path"]
        file_count = int(row["file_count"] or 0)
        word_count = int(row["word_count"] or 0)
        chunk_count = int(row["chunk_count"] or 0)
        detail = row["detail"] or (
            f"{file_count} {'file' if file_count == 1 else 'files'} - "
            f"{word_count:,} words - {chunk_count:,} chunks"
        )
        status = row["status"] or ("ready" if chunk_count else "needsReview")

        materials.append(
            {
                "id": str(row["id"]),
                "title": title,
                "detail": detail,
                "status": status,
                "kind": row["kind"] or "folder",
                "path": path_value,
                "addedAt": row["added_at"] or now_iso(),
                "indexedAt": row["indexed_at"],
                "isActive": bool(row["is_active"]) if row["is_active"] is not None else status == "ready",
                "fileCount": file_count,
                "sizeBytes": int(row["size_bytes"] or 0),
                "wordCount": word_count,
                "pageCount": row["page_count"],
                "chunkCount": chunk_count,
                "chunkSize": row["chunk_size"],
                "embeddingModel": row["embedding_model"],
                "embeddingModelId": row["embedding_model_id"],
                "embeddingModelName": row["embedding_model_name"],
                "cleaningProfileId": row["cleaning_profile_id"],
                "cleaningProfileName": row["cleaning_profile_name"],
                "cleaningProfileVersion": row["cleaning_profile_version"],
                "cleaningRuleIds": parse_json_string_list(row["cleaning_rule_ids_json"]),
                "error": row["error"],
            }
        )

    return materials


def set_material_active(user_data_path: str, material_id: str, is_active: bool) -> None:
    init_db(user_data_path)
    collection_id = collection_id_from_material_id(material_id)
    if collection_id is None:
        return

    with connect(user_data_path) as conn:
        row = conn.execute("SELECT id, name FROM collections WHERE id = ?", (collection_id,)).fetchone()
        if not row:
            return
        conn.execute(
            """
            INSERT INTO tokensmith_collection_state(collection_id, status, kind, added_at, is_active)
            VALUES (?, 'ready', 'folder', ?, ?)
            ON CONFLICT(collection_id) DO UPDATE SET is_active = excluded.is_active
            """,
            (collection_id, now_iso(), 1 if is_active else 0),
        )


def has_chunks(user_data_path: str, material_ids: Optional[Sequence[str]] = None) -> bool:
    init_db(user_data_path)

    with connect(user_data_path) as conn:
        if not material_ids:
            row = conn.execute("SELECT 1 FROM chunks LIMIT 1").fetchone()
        else:
            placeholders = ",".join("?" for _ in material_ids)
            row = conn.execute(
                f"""
                SELECT 1
                FROM chunks ch
                JOIN documents d ON d.id = ch.document_id
                JOIN folders f ON f.id = d.folder_id
                JOIN collection_items ci ON ci.folder_id = f.id
                WHERE CAST(ci.collection_id AS TEXT) IN ({placeholders})
                LIMIT 1
                """,
                list(material_ids),
            ).fetchone()

    return row is not None


def dump_index(user_data_path: str) -> Dict[str, Any]:
    init_db(user_data_path)

    with connect(user_data_path) as conn:
        documents = [dict(row) for row in conn.execute("SELECT * FROM documents ORDER BY document_path").fetchall()]
        chunk_rows = conn.execute(
            """
            SELECT
                ch.*,
                e.model AS embedding_model,
                e.embedding AS embedding,
                CAST(ci.collection_id AS TEXT) AS material_id,
                col.name AS material_title,
                d.document_path AS path
            FROM chunks ch
            JOIN documents d ON d.id = ch.document_id
            JOIN folders f ON f.id = d.folder_id
            LEFT JOIN collection_items ci ON ci.folder_id = f.id
            LEFT JOIN collections col ON col.id = ci.collection_id
            LEFT JOIN embeddings e ON e.chunk_id = ch.id
            ORDER BY ch.id
            """
        ).fetchall()

    chunks: List[Dict[str, Any]] = []
    for row in chunk_rows:
        chunk = dict(row)
        embedding = chunk.pop("embedding", None)
        embedding_model = chunk.get("embedding_model")
        if embedding and embedding_model:
            chunk["embeddings"] = {embedding_model: blob_to_vector(embedding).astype(float).tolist()}
        chunk["rowid"] = chunk["id"]
        chunk["text"] = chunk["chunk_text"]
        chunk["document_title"] = chunk.get("title") or Path(str(chunk.get("path") or "")).stem
        chunk["sectionHeader"] = chunk.get("section_header")
        chunks.append(chunk)

    return {"version": SCHEMA_VERSION, "documents": documents, "chunks": chunks, "updatedAt": now_iso()}
