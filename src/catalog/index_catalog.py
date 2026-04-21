"""
SQLite catalog for indexed chunk metadata.

This module supports:
- writing index metadata during build (dual-write with pickle artifacts)
- loading chunk metadata at startup with pickle fallback in retriever
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS index_builds (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            index_prefix TEXT NOT NULL,
            artifacts_dir TEXT NOT NULL,
            source_markdown TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS chunk_metadata (
            build_id INTEGER NOT NULL,
            chunk_id INTEGER NOT NULL,
            source_path TEXT,
            chunk_text TEXT,
            metadata_json TEXT NOT NULL,
            PRIMARY KEY (build_id, chunk_id),
            FOREIGN KEY (build_id) REFERENCES index_builds(id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_index_builds_prefix ON index_builds(index_prefix, id DESC);
        CREATE INDEX IF NOT EXISTS idx_chunk_metadata_build ON chunk_metadata(build_id, chunk_id);
        """
    )


class IndexCatalog:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)

    def init(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(self.db_path)) as conn:
            _ensure_schema(conn)
            conn.commit()

    def write_build(
        self,
        *,
        index_prefix: str,
        artifacts_dir: Path,
        source_markdown: str,
        chunks: List[str],
        sources: List[str],
        metadata: List[Dict[str, Any]],
    ) -> int:
        if not (len(chunks) == len(sources) == len(metadata)):
            raise ValueError("chunks, sources, metadata lengths must match")
        self.init()
        with sqlite3.connect(str(self.db_path)) as conn:
            _ensure_schema(conn)
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO index_builds (index_prefix, artifacts_dir, source_markdown)
                VALUES (?, ?, ?)
                """,
                (index_prefix, str(artifacts_dir), source_markdown),
            )
            build_id = int(cur.lastrowid)
            for chunk_id, (chunk_text, source_path, meta) in enumerate(
                zip(chunks, sources, metadata)
            ):
                cur.execute(
                    """
                    INSERT INTO chunk_metadata (build_id, chunk_id, source_path, chunk_text, metadata_json)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        build_id,
                        int(chunk_id),
                        str(source_path),
                        chunk_text,
                        json.dumps(meta, ensure_ascii=False, default=str),
                    ),
                )
            conn.commit()
            return build_id

    def load_latest_build(
        self,
        *,
        index_prefix: str,
    ) -> Optional[tuple[List[str], List[str], List[Dict[str, Any]]]]:
        if not self.db_path.exists():
            return None
        with sqlite3.connect(str(self.db_path)) as conn:
            _ensure_schema(conn)
            row = conn.execute(
                """
                SELECT id FROM index_builds
                WHERE index_prefix = ?
                ORDER BY id DESC
                LIMIT 1
                """,
                (index_prefix,),
            ).fetchone()
            if row is None:
                return None
            build_id = int(row[0])
            rows = conn.execute(
                """
                SELECT chunk_text, source_path, metadata_json
                FROM chunk_metadata
                WHERE build_id = ?
                ORDER BY chunk_id ASC
                """,
                (build_id,),
            ).fetchall()
            chunks: List[str] = []
            sources: List[str] = []
            metadata: List[Dict[str, Any]] = []
            for chunk_text, source_path, metadata_json in rows:
                chunks.append(chunk_text)
                sources.append(source_path)
                metadata.append(json.loads(metadata_json))
            return chunks, sources, metadata
