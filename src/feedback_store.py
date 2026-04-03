import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

DB_PATH = Path("index/cache/embeddings.db")


def init_feedback_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS answers (
                answer_id TEXT PRIMARY KEY,
                answer_seq INTEGER,
                session_id TEXT,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                created_at TEXT NOT NULL,
                retrieval_json TEXT,
                model TEXT,
                prompt_mode TEXT
            );
            """
        )
        try:
            conn.execute("ALTER TABLE answers ADD COLUMN session_id TEXT;")
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute("ALTER TABLE answers ADD COLUMN answer_seq INTEGER;")
        except sqlite3.OperationalError:
            pass
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback (
                feedback_id INTEGER PRIMARY KEY AUTOINCREMENT,
                answer_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                vote INTEGER NOT NULL CHECK (vote IN (1, -1)),
                reason TEXT,
                created_at TEXT NOT NULL,
                UNIQUE(answer_id, session_id),
                FOREIGN KEY(answer_id) REFERENCES answers(answer_id)
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS user_profile (
                session_id TEXT PRIMARY KEY,
                summary_json TEXT,
                updated_at TEXT NOT NULL
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS user_topic_state (
                session_id TEXT NOT NULL,
                topic TEXT NOT NULL,
                difficulty TEXT NOT NULL,
                confidence REAL NOT NULL,
                evidence_json TEXT,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (session_id, topic)
            );
            """
        )
        conn.commit()
    finally:
        conn.close()


def save_answer(
    answer_id: str,
    session_id: Optional[str],
    question: str,
    answer: str,
    retrieval_info: Optional[Dict[str, Any]] = None,
    model: Optional[str] = None,
    prompt_mode: Optional[str] = None,
) -> None:
    payload = json.dumps(retrieval_info or {}, ensure_ascii=False)
    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.execute(
            """
            INSERT INTO answers (answer_id, session_id, question, answer, created_at, retrieval_json, model, prompt_mode)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                answer_id,
                session_id,
                question,
                answer,
                datetime.utcnow().isoformat(),
                payload,
                model,
                prompt_mode,
            ),
        )
        try:
            conn.execute(
                "UPDATE answers SET answer_seq=? WHERE answer_id=?",
                (cursor.lastrowid, answer_id),
            )
        except sqlite3.OperationalError:
            pass
        conn.commit()
    finally:
        conn.close()


def save_feedback(
    answer_id: str,
    session_id: str,
    vote: int,
    reason: Optional[str] = None,
) -> None:
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(
            """
            INSERT INTO feedback (answer_id, session_id, vote, reason, created_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(answer_id, session_id)
            DO UPDATE SET vote=excluded.vote, reason=excluded.reason, created_at=excluded.created_at
            """,
            (
                answer_id,
                session_id,
                vote,
                reason,
                datetime.utcnow().isoformat(),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def get_answer_question(answer_id: str) -> Optional[str]:
    conn = sqlite3.connect(DB_PATH)
    try:
        row = conn.execute(
            "SELECT question FROM answers WHERE answer_id=?",
            (answer_id,),
        ).fetchone()
        return row[0] if row else None
    finally:
        conn.close()


def _difficulty_rank(value: str) -> int:
    order = {"easy": 0, "medium": 1, "hard": 2}
    return order.get(value, 1)


def update_user_topic_state(
    session_id: str,
    topic: str,
    difficulty: str,
    delta_confidence: float,
    evidence: Optional[Dict[str, Any]] = None,
    max_evidence: int = 5,
) -> None:
    conn = sqlite3.connect(DB_PATH)
    try:
        row = conn.execute(
            """
            SELECT difficulty, confidence, evidence_json
            FROM user_topic_state
            WHERE session_id=? AND topic=?
            """,
            (session_id, topic),
        ).fetchone()

        now = datetime.utcnow().isoformat()
        if row:
            current_difficulty, confidence, evidence_json = row
            confidence = max(0.0, min(1.0, float(confidence) + float(delta_confidence)))
            merged_difficulty = (
                difficulty
                if _difficulty_rank(difficulty) >= _difficulty_rank(current_difficulty)
                else current_difficulty
            )
            existing_evidence = json.loads(evidence_json) if evidence_json else []
        else:
            confidence = max(0.0, min(1.0, 0.5 + float(delta_confidence)))
            merged_difficulty = difficulty
            existing_evidence = []

        if evidence:
            evidence = dict(evidence)
            evidence.setdefault("timestamp", now)
            existing_evidence.append(evidence)
            existing_evidence = existing_evidence[-max_evidence:]

        conn.execute(
            """
            INSERT INTO user_topic_state (session_id, topic, difficulty, confidence, evidence_json, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id, topic)
            DO UPDATE SET
                difficulty=excluded.difficulty,
                confidence=excluded.confidence,
                evidence_json=excluded.evidence_json,
                updated_at=excluded.updated_at
            """,
            (
                session_id,
                topic,
                merged_difficulty,
                confidence,
                json.dumps(existing_evidence, ensure_ascii=False),
                now,
            ),
        )
        conn.commit()
    finally:
        conn.close()
