import sqlite3
import json
import time
from pathlib import Path
from typing import List, Dict, Optional, Any

class Database:
    def __init__(self, db_path: str = "data/tokensmith.db"):
        """Initialize the database connection and schema."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self):
        """Create necessary tables if they don't exist."""
        with self.conn:
            # Sessions table
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at REAL DEFAULT (unixepoch())
                )
            """)
            
            # Messages table
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp REAL DEFAULT (unixepoch()),
                    FOREIGN KEY (session_id) REFERENCES sessions (id)
                )
            """)
            
            # Cache table
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS response_cache (
                    query_hash TEXT PRIMARY KEY,
                    query TEXT NOT NULL,
                    response TEXT NOT NULL,
                    timestamp REAL DEFAULT (unixepoch())
                )
            """)

    def create_session(self) -> int:
        """Create a new chat session and return its ID."""
        with self.conn:
            cursor = self.conn.execute("INSERT INTO sessions DEFAULT VALUES")
            return cursor.lastrowid

    def add_message(self, session_id: int, role: str, content: str):
        """Add a message to the chat history."""
        with self.conn:
            self.conn.execute(
                "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
                (session_id, role, content)
            )

    def get_sessions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieve recent sessions with their message counts."""
        cursor = self.conn.execute(
            """
            SELECT s.id, s.created_at, COUNT(m.id) as message_count 
            FROM sessions s 
            LEFT JOIN messages m ON s.id = m.session_id 
            GROUP BY s.id 
            ORDER BY s.created_at DESC 
            LIMIT ?
            """,
            (limit,)
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_history(self, session_id: int) -> List[Dict[str, Any]]:
        """Retrieve chat history for a session."""
        cursor = self.conn.execute(
            "SELECT role, content, timestamp FROM messages WHERE session_id = ? ORDER BY timestamp ASC",
            (session_id,)
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_cached_response(self, query: str) -> Optional[str]:
        """Retrieve a cached response for a query."""
        # Simple normalization: strip whitespace and lowercase
        normalized_query = query.strip().lower()
        cursor = self.conn.execute(
            "SELECT response FROM response_cache WHERE query = ?",
            (normalized_query,)
        )
        row = cursor.fetchone()
        return row['response'] if row else None

    def cache_response(self, query: str, response: str):
        """Cache a response for a query."""
        normalized_query = query.strip().lower()
        # Simple hash for the primary key (or just use the query itself if unique enough)
        # Here we use the query itself as the key logic, but let's just use the normalized query as the key
        # to avoid complex hashing for now, or we can use a hash function.
        # Let's just use the normalized query as the unique constraint logic.
        
        # We'll use a simple hash of the normalized query for the ID to keep it clean, 
        # but actually the table definition has query_hash. Let's just use the normalized query string as the hash for now
        # or actually implement a hash.
        import hashlib
        query_hash = hashlib.sha256(normalized_query.encode()).hexdigest()
        
        with self.conn:
            self.conn.execute(
                """
                INSERT OR REPLACE INTO response_cache (query_hash, query, response)
                VALUES (?, ?, ?)
                """,
                (query_hash, normalized_query, response)
            )

    def close(self):
        """Close the database connection."""
        self.conn.close()
