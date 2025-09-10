

import sqlite3
import json
import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path

@dataclass
class FeedbackEntry:
    id: Optional[int] = None
    timestamp: str = ""
    query: str = ""
    answer: str = ""
    retrieved_chunks: str = ""
    thumbs_up: Optional[bool] = None
    comment: str = ""
    rating: Optional[int] = None
    improvement_suggestions: str = ""
    session_id: str = ""
    prompt_style: str = "default"

class FeedbackDB:
    
    def __init__(self, db_path: str = "feedback.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    query TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    retrieved_chunks TEXT NOT NULL,
                    thumbs_up INTEGER,
                    comment TEXT,
                    rating INTEGER,
                    improvement_suggestions TEXT,
                    session_id TEXT,
                    prompt_style TEXT DEFAULT 'default',
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            

            try:
                conn.execute("ALTER TABLE feedback ADD COLUMN prompt_style TEXT DEFAULT 'default'")
            except sqlite3.OperationalError:

                pass
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS system_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    metadata TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS improvement_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    improvement_type TEXT NOT NULL,
                    description TEXT NOT NULL,
                    before_value TEXT,
                    after_value TEXT,
                    feedback_count INTEGER,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
    
    def add_feedback(self, feedback: FeedbackEntry) -> int:
        feedback.timestamp = datetime.datetime.now().isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO feedback 
                (timestamp, query, answer, retrieved_chunks, thumbs_up, comment, 
                 rating, improvement_suggestions, session_id, prompt_style)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                feedback.timestamp,
                feedback.query,
                feedback.answer,
                feedback.retrieved_chunks,
                feedback.thumbs_up,
                feedback.comment,
                feedback.rating,
                feedback.improvement_suggestions,
                feedback.session_id,
                feedback.prompt_style
            ))
            return cursor.lastrowid
    
    def get_feedback_stats(self) -> Dict:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total_feedback,
                    AVG(CASE WHEN thumbs_up = 1 THEN 1.0 ELSE 0.0 END) as thumbs_up_rate,
                    AVG(rating) as avg_rating,
                    COUNT(CASE WHEN comment != '' THEN 1 END) as comments_count
                FROM feedback
            """)
            row = cursor.fetchone()
            
            return {
                "total_feedback": row[0] or 0,
                "thumbs_up_rate": row[1] or 0.0,
                "avg_rating": row[2] or 0.0,
                "comments_count": row[3] or 0
            }
    
    def get_recent_feedback(self, limit: int = 50) -> List[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM feedback 
                ORDER BY created_at DESC 
                LIMIT ?
            """, (limit,))
            return [dict(row) for row in cursor.fetchall()]
    
    def get_negative_feedback(self) -> List[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM feedback 
                WHERE thumbs_up = 0 OR rating < 3 OR comment != ''
                ORDER BY created_at DESC
            """)
            return [dict(row) for row in cursor.fetchall()]
    
    def get_query_patterns(self) -> List[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT 
                    query,
                    COUNT(*) as frequency,
                    AVG(CASE WHEN thumbs_up = 1 THEN 1.0 ELSE 0.0 END) as success_rate,
                    AVG(rating) as avg_rating
                FROM feedback 
                GROUP BY query
                HAVING COUNT(*) > 1
                ORDER BY frequency DESC, success_rate ASC
            """)
            return [dict(zip([col[0] for col in cursor.description], row)) 
                   for row in cursor.fetchall()]
    
    def add_system_metric(self, metric_name: str, value: float, metadata: str = ""):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO system_metrics (timestamp, metric_name, metric_value, metadata)
                VALUES (?, ?, ?, ?)
            """, (datetime.datetime.now().isoformat(), metric_name, value, metadata))
    
    def log_improvement(self, improvement_type: str, description: str, 
                       before_value: str = "", after_value: str = "", 
                       feedback_count: int = 0):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO improvement_log 
                (timestamp, improvement_type, description, before_value, after_value, feedback_count)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (datetime.datetime.now().isoformat(), improvement_type, description,
                 before_value, after_value, feedback_count))
