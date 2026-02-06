import sqlite3
from pathlib import Path
from typing import List, Dict

def init_db(db_path: Path) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.executescript("""
            DROP TABLE IF EXISTS paragraphs;
            DROP TABLE IF EXISTS sections;
            CREATE TABLE sections (
                id INTEGER PRIMARY KEY, heading TEXT, section_summary TEXT,
                one_line_summary TEXT, prev_section_id INTEGER, next_section_id INTEGER,
                num_paragraphs INTEGER, content_length INTEGER
            );
            CREATE TABLE paragraphs (
                id INTEGER PRIMARY KEY, section_id INTEGER, para_index INTEGER,
                one_line_summary TEXT, raw_text TEXT,
                FOREIGN KEY(section_id) REFERENCES sections(id)
            );
            CREATE INDEX idx_paragraphs_section ON paragraphs(section_id, para_index);
        """)

def save_section_to_db(db_path: Path, section_id: int, heading: str, summary: str, one_line: str, paras: List[Dict]):
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO sections (id, heading, section_summary, one_line_summary, content_length) VALUES (?, ?, ?, ?, ?)",
            (section_id, heading, summary, one_line, 0) # simplified
        )
        for p in paras:
            cur.execute(
                "INSERT INTO paragraphs (section_id, para_index, one_line_summary, raw_text) VALUES (?, ?, ?, ?)",
                (section_id, p['index'], p['summary'], p['text'])
            )
        conn.commit()
