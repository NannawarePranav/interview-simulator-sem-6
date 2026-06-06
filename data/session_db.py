"""
session_db.py — SQLite session persistence for AI Mock Interviewer

Schema:
  sessions      : session_id, timestamp, candidate_name, readiness_level, overall_score
  question_logs : id, session_id, topic, difficulty, question, answer, score, time_taken
"""
import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sessions.db')


def _get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create tables if they don't already exist."""
    conn = _get_connection()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id      TEXT PRIMARY KEY,
            timestamp       TEXT,
            candidate_name  TEXT,
            readiness_level TEXT,
            overall_score   REAL
        );

        CREATE TABLE IF NOT EXISTS question_logs (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id  TEXT,
            topic       TEXT,
            difficulty  TEXT,
            question    TEXT,
            answer      TEXT,
            score       REAL,
            time_taken  INTEGER
        );
    """)
    conn.commit()
    conn.close()


def save_session(session_id: str, candidate_name: str,
                 readiness_level: str, overall_score: float):
    conn = _get_connection()
    conn.execute(
        """INSERT OR REPLACE INTO sessions
           (session_id, timestamp, candidate_name, readiness_level, overall_score)
           VALUES (?, ?, ?, ?, ?)""",
        (session_id, datetime.utcnow().isoformat(), candidate_name,
         readiness_level, overall_score)
    )
    conn.commit()
    conn.close()


def save_question_log(session_id: str, topic: str, difficulty: str,
                      question: str, answer: str, score: float,
                      time_taken: int = 0):
    conn = _get_connection()
    conn.execute(
        """INSERT INTO question_logs
           (session_id, topic, difficulty, question, answer, score, time_taken)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (session_id, topic, difficulty, question, answer, score, time_taken)
    )
    conn.commit()
    conn.close()


def get_recent_sessions(limit: int = 10) -> list:
    conn = _get_connection()
    rows = conn.execute(
        """SELECT session_id, timestamp, candidate_name, readiness_level, overall_score
           FROM sessions ORDER BY timestamp DESC LIMIT ?""",
        (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# Auto-init on import
init_db()
