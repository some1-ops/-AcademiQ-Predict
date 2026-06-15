"""
database.py — SQLite schema initialization, user management, and
dataset/training-run logging for the Prediction System.
"""

import sqlite3
import hashlib
import os
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DB_PATH  = DATA_DIR / "students.db"
UPLOADS_DIR = DATA_DIR / "uploads"


def _ensure_dirs():
    DATA_DIR.mkdir(exist_ok=True)
    UPLOADS_DIR.mkdir(exist_ok=True)


def _hash(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


# ── Connection ────────────────────────────────────────────────────────────────
def get_connection() -> sqlite3.Connection:
    _ensure_dirs()
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


# ── Schema Init ───────────────────────────────────────────────────────────────
def init_db():
    """Create all tables and seed default users if they don't exist."""
    _ensure_dirs()
    conn = get_connection()
    cur  = conn.cursor()

    # Users table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            username   TEXT    UNIQUE NOT NULL,
            password   TEXT    NOT NULL,
            role       TEXT    NOT NULL CHECK(role IN ('admin','student')),
            created_at TEXT    DEFAULT (datetime('now'))
        )
    """)

    # Datasets metadata table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS datasets (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            filename    TEXT    NOT NULL,
            row_count   INTEGER,
            col_count   INTEGER,
            uploaded_at TEXT    DEFAULT (datetime('now')),
            uploaded_by TEXT,
            file_path   TEXT
        )
    """)

    # Training runs log
    cur.execute("""
        CREATE TABLE IF NOT EXISTS training_runs (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_id   INTEGER,
            accuracy     REAL,
            precision_s  REAL,
            recall_s     REAL,
            f1_score     REAL,
            train_size   INTEGER,
            test_size    INTEGER,
            trained_at   TEXT    DEFAULT (datetime('now')),
            trained_by   TEXT,
            model_path   TEXT
        )
    """)

    # Predictions log
    cur.execute("""
        CREATE TABLE IF NOT EXISTS prediction_log (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id    TEXT,
            input_data    TEXT,
            predicted     TEXT,
            predicted_at  TEXT    DEFAULT (datetime('now')),
            predicted_by  TEXT
        )
    """)

    conn.commit()

    # Seed default users
    _seed_users(cur, conn)
    conn.close()


def _seed_users(cur, conn):
    default_users = [
        ("admin",   _hash("admin123"),   "admin"),
        ("student", _hash("student123"), "student"),
    ]
    for username, pw_hash, role in default_users:
        cur.execute(
            "INSERT OR IGNORE INTO users (username, password, role) VALUES (?,?,?)",
            (username, pw_hash, role)
        )
    conn.commit()


# ── Auth helpers ──────────────────────────────────────────────────────────────
def verify_user(username: str, password: str):
    """Return user Row if credentials match, else None."""
    conn = get_connection()
    cur  = conn.cursor()
    cur.execute(
        "SELECT * FROM users WHERE username=? AND password=?",
        (username, _hash(password))
    )
    row = cur.fetchone()
    conn.close()
    return row


# ── Dataset helpers ───────────────────────────────────────────────────────────
def log_dataset(filename: str, row_count: int, col_count: int,
                file_path: str, uploaded_by: str) -> int:
    conn = get_connection()
    cur  = conn.cursor()
    cur.execute(
        """INSERT INTO datasets (filename, row_count, col_count, file_path, uploaded_by)
           VALUES (?,?,?,?,?)""",
        (filename, row_count, col_count, file_path, uploaded_by)
    )
    conn.commit()
    row_id = cur.lastrowid
    conn.close()
    return row_id


def get_datasets():
    conn = get_connection()
    cur  = conn.cursor()
    cur.execute("SELECT * FROM datasets ORDER BY uploaded_at DESC")
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


# ── Training run helpers ──────────────────────────────────────────────────────
def log_training_run(dataset_id, accuracy, precision_s, recall_s, f1_score,
                     train_size, test_size, trained_by, model_path) -> int:
    conn = get_connection()
    cur  = conn.cursor()
    cur.execute(
        """INSERT INTO training_runs
           (dataset_id, accuracy, precision_s, recall_s, f1_score,
            train_size, test_size, trained_by, model_path)
           VALUES (?,?,?,?,?,?,?,?,?)""",
        (dataset_id, accuracy, precision_s, recall_s, f1_score,
         train_size, test_size, trained_by, model_path)
    )
    conn.commit()
    row_id = cur.lastrowid
    conn.close()
    return row_id


def get_latest_training_run():
    conn = get_connection()
    cur  = conn.cursor()
    cur.execute("SELECT * FROM training_runs ORDER BY trained_at DESC LIMIT 1")
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None


# ── Prediction log helpers ────────────────────────────────────────────────────
def log_prediction(student_id, input_data, predicted, predicted_by):
    conn = get_connection()
    cur  = conn.cursor()
    cur.execute(
        """INSERT INTO prediction_log (student_id, input_data, predicted, predicted_by)
           VALUES (?,?,?,?)""",
        (student_id, str(input_data), predicted, predicted_by)
    )
    conn.commit()
    conn.close()
