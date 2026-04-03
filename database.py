import sqlite3
import json
import os

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "clipify.db")


def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    conn = get_db()
    try:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                email             TEXT    UNIQUE NOT NULL,
                name              TEXT    NOT NULL,
                password_hash     TEXT    NOT NULL,
                created_at        TEXT    DEFAULT (datetime('now')),
                default_annotate  INTEGER DEFAULT 0,
                default_highlight INTEGER DEFAULT 1,
                default_emojis    INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS history (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                filename    TEXT,
                mode        TEXT    NOT NULL,
                clips_count INTEGER DEFAULT 0,
                clips_json  TEXT,
                created_at  TEXT    DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS password_reset_tokens (
                token       TEXT    PRIMARY KEY,
                user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                expires_at  TEXT    NOT NULL,
                used        INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS cache (
                file_hash   TEXT    NOT NULL,
                mode        TEXT    NOT NULL,
                clip_length TEXT    NOT NULL,
                max_clips   INTEGER NOT NULL,
                result_json TEXT    NOT NULL,
                created_at  TEXT    DEFAULT (datetime('now')),
                PRIMARY KEY (file_hash, mode, clip_length, max_clips)
            );
        """)
        conn.commit()
    finally:
        conn.close()


# ── Users ─────────────────────────────────────────────────────────────────────

def create_user(email: str, name: str, password_hash: str) -> int:
    conn = get_db()
    try:
        cur = conn.execute(
            "INSERT INTO users (email, name, password_hash) VALUES (?, ?, ?)",
            (email.lower().strip(), name.strip(), password_hash),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def get_user_by_email(email: str):
    conn = get_db()
    try:
        return conn.execute(
            "SELECT * FROM users WHERE email = ?", (email.lower().strip(),)
        ).fetchone()
    finally:
        conn.close()


def get_user_by_id(user_id: int):
    conn = get_db()
    try:
        return conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    finally:
        conn.close()


def update_user(user_id: int, **fields):
    if not fields:
        return
    set_clause = ", ".join(f"{k} = ?" for k in fields)
    values = list(fields.values()) + [user_id]
    conn = get_db()
    try:
        conn.execute(f"UPDATE users SET {set_clause} WHERE id = ?", values)
        conn.commit()
    finally:
        conn.close()


# ── History ───────────────────────────────────────────────────────────────────

def add_history(user_id: int, filename: str, mode: str, clips_count: int, clips_json: str) -> int:
    conn = get_db()
    try:
        cur = conn.execute(
            "INSERT INTO history (user_id, filename, mode, clips_count, clips_json) VALUES (?, ?, ?, ?, ?)",
            (user_id, filename, mode, clips_count, clips_json),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def get_history(user_id: int) -> list:
    conn = get_db()
    try:
        rows = conn.execute(
            "SELECT * FROM history WHERE user_id = ? ORDER BY created_at DESC",
            (user_id,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def delete_history_item(item_id: int, user_id: int) -> bool:
    conn = get_db()
    try:
        cur = conn.execute(
            "DELETE FROM history WHERE id = ? AND user_id = ?", (item_id, user_id)
        )
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


# ── Password reset tokens ──────────────────────────────────────────────────────

def create_reset_token(user_id: int, token: str, expires_at: str) -> None:
    conn = get_db()
    try:
        # Invalidate any existing unused tokens for this user first
        conn.execute("UPDATE password_reset_tokens SET used = 1 WHERE user_id = ? AND used = 0", (user_id,))
        conn.execute(
            "INSERT INTO password_reset_tokens (token, user_id, expires_at) VALUES (?, ?, ?)",
            (token, user_id, expires_at),
        )
        conn.commit()
    finally:
        conn.close()


def get_reset_token(token: str):
    conn = get_db()
    try:
        return conn.execute(
            "SELECT * FROM password_reset_tokens WHERE token = ? AND used = 0",
            (token,),
        ).fetchone()
    finally:
        conn.close()


def consume_reset_token(token: str) -> None:
    conn = get_db()
    try:
        conn.execute("UPDATE password_reset_tokens SET used = 1 WHERE token = ?", (token,))
        conn.commit()
    finally:
        conn.close()


# ── Cache ─────────────────────────────────────────────────────────────────────

def get_cache(file_hash: str, mode: str, clip_length: str, max_clips: int):
    conn = get_db()
    try:
        row = conn.execute(
            "SELECT result_json FROM cache WHERE file_hash=? AND mode=? AND clip_length=? AND max_clips=?",
            (file_hash, mode, clip_length, max_clips),
        ).fetchone()
        return json.loads(row["result_json"]) if row else None
    finally:
        conn.close()


def set_cache(file_hash: str, mode: str, clip_length: str, max_clips: int, result: dict) -> None:
    conn = get_db()
    try:
        conn.execute(
            """INSERT OR REPLACE INTO cache (file_hash, mode, clip_length, max_clips, result_json)
               VALUES (?, ?, ?, ?, ?)""",
            (file_hash, mode, clip_length, max_clips, json.dumps(result)),
        )
        conn.commit()
    finally:
        conn.close()


def get_user_stats(user_id: int) -> dict:
    conn = get_db()
    try:
        row = conn.execute(
            "SELECT COUNT(*) as jobs, COALESCE(SUM(clips_count), 0) as clips FROM history WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        return {"jobs": row["jobs"], "clips": row["clips"]}
    finally:
        conn.close()
