from __future__ import annotations

import sqlite3

import numpy as np

_CREATE_TABLE = """\
CREATE TABLE IF NOT EXISTS embeddings (
    file_path TEXT PRIMARY KEY,
    content_hash TEXT NOT NULL,
    embedding BLOB NOT NULL,
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
)
"""


class EmbeddingCache:
    """SQLite-backed cache for article embeddings."""

    def __init__(self, db_path: str) -> None:
        self.conn = sqlite3.connect(db_path)
        self.conn.execute(_CREATE_TABLE)
        self.conn.commit()

    def get(self, file_path: str, content_hash: str) -> np.ndarray | None:
        """Return cached embedding if hash matches, else None."""
        row = self.conn.execute(
            "SELECT embedding FROM embeddings WHERE file_path = ? AND content_hash = ?",
            (file_path, content_hash),
        ).fetchone()
        if row is None:
            return None
        return np.frombuffer(row[0], dtype=np.float32).copy()

    def put(self, file_path: str, content_hash: str, embedding: np.ndarray) -> None:
        """Insert or replace a cached embedding."""
        self.conn.execute(
            "INSERT OR REPLACE INTO embeddings"
            " (file_path, content_hash, embedding)"
            " VALUES (?, ?, ?)",
            (file_path, content_hash, embedding.astype(np.float32).tobytes()),
        )
        self.conn.commit()

    def put_batch(
        self,
        entries: list[tuple[str, str, np.ndarray]],
    ) -> None:
        """Batch insert embeddings in a single transaction."""
        self.conn.executemany(
            "INSERT OR REPLACE INTO embeddings"
            " (file_path, content_hash, embedding)"
            " VALUES (?, ?, ?)",
            [(fp, ch, emb.astype(np.float32).tobytes()) for fp, ch, emb in entries],
        )
        self.conn.commit()

    def prune(self, active_paths: set[str]) -> int:
        """Delete rows for paths not in active_paths. Returns count deleted."""
        cursor = self.conn.execute("SELECT file_path FROM embeddings")
        all_paths = {row[0] for row in cursor.fetchall()}
        stale = all_paths - active_paths
        if stale:
            placeholders = ",".join("?" for _ in stale)
            self.conn.execute(
                f"DELETE FROM embeddings WHERE file_path IN ({placeholders})",
                list(stale),
            )
            self.conn.commit()
        return len(stale)

    def close(self) -> None:
        """Close the database connection."""
        self.conn.close()
