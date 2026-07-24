"""SQLite persistence — one WAL file holds the whole engine state.

Everything here is DERIVED data (the caller's source of truth is wherever the
memories actually live — markdown files, a DB, an app). Dropping the file and
re-indexing is always safe, which is what makes the engine adoptable next to
an existing store without migration risk.

Concurrency: ONE ``sqlite3.Connection`` is shared across threads
(``check_same_thread=False``), so EVERY method — reads included — takes the
re-entrant lock before touching it. Multi-statement writes commit inside a
``try/except`` that rolls back on failure, so a mid-write exception can never
leave a partial transaction to be committed by the next writer.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

_SCHEMA = """
CREATE TABLE IF NOT EXISTS nodes(
  id TEXT PRIMARY KEY, kind TEXT DEFAULT 'note', title TEXT DEFAULT '',
  tags TEXT DEFAULT '[]', text_len INT DEFAULT 0, updated_at REAL,
  access_count INT DEFAULT 0, last_access REAL DEFAULT 0,
  pinned INT DEFAULT 0, importance REAL DEFAULT 1.0
);
CREATE TABLE IF NOT EXISTS postings(
  term TEXT, node_id TEXT, tf REAL, PRIMARY KEY(term, node_id)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS idx_postings_node ON postings(node_id);
CREATE TABLE IF NOT EXISTS vectors(node_id TEXT PRIMARY KEY, dim INT, vec BLOB);
CREATE TABLE IF NOT EXISTS teacher_vecs(node_id TEXT PRIMARY KEY, model TEXT, dim INT, vec BLOB);
CREATE TABLE IF NOT EXISTS edges(
  src TEXT, dst TEXT, etype INT, w REAL, updated REAL,
  PRIMARY KEY(src, dst, etype)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(src);
CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(dst);
CREATE TABLE IF NOT EXISTS params(key TEXT PRIMARY KEY, blob BLOB);
CREATE TABLE IF NOT EXISTS feedback(
  rowid INTEGER PRIMARY KEY AUTOINCREMENT,
  ts REAL, query_hash TEXT, node_id TEXT, features BLOB,
  shown INT DEFAULT 1, used INT DEFAULT 0, label_src TEXT DEFAULT ''
);
"""

EDGE_LINK, EDGE_TAG, EDGE_KNN, EDGE_COACCESS = 0, 1, 2, 3


class Store:
    """Thread-safe (single re-entrant lock) wrapper over the SQLite state."""

    def __init__(self, path: str = ":memory:") -> None:
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.RLock()
        with self._lock:
            if path != ":memory:":
                self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.executescript(_SCHEMA)
            self._migrate()
            self._conn.commit()

    def _migrate(self) -> None:
        """Idempotent column additions for DBs created by older versions.
        ADD COLUMN with a constant default is O(1) in SQLite — safe on live
        vaults. trust: per-item reliability prior in [0,1], neutral 0.5;
        trust_updated: timestamp of the last trust write (drives the lazy
        decay-to-neutral that stops stale reinforcement from ossifying)."""
        cols = {r[1] for r in self._conn.execute("PRAGMA table_info(nodes)")}
        if "trust" not in cols:
            self._conn.execute(
                "ALTER TABLE nodes ADD COLUMN trust REAL DEFAULT 0.5")
        if "trust_updated" not in cols:
            self._conn.execute(
                "ALTER TABLE nodes ADD COLUMN trust_updated REAL DEFAULT 0")
        if "content_sha" not in cols:
            # Idempotence key for index(): digest of everything that affects
            # derived state. Lets a re-index of UNCHANGED content short-circuit
            # to a metadata touch instead of a full tokenize+embed+edges+fsync
            # transaction (the difference between a session wake that re-scans
            # its vault in milliseconds vs minutes).
            self._conn.execute(
                "ALTER TABLE nodes ADD COLUMN content_sha TEXT DEFAULT ''")

    # ── transactional write helper ───────────────────────────────────
    def _write(self, fn) -> Any:
        """Run *fn(conn)* under the lock, commit, and roll back on any error
        so a partial multi-statement write is never left for the next commit."""
        with self._lock:
            try:
                result = fn(self._conn)
                self._conn.commit()
                return result
            except Exception:
                self._conn.rollback()
                raise

    def _read(self, sql: str, params: Sequence[Any] = ()) -> List[Tuple]:
        with self._lock:
            return list(self._conn.execute(sql, params).fetchall())

    # ── nodes ────────────────────────────────────────────────────────
    def upsert_node(
        self, node_id: str, *, kind: str, title: str, tags: Sequence[str],
        text_len: int, updated_at: float, pinned: bool, importance: float,
    ) -> None:
        self._write(lambda c: c.execute(
            "INSERT INTO nodes(id,kind,title,tags,text_len,updated_at,pinned,importance)"
            " VALUES(?,?,?,?,?,?,?,?)"
            " ON CONFLICT(id) DO UPDATE SET kind=excluded.kind,title=excluded.title,"
            " tags=excluded.tags,text_len=excluded.text_len,updated_at=excluded.updated_at,"
            " pinned=excluded.pinned,importance=excluded.importance",
            (node_id, kind, title, json.dumps(list(tags), ensure_ascii=False),
             text_len, updated_at, int(pinned), importance)))

    def touch_node(self, node_id: str, updated_at: float) -> None:
        """Refresh a node's updated_at without touching derived state — the
        cheap path for a re-index whose content is byte-identical."""
        self._write(lambda c: c.execute(
            "UPDATE nodes SET updated_at=? WHERE id=?", (updated_at, node_id)))

    def index_atomic(
        self, node_id: str, *, kind: str, title: str, tags: Sequence[str],
        text_len: int, updated_at: float, pinned: bool, importance: float,
        tf: Dict[str, float], vec: bytes, dim: int,
        edges: Sequence[Tuple[int, Sequence[Tuple[str, float]]]],
        teacher: Optional[Tuple[str, bytes, int]] = None,
        text_param: Optional[Tuple[str, bytes]] = None,
        content_sha: str = "",
    ) -> None:
        """Write one memory's node + postings + vector + edges (+ optional
        teacher / distill-text) in a SINGLE transaction, so a mid-index failure
        (disk-full, crash) rolls the whole node back instead of leaving an
        orphan node row with no vector or postings."""
        ts = time.time()

        def _do(c):
            c.execute(
                "INSERT INTO nodes(id,kind,title,tags,text_len,updated_at,pinned,importance,content_sha)"
                " VALUES(?,?,?,?,?,?,?,?,?)"
                " ON CONFLICT(id) DO UPDATE SET kind=excluded.kind,title=excluded.title,"
                " tags=excluded.tags,text_len=excluded.text_len,updated_at=excluded.updated_at,"
                " pinned=excluded.pinned,importance=excluded.importance,"
                " content_sha=excluded.content_sha",
                (node_id, kind, title, json.dumps(list(tags), ensure_ascii=False),
                 text_len, updated_at, int(pinned), importance, content_sha))
            c.execute("DELETE FROM postings WHERE node_id=?", (node_id,))
            c.executemany(
                "INSERT OR REPLACE INTO postings(term,node_id,tf) VALUES(?,?,?)",
                [(t, node_id, f) for t, f in tf.items()])
            c.execute("INSERT OR REPLACE INTO vectors(node_id,dim,vec) VALUES(?,?,?)",
                      (node_id, dim, vec))
            for etype, rows in edges:
                c.execute("DELETE FROM edges WHERE src=? AND etype=?", (node_id, etype))
                c.executemany(
                    "INSERT OR REPLACE INTO edges(src,dst,etype,w,updated) VALUES(?,?,?,?,?)",
                    [(node_id, d, etype, w, ts) for d, w in rows])
            if teacher is not None:
                model, tvec, tdim = teacher
                c.execute("INSERT OR REPLACE INTO teacher_vecs(node_id,model,dim,vec)"
                          " VALUES(?,?,?,?)", (node_id, model, tdim, tvec))
            if text_param is not None:
                key, blob = text_param
                c.execute("INSERT OR REPLACE INTO params(key,blob) VALUES(?,?)", (key, blob))
        self._write(_do)

    _NODE_COLS = ("id,kind,title,tags,text_len,updated_at,access_count,last_access,"
                  "pinned,importance,trust,trust_updated,content_sha")

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        rows = self._read(
            f"SELECT {self._NODE_COLS} FROM nodes WHERE id=?", (node_id,))
        return self._node_row(rows[0]) if rows else None

    def nodes(self, ids: Optional[Iterable[str]] = None) -> List[Dict[str, Any]]:
        if ids is None:
            rows = self._read(f"SELECT {self._NODE_COLS} FROM nodes")
            return [self._node_row(r) for r in rows]
        ids = list(ids)
        if not ids:
            return []
        q = ",".join("?" for _ in ids)
        rows = self._read(
            f"SELECT {self._NODE_COLS} FROM nodes WHERE id IN ({q})", ids)
        return [self._node_row(r) for r in rows]

    @staticmethod
    def _node_row(r: Tuple) -> Dict[str, Any]:
        return {
            "id": r[0], "kind": r[1], "title": r[2], "tags": json.loads(r[3] or "[]"),
            "text_len": r[4], "updated_at": r[5], "access_count": r[6],
            "last_access": r[7], "pinned": bool(r[8]), "importance": r[9],
            "trust": 0.5 if r[10] is None else float(r[10]),
            "trust_updated": float(r[11] or 0.0),
            "content_sha": r[12] or "",
        }

    def set_trust(self, node_id: str, trust: float, ts: float) -> None:
        self._write(lambda c: c.execute(
            "UPDATE nodes SET trust=?, trust_updated=? WHERE id=?",
            (trust, ts, node_id)))

    def remove_node(self, node_id: str) -> None:
        def _do(c):
            for sql in (
                "DELETE FROM nodes WHERE id=?", "DELETE FROM postings WHERE node_id=?",
                "DELETE FROM vectors WHERE node_id=?", "DELETE FROM teacher_vecs WHERE node_id=?",
                "DELETE FROM edges WHERE src=?", "DELETE FROM feedback WHERE node_id=?",
                "DELETE FROM edges WHERE dst=?",
            ):
                c.execute(sql, (node_id,))
        self._write(_do)

    def touch_access(self, ids: Iterable[str], ts: Optional[float] = None) -> None:
        ts = ts or time.time()
        rows = [(ts, i) for i in ids]
        self._write(lambda c: c.executemany(
            "UPDATE nodes SET access_count=access_count+1,last_access=? WHERE id=?", rows))

    def count_nodes(self) -> int:
        return int(self._read("SELECT COUNT(*) FROM nodes")[0][0])

    # ── postings (BM25) ──────────────────────────────────────────────
    def replace_postings(self, node_id: str, tf: Dict[str, float]) -> None:
        def _do(c):
            c.execute("DELETE FROM postings WHERE node_id=?", (node_id,))
            c.executemany(
                "INSERT OR REPLACE INTO postings(term,node_id,tf) VALUES(?,?,?)",
                [(t, node_id, f) for t, f in tf.items()])
        self._write(_do)

    def postings_for_terms(self, terms: Sequence[str]) -> Dict[str, List[Tuple[str, float]]]:
        if not terms:
            return {}
        q = ",".join("?" for _ in terms)
        out: Dict[str, List[Tuple[str, float]]] = {}
        for term, node_id, tf in self._read(
                f"SELECT term,node_id,tf FROM postings WHERE term IN ({q})", list(terms)):
            out.setdefault(term, []).append((node_id, tf))
        return out

    def doc_lens(self) -> Dict[str, int]:
        return {r[0]: r[1] for r in self._read("SELECT id,text_len FROM nodes")}

    # ── vectors ──────────────────────────────────────────────────────
    def put_vector(self, node_id: str, vec: bytes, dim: int) -> None:
        self._write(lambda c: c.execute(
            "INSERT OR REPLACE INTO vectors(node_id,dim,vec) VALUES(?,?,?)",
            (node_id, dim, vec)))

    def all_vectors(self) -> List[Tuple[str, int, bytes]]:
        return self._read("SELECT node_id,dim,vec FROM vectors")

    def swap_embedder_and_vectors(
        self, embedder_blob: bytes, vec_rows: Sequence[Tuple[str, int, bytes]]
    ) -> None:
        """ATOMIC distill swap: replace the embedder param AND re-embed every
        vector in ONE transaction. A crash rolls the whole thing back, so the
        stored table and the vectors it produced are never left inconsistent."""
        def _do(c):
            c.execute("INSERT OR REPLACE INTO params(key,blob) VALUES('embedder',?)",
                      (embedder_blob,))
            c.executemany(
                "INSERT OR REPLACE INTO vectors(node_id,dim,vec) VALUES(?,?,?)", vec_rows)
        self._write(_do)

    def put_teacher(self, node_id: str, model: str, vec: bytes, dim: int) -> None:
        self._write(lambda c: c.execute(
            "INSERT OR REPLACE INTO teacher_vecs(node_id,model,dim,vec) VALUES(?,?,?,?)",
            (node_id, model, dim, vec)))

    def teachers(self) -> List[Tuple[str, str, int, bytes]]:
        return self._read("SELECT node_id,model,dim,vec FROM teacher_vecs")

    # ── edges ────────────────────────────────────────────────────────
    def upsert_edges(self, rows: Iterable[Tuple[str, str, int, float]]) -> None:
        ts = time.time()
        data = [(s, d, t, w, ts) for s, d, t, w in rows]
        self._write(lambda c: c.executemany(
            "INSERT INTO edges(src,dst,etype,w,updated) VALUES(?,?,?,?,?)"
            " ON CONFLICT(src,dst,etype) DO UPDATE SET w=excluded.w,updated=excluded.updated",
            data))

    def replace_edges_from(self, node_id: str, etype: int,
                           rows: Iterable[Tuple[str, float]]) -> None:
        ts = time.time()
        data = [(node_id, d, etype, w, ts) for d, w in rows]

        def _do(c):
            c.execute("DELETE FROM edges WHERE src=? AND etype=?", (node_id, etype))
            c.executemany(
                "INSERT OR REPLACE INTO edges(src,dst,etype,w,updated) VALUES(?,?,?,?,?)",
                data)
        self._write(_do)

    def edges_by_type(self, etype: int) -> List[Tuple[str, str, float, float]]:
        return self._read("SELECT src,dst,w,updated FROM edges WHERE etype=?", (etype,))

    def get_edge(self, src: str, dst: str, etype: int) -> Optional[Tuple[float, float]]:
        rows = self._read(
            "SELECT w,updated FROM edges WHERE src=? AND dst=? AND etype=?", (src, dst, etype))
        return (rows[0][0], rows[0][1]) if rows else None

    def set_edge(self, src: str, dst: str, etype: int, w: float) -> None:
        self.upsert_edges([(src, dst, etype, w)])

    def prune_edges(self, etype: int, floor: float) -> int:
        def _do(c):
            return c.execute("DELETE FROM edges WHERE etype=? AND w<?", (etype, floor)).rowcount
        return self._write(_do)

    # ── params / feedback ────────────────────────────────────────────
    def put_param(self, key: str, blob: bytes) -> None:
        self._write(lambda c: c.execute(
            "INSERT OR REPLACE INTO params(key,blob) VALUES(?,?)", (key, blob)))

    def get_param(self, key: str) -> Optional[bytes]:
        rows = self._read("SELECT blob FROM params WHERE key=?", (key,))
        return rows[0][0] if rows else None

    def delete_param(self, key: str) -> None:
        self._write(lambda c: c.execute("DELETE FROM params WHERE key=?", (key,)))

    def add_feedback(self, query_hash: str, node_id: str, features: bytes,
                     used: bool, label_src: str, cap: int) -> None:
        def _do(c):
            c.execute(
                "INSERT INTO feedback(ts,query_hash,node_id,features,shown,used,label_src)"
                " VALUES(?,?,?,?,1,?,?)",
                (time.time(), query_hash, node_id, features, int(used), label_src))
            c.execute(
                "DELETE FROM feedback WHERE rowid <= "
                "(SELECT MAX(rowid) FROM feedback) - ?", (cap,))
        self._write(_do)

    def feedback_rows(self, limit: int) -> List[Tuple[str, bytes, int]]:
        return self._read(
            "SELECT query_hash,features,used FROM feedback ORDER BY rowid DESC LIMIT ?",
            (limit,))

    def feedback_count(self) -> int:
        return int(self._read("SELECT COUNT(*) FROM feedback")[0][0])

    def close(self) -> None:
        with self._lock:
            self._conn.commit()
            self._conn.close()
