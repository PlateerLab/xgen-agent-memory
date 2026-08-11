"""SynapseMemory — the public engine: index / search / feedback / distill.

One object per vault. All state — vectors, edges, the learned ranker AND the
embedding table — lives in the single `<path>` SQLite file. Zero network
calls; every operation is CPU-milliseconds. All public methods are thread-safe
(one re-entrant lock guards the caches, ranker, and embedder).

    mem = SynapseMemory.open(path="vault/synapse.db")     # or from_env()
    mem.index("note-1", "본문…", title="제목", tags=["게임"], links=["note-0"])
    hits = mem.search("리듬게임 판정", top_k=8)
    mem.feedback(hits[0].query_token, used_ids=["note-1"])  # ← 온라인 학습
    mem.distill()                                           # teacher가 있을 때만
"""

from __future__ import annotations

import hashlib
import logging
import math
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ._rwlock import RWLock
from .bm25 import bm25_scores, term_frequencies, top_n
from .config import SynapseConfig
from .embedder import HashEmbedder, pack_vec, unpack_vec
from .graph import (
    build_type_adjacency,
    derive_knn_edges,
    derive_tag_edges,
    ppr_features,
    reinforce_coaccess,
)
from .ranker import FEATURES, OnlineRanker
from .store import EDGE_COACCESS, EDGE_KNN, EDGE_LINK, EDGE_TAG, Store
from .tokenizer import lexical_tokens

logger = logging.getLogger(__name__)

#: Bump when the MEANING of the tokenizer or embedding pipeline changes in a
#: way the config fields alone do not capture — a new stemmer, a changed
#: n-gram rule, a different normalisation. Every vault then re-derives once.
#:
#: 2: Porter stemming for Latin words; Latin trigrams moved out of BM25.
#:    Needed as its own lever because 1.9.0 recorded the *new* geometry
#:    against rows derived by the *old* tokenizer — from then on the
#:    fingerprints matched and nothing could notice.
_GEOMETRY_VERSION = 2

_KIND_PRIOR = {"fact": 1.0, "insight": 0.8, "note": 0.5, "digest": 0.4, "turn": 0.2}


@dataclass
class SearchHit:
    id: str
    score: float
    title: str = ""
    kind: str = "note"
    features: Dict[str, float] = field(default_factory=dict)
    sources: List[str] = field(default_factory=list)  # which retrievers hit
    #: Opaque token identifying this query for feedback().
    query_token: str = ""


class SynapseMemory:
    def __init__(self, config: Optional[SynapseConfig] = None, **overrides: Any) -> None:
        cfg = config or SynapseConfig()
        if overrides:
            cfg = SynapseConfig(
                **{
                    **cfg.__dict__,
                    **{
                        k: v
                        for k, v in overrides.items()
                        if k in SynapseConfig.__dataclass_fields__
                    },
                }
            )
        self.cfg = cfg
        self.store = Store(cfg.path)
        self._invalidate_on_geometry_change()
        self.embedder = self._load_embedder()
        self.ranker = self._load_ranker()
        # Incremental caches — the difference between O(N) and O(N²) bulk
        # indexing, and between ~8ms and ~50ms queries at a few thousand
        # nodes. All derived from SQLite; safe to drop at any time.
        self._vec_cache: Optional[Dict[str, np.ndarray]] = None
        self._vec_matrix: Optional[tuple] = None  # (ids, np.ndarray)
        self._doclen_cache: Optional[Dict[str, int]] = None
        self._tag_cache: Optional[Dict[str, List[str]]] = None
        self._adj_cache: Dict[int, Optional[list]] = {}
        #: query_token → {"hash":…, "features": {node_id: np.ndarray}, "shown": [ids]}
        self._recent_queries: Dict[str, Dict[str, Any]] = {}
        self._rng = random.Random(cfg.seed)
        # Readers run together; writers are exclusive; arrival order decides
        # (see `_rwlock` for why fair, not writer-preferring). A wedged write
        # still blocks readers — only `search(timeout=)` bounds that.
        #
        # NOT reentrant, unlike the RLock it replaces: a locked method must
        # never call another locked method (see ``_get_text_unlocked``).
        self._lock = RWLock()
        # `_recent_queries` is touched from BOTH sides — search inserts under
        # the read lock, feedback/remove delete under the write lock — so it
        # needs a mutex of its own.
        self._rq_lock = threading.Lock()
        # Derived caches are built lazily, including from read paths where
        # several threads may now arrive at once. One builder at a time:
        # rebuilding an 8.7k-vector matrix twice is pure waste.
        #
        # REENTRANT on purpose — the builders nest (`_vector_matrix` needs
        # `_vectors`), and a plain mutex there deadlocks a thread against
        # itself. This guards duplicated work, not correctness, so
        # reentrancy costs nothing.
        self._cache_build_lock = threading.RLock()

    def _geometry(self) -> str:
        """Everything a stored row's derived state depends on.

        Not the same question as "did the note change" — this is "would the
        same note produce different postings and vectors today". Embedding
        width, tokenizer streams, stemming: change any of them and every row
        in the file is stale, however untouched the notes are.
        """
        return (
            f"v{_GEOMETRY_VERSION}:"
            f"{self.cfg.dim}:{self.cfg.vocab_size}:"
            f"{tuple(self.cfg.char_ngrams)}:{self.cfg.suffix_strip}:"
            f"{self.cfg.cross_space}:{self.cfg.latin_stemming}:"
            f"{self.cfg.latin_ngram_min_len}"
        )

    def _invalidate_on_geometry_change(self) -> None:
        """Clear every digest when the geometry moved, so the host is told.

        Putting the geometry into ``content_sha`` is not enough on its own:
        the digest is only consulted for notes a HOST decides to offer, and
        a host that diffs on timestamps never offers an untouched note. The
        result was a vault half-converted to a new tokenization while every
        signal said it was in sync.

        An empty digest in ``manifest()`` is the contract that fixes it —
        "indexed, derived state unknown" — and a host reading it re-offers
        exactly those notes. One UPDATE, once, on the boot after a change.
        """
        current = self._geometry()
        stored = self.store.get_param("geometry")
        previous = stored.decode("utf-8", "replace") if stored else ""
        if previous == current:
            return
        # No recorded geometry means a vault written before this was tracked
        # — which is to say, derived by SOME geometry nobody can name. The
        # first version of this treated "unknown" as "fine" to avoid a mass
        # re-index on upgrade, and the result was the upgrade re-indexing
        # nothing at all: every row kept a digest that described a
        # tokenization no longer in use. Derived data is rebuildable and a
        # wrong assumption of freshness is not, so unknown counts as stale.
        n = self.store.clear_content_shas()
        if n:
            logger.info(
                "synapse: geometry changed (%s → %s) — %d rows marked stale",
                previous or "<unrecorded>",
                current,
                n,
            )
        self.store.put_param("geometry", current.encode("utf-8"))

    # ── construction helpers ─────────────────────────────────────────
    @classmethod
    def open(cls, path: str = "synapse.db", **overrides: Any) -> "SynapseMemory":
        return cls(SynapseConfig(path=path), **overrides)

    @classmethod
    def from_env(cls, *, dotenv: Optional[str] = None, **overrides: Any) -> "SynapseMemory":
        """Configure via GMA_* env vars (optionally loading a .env first)."""
        return cls(SynapseConfig.from_env(dotenv=dotenv, **overrides))

    def _load_embedder(self) -> HashEmbedder:
        kw = dict(
            seed=self.cfg.seed,
            char_ngrams=self.cfg.char_ngrams,
            jamo_ngrams=self.cfg.jamo_ngrams,
            suffix_strip=self.cfg.suffix_strip,
        )
        blob = self.store.get_param("embedder")
        if blob:
            try:
                return HashEmbedder.loads(blob, **kw)
            except Exception:
                pass
        return HashEmbedder(self.cfg.vocab_size, self.cfg.dim, **kw)

    def _load_ranker(self) -> OnlineRanker:
        blob = self.store.get_param("ranker")
        if blob:
            try:
                return OnlineRanker.loads(blob)
            except Exception:
                pass
        return OnlineRanker(
            hidden=self.cfg.hidden,
            lr=self.cfg.lr,
            l2=self.cfg.l2,
            blend_min_events=self.cfg.blend_min_events,
        )

    # ── write path ───────────────────────────────────────────────────
    def index(
        self,
        node_id: str,
        text: str,
        *,
        title: str = "",
        kind: str = "note",
        tags: Sequence[str] = (),
        links: Sequence[str] = (),
        importance: float = 1.0,
        pinned: bool = False,
        updated_at: Optional[float] = None,
        teacher_vec: Optional[Sequence[float]] = None,
        teacher_model: str = "",
    ) -> None:
        """Index (or re-index) one memory. Idempotent by *node_id*.

        *teacher_vec* is an ALREADY-COMPUTED higher-quality embedding the
        caller happens to have (e.g. a stored API embedding) — used only as a
        distillation label, never required.
        """
        with self._lock.write():
            prepared = self._prepare_index(
                node_id,
                text,
                title=title,
                kind=kind,
                tags=tags,
                links=links,
                importance=importance,
                pinned=pinned,
                updated_at=updated_at,
                teacher_vec=teacher_vec,
                teacher_model=teacher_model,
            )
            if prepared is None:
                return
            row, cache_delta, touch = prepared
            if row is None:
                if touch is not None:
                    self.store.touch_node(node_id, touch)
                return
            self.store.index_atomic(**row)
            self._apply_cache_delta(cache_delta)

    _MISSING = object()

    def _prepare_index(
        self,
        node_id: str,
        text: str,
        *,
        title: str = "",
        kind: str = "note",
        tags: Sequence[str] = (),
        links: Sequence[str] = (),
        importance: float = 1.0,
        pinned: bool = False,
        updated_at: Optional[float] = None,
        teacher_vec: Optional[Sequence[float]] = None,
        teacher_model: str = "",
        existing: Any = _MISSING,
    ) -> Optional[tuple]:
        """Derive one memory's state WITHOUT writing anything.

        Returns ``(row, cache_delta, touch)``:
          * ``row`` — kwargs for ``store.index_atomic``/``index_atomic_many``,
            or ``None`` when the digest says nothing derived would change;
          * ``cache_delta`` — what the in-memory caches must learn once the
            write commits;
          * ``touch`` — the timestamp to record when only the clock moved.

        Split out of ``index()`` so a batch can compute many of these and then
        commit them together. Compute is 2-3 ms per note; the commit was
        measured at 43.7 ms, so the split is where a catch-up stops being
        dominated by fsync.

        *existing* lets a batch caller supply the node row it already fetched
        — otherwise this issues one SELECT per note, which is the same
        per-item cost the batch exists to remove.
        """
        if existing is self._MISSING:
            existing = self.store.get_node(node_id)  # None ⇒ fresh insert
        body = f"{title}\n{text}" if title else text
        # Idempotence short-circuit: hosts commonly re-scan their whole
        # vault on session wake ("index everything, just in case"). When
        # NOTHING that affects derived state changed, a full re-index
        # (tokenize + embed + edge derivation + fsync'd transaction) is
        # pure waste — measured as the difference between a wake-time
        # backfill of a real vault taking minutes vs milliseconds. The
        # digest covers every input the derived rows depend on, config
        # geometry included (a dim/vocab change must rebuild vectors).
        sha_basis = "\x1f".join(
            (
                body,
                kind,
                ",".join(sorted(tags)),
                ",".join(sorted(links)),
                f"{importance:.6f}",
                str(bool(pinned)),
                f"{self.cfg.dim}:{self.cfg.vocab_size}",
                # Tokenizer geometry belongs in the digest for the same reason
                # the embedding geometry does: change it and every derived row
                # is stale. Without this, flipping `latin_stemming` or
                # `char_ngrams` left the OLD postings in place — the digest said
                # "unchanged", nothing re-indexed, and the vault silently became
                # a mix of two tokenizations that no longer agreed with the
                # query view.
                f"{tuple(self.cfg.char_ngrams)}:{self.cfg.suffix_strip}:"
                f"{self.cfg.cross_space}:{self.cfg.latin_stemming}:"
                f"{self.cfg.latin_ngram_min_len}",
                str(teacher_model) if teacher_vec is not None else "",
            )
        )
        content_sha = hashlib.sha1(sha_basis.encode("utf-8", "replace")).hexdigest()
        if existing is not None and existing.get("content_sha") == content_sha:
            touch = (
                updated_at
                if updated_at is not None and updated_at != existing.get("updated_at")
                else None
            )
            return None, None, touch
        # ── compute everything FIRST (reads + numpy), commit ONCE ──
        # LEXICAL stream → BM25 postings (words + stems + syllable bigrams
        # + cross-space bigrams). The jamo-augmented EMBEDDING stream lives
        # only inside the hash embedder.
        tok_kw = dict(
            char_ngrams=self.cfg.char_ngrams,
            suffix_strip=self.cfg.suffix_strip,
            cross_space=self.cfg.cross_space,
            latin_ngram_min_len=self.cfg.latin_ngram_min_len,
            latin_stemming=self.cfg.latin_stemming,
        )
        tokens = lexical_tokens(body, limit=self.cfg.max_doc_tokens, **tok_kw)
        # BM25F-lite: title terms weigh title_boost× (the title already
        # appears once inside `body`, so the extra weight is title_boost−1).
        tf = term_frequencies(tokens)
        if title and self.cfg.title_boost > 1.0:
            extra = self.cfg.title_boost - 1.0
            for t in set(lexical_tokens(title, limit=64, **tok_kw)):
                tf[t] = tf.get(t, 0.0) + extra
        vec = self.embedder.embed(body, limit=self.cfg.max_doc_tokens)
        # Edges. LINK is stored ONE-directional and symmetrized at query
        # time (build_type_adjacency), so a changed link set can't orphan a
        # reverse edge. TAG/KNN derived from current graph state.
        tag_edges = derive_tag_edges(
            self._tags_map(), self._n_docs(), node_id, tags, fanout=self.cfg.tag_fanout
        )
        knn_edges = derive_knn_edges(
            vec,
            self._vectors(),
            node_id,
            k=self.cfg.knn_edges,
            min_sim=self.cfg.knn_min_sim,
            sample_cap=self.cfg.knn_sample_cap,
        )
        teacher = None
        if teacher_vec is not None:
            tv = np.asarray(teacher_vec, dtype=np.float32)
            teacher = (teacher_model, pack_vec(tv), int(tv.shape[0]))
        text_param = (
            (f"text:{node_id}", body[: self.cfg.store_text_maxlen].encode("utf-8"))
            if self.cfg.store_text
            else None
        )

        row = dict(
            node_id=node_id,
            kind=kind,
            title=title,
            tags=tags,
            text_len=len(tokens),
            updated_at=updated_at or time.time(),
            pinned=pinned,
            importance=importance,
            tf=tf,
            vec=pack_vec(vec),
            dim=self.cfg.dim,
            edges=[
                (EDGE_LINK, [(dst, 1.0) for dst in links]),
                (EDGE_TAG, tag_edges),
                (EDGE_KNN, knn_edges),
            ],
            teacher=teacher,
            text_param=text_param,
            content_sha=content_sha,
        )
        cache_delta = dict(
            node_id=node_id,
            vec=vec,
            text_len=len(tokens),
            tags=tuple(tags),
            old_tags=tuple(existing["tags"]) if existing is not None else (),
        )
        return row, cache_delta, None

    def _apply_cache_delta(self, delta: Dict[str, Any]) -> None:
        """Teach the in-memory caches what a committed write changed.

        Separate from the write so a batch applies one delta per note AFTER
        its chunk commits — a cache that learned about a rolled-back write
        would answer for rows the database does not have.
        """
        node_id = delta["node_id"]
        if self._vec_cache is not None:
            self._vec_cache[node_id] = delta["vec"]
        self._vec_matrix = None
        if self._doclen_cache is not None:
            self._doclen_cache[node_id] = delta["text_len"]
        # Tag cache: INCREMENTAL on both insert and re-index. On a re-index
        # remove the node from its OLD tags then add the new ones —
        # O(#tags), NOT an O(N) full-cache rebuild (which made re-indexing a
        # big corpus O(N²): 541 ms/re-index at 40k).
        if self._tag_cache is not None:
            for t in delta["old_tags"]:
                lst = self._tag_cache.get(t)
                if lst and node_id in lst:
                    lst.remove(node_id)
            for t in delta["tags"]:
                lst = self._tag_cache.setdefault(t, [])
                if node_id not in lst:
                    lst.append(node_id)
        self._adj_cache.clear()

    def index_many(
        self,
        items: Sequence[Dict[str, Any]],
        *,
        chunk_size: int = 200,
    ) -> Dict[str, int]:
        """Index many memories, amortising the commit across a chunk.

        Each item is a mapping with ``node_id`` and ``text`` plus any keyword
        ``index()`` accepts. Unchanged content short-circuits exactly as it
        does one at a time — the digest decides, not the caller.

        Why this exists: ``index()`` in a loop pays one fsync per note. On a
        real deployment that fsync measured 43.7 ms against 2-3 ms of actual
        indexing, so a catch-up was 97% commit. Here the whole chunk computes
        first, then commits once.

        kNN edges still see prior members of the same batch, because the
        in-memory vector cache is updated as each note is prepared — the
        result is identical to indexing them one by one, minus the fsyncs.

        Returns ``{"indexed": n, "touched": n, "skipped": n}``.
        """
        out = {"indexed": 0, "touched": 0, "skipped": 0}
        if not items:
            return out
        with self._lock.write():
            for start in range(0, len(items), max(1, chunk_size)):
                chunk = items[start : start + max(1, chunk_size)]
                # One SELECT for the whole chunk instead of one per note —
                # otherwise the batch still pays N round trips to discover
                # that N notes are unchanged.
                known = {
                    n["id"]: n
                    for n in self.store.nodes([it["node_id"] for it in chunk if it.get("node_id")])
                }
                rows: List[Dict[str, Any]] = []
                deltas: List[Dict[str, Any]] = []
                touches: List[Tuple[str, float]] = []
                for item in chunk:
                    kwargs = dict(item)
                    node_id = kwargs.pop("node_id")
                    text = kwargs.pop("text", "")
                    prepared = self._prepare_index(
                        node_id, text, existing=known.get(node_id), **kwargs
                    )
                    row, delta, touch = prepared
                    if row is None:
                        if touch is not None:
                            touches.append((node_id, touch))
                            out["touched"] += 1
                        else:
                            out["skipped"] += 1
                        continue
                    rows.append(row)
                    deltas.append(delta)
                    # Visible to the NEXT note's kNN in this same chunk.
                    if self._vec_cache is not None:
                        self._vec_cache[node_id] = delta["vec"]
                if touches:
                    self.store.touch_nodes(touches)
                if rows:
                    self.store.index_atomic_many(rows)
                    for delta in deltas:
                        self._apply_cache_delta(delta)
                    out["indexed"] += len(rows)
        return out

    def manifest(self) -> Dict[str, Tuple[float, str]]:
        """``{node_id: (updated_at, content_sha)}`` — what is already indexed.

        The primitive an incremental host needs: with it, "what changed" is a
        set difference against metadata the host already holds; without it the
        only way to ask is to re-read every note and re-derive its digest.
        """
        with self._lock.read():
            return self.store.manifest()

    def catalog_counts(
        self, *, by: str = "kind", kind: Optional[str] = None
    ) -> List[Tuple[str, int]]:
        """``[(key, count)]`` by ``kind`` or by calendar ``day``.

        The cheap end of a progressive browser: "how much is there" without
        reading anything. See ``Store.catalog_counts``.
        """
        with self._lock.read():
            return self.store.catalog_counts(by=by, kind=kind)

    def catalog_page(
        self,
        *,
        day: Optional[str] = None,
        kind: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """One page of note metadata — no bodies. See ``Store.catalog_page``."""
        with self._lock.read():
            return self.store.catalog_page(day=day, kind=kind, limit=limit, offset=offset)

    def neighbourhood(
        self,
        node_ids: Sequence[str],
        *,
        depth: int = 1,
        max_nodes: int = 400,
        max_edges: int = 4000,
    ) -> Dict[str, Any]:
        """The subgraph around *node_ids* — nodes + typed edges, both capped.

        A graph view is read at a screen's worth of detail; shipping the
        whole vault (5,384 nodes, 4.3 MB) is a download, not a view. Depth is
        expanded breadth-first and stops at ``max_nodes``, so the caller gets
        a bounded answer instead of an unbounded one truncated by whatever
        gives out first.
        """
        seed = [n for n in node_ids if n]
        if not seed:
            return {"nodes": [], "edges": [], "truncated": False}
        with self._lock.read():
            frontier = list(dict.fromkeys(seed))
            seen = dict.fromkeys(frontier)
            edges: List[Dict[str, Any]] = []
            truncated = False
            for _ in range(max(0, depth)):
                if not frontier or len(seen) >= max_nodes:
                    break
                rows = self.store.edges_touching(frontier, limit=max_edges)
                nxt: List[str] = []
                for src, dst, etype, w in rows:
                    edges.append({"src": src, "dst": dst, "type": etype, "w": w})
                    for end in (src, dst):
                        if end not in seen:
                            if len(seen) >= max_nodes:
                                truncated = True
                                continue
                            seen[end] = None
                            nxt.append(end)
                frontier = nxt
            nodes = self.store.nodes(list(seen))
        return {
            "nodes": [
                {
                    "id": n["id"],
                    "kind": n["kind"],
                    "title": n["title"],
                    "updated_at": n["updated_at"],
                    "text_len": n["text_len"],
                    "pinned": n["pinned"],
                    "importance": n["importance"],
                }
                for n in nodes
            ],
            "edges": edges,
            "truncated": truncated or len(edges) >= max_edges,
        }

    def remove_many(self, node_ids: Sequence[str]) -> int:
        """Delete many memories in one transaction.

        Reaping is inherently bulk — a vault whose deletions never reached the
        index presents thousands at once — and one fsync per node would turn
        the cleanup into its own outage.
        """
        ids = [n for n in node_ids if n]
        if not ids:
            return 0
        with self._lock.write():
            removed = self.store.remove_nodes(ids)
            gone = set(ids)
            for node_id in ids:
                if self._vec_cache is not None:
                    self._vec_cache.pop(node_id, None)
                if self._doclen_cache is not None:
                    self._doclen_cache.pop(node_id, None)
            self._vec_matrix = None
            self._tag_cache = None
            self._adj_cache.clear()
            with self._rq_lock:
                for tok in list(self._recent_queries):
                    if gone & set(self._recent_queries[tok]["features"]):
                        del self._recent_queries[tok]
            return removed

    def remove(self, node_id: str) -> None:
        with self._lock.write():
            self.store.remove_node(node_id)
            self.store.delete_param(f"text:{node_id}")
            if self._vec_cache is not None:
                self._vec_cache.pop(node_id, None)
            self._vec_matrix = None
            if self._doclen_cache is not None:
                self._doclen_cache.pop(node_id, None)
            self._tag_cache = None
            self._adj_cache.clear()
            # Drop any pending feedback tokens that reference this node — else
            # feedback() could reinforce co-access edges or re-insert feedback
            # rows pointing at a now-deleted node.
            with self._rq_lock:
                for tok in list(self._recent_queries):
                    if node_id in self._recent_queries[tok]["features"]:
                        del self._recent_queries[tok]

    # ── read path ────────────────────────────────────────────────────
    #: Default patience for a RETRIEVAL. A turn can proceed without memory;
    #: it cannot proceed while blocked behind a write that never returns.
    #: Long enough that a normal index chunk is simply waited out.
    SEARCH_TIMEOUT_S: Optional[float] = 20.0

    def search(
        self,
        query: str,
        *,
        top_k: Optional[int] = None,
        kinds: Optional[Sequence[str]] = None,
        timeout: Any = _MISSING,
    ) -> List[SearchHit]:
        """Retrieve. Raises ``MemoryBusy`` if a write holds the engine longer
        than ``timeout`` (default ``SEARCH_TIMEOUT_S``; pass ``None`` to wait
        forever, which is what the callers who cannot degrade should do)."""
        wait = self.SEARCH_TIMEOUT_S if timeout is self._MISSING else timeout
        with self._lock.read(timeout=wait):
            return self._search(query, top_k=top_k, kinds=kinds)

    def _search(
        self, query: str, *, top_k: Optional[int] = None, kinds: Optional[Sequence[str]] = None
    ) -> List[SearchHit]:
        # `is None` (not truthiness) so an explicit top_k=0 means "0 results",
        # and clamp negatives to 0 rather than slicing from the tail.
        top_k = self.cfg.top_k if top_k is None else max(0, top_k)
        q_tokens = lexical_tokens(
            query,
            char_ngrams=self.cfg.char_ngrams,
            suffix_strip=self.cfg.suffix_strip,
            cross_space=self.cfg.cross_space,
            limit=self.cfg.max_query_tokens,
        )
        now = time.time()

        # ① seeds — BM25 ∪ cosine, fused by RRF.
        bm25 = bm25_scores(
            self.store, q_tokens, doc_lens=self._doclens(), k1=self.cfg.bm25_k1, b=self.cfg.bm25_b
        )
        q_vec = self.embedder.embed(query, limit=self.cfg.max_query_tokens)
        ids, matrix = self._vector_matrix()
        cos: Dict[str, float] = {}
        if ids:
            sims = matrix @ q_vec
            # Only the top slice matters; argpartition keeps this O(N).
            k = min(len(ids), max(self.cfg.vector_seed_k * 4, 64))
            for j in np.argpartition(-sims, k - 1)[:k]:
                cos[ids[int(j)]] = float(sims[int(j)])
        bm_top = top_n(bm25, self.cfg.bm25_seed_k)
        cos_top = sorted(cos, key=lambda k: -cos[k])[: self.cfg.vector_seed_k]
        rrf: Dict[str, float] = {}
        for rank, nid in enumerate(bm_top):
            rrf[nid] = rrf.get(nid, 0.0) + 1.0 / (60 + rank)
        for rank, nid in enumerate(cos_top):
            rrf[nid] = rrf.get(nid, 0.0) + 1.0 / (60 + rank)
        seeds = {nid: score for nid, score in rrf.items()}
        if not seeds:
            return []

        # ② graph expansion — per-type PPR from the seed distribution.
        ppr = ppr_features(
            self._adjacencies(), seeds, alpha=self.cfg.ppr_alpha, iters=self.cfg.ppr_iters
        )
        candidates = set(seeds)
        expansion_pool: Dict[str, float] = {}
        for etype_scores in ppr.values():
            for nid, s in etype_scores.items():
                if nid not in candidates:
                    expansion_pool[nid] = max(expansion_pool.get(nid, 0.0), s)
        for nid in sorted(expansion_pool, key=lambda k: -expansion_pool[k])[
            : self.cfg.graph_expand_k
        ]:
            candidates.add(nid)

        # ③ features + ranking.
        meta = {n["id"]: n for n in self.store.nodes(candidates)}
        q_words = set(w for w in q_tokens if len(w) > 1)
        feats: Dict[str, np.ndarray] = {}
        scored: List[SearchHit] = []
        for nid in candidates:
            node = meta.get(nid)
            if node is None:
                continue
            if kinds and node["kind"] not in kinds:
                continue
            age_days = max(0.0, (now - (node["updated_at"] or now)) / 86400.0)
            title_words = set(
                lexical_tokens(node["title"], char_ngrams=(), cross_space=False, limit=32)
            )
            x = np.array(
                [
                    bm25.get(nid, 0.0),
                    cos.get(nid, 0.0),
                    rrf.get(nid, 0.0),
                    ppr[EDGE_LINK].get(nid, 0.0),
                    ppr[EDGE_TAG].get(nid, 0.0),
                    ppr[EDGE_KNN].get(nid, 0.0),
                    ppr[EDGE_COACCESS].get(nid, 0.0),
                    1.0 / (1.0 + math.log1p(age_days)),
                    math.log1p(node["access_count"]),
                    node["importance"],
                    1.0 if node["pinned"] else 0.0,
                    1.0 if q_words & title_words else 0.0,
                    _KIND_PRIOR.get(node["kind"], 0.5),
                    min(1.0, node["text_len"] / 512.0),
                ],
                dtype=np.float32,
            )
            # NOTE: do NOT observe() here — updating the normalization stats
            # mid-ranking makes a candidate perturb its own (and later
            # candidates') z-scores, so search is non-idempotent. Score against
            # the current stats, collect the features, and fold them into the
            # running normalization AFTER ranking (below).
            feats[nid] = x
            score = self.ranker.score(x)
            # Per-item trust prior — additive, so negatively-scored candidates
            # are pushed the same direction as positive ones (a multiplier
            # would invert the effect below zero). Neutral trust (0.5) adds 0.
            if self.cfg.trust_weight > 0:
                score += (self._effective_trust(node, now) - 0.5) * 2.0 * self.cfg.trust_weight
            sources = []
            if nid in bm25:
                sources.append("bm25")
            if nid in cos and cos[nid] > 0:
                sources.append("vector")
            if nid not in seeds:
                sources.append("graph")
            scored.append(
                SearchHit(
                    id=nid,
                    score=score,
                    title=node["title"],
                    kind=node["kind"],
                    features={f: float(v) for f, v in zip(FEATURES, x)},
                    sources=sources,
                )
            )
        scored.sort(key=lambda h: -h.score)
        result = scored[:top_k]

        # Fold this query's candidate features into the running normalization
        # AFTER ranking — so the scores above are idempotent w.r.t. this call.
        for x in feats.values():
            self.ranker.observe(x)

        # ε-exploration: swap the tail slot with a random non-shown candidate.
        if len(scored) > top_k and result and self._rng.random() < self.cfg.epsilon:
            result[-1] = self._rng.choice(scored[top_k:])

        # Register for feedback.
        token = hashlib.sha1(f"{query}|{now}".encode()).hexdigest()[:16]
        for h in result:
            h.query_token = token
        # Search runs under the READ lock, so several of these can land at
        # once; the bounded dict needs its own mutex.
        with self._rq_lock:
            self._recent_queries[token] = {
                "hash": hashlib.sha1(query.encode()).hexdigest()[:16],
                "features": {h.id: feats[h.id] for h in result if h.id in feats},
                "shown": [h.id for h in result],
            }
            if len(self._recent_queries) > 64:
                self._recent_queries.pop(next(iter(self._recent_queries)))
        self.store.touch_access([h.id for h in result], ts=now)
        return result

    # ── learning path ────────────────────────────────────────────────
    def feedback(
        self,
        query_token: str,
        *,
        used_ids: Sequence[str] = (),
        ignored_ids: Optional[Sequence[str]] = None,
        label_src: str = "implicit",
    ) -> Dict[str, float]:
        """Report which shown memories were actually USED.

        *ignored_ids* defaults to shown-minus-used. Triggers: Hebbian
        co-access reinforcement, pairwise ranker SGD (event + small replay),
        and persists everything. Cost: microseconds-to-ms.
        """
        with self._lock.write():
            return self._feedback(
                query_token, used_ids=used_ids, ignored_ids=ignored_ids, label_src=label_src
            )

    def _feedback(
        self,
        query_token: str,
        *,
        used_ids: Sequence[str] = (),
        ignored_ids: Optional[Sequence[str]] = None,
        label_src: str = "implicit",
    ) -> Dict[str, float]:
        with self._rq_lock:
            q = self._recent_queries.get(query_token)
        if q is None:
            return {"applied": 0.0}
        used = [i for i in used_ids if i in q["features"]]
        ignored = (
            list(ignored_ids)
            if ignored_ids is not None
            else [i for i in q["shown"] if i not in set(used_ids)]
        )
        ignored = [i for i in ignored if i in q["features"]]

        # ① Hebbian graph learning.
        if len(used) >= 2:
            reinforce_coaccess(
                self.store,
                used,
                eta=self.cfg.hebb_eta,
                decay=self.cfg.hebb_decay,
                prune=self.cfg.hebb_prune,
            )
            self._adj_cache.pop(EDGE_COACCESS, None)
        # ①b Trust — same policy as learn(): used items gain reliability.
        for nid in used:
            self._trust_feedback_unlocked(nid, True)

        # ② Ranker. Split by QUERY: a deterministic ~25% of queries are HELD
        # OUT — refereed for the blend gate, never trained on (and never added
        # to the replay buffer), so the gate's McNemar test is truly
        # out-of-sample. The rest train the live weights + feed replay.
        is_eval = int(q["hash"], 16) % 4 == 0
        loss = 0.0
        pairs = 0
        for u in used:
            for g in ignored:
                if is_eval:
                    self.ranker.referee_pair(q["features"][u], q["features"][g])
                else:
                    loss += self.ranker.update_pair(q["features"][u], q["features"][g])
                    pairs += 1
        if not is_eval:
            for nid in used:
                self.store.add_feedback(
                    q["hash"],
                    nid,
                    q["features"][nid].tobytes(),
                    True,
                    label_src,
                    self.cfg.replay_cap,
                )
            for nid in ignored:
                self.store.add_feedback(
                    q["hash"],
                    nid,
                    q["features"][nid].tobytes(),
                    False,
                    label_src,
                    self.cfg.replay_cap,
                )
            pairs += self._replay(8)
        self._persist_ranker()  # ranker only — the embedder is untouched here
        return {
            "applied": 1.0,
            "pairs": float(pairs),
            "loss": loss / max(1, pairs),
            **self.ranker.stats(),
        }

    def _replay(self, n_pairs: int) -> int:
        rows = self.store.feedback_rows(256)
        pos = [(q, np.frombuffer(f, dtype=np.float32)) for q, f, u in rows if u]
        neg = [(q, np.frombuffer(f, dtype=np.float32)) for q, f, u in rows if not u]
        by_q: Dict[str, Dict[str, list]] = {}
        for q, x in pos:
            by_q.setdefault(q, {"p": [], "n": []})["p"].append(x)
        for q, x in neg:
            by_q.setdefault(q, {"p": [], "n": []})["n"].append(x)
        eligible = [(q, d) for q, d in by_q.items() if d["p"] and d["n"]]
        done = 0
        while eligible and done < n_pairs:
            q, d = self._rng.choice(eligible)
            self.ranker.update_pair(self._rng.choice(d["p"]), self._rng.choice(d["n"]))
            done += 1
        return done

    # ── per-item trust (reliability prior) ──────────────────────────
    def _effective_trust(self, node: Dict[str, Any], now: float) -> float:
        """Trust with lazy decay TOWARD NEUTRAL (0.5).

        The anti-ossification mechanism: a reinforcement that is never
        re-confirmed fades back to neutral instead of hardening into a
        permanent belief the engine cites against itself long after the
        world changed. Pure read — nothing is written here."""
        t = float(node.get("trust", 0.5))
        hl = self.cfg.trust_half_life_days
        if hl <= 0 or t == 0.5:
            return t
        ts = float(node.get("trust_updated") or 0.0)
        if ts <= 0:
            return t
        age_days = max(0.0, (now - ts) / 86400.0)
        return 0.5 + (t - 0.5) * (0.5 ** (age_days / hl))

    def trust_feedback(
        self, node_id: str, helpful: bool, *, now: Optional[float] = None
    ) -> Optional[float]:
        """Explicit per-item reliability feedback, asymmetric (loss-averse):
        helpful +trust_helpful, unhelpful −trust_unhelpful (2× by default).
        The stored value is decayed to *now* first, so stale trust doesn't
        anchor the update. Returns the new trust, or None if unknown id."""
        with self._lock.write():
            return self._trust_feedback_unlocked(node_id, helpful, now=now)

    def _trust_feedback_unlocked(
        self, node_id: str, helpful: bool, *, now: Optional[float] = None
    ) -> Optional[float]:
        """The body of ``trust_feedback`` for callers ALREADY holding the
        write lock.

        The lock is not reentrant. ``_feedback`` and ``_learn`` both update
        per-item trust as part of one learning step, and both already run
        under it — calling the public method there self-deadlocks.
        """
        node = self.store.get_node(node_id)
        if node is None:
            return None
        ts = now if now is not None else time.time()
        eff = self._effective_trust(node, ts)
        delta = self.cfg.trust_helpful if helpful else -self.cfg.trust_unhelpful
        t = min(1.0, max(0.0, eff + delta))
        self.store.set_trust(node_id, t, ts)
        return t

    def learn(
        self,
        query_key: str,
        *,
        positives: Sequence[tuple],
        negatives: Sequence[Sequence[float]],
        label_src: str = "implicit",
    ) -> Dict[str, float]:
        """Feature-level feedback — the CROSS-TURN-SAFE learning entry point.

        Unlike :meth:`feedback` (which looks the query's candidate features up
        in the bounded ``_recent_queries`` cache and so silently no-ops once the
        query has been evicted), ``learn`` takes the feature vectors from the
        caller. A host that remembers "note X was shown for query Q with feature
        vector f" can therefore reinforce it any number of turns later — when the
        user finally *edits* or *cites* that note — with no dependency on how
        many searches happened in between.

        Parameters
        ----------
        query_key : str
            A stable identifier for the originating query. Only its hash is used,
            to place the whole query on the SAME side of the blend gate's
            out-of-sample hold-out (so a query is either trained-on or refereed,
            never both — the property that makes the gate immune to label noise).
        positives : sequence of ``(node_id, feature_vector)``
            The memories a trusted external signal says were actually useful.
            ``node_id`` drives Hebbian co-access (ids that were useful *together*
            get linked); the feature vector drives the pairwise ranker.
        negatives : sequence of feature_vector
            Features of memories shown for the same query but NOT flagged useful
            (the ranker learns useful > shown-but-unused). Caller must never pass
            "every result" as positive — that would train the ranker to rubber-
            stamp whatever the current retriever already surfaces.
        """
        with self._lock.write():
            return self._learn(query_key, positives, negatives, label_src)

    def _learn(
        self,
        query_key: str,
        positives: Sequence[tuple],
        negatives: Sequence[Sequence[float]],
        label_src: str,
    ) -> Dict[str, float]:
        pos = [(pid, np.asarray(f, dtype=np.float32)) for pid, f in positives if f is not None]
        neg = [np.asarray(f, dtype=np.float32) for f in negatives if f is not None]
        if not pos or not neg:
            return {"applied": 0.0, "pairs": 0.0}
        qh = hashlib.sha1(query_key.encode("utf-8")).hexdigest()[:16]

        # ① Hebbian — memories confirmed useful TOGETHER get a co-access edge.
        pos_ids = [pid for pid, _ in pos if pid]
        if len(pos_ids) >= 2:
            reinforce_coaccess(
                self.store,
                pos_ids,
                eta=self.cfg.hebb_eta,
                decay=self.cfg.hebb_decay,
                prune=self.cfg.hebb_prune,
            )
            self._adj_cache.pop(EDGE_COACCESS, None)
        # ①b Trust — a confirmed-useful item gains reliability. (Negatives do
        # NOT lose trust here: shown-but-unflagged is a ranking contrast, not
        # evidence the memory itself is wrong — explicit unhelpful feedback
        # goes through trust_feedback().)
        for pid in pos_ids:
            self._trust_feedback_unlocked(pid, True)

        # ② Ranker — same query-level hold-out split as feedback(): a
        # deterministic ~25% of queries referee the blend gate (never trained).
        is_eval = int(qh, 16) % 4 == 0
        loss = 0.0
        pairs = 0
        for _, pf in pos:
            for nf in neg:
                if is_eval:
                    self.ranker.referee_pair(pf, nf)
                else:
                    loss += self.ranker.update_pair(pf, nf)
                    pairs += 1
        if not is_eval:
            for pid, pf in pos:
                self.store.add_feedback(
                    qh, pid or "pos", pf.tobytes(), True, label_src, self.cfg.replay_cap
                )
            for i, nf in enumerate(neg):
                self.store.add_feedback(
                    qh, f"neg{i}", nf.tobytes(), False, label_src, self.cfg.replay_cap
                )
            pairs += self._replay(8)
        self._persist_ranker()
        return {
            "applied": 1.0,
            "pairs": float(pairs),
            "loss": loss / max(1, pairs),
            **self.ranker.stats(),
        }

    # ── compositional entity join (AND/OR retrieval) ─────────────────
    def search_join(
        self, entities: Sequence[str], *, mode: str = "and", top_k: Optional[int] = None
    ) -> List[SearchHit]:
        """Memories related to ALL (*and*) or ANY (*or*) of *entities*.

        Additive fusion (plain BM25 over "e1 e2") lets one strong single-
        entity match dominate; the intersection question — "what touches
        BOTH X and Y?" — needs the WEAKEST-LINK score. Per entity we build a
        relatedness map (normalized BM25 ∪ graph PPR from that entity's
        lexical seeds, so a memory can qualify through a LINK/TAG/KNN
        connection even without mentioning the entity verbatim), then score
        candidates by ``min`` across entities (AND) or ``mean`` (OR)."""
        if mode not in ("and", "or"):
            raise ValueError(f"mode must be 'and' or 'or', got {mode!r}")
        with self._lock.read():
            top_k = self.cfg.top_k if top_k is None else max(0, top_k)
            ents = [e for e in entities if e and e.strip()]
            if not ents:
                return []
            rels: List[Dict[str, float]] = []
            for ent in ents:
                toks = lexical_tokens(
                    ent,
                    char_ngrams=self.cfg.char_ngrams,
                    suffix_strip=self.cfg.suffix_strip,
                    cross_space=self.cfg.cross_space,
                    limit=self.cfg.max_query_tokens,
                )
                bm = bm25_scores(
                    self.store,
                    toks,
                    doc_lens=self._doclens(),
                    k1=self.cfg.bm25_k1,
                    b=self.cfg.bm25_b,
                )
                rel: Dict[str, float] = {}
                if bm:
                    mx = max(bm.values())
                    if mx > 0:
                        for n, s in bm.items():
                            rel[n] = s / mx
                # graph reach: PPR seeded by the entity's lexical top hits —
                # slightly discounted so verbatim mentions outrank hops.
                seeds = {n: bm[n] for n in top_n(bm, self.cfg.bm25_seed_k)}
                if seeds:
                    ppr = ppr_features(
                        self._adjacencies(),
                        seeds,
                        alpha=self.cfg.ppr_alpha,
                        iters=self.cfg.ppr_iters,
                    )
                    flat: Dict[str, float] = {}
                    for etype_scores in ppr.values():
                        for n, s in etype_scores.items():
                            if s > flat.get(n, 0.0):
                                flat[n] = s
                    if flat:
                        pmx = max(flat.values())
                        if pmx > 0:
                            for n, s in flat.items():
                                cand = 0.8 * s / pmx
                                if cand > rel.get(n, 0.0):
                                    rel[n] = cand
                rels.append(rel)

            if mode == "and":
                common = set(rels[0])
                for r in rels[1:]:
                    common &= set(r)
                scored = [(n, min(r[n] for r in rels)) for n in common]
            else:
                every = set().union(*rels)
                scored = [(n, sum(r.get(n, 0.0) for r in rels) / len(rels)) for n in every]
            scored.sort(key=lambda t: -t[1])
            picked = scored[:top_k]
            meta = {m["id"]: m for m in self.store.nodes([n for n, _ in picked])}
            out: List[SearchHit] = []
            for nid, s in picked:
                node = meta.get(nid) or {}
                out.append(
                    SearchHit(
                        id=nid,
                        score=float(s),
                        title=node.get("title", ""),
                        kind=node.get("kind", "note"),
                        features={},
                        sources=[f"join:{mode}"],
                    )
                )
            return out

    # ── contradiction detection (store hygiene) ─────────────────────
    #: Deterministic negation markers (ko + en). Bag-of-ngram similarity is
    #: nearly blind to negation ("작동한다" vs "작동하지 않는다" score as
    #: highly similar), so marker ASYMMETRY is the primary divergence signal
    #: for true contradictions; (1 − cosine) alone only catches topic drift.
    _NEG_MARKERS = (
        "않",
        "안 ",
        "안된",
        "안 된",
        "안됨",
        "못 ",
        "못한",
        "못함",
        "없",
        "아니",
        "불가",
        "금지",
        "말 것",
        "실패",
        "not ",
        "no ",
        "never",
        "don't",
        "doesn't",
        "can't",
        "cannot",
        "won't",
        "broken",
        "fails",
        "failed",
        "disabled",
        "unavailable",
    )

    @classmethod
    def _has_negation(cls, text: str) -> bool:
        low = text.lower()
        return any(m in low for m in cls._NEG_MARKERS)

    def contradictions(
        self, node_id: str, *, top_k: int = 5, min_score: float = 0.15, candidates: int = 64
    ) -> List[Dict[str, Any]]:
        """Memories that likely CONFLICT with *node_id* — store hygiene.

        score = word_jaccard × divergence, where divergence = (1 − cosine)
        plus a +0.5 boost when exactly one side carries a negation marker.
        High topical overlap + diverging/negated content = probable conflict;
        near-duplicates (high overlap, high cosine, same polarity) score ~0.

        Diagnostic only — never deletes. Needs ``store_text=True`` (nodes
        without stored text fall back to their title). Candidate generation
        reuses BM25 with the node's own words as the query, so cost is one
        search, not O(N²)."""
        with self._lock.read():
            node = self.store.get_node(node_id)
            if node is None:
                return []
            text = self._get_text_unlocked(node_id) or node.get("title") or ""
            if not text:
                return []
            words = set(
                w
                for w in lexical_tokens(
                    text,
                    char_ngrams=(),
                    cross_space=False,
                    suffix_strip=self.cfg.suffix_strip,
                    limit=self.cfg.max_doc_tokens,
                )
                if len(w) > 1
            )
            if not words:
                return []
            neg_self = self._has_negation(text)
            vec_self = self.embedder.embed(text, limit=self.cfg.max_doc_tokens)

            # Candidates: highest lexical overlap via BM25 on our own words.
            bm25 = bm25_scores(
                self.store,
                list(words),
                doc_lens=self._doclens(),
                k1=self.cfg.bm25_k1,
                b=self.cfg.bm25_b,
            )
            bm25.pop(node_id, None)
            cand_ids = top_n(bm25, candidates)

            out: List[Dict[str, Any]] = []
            for cid in cand_ids:
                ctext = self._get_text_unlocked(cid)
                if ctext is None:
                    cnode = self.store.get_node(cid)
                    ctext = (cnode.get("title") if cnode else "") or ""
                if not ctext:
                    continue
                cwords = set(
                    w
                    for w in lexical_tokens(
                        ctext,
                        char_ngrams=(),
                        cross_space=False,
                        suffix_strip=self.cfg.suffix_strip,
                        limit=self.cfg.max_doc_tokens,
                    )
                    if len(w) > 1
                )
                if not cwords:
                    continue
                inter = len(words & cwords)
                union = len(words | cwords)
                jaccard = inter / union if union else 0.0
                if jaccard <= 0.0:
                    continue
                cvec = self.embedder.embed(ctext, limit=self.cfg.max_doc_tokens)
                cos = float(np.dot(vec_self, cvec))
                divergence = max(0.0, 1.0 - cos)
                if self._has_negation(ctext) != neg_self:
                    divergence += 0.5
                score = jaccard * divergence
                if score >= min_score:
                    cnode = self.store.get_node(cid) or {}
                    out.append(
                        {
                            "id": cid,
                            "score": round(score, 4),
                            "jaccard": round(jaccard, 4),
                            "cosine": round(cos, 4),
                            "negation_flip": self._has_negation(ctext) != neg_self,
                            "title": cnode.get("title", ""),
                        }
                    )
            out.sort(key=lambda d: -d["score"])
            return out[:top_k]

    # ── distillation ────────────────────────────────────────────────
    def distill(self, *, epochs: Optional[int] = None) -> Dict[str, Any]:
        """Fit the embedding table to stored teacher vectors, then ATOMICALLY
        swap the table and re-embed every stored-text node in one transaction.
        Needs ``store_text=True`` (the whole corpus must be re-embeddable) and
        enough teacher pairs. Bounded batch job — run at idle / close."""
        with self._lock.write():
            if not self.cfg.store_text:
                return {"trained": 0.0, "reason_no_text": 1.0}
            teachers = self.store.teachers()
            texts = self._distill_texts()
            pairs = [
                (texts[nid], unpack_vec(blob, dim))
                for nid, _m, dim, blob in teachers
                if texts.get(nid)
            ]
            metrics = self.embedder.distill(
                pairs,
                epochs=epochs or self.cfg.distill_epochs,
                lr=self.cfg.distill_lr,
                batch=self.cfg.distill_batch,
            )
            candidate = metrics.pop("candidate", None)
            if candidate is not None:
                # Re-embedding must cover EVERY node, else the un-covered ones
                # keep OLD-table vectors that the NEW query embedder can't match.
                all_ids = {n["id"] for n in self.store.nodes()}
                if not all_ids.issubset(texts.keys()):
                    metrics["swapped"] = 0.0
                    metrics["reason_incomplete_text"] = 1.0
                    self._persist_ranker()
                    return metrics
                # Build the candidate on a SCRATCH embedder and swap it in ONLY
                # after the store commit succeeds. If the commit rolls back, the
                # live embedder + on-disk table + vectors all stay OLD together —
                # no in-memory/disk mismatch for the rest of the session.
                scratch = HashEmbedder(
                    self.cfg.vocab_size,
                    self.cfg.dim,
                    seed=self.cfg.seed,
                    char_ngrams=self.cfg.char_ngrams,
                    jamo_ngrams=self.cfg.jamo_ngrams,
                    suffix_strip=self.cfg.suffix_strip,
                )
                scratch.table = candidate
                rows = [
                    (nid, self.cfg.dim, pack_vec(scratch.embed(txt, limit=self.cfg.max_doc_tokens)))
                    for nid, txt in texts.items()
                ]
                self.store.swap_embedder_and_vectors(scratch.dumps(), rows)
                self.embedder = scratch  # commit succeeded → adopt atomically
                self._vec_cache = None
                self._vec_matrix = None
            self._persist_ranker()
            return metrics

    # ── misc ─────────────────────────────────────────────────────────
    def stats(self) -> Dict[str, Any]:
        with self._lock.read():
            return {
                "nodes": self.store.count_nodes(),
                "feedback_rows": self.store.feedback_count(),
                "edges": {
                    name: len(self.store.edges_by_type(t))
                    for name, t in (
                        ("link", EDGE_LINK),
                        ("tag", EDGE_TAG),
                        ("knn", EDGE_KNN),
                        ("coaccess", EDGE_COACCESS),
                    )
                },
                "ranker": self.ranker.stats(),
                "dim": self.cfg.dim,
            }

    def close(self) -> None:
        with self._lock.write():
            self._persist_models()
            self.store.close()

    def __enter__(self) -> "SynapseMemory":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ── internals ────────────────────────────────────────────────────
    def _vectors(self) -> Dict[str, np.ndarray]:
        # Double-checked: several readers can now arrive here at once, and
        # materialising 8.7k vectors twice is pure waste. The second check
        # inside the mutex is what makes the first one safe to skip.
        if self._vec_cache is None:
            with self._cache_build_lock:
                if self._vec_cache is None:
                    self._vec_cache = {
                        nid: unpack_vec(blob, dim) for nid, dim, blob in self.store.all_vectors()
                    }
        return self._vec_cache

    def _vector_matrix(self) -> tuple:
        """(ids, row-stacked matrix) — cached; rebuilt lazily after writes."""
        if self._vec_matrix is None:
            with self._cache_build_lock:
                if self._vec_matrix is None:
                    vectors = self._vectors()
                    ids = list(vectors.keys())
                    matrix = (
                        np.stack([vectors[i] for i in ids])
                        if ids
                        else np.zeros((0, self.cfg.dim), dtype=np.float32)
                    )
                    self._vec_matrix = (ids, matrix)
        return self._vec_matrix

    def _doclens(self) -> Dict[str, int]:
        if self._doclen_cache is None:
            with self._cache_build_lock:
                if self._doclen_cache is None:
                    self._doclen_cache = self.store.doc_lens()
        return self._doclen_cache

    def _n_docs(self) -> int:
        return len(self._doclens())

    def _tags_map(self) -> Dict[str, List[str]]:
        if self._tag_cache is None:
            with self._cache_build_lock:
                if self._tag_cache is None:
                    tag_map: Dict[str, List[str]] = {}
                    for node in self.store.nodes():
                        for t in node["tags"]:
                            tag_map.setdefault(t, []).append(node["id"])
                    self._tag_cache = tag_map
        return self._tag_cache

    def _adjacencies(self) -> Dict[int, dict]:
        out: Dict[int, dict] = {}
        for etype in (EDGE_LINK, EDGE_TAG, EDGE_KNN, EDGE_COACCESS):
            cached = self._adj_cache.get(etype)
            if cached is None:
                with self._cache_build_lock:
                    cached = self._adj_cache.get(etype)
                    if cached is None:
                        cached = build_type_adjacency(
                            self.store.edges_by_type(etype),
                            etype,
                            coaccess_decay=self.cfg.hebb_decay,
                        )
                        self._adj_cache[etype] = cached
            out[etype] = cached
        return out

    def _persist_ranker(self) -> None:
        """Persist ONLY the ranker — tiny (~1 KB), safe to write every feedback."""
        self.store.put_param("ranker", self.ranker.dumps())

    def _persist_embedder(self) -> None:
        """Persist the embedding table (~fp16, tens of MB) — ONLY after
        distillation actually mutates it, never on the feedback hot path
        (writing it every feedback was a 14 s/20-call zlib bottleneck). Lives
        in the same db as the vectors so distill can swap both atomically."""
        self.store.put_param("embedder", self.embedder.dumps())

    def _persist_models(self) -> None:
        self._persist_ranker()
        self._persist_embedder()

    def get_text(self, node_id: str) -> Optional[str]:
        """The stored (bounded) body of a node, or None. Requires
        ``store_text=True``. Lets a host return the actual text alongside a
        search hit — e.g. to fill a retrieval result's ``content`` — without
        keeping the corpus in a second place."""
        with self._lock.read():
            return self._get_text_unlocked(node_id)

    def _get_text_unlocked(self, node_id: str) -> Optional[str]:
        """The body of ``get_text`` for callers ALREADY holding the lock.

        The lock is not reentrant, so ``contradictions`` — which runs under
        the read lock and needs several bodies — must not call the public
        method or it deadlocks against the next waiting writer.
        """
        blob = self.store.get_param(f"text:{node_id}")
        return blob.decode("utf-8", "replace") if blob else None

    def _save_text_for_distill(self, node_id: str, body: str) -> None:
        # Distillation needs the text back; store a bounded copy in params-space.
        key = f"text:{node_id}"
        self.store.put_param(key, body[: self.cfg.store_text_maxlen].encode("utf-8"))

    def _distill_texts(self) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for node in self.store.nodes():
            blob = self.store.get_param(f"text:{node['id']}")
            if blob:
                out[node["id"]] = blob.decode("utf-8", "replace")
        return out
