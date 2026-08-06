"""Typed-edge graph: derivation, per-type Personalized PageRank, Hebbian edges.

Edge types
  LINK (0)     — explicit links the caller declares (wikilinks, citations)
  TAG (1)      — shared tags, IDF-weighted, fanout-capped (anti-hub)
  KNN (2)      — semantic nearest neighbours in the local embedding space,
                 frozen at index time
  COACCESS (3) — LEARNED: Hebbian reinforcement of pairs that were retrieved
                 together and both actually used, with lazy time decay

Retrieval runs one PPR per edge type so each type becomes its own ranking
feature — the ranker learns the type weights, no differentiable graph needed.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

from .store import EDGE_COACCESS, EDGE_KNN, EDGE_LINK, EDGE_TAG, Store

_WEEK = 7 * 24 * 3600.0


#: A tag on more than this FRACTION of the corpus is a "meta tag" (status,
#: type markers like execution/success/note) — it carries no topical signal
#: and, worse, densifies the tag graph toward a clique that makes PPR blow up.
#: Such tags are excluded from edge derivation.
_TAG_DF_CUTOFF = 0.30
#: Absolute floor so tiny corpora don't drop everything.
_TAG_DF_MIN_DOCS = 40


def derive_tag_edges(tag_members: Dict[str, List[str]], n_docs: int, node_id: str,
                     tags: Sequence[str], *, fanout: int = 6) -> List[Tuple[str, float]]:
    """IDF-weighted shared-tag edges from *node_id* to its tag-mates.

    *tag_members* is the engine's cached {tag: [node_ids]} map — passing it in
    keeps bulk indexing O(N) instead of O(N²). Meta tags (present on a large
    fraction of the corpus) are skipped: no topical signal, and they would
    otherwise turn the tag graph into a near-clique."""
    if not tags:
        return []
    n = max(1, n_docs)
    cutoff = max(_TAG_DF_MIN_DOCS, int(n * _TAG_DF_CUTOFF))
    weights: Dict[str, float] = {}
    for tag in tags:
        members = tag_members.get(tag, [])
        df = len(members)
        if df < 2 or df > cutoff:
            continue
        idf = math.log(1.0 + n / df)
        # Fanout cap: hub tags connect only a bounded number of mates.
        mates = [m for m in members if m != node_id][-fanout:]
        for m in mates:
            weights[m] = max(weights.get(m, 0.0), 0.5 * idf)
    return sorted(weights.items(), key=lambda kv: -kv[1])[:fanout * 2]


def derive_knn_edges(query_vec, vectors: Dict[str, "object"], node_id: str,
                     *, k: int = 6, min_sim: float = 0.25,
                     sample_cap: int = 4096) -> List[Tuple[str, float]]:
    """Semantic neighbours of a freshly indexed node (local embedding space).

    Scanning EVERY vector per index makes bulk indexing O(N²). KNN edges are a
    graph-expansion aid, not exact — so above *sample_cap* we compare only
    against the most recently indexed vectors (dict keeps insertion order),
    which keeps indexing linear and biases toward recent, likely-more-relevant
    memories. `argpartition` finds the top-k in O(N) instead of a full sort."""
    import numpy as np

    ids = [i for i in vectors.keys() if i != node_id]
    if not ids:
        return []
    if len(ids) > sample_cap:
        ids = ids[-sample_cap:]
    mat = np.stack([vectors[i] for i in ids])
    sims = mat @ query_vec
    if len(sims) > k:
        top = np.argpartition(-sims, k)[:k]
    else:
        top = np.arange(len(sims))
    order = top[np.argsort(-sims[top])]
    return [(ids[int(j)], float(sims[int(j)])) for j in order if sims[int(j)] >= min_sim]


# ── Hebbian co-access ────────────────────────────────────────────────


def reinforce_coaccess(store: Store, used_ids: Sequence[str], *, eta: float = 0.3,
                       decay: float = 0.9, prune: float = 0.05) -> int:
    """Strengthen COACCESS edges between every pair of *used* memories.

    Lazy decay: the stored weight is first decayed by elapsed weeks since its
    last update, then reinforced — no background job ever runs.
    """
    now = time.time()
    updates: List[Tuple[str, str, int, float]] = []
    ids = list(dict.fromkeys(used_ids))
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            for src, dst in ((a, b), (b, a)):
                row = store.get_edge(src, dst, EDGE_COACCESS)
                w = 0.0
                if row is not None:
                    w0, updated = row
                    w = w0 * (decay ** max(0.0, (now - updated) / _WEEK))
                w = w + eta * (1.0 - w)
                updates.append((src, dst, EDGE_COACCESS, w))
    if updates:
        store.upsert_edges(updates)
        store.prune_edges(EDGE_COACCESS, prune)
    return len(updates)


# ── per-type PPR ─────────────────────────────────────────────────────


def build_adjacency(
    edges: Iterable[Tuple[str, str, float]],
) -> Dict[str, List[Tuple[str, float]]]:
    """Row-normalized adjacency {src: [(dst, p)]} — built ONCE per graph
    mutation (the engine caches it), so queries never touch raw edge rows."""
    out: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
    sums: Dict[str, float] = defaultdict(float)
    for src, dst, w in edges:
        if w <= 0:
            continue
        out[src].append((dst, w))
        sums[src] += w
    return {src: [(dst, w / sums[src]) for dst, w in lst] for src, lst in out.items()}


def personalized_pagerank(
    adj: Dict[str, List[Tuple[str, float]]],
    seeds: Dict[str, float],
    *,
    alpha: float = 0.5,
    iters: int = 20,
    tol: float = 1e-5,
) -> Dict[str, float]:
    """Sparse power iteration over a prebuilt row-normalized adjacency.

    Frontier-limited: only mass reachable from the seeds is ever touched, and
    dangling mass returns to the seed distribution."""
    if not seeds:
        return {}
    s_total = sum(seeds.values()) or 1.0
    restart = {k: v / s_total for k, v in seeds.items()}
    rank = dict(restart)
    for _ in range(iters):
        nxt: Dict[str, float] = defaultdict(float)
        dangling = 0.0
        for node, r in rank.items():
            share = (1.0 - alpha) * r
            outs = adj.get(node)
            if outs:
                for dst, p in outs:
                    nxt[dst] += share * p
            else:
                dangling += share
        # Restart + dangling mass in one pass over the (small) seed dict.
        for k, v in restart.items():
            nxt[k] += alpha * v + dangling * v
        # L1 convergence — a rank key absent from nxt contributes its own
        # value; nxt keys are a superset of the mass that moved, so summing
        # over nxt (plus rank-only keys) is exact without a full set union.
        delta = 0.0
        for k, nv in nxt.items():
            delta += abs(nv - rank.get(k, 0.0))
        for k, rv in rank.items():
            if k not in nxt:
                delta += rv
        rank = dict(nxt)
        if delta < tol:
            break
    return rank


def build_type_adjacency(
    rows: List[Tuple[str, str, float, float]],
    etype: int,
    *,
    now: float | None = None,
    coaccess_decay: float = 0.9,
) -> Dict[str, List[Tuple[str, float]]]:
    """Edge rows (src, dst, w, updated) → normalized adjacency for one type.

    COACCESS weights are decayed at build time (lazy — the engine invalidates
    this cache on every reinforcement, so drift within a cache lifetime is
    negligible and no maintenance job ever runs)."""
    now = now or time.time()
    if etype == EDGE_COACCESS:
        edges = [(s, d, w * (coaccess_decay ** max(0.0, (now - u) / _WEEK)))
                 for s, d, w, u in rows]
    elif etype == EDGE_LINK:
        # LINK is stored one-directional (what each node declares); treat it as
        # UNDIRECTED for propagation. Symmetrizing here — instead of persisting
        # reverse edges — means a node's link set is fully described by its own
        # rows, so re-indexing can never orphan a reverse edge. Dedup by pair
        # (max weight) so a MUTUAL link (A→B and B→A both declared) counts once,
        # not twice, in the row-normalized adjacency.
        pair_w: Dict[Tuple[str, str], float] = {}
        for s, d, w, _u in rows:
            for a, b in ((s, d), (d, s)):
                key = (a, b)
                if w > pair_w.get(key, 0.0):
                    pair_w[key] = w
        edges = [(a, b, w) for (a, b), w in pair_w.items()]
    else:
        edges = [(s, d, w) for s, d, w, _u in rows]
    return build_adjacency(edges)


def ppr_features(
    adjacencies: Dict[int, Dict[str, List[Tuple[str, float]]]],
    seeds: Dict[str, float],
    *,
    alpha: float = 0.5,
    iters: int = 20,
) -> Dict[int, Dict[str, float]]:
    """One PPR per edge type over prebuilt adjacencies → {etype: {node: score}}."""
    result: Dict[int, Dict[str, float]] = {}
    for etype in (EDGE_LINK, EDGE_TAG, EDGE_KNN, EDGE_COACCESS):
        adj = adjacencies.get(etype)
        if not adj:
            result[etype] = {}
            continue
        result[etype] = personalized_pagerank(adj, seeds, alpha=alpha, iters=iters)
    return result
