"""BM25 over the SQLite postings table (Okapi, query-terms-only fetch).

Only the postings for the QUERY's terms are pulled — never the full index —
so scoring cost tracks query length, not corpus size.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Dict, List, Optional, Sequence

from .store import Store


def bm25_scores(
    store: Store,
    query_terms: Sequence[str],
    *,
    doc_lens: Optional[Dict[str, int]] = None,
    k1: float = 1.2,
    b: float = 0.75,
) -> Dict[str, float]:
    """Return {node_id: bm25} for every document matching ≥1 query term.

    Pass the engine's cached *doc_lens* to skip the per-query table scan."""
    if not query_terms:
        return {}
    if doc_lens is None:
        doc_lens = store.doc_lens()
    n_docs = max(1, len(doc_lens))
    avg_len = max(1.0, sum(doc_lens.values()) / n_docs)
    unique_terms = list(dict.fromkeys(query_terms))
    postings = store.postings_for_terms(unique_terms)
    # Query-side term frequency softens repeated n-grams from long queries.
    qtf = Counter(query_terms)

    scores: Dict[str, float] = {}
    for term in unique_terms:
        rows = postings.get(term)
        if not rows:
            continue
        df = len(rows)
        idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
        wq = 1.0 + math.log(qtf[term])
        for node_id, tf in rows:
            dl = doc_lens.get(node_id, avg_len)
            denom = tf + k1 * (1.0 - b + b * dl / avg_len)
            scores[node_id] = scores.get(node_id, 0.0) + wq * idf * (tf * (k1 + 1.0)) / denom
    return scores


def term_frequencies(tokens: Sequence[str]) -> Dict[str, float]:
    """Sub-linear tf map for indexing (1 + log(count))."""
    return {t: 1.0 + math.log(c) for t, c in Counter(tokens).items()}


def top_n(scores: Dict[str, float], n: int) -> List[str]:
    return [k for k, _ in sorted(scores.items(), key=lambda kv: -kv[1])[:n]]
