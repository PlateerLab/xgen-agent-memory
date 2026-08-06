"""Configuration — explicit args first, environment second, defaults last.

Every knob can arrive three ways (highest precedence first):

1. Constructor arguments: ``SynapseMemory(path=..., dim=...)``
2. Environment variables with the ``GMA_`` prefix (a ``.env`` file can be
   loaded with :func:`load_dotenv` — a tiny dependency-free parser)
3. The dataclass defaults below

The engine itself never talks to the network; the only "credential"-shaped
value is the optional teacher-embedding source used for distillation, and even
that is data the caller hands in (vectors), not a key this library uses.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Optional

_ENV_PREFIX = "GMA_"


def load_dotenv(path: str | os.PathLike = ".env", *, override: bool = False) -> Dict[str, str]:
    """Minimal ``.env`` loader (KEY=VALUE lines, ``#`` comments, quotes).

    Returns the parsed map and sets ``os.environ`` (existing values win unless
    ``override``). No dependency on python-dotenv.
    """
    parsed: Dict[str, str] = {}
    p = Path(path)
    if not p.is_file():
        return parsed
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'\"")
        if not key:
            continue
        parsed[key] = value
        if override or key not in os.environ:
            os.environ[key] = value
    return parsed


@dataclass
class SynapseConfig:
    """All engine knobs. See module docstring for the precedence rules."""

    # ── storage ──
    #: SQLite database path. ``":memory:"`` for ephemeral (tests). The learned
    #: embedding table lives INSIDE this db (a params row), so a distill can
    #: swap the table and re-embed every vector in ONE atomic transaction — no
    #: sidecar file to fall out of sync, and two instances can't clobber it.
    path: str = "synapse.db"
    #: Keep a copy of each note's text (bounded) so ``distill()`` can re-embed
    #: the whole corpus with an improved table, and so a host can read a hit's
    #: text back via ``get_text()``. Cleaned up on ``remove()``. Set False when
    #: you never distill and never need the text back.
    store_text: bool = True
    #: Max characters of each body kept when ``store_text`` — cap on the text
    #: duplicated into the db. Raise it when using ``get_text()`` to surface
    #: full note bodies in search results.
    store_text_maxlen: int = 4000

    # ── embedding layer ──
    #: Hash-bucket vocabulary size (power of two).
    vocab_size: int = 65_536
    #: Local embedding dimension.
    dim: int = 256
    #: Deterministic seed for the initial hash-projection table.
    seed: int = 41

    # ── tokenizer ──
    #: Syllable n-gram sizes for the LEXICAL (BM25) stream. Bigrams only by
    #: default — the NTCIR-validated sweet spot; trigrams cost 2-3× for no
    #: measured gain.
    char_ngrams: tuple = (2,)
    #: Jamo n-gram sizes for the EMBEDDING stream only (typo robustness).
    #: 3/5 per the ACL 2018 Korean subword recipe; jamo bigrams hurt.
    jamo_ngrams: tuple = (3, 5)
    #: Additionally index a guarded 조사-stripped stem next to each surface word.
    suffix_strip: bool = True
    #: Cross-space syllable bigrams (붙여쓰기 robustness) in the BM25 stream.
    cross_space: bool = True
    #: BM25F-lite: title terms count this many times in the postings.
    title_boost: float = 2.0
    #: Per-document token cap (indexing) / per-query cap.
    max_doc_tokens: int = 2048
    max_query_tokens: int = 256

    # ── retrieval ──
    #: Seed set sizes for the two retrievers before fusion.
    bm25_seed_k: int = 24
    vector_seed_k: int = 24
    #: BM25 parameters.
    bm25_k1: float = 1.2
    bm25_b: float = 0.75
    #: PPR restart probability and iteration cap (per edge type).
    ppr_alpha: float = 0.5
    ppr_iters: int = 20
    #: Graph expansion frontier cap (nodes added beyond seeds).
    graph_expand_k: int = 16
    #: Final results returned by default.
    top_k: int = 8

    # ── graph edges ──
    #: Semantic k-NN edges derived at index time.
    knn_edges: int = 6
    knn_min_sim: float = 0.25
    #: Above this many vectors, a new node's k-NN is computed only against the
    #: most recently indexed `knn_sample_cap` — keeps bulk indexing linear
    #: instead of O(N²). Raise for exact k-NN on large static corpora.
    knn_sample_cap: int = 4096
    #: Tag-edge fanout cap per tag (anti-hub).
    tag_fanout: int = 6
    #: Hebbian co-access learning rate / weekly decay / prune floor.
    hebb_eta: float = 0.3
    hebb_decay: float = 0.9
    hebb_prune: float = 0.05

    # ── ranker / learning ──
    #: Hidden width of the tiny MLP.
    hidden: int = 16
    #: Online SGD learning rate + L2.
    lr: float = 0.05
    l2: float = 1e-4
    #: Replay buffer size (feedback rows kept for mini-batch refreshes).
    replay_cap: int = 4096
    #: Blend gate: the learned score joins the heuristic only after this many
    #: labelled events AND a better online AUC (safety floor).
    blend_min_events: int = 100
    #: Exploration: probability of swapping one tail slot with a random candidate.
    epsilon: float = 0.05

    # ── per-item trust (reliability prior, orthogonal to the ranker) ──
    #: Asymmetric trust updates: confirmed-useful +helpful, explicitly-unhelpful
    #: −unhelpful (loss-averse — a bad memory should fall faster than a good
    #: one rises). Trust lives in [0,1], neutral 0.5.
    trust_helpful: float = 0.05
    trust_unhelpful: float = 0.10
    #: Lazy decay of trust TOWARD NEUTRAL with this half-life (days). This is
    #: the anti-ossification mechanism: reinforcement that is never re-confirmed
    #: fades back to 0.5 instead of hardening into a permanent belief. 0 = off.
    trust_half_life_days: float = 45.0
    #: Score bonus scale: final_score += (effective_trust − 0.5) × 2 × weight,
    #: i.e. a fully trusted item gains +weight, a fully distrusted one −weight
    #: (additive — ranker scores can be negative, so a multiplier would invert
    #: the effect on negatively-scored candidates).
    trust_weight: float = 2.0

    # ── distillation ──
    #: Distillation mini-batch epochs / learning rate (Adam).
    distill_epochs: int = 4
    distill_lr: float = 1e-3
    distill_batch: int = 64

    #: Free-form extras for forward compatibility.
    extras: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Fail fast with a clear message instead of a cryptic numpy traceback
        (dim<0 → 'negative dimensions', vocab_size=0 → modulo-by-zero) or a
        silent NaN weight-poisoning (lr=inf) far downstream."""
        import math as _m
        if self.dim < 1:
            raise ValueError(f"dim must be ≥ 1, got {self.dim}")
        if self.vocab_size < 2:
            raise ValueError(f"vocab_size must be ≥ 2, got {self.vocab_size}")
        if self.hidden < 1:
            raise ValueError(f"hidden must be ≥ 1, got {self.hidden}")
        if not (_m.isfinite(self.lr) and self.lr > 0):
            raise ValueError(f"lr must be a positive finite float, got {self.lr}")
        if not (_m.isfinite(self.l2) and self.l2 >= 0):
            raise ValueError(f"l2 must be a non-negative finite float, got {self.l2}")
        if self.top_k < 1:
            raise ValueError(f"top_k must be ≥ 1, got {self.top_k}")
        if not (0.0 <= self.epsilon <= 1.0):
            raise ValueError(f"epsilon must be in [0, 1], got {self.epsilon}")
        if not (0.0 <= self.trust_helpful <= 1.0):
            raise ValueError(f"trust_helpful must be in [0, 1], got {self.trust_helpful}")
        if not (0.0 <= self.trust_unhelpful <= 1.0):
            raise ValueError(f"trust_unhelpful must be in [0, 1], got {self.trust_unhelpful}")
        if not (_m.isfinite(self.trust_half_life_days) and self.trust_half_life_days >= 0):
            raise ValueError(f"trust_half_life_days must be ≥ 0, got {self.trust_half_life_days}")
        if not (_m.isfinite(self.trust_weight) and self.trust_weight >= 0):
            raise ValueError(f"trust_weight must be ≥ 0, got {self.trust_weight}")

    @classmethod
    def from_env(cls, *, dotenv: Optional[str] = None, **overrides: Any) -> "SynapseConfig":
        """Build a config from ``GMA_*`` environment variables + overrides.

        ``GMA_PATH``, ``GMA_DIM``, ``GMA_VOCAB_SIZE``, … (upper-cased field
        names). Explicit ``overrides`` win over env; env wins over defaults.
        """
        if dotenv:
            load_dotenv(dotenv)
        values: Dict[str, Any] = {}
        for f in fields(cls):
            if f.name == "extras":
                continue
            raw = os.environ.get(_ENV_PREFIX + f.name.upper())
            if raw is None:
                continue
            values[f.name] = _coerce(raw, f.type)
        values.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**values)


def _coerce(raw: str, annotation: Any) -> Any:
    text = str(annotation)
    if "bool" in text:
        return raw.strip().lower() in ("1", "true", "yes", "on")
    if "int" in text:
        return int(raw)
    if "float" in text:
        return float(raw)
    if "tuple" in text:
        return tuple(int(x) for x in raw.replace(",", " ").split())
    return raw
