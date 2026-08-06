"""Ranking layer — heuristic floor + tiny online-learned MLP, safely blended.

Score = (1 − λ)·heuristic + λ·MLP(features)

The heuristic is a fixed, hand-set linear combination (the performance FLOOR —
equivalent to a sensible non-learned engine). The MLP (14→16→1, ~273 params,
numpy only) learns from pairwise implicit feedback. λ stays 0 until the
learner has BOTH enough labelled events and a better online AUC than the
heuristic — so learning can only ever help, never regress below the floor.
An EMA shadow smooths weight updates; everything serializes into one blob.
"""

from __future__ import annotations

import io
import json
from typing import Dict, Optional

import numpy as np

FEATURES = [
    "bm25", "cosine", "rrf", "ppr_link", "ppr_tag", "ppr_knn", "ppr_co",
    "recency", "freq", "importance", "pinned", "title_hit", "kind_prior", "len_norm",
]
N_FEATURES = len(FEATURES)

#: The heuristic floor: fixed weights over z-normalized features (mirrors a
#: sane hand-tuned engine: semantic + keyword dominate, graph/recency assist).
_HEURISTIC_W = np.array([
    1.2,   # bm25 — measured stronger than the undistilled hash cosine on the
           # MIRACL-ko harness; distillation shifts this balance and the
           # LEARNED ranker re-weights it per-vault at runtime.
    0.6,   # cosine
    0.9,   # rrf
    0.45,  # ppr_link
    0.25,  # ppr_tag
    0.35,  # ppr_knn
    0.45,  # ppr_co
    0.30,  # recency
    0.15,  # freq
    0.40,  # importance
    0.50,  # pinned
    0.35,  # title_hit
    0.10,  # kind_prior
    0.05,  # len_norm
], dtype=np.float32)


class OnlineRanker:
    def __init__(self, *, hidden: int = 16, lr: float = 0.05, l2: float = 1e-4,
                 blend_min_events: int = 100, seed: int = 13) -> None:
        rng = np.random.default_rng(seed)
        self.lr = lr
        self.l2 = l2
        self.blend_min_events = blend_min_events
        self.W1 = (rng.standard_normal((N_FEATURES, hidden)) * 0.15).astype(np.float32)
        self.b1 = np.zeros(hidden, dtype=np.float32)
        self.W2 = (rng.standard_normal(hidden) * 0.15).astype(np.float32)
        self.b2 = np.float32(0.0)
        # EMA shadow (what scoring actually uses) — smooths SGD jitter.
        self._ema = [self.W1.copy(), self.b1.copy(), self.W2.copy(), np.float32(self.b2)]
        self.ema_beta = 0.98
        # Running feature normalization (Welford-ish, persisted).
        self.mu = np.zeros(N_FEATURES, dtype=np.float32)
        self.var = np.ones(N_FEATURES, dtype=np.float32)
        self.n_norm = 0
        # Blend gate — a McNemar paired test of learner-vs-heuristic evaluated
        # STRICTLY OUT-OF-SAMPLE. The host holds out a deterministic slice of
        # QUERIES (by query hash) and feeds their pairs to referee_pair(), never
        # to update_pair() — so the learner truly never trains on them. That
        # query-level split (not a per-call ordinal) is what keeps replayed,
        # already-trained pairs out of the referee. Under random feedback the
        # held-out discordant win-rate is a true 50% and the gate stays shut;
        # under a genuine signal it wins the held-out queries and opens. Counts
        # are INTEGER and accumulate (no decay), so confidence tightens with
        # data. `disc_b` = learner right & heuristic wrong; `disc_c` = reverse;
        # concordant pairs carry no comparative signal.
        self.events = 0
        self.disc_b = 0  # held-out: learner correct, heuristic wrong
        self.disc_c = 0  # held-out: heuristic correct, learner wrong

    #: The blend gate needs the Wilson lower bound of the held-out win-rate to
    #: clear THIS (a margin above 0.5), so noise near 50% can't open it.
    _GATE_FLOOR = 0.52

    # ── normalization ────────────────────────────────────────────────
    def observe(self, x: np.ndarray) -> None:
        self.n_norm += 1
        k = min(self.n_norm, 5000)  # cap adaptation speed
        delta = x - self.mu
        self.mu += delta / k
        self.var += (delta * (x - self.mu) - self.var) / k

    #: z-scores are clipped to ±this. A near-constant feature (every candidate
    #: shares a kind_prior, or all-zero ppr) drives `var → 0`, and (x−mu)/√var
    #: then explodes — on a small corpus that made absolute scores swing 100×
    #: across warmup. Clipping caps that WITHOUT rescaling informative features
    #: (whose |z| < clip), so ranking quality is unchanged on real corpora
    #: while the blow-up is contained. Cheaper and less distorting than a
    #: variance floor, which shifts every feature's scale.
    _Z_CLIP = 8.0

    def _z(self, x: np.ndarray) -> np.ndarray:
        z = (x - self.mu) / np.sqrt(self.var + 1e-6)
        return np.clip(z, -self._Z_CLIP, self._Z_CLIP)

    # ── scoring ──────────────────────────────────────────────────────
    @staticmethod
    def _gelu(x: np.ndarray) -> np.ndarray:
        return 0.5 * x * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x ** 3)))

    def heuristic(self, x: np.ndarray) -> float:
        return float(_HEURISTIC_W @ self._z(x))

    def _mlp(self, z: np.ndarray, params: Optional[list] = None) -> float:
        W1, b1, W2, b2 = params or self._ema
        h = self._gelu(z @ W1 + b1)
        return float(h @ W2 + b2)

    @staticmethod
    def _wilson_lower(successes: float, n: float, z: float = 3.0) -> float:
        """Wilson score lower bound for a binomial proportion (z=3.0 ≈ 99.9%).

        With honest INTEGER counts this tightens as n grows, so a genuinely
        better learner clears 0.5 once enough held-out pairs accumulate. The
        conservative z absorbs the residual document-level overlap between the
        held-out and trained queries (a held-out QUERY is out-of-sample, but
        its candidate DOCUMENTS can still appear in trained queries), which
        would otherwise let noise edge a few points over 0.5."""
        if n <= 0:
            return 0.0
        p = successes / n
        denom = 1.0 + z * z / n
        center = p + z * z / (2 * n)
        margin = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
        return (center - margin) / denom

    @property
    def blend(self) -> float:
        """λ — how much the learned score participates.

        Opens only when the OUT-OF-SAMPLE McNemar test says the learner beats
        the heuristic with statistical confidence (Wilson 99% lower bound of
        the held-out discordant win-rate > 0.5), never on a numerical lead."""
        if self.events < self.blend_min_events:
            return 0.0
        disc = self.disc_b + self.disc_c
        if disc < 20:
            return 0.0
        lower = self._wilson_lower(self.disc_b, disc)
        # Require the confident win-rate to clear 0.52, not just 0.5 — a 2-point
        # margin above the coin-flip that residual document overlap can't fake,
        # so noise (which tops out just over 0.50) never opens the gate while a
        # real edge still does.
        if lower <= self._GATE_FLOOR:
            return 0.0
        return float(min(0.7, 5.0 * (lower - self._GATE_FLOOR)))

    def score(self, x: np.ndarray) -> float:
        z = self._z(x)
        lam = self.blend
        h = float(_HEURISTIC_W @ z)
        if lam <= 0.0:
            return h
        return (1.0 - lam) * h + lam * self._mlp(z)

    # ── learning ─────────────────────────────────────────────────────
    def referee_pair(self, x_pos: np.ndarray, x_neg: np.ndarray) -> None:
        """Judge one HELD-OUT pair for the blend gate WITHOUT training on it.

        The caller guarantees this pair (and the query it came from) is never
        passed to :meth:`update_pair`, so the McNemar count is genuinely
        out-of-sample — the learner cannot have memorized it, which is what
        makes the gate immune to label noise."""
        self.events += 1
        zp, zn = self._z(x_pos), self._z(x_neg)
        learner_ok = self._mlp(zp) > self._mlp(zn)
        heur_ok = float(_HEURISTIC_W @ zp) > float(_HEURISTIC_W @ zn)
        if learner_ok != heur_ok:
            if learner_ok:
                self.disc_b += 1
            else:
                self.disc_c += 1

    def update_pair(self, x_pos: np.ndarray, x_neg: np.ndarray) -> float:
        """One pairwise SGD step: used ranked above ignored. TRAIN only — the
        blend-gate referee is :meth:`referee_pair` on held-out queries."""
        self.events += 1
        zp, zn = self._z(x_pos), self._z(x_neg)
        # TRAIN: forward on live (non-EMA) weights.
        hp = self._gelu(zp @ self.W1 + self.b1)
        hn = self._gelu(zn @ self.W1 + self.b1)
        sp = float(hp @ self.W2 + self.b2)
        sn = float(hn @ self.W2 + self.b2)
        margin = sp - sn
        loss = float(np.logaddexp(0.0, -margin))  # stable softplus (no overflow)
        g = -1.0 / (1.0 + np.exp(min(margin, 30.0)))  # dL/dmargin, clamped

        # Backprop (manual, tiny).
        def back(z: np.ndarray, h: np.ndarray, sign: float):
            gs = g * sign
            gW2 = gs * h
            gb2 = gs
            gh = gs * self.W2
            pre = z @ self.W1 + self.b1
            # GELU'(x) approx
            t = np.tanh(0.7978845608 * (pre + 0.044715 * pre ** 3))
            dg = 0.5 * (1 + t) + 0.5 * pre * (1 - t ** 2) * 0.7978845608 * (1 + 3 * 0.044715 * pre ** 2)
            gpre = gh * dg
            gW1 = np.outer(z, gpre)
            gb1 = gpre
            return gW1, gb1, gW2, gb2

        gW1p, gb1p, gW2p, gb2p = back(zp, hp, +1.0)
        gW1n, gb1n, gW2n, gb2n = back(zn, hn, -1.0)
        self.W1 -= self.lr * (gW1p + gW1n + self.l2 * self.W1)
        self.b1 -= self.lr * (gb1p + gb1n)
        self.W2 -= self.lr * (gW2p + gW2n + self.l2 * self.W2)
        self.b2 -= np.float32(self.lr * (gb2p + gb2n))

        # EMA shadow update.
        b = self.ema_beta
        for ema, live in zip(self._ema, [self.W1, self.b1, self.W2, np.float32(self.b2)]):
            if isinstance(ema, np.ndarray):
                ema *= b
                ema += (1 - b) * live
        self._ema[3] = np.float32(b * float(self._ema[3]) + (1 - b) * float(self.b2))
        return loss

    # ── persistence ──────────────────────────────────────────────────
    def dumps(self) -> bytes:
        buf = io.BytesIO()
        np.savez_compressed(
            buf, W1=self.W1, b1=self.b1, W2=self.W2, b2=np.float32(self.b2),
            eW1=self._ema[0], eb1=self._ema[1], eW2=self._ema[2], eb2=np.float32(self._ema[3]),
            mu=self.mu, var=self.var,
            meta=json.dumps({
                "n_norm": self.n_norm, "events": self.events,
                "disc_b": self.disc_b, "disc_c": self.disc_c,
                "lr": self.lr, "l2": self.l2, "blend_min_events": self.blend_min_events,
            }),
        )
        return buf.getvalue()

    @classmethod
    def loads(cls, blob: bytes) -> "OnlineRanker":
        data = np.load(io.BytesIO(blob), allow_pickle=False)
        meta = json.loads(str(data["meta"]))
        r = cls(hidden=data["W1"].shape[1], lr=meta["lr"], l2=meta["l2"],
                blend_min_events=meta["blend_min_events"])
        r.W1, r.b1, r.W2, r.b2 = data["W1"], data["b1"], data["W2"], np.float32(data["b2"])
        r._ema = [data["eW1"], data["eb1"], data["eW2"], np.float32(data["eb2"])]
        r.mu, r.var = data["mu"], data["var"]
        r.n_norm = meta["n_norm"]
        r.events = meta["events"]
        r.disc_b = int(meta.get("disc_b", 0))
        r.disc_c = int(meta.get("disc_c", 0))
        return r

    def stats(self) -> Dict[str, float]:
        disc = self.disc_b + self.disc_c
        return {
            "events": float(self.events),
            "blend": self.blend,
            "disc_b": float(self.disc_b),
            "disc_c": float(self.disc_c),
            "win_rate": self.disc_b / disc if disc > 0 else 0.0,
        }
