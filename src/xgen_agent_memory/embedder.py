"""Hash-bucket static embedding layer — lookup + mean-pool, and distillation.

Inference is `L2norm(mean_i W[h(tok_i)])`: no matmul, no network, O(tokens).
The table starts as a deterministic random projection (already a usable weak
semantic space thanks to shared sub-word n-grams) and can be DISTILLED against
teacher vectors the caller already has (e.g. previously paid-for API
embeddings) — the library never calls an embedding API itself.

Distillation objective (Adam, mini-batch, CPU seconds):

    min_{W,P}  Σ ‖ P · e_W(text) − t ‖²   with e_W = mean-pooled table lookup

`P ∈ R^{d_t×d}` maps the local space into the teacher's; only `W` is used at
query time (P is a training-time bridge). A quality gate keeps the old table
unless the new one actually correlates better with the teacher's geometry.
"""

from __future__ import annotations

import io
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from .tokenizer import fnv1a_pair, tokenize


class HashEmbedder:
    def __init__(self, vocab_size: int, dim: int, *, seed: int = 41,
                 char_ngrams: Sequence[int] = (2, 3),
                 jamo_ngrams: Sequence[int] = (3, 5),
                 suffix_strip: bool = True) -> None:
        self.vocab_size = vocab_size
        self.dim = dim
        self.char_ngrams = tuple(char_ngrams)
        self.jamo_ngrams = tuple(jamo_ngrams)
        self.suffix_strip = suffix_strip
        rng = np.random.default_rng(seed)
        # fp32 master table; persisted as fp16 to halve disk. Scaled so that
        # mean-pooled vectors have a sane norm pre-normalization.
        self.table = (rng.standard_normal((vocab_size, dim)) / np.sqrt(dim)).astype(np.float32)

    # ── inference ────────────────────────────────────────────────────
    def bucket_ids(self, text: str, *, limit: int = 2048) -> np.ndarray:
        """Bloom-style k=2 bucket ids, shape (n_tokens, 2).

        Two independent hash views per token make total collisions ~(1/B)²
        (NeurIPS 2017 hash embeddings) — a single hash over a jamo+syllable
        n-gram vocabulary collides badly at 2^16 buckets."""
        toks = tokenize(text, char_ngrams=self.char_ngrams,
                        jamo_ngrams=self.jamo_ngrams,
                        suffix_strip=self.suffix_strip, limit=limit)
        if not toks:
            return np.zeros((0, 2), dtype=np.int64)
        return np.array([fnv1a_pair(t, self.vocab_size) for t in toks], dtype=np.int64)

    def embed(self, text: str, *, limit: int = 2048) -> np.ndarray:
        ids = self.bucket_ids(text, limit=limit)
        if ids.size == 0:
            return np.zeros(self.dim, dtype=np.float32)
        # e_token = (W[h1] + W[h2]) / 2, mean-pooled over tokens.
        v = (self.table[ids[:, 0]] + self.table[ids[:, 1]]).mean(axis=0) * 0.5
        n = float(np.linalg.norm(v))
        return (v / n).astype(np.float32) if n > 1e-9 else v.astype(np.float32)

    # ── distillation ────────────────────────────────────────────────
    #: Below this many (text, teacher) pairs the held-out geometry correlation
    #: is too noisy to trust — distillation trains nothing and reports it.
    MIN_DISTILL_PAIRS = 24
    #: The candidate table is accepted only if it beats the current one by more
    #: than this correlation margin — guards against swapping on holdout noise.
    DISTILL_MARGIN = 0.01

    def distill(
        self,
        pairs: Sequence[Tuple[str, np.ndarray]],
        *,
        epochs: int = 4,
        lr: float = 1e-3,
        batch: int = 64,
        seed: int = 7,
        holdout: float = 0.2,
    ) -> Dict[str, Any]:
        """Fit a CANDIDATE table to teacher vectors. Does NOT mutate
        ``self.table`` — returns the candidate under ``metrics['candidate']``
        (an ndarray) only when it clears a held-out margin, else ``None``. The
        caller applies + re-embeds atomically, so a partial distill is never
        persisted. Needs ≥ MIN_DISTILL_PAIRS with a ≥5-item holdout."""
        if len(pairs) < self.MIN_DISTILL_PAIRS:
            return {"trained": 0.0, "reason_insufficient": 1.0,
                    "pairs": float(len(pairs)), "candidate": None}
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(pairs))
        n_hold = max(5, int(len(pairs) * holdout))
        hold = [pairs[i] for i in idx[:n_hold]]
        train = [pairs[i] for i in idx[n_hold:]]
        d_t = int(hold[0][1].shape[0])

        # Precompute bucket-id lists once.
        train_ids = [self.bucket_ids(t) for t, _ in train]
        teachers = np.stack([v / (np.linalg.norm(v) + 1e-9) for _, v in train]).astype(np.float32)

        W = self.table.copy()
        P = (rng.standard_normal((d_t, self.dim)) / np.sqrt(self.dim)).astype(np.float32)
        mW = np.zeros_like(W); vW = np.zeros_like(W)
        mP = np.zeros_like(P); vP = np.zeros_like(P)
        b1, b2, eps = 0.9, 0.999, 1e-8
        step = 0

        before = self._geometry_corr(hold, self.table)
        order = np.arange(len(train))
        for _ in range(epochs):
            rng.shuffle(order)
            for start in range(0, len(order), batch):
                chunk = order[start : start + batch]
                gW = np.zeros_like(W)
                gP = np.zeros_like(P)
                for j in chunk:
                    ids = train_ids[j]
                    if ids.size == 0:
                        continue
                    e = (W[ids[:, 0]] + W[ids[:, 1]]).mean(axis=0) * 0.5  # (d,)
                    pred = P @ e                                # (d_t,)
                    err = pred - teachers[j]                    # (d_t,)
                    gP += np.outer(err, e)
                    ge = P.T @ err * (0.5 / ids.shape[0])       # (d,)
                    np.add.at(gW, ids[:, 0], ge)
                    np.add.at(gW, ids[:, 1], ge)
                scale = 1.0 / max(1, len(chunk))
                gW *= scale
                gP *= scale
                step += 1
                for grad, param, m, v in ((gW, W, mW, vW), (gP, P, mP, vP)):
                    m *= b1; m += (1 - b1) * grad
                    v *= b2; v += (1 - b2) * grad * grad
                    mh = m / (1 - b1 ** step)
                    vh = v / (1 - b2 ** step)
                    param -= lr * mh / (np.sqrt(vh) + eps)

        after = self._geometry_corr(hold, W)
        accept = after > before + self.DISTILL_MARGIN
        return {"trained": 1.0, "corr_before": before, "corr_after": after,
                "swapped": 1.0 if accept else 0.0, "pairs": float(len(pairs)),
                "candidate": W if accept else None}

    def _geometry_corr(self, hold: Sequence[Tuple[str, np.ndarray]], table: np.ndarray) -> float:
        """Pearson corr between local and teacher pairwise cosine matrices."""
        if len(hold) < 2:
            return 0.0
        local: List[np.ndarray] = []
        teach: List[np.ndarray] = []
        for text, tvec in hold:
            ids = self.bucket_ids(text)
            if ids.size == 0:
                continue
            v = (table[ids[:, 0]] + table[ids[:, 1]]).mean(axis=0) * 0.5
            v = v / (np.linalg.norm(v) + 1e-9)
            local.append(v)
            teach.append(tvec / (np.linalg.norm(tvec) + 1e-9))
        if len(local) < 2:
            return 0.0
        L = np.stack(local); T = np.stack(teach)
        sl = (L @ L.T)[np.triu_indices(len(local), k=1)]
        st = (T @ T.T)[np.triu_indices(len(local), k=1)]
        if sl.std() < 1e-9 or st.std() < 1e-9:
            return 0.0
        return float(np.corrcoef(sl, st)[0, 1])

    # ── persistence ──────────────────────────────────────────────────
    def dumps(self) -> bytes:
        buf = io.BytesIO()
        np.savez_compressed(buf, table=self.table.astype(np.float16),
                            meta=np.array([self.vocab_size, self.dim], dtype=np.int64))
        return buf.getvalue()

    @classmethod
    def loads(cls, blob: bytes, *, seed: int = 41,
              char_ngrams: Sequence[int] = (2, 3),
              jamo_ngrams: Sequence[int] = (3, 5),
              suffix_strip: bool = True) -> "HashEmbedder":
        data = np.load(io.BytesIO(blob))
        vocab_size, dim = (int(x) for x in data["meta"])
        emb = cls(vocab_size, dim, seed=seed, char_ngrams=char_ngrams,
                  jamo_ngrams=jamo_ngrams, suffix_strip=suffix_strip)
        emb.table = data["table"].astype(np.float32)
        return emb

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            f.write(self.dumps())

    @classmethod
    def load(cls, path: str, **kw) -> "HashEmbedder":
        with open(path, "rb") as f:
            return cls.loads(f.read(), **kw)


def cosine_matrix(query_vec: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """Cosines of one query against row-normalized matrix (rows already unit)."""
    return vectors @ query_vec


def pack_vec(vec: np.ndarray) -> bytes:
    return vec.astype(np.float16).tobytes()


def unpack_vec(blob: bytes, dim: int) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float16, count=dim).astype(np.float32)
