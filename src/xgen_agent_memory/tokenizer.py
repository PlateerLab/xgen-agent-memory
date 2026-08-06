"""Tokenizer — two token streams tuned by NTCIR/ACL evidence for Korean.

LEXICAL stream (BM25 postings) — precision-oriented:
  * surface words (NFKC + casefold)
  * guarded 조사-stripped stems (받침 agreement, ≥2-syllable stems; additive —
    surface AND stem are indexed, a wrong strip only adds one noisy term)
  * overlapping SYLLABLE BIGRAMS within each Hangul word — the no-analyzer
    gold standard (NTCIR-3/4/5: bigrams match or beat morphological analysis;
    trigrams cost 2-3× for no measured gain, so none are generated)
  * CROSS-SPACE bigrams over each run of consecutive Hangul words with the
    spaces removed — 붙여쓰기 robustness (the production char_filter trick)

EMBEDDING stream (hash embedder) — recall-oriented, adds:
  * jamo 3/5-grams over decomposed 초성/중성/종성 with an explicit empty-종성
    pad (ACL 2018 recipe: jamo 3-5 helps, jamo bigrams HURT — none generated).
    Jamo stays OUT of the BM25 index: it inflates postings ~3× and adds
    vowel-noise to exact matching, but is exactly right for the fuzzy
    semantic space.

Both streams share normalization so index and query views never diverge.
"""

from __future__ import annotations

import re
from typing import Iterable, List

from .hangul import has_hangul, normalize, strip_suffix, to_jamo

_WORD = re.compile(r"[\w']+", re.UNICODE)

#: Tokens that carry no signal (tiny stoplist — BM25's IDF handles the rest).
_STOP = {"the", "a", "an", "of", "to", "and", "is", "in", "it", "i", "you"}

#: Marker prefix for jamo grams — never collides with syllable grams.
_JAMO_MARK = "ⱼ"
#: Marker for cross-space bigrams — kept distinct from in-word bigrams so
#: their IDF is computed on their own distribution.
_XSPACE_MARK = "ₓ"


def _ngrams(word: str, sizes: Iterable[int]) -> List[str]:
    out: List[str] = []
    for n in sizes:
        if len(word) <= n:
            continue
        out.extend(word[i : i + n] for i in range(len(word) - n + 1))
    return out


#: A single "word" with no internal spaces (a base64 blob, a long URL, a run
#: of CJK) is truncated to this before n-gramming. Without it the per-word
#: n-gram list (and the 3× jamo expansion) grows O(len), so one megabyte-long
#: token is an OOM/hang DoS even though the OUTPUT token cap looks bounded.
#: 128 chars keeps every real word intact while defanging pathological ones —
#: matching on the first 128 chars of a blob is plenty for retrieval.
_MAX_WORD_LEN = 128
#: Only the first this-many characters of a document/query are scanned. A note
#: with real signal in its first ~half-MB is already far beyond any prompt
#: budget; scanning a multi-megabyte blob in full is pure DoS surface.
_MAX_TEXT_SCAN = 500_000


def _words(text: str) -> List[str]:
    if len(text) > _MAX_TEXT_SCAN:
        text = text[:_MAX_TEXT_SCAN]
    out = []
    for match in _WORD.finditer(normalize(text)):
        w = match.group()
        if w in _STOP or (w.isdigit() and len(w) > 6):
            continue
        out.append(w if len(w) <= _MAX_WORD_LEN else w[:_MAX_WORD_LEN])
    return out


def lexical_tokens(
    text: str,
    *,
    char_ngrams: Iterable[int] = (2,),
    suffix_strip: bool = True,
    cross_space: bool = True,
    limit: int = 2048,
) -> List[str]:
    """BM25 stream: words + guarded stems + syllable bigrams + cross-space
    bigrams. See module docstring for the evidence behind each choice."""
    tokens: List[str] = []
    run: List[str] = []  # consecutive Hangul words → cross-space bigrams

    def flush_run() -> None:
        if cross_space and len(run) > 1:
            joined = "".join(run)
            # Only the bigrams that CROSS word boundaries are new information.
            boundaries = set()
            pos = 0
            for w in run[:-1]:
                pos += len(w)
                boundaries.add(pos - 1)  # bigram starting here spans the gap
            tokens.extend(
                _XSPACE_MARK + joined[i : i + 2]
                for i in range(len(joined) - 1) if i in boundaries
            )
        run.clear()

    for word in _words(text):
        korean = has_hangul(word)
        tokens.append(word)
        if korean:
            if suffix_strip:
                stem = strip_suffix(word)
                if stem != word:
                    tokens.append(stem)
            tokens.extend(_ngrams(word, char_ngrams))
            run.append(word)
        else:
            flush_run()
            if len(word) > 3:
                tokens.extend(_ngrams(word, (3,)))
        if len(tokens) >= limit:
            break
    flush_run()
    return tokens[:limit]


def embed_tokens(
    text: str,
    *,
    char_ngrams: Iterable[int] = (2,),
    jamo_ngrams: Iterable[int] = (3, 5),
    suffix_strip: bool = True,
    limit: int = 2048,
) -> List[str]:
    """Embedding stream: the lexical stream + padded jamo n-grams."""
    tokens = lexical_tokens(text, char_ngrams=char_ngrams,
                            suffix_strip=suffix_strip, cross_space=False,
                            limit=limit)
    jamo_sizes = tuple(jamo_ngrams)
    if jamo_sizes:
        for word in _words(text):
            if not has_hangul(word):
                continue
            jam = to_jamo(word, pad_tail=True)
            tokens.extend(_JAMO_MARK + g for g in _ngrams(jam, jamo_sizes))
            if len(tokens) >= limit:
                break
    return tokens[:limit]


def tokenize(text: str, *, char_ngrams: Iterable[int] = (2,),
             jamo_ngrams: Iterable[int] = (3, 5), suffix_strip: bool = True,
             limit: int = 2048) -> List[str]:
    """Back-compat alias for the embedding stream."""
    return embed_tokens(text, char_ngrams=char_ngrams, jamo_ngrams=jamo_ngrams,
                        suffix_strip=suffix_strip, limit=limit)


def fnv1a(token: str, buckets: int) -> int:
    """FNV-1a 64-bit hash → bucket index (stable across processes/platforms)."""
    h = 0xCBF29CE484222325
    for byte in token.encode("utf-8"):
        h ^= byte
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h % buckets


def fnv1a_pair(token: str, buckets: int) -> tuple:
    """TWO independent bucket indexes (Bloom-style k=2 hashing).

    Hash-embedding evidence (NeurIPS 2017): k≥2 hash functions make total
    collisions ~(1/B)^k — a single hash over a large n-gram vocabulary
    collides badly."""
    h = 0xCBF29CE484222325
    for byte in token.encode("utf-8"):
        h ^= byte
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    # Split the 64-bit state into two independent 32-bit views.
    return h % buckets, ((h >> 32) ^ (h & 0xFFFFFFFF)) % buckets
