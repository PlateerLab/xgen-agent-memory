"""Hangul utilities — jamo decomposition, normalization, light suffix stripping.

Pure unicode arithmetic, no dependencies. A precomposed syllable U+AC00..U+D7A3
decomposes as::

    idx = code - 0xAC00
    초성 = idx // (21*28)      # 19 leads
    중성 = (idx % (21*28)) // 28   # 21 vowels
    종성 = idx % 28            # 27 tails + none

Jamo-level text makes matching robust to typos and 활용 (conjugation): 먹었다 /
먹는다 share the jamo prefix ㅁㅓㄱ even though they share no syllable, and a
single-jamo typo (게임/개임) still overlaps heavily at the jamo n-gram level.
"""

from __future__ import annotations

import unicodedata
from typing import List

_BASE = 0xAC00
_LAST = 0xD7A3

_LEADS = [
    "ㄱ", "ㄲ", "ㄴ", "ㄷ", "ㄸ", "ㄹ", "ㅁ", "ㅂ", "ㅃ", "ㅅ",
    "ㅆ", "ㅇ", "ㅈ", "ㅉ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ",
]
_VOWELS = [
    "ㅏ", "ㅐ", "ㅑ", "ㅒ", "ㅓ", "ㅔ", "ㅕ", "ㅖ", "ㅗ", "ㅘ",
    "ㅙ", "ㅚ", "ㅛ", "ㅜ", "ㅝ", "ㅞ", "ㅟ", "ㅠ", "ㅡ", "ㅢ", "ㅣ",
]
_TAILS = [
    "", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ",
    "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ", "ㅅ",
    "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ",
]


def is_hangul_syllable(ch: str) -> bool:
    return _BASE <= ord(ch) <= _LAST


def has_hangul(text: str) -> bool:
    return any(_BASE <= ord(c) <= _LAST for c in text)


#: Explicit empty-종성 pad — with it every syllable is exactly 3 jamo, which
#: measurably improves jamo n-gram quality (ACL 2018 Korean subword recipe)
#: and keeps typo edits positionally aligned.
_PAD = "ᴥ"


def to_jamo(text: str, *, pad_tail: bool = False) -> str:
    """Decompose every precomposed syllable into its jamo sequence.

    Non-Hangul characters pass through unchanged. 먹었다 → ㅁㅓㄱㅇㅓㅆㄷㅏ
    (+pad_tail: ㅁㅓㄱㅇㅓㅆㄷㅏᴥ). Emits COMPATIBILITY jamo (U+3131 block) —
    what users actually type — never NFD's conjoining jamo (U+1100 block),
    which would silently fail to match typed queries."""
    out: List[str] = []
    for ch in text:
        code = ord(ch)
        if _BASE <= code <= _LAST:
            idx = code - _BASE
            out.append(_LEADS[idx // 588])
            out.append(_VOWELS[(idx % 588) // 28])
            tail = _TAILS[idx % 28]
            if tail:
                out.append(tail)
            elif pad_tail:
                out.append(_PAD)
        else:
            out.append(ch)
    return "".join(out)


def has_jongseong(ch: str) -> bool:
    """True when the syllable ends in a 받침 (tail consonant)."""
    return is_hangul_syllable(ch) and (ord(ch) - _BASE) % 28 != 0


def normalize(text: str) -> str:
    """NFKC (full-width→half-width, compatibility jamo unification) + casefold."""
    return unicodedata.normalize("NFKC", text).casefold()


# ── guarded suffix stripping (조사/어미) ──────────────────────────────
#
# Longest-match against a CLOSED list, ONE strip per word, with three guards
# (per the Korean word-unit-indexing literature / KR100401466B1 blueprint):
#   1. the stem keeps ≥2 syllables ('가게' never loses '게')
#   2. multi-syllable particles strip freely; SINGLE-syllable particles are
#      gated by 받침 agreement — 이/은/을/과 attach only after a tail
#      consonant, 가/는/를/와/로 only after a vowel; a mismatch means the
#      syllable is part of the word, not a particle
#   3. '의' is never stripped alone (의사, 회의, 주의 …) — only inside
#      multi-syllable forms like 에서의
# 어미 stripping is restricted to the 하다/되다 families + formal endings —
# broad ending removal needs a verb dictionary and explodes error rates.
# Additive by design: surface AND stem are both indexed, so a wrong strip can
# only add one slightly-noisy term, never lose the original.

_JOSA_MULTI = [
    "에서의", "으로의", "까지의", "에게서", "으로써", "으로서", "이라고", "라고는",
    "에서는", "에서도", "에서만", "으로는", "으로도", "으로만", "이라는", "이라도",
    "한테서", "부터는", "까지는", "까지도", "마저도", "밖에는",
    "에서", "에게", "한테", "께서", "으로", "라고", "이라", "라는", "부터", "까지",
    "마저", "조차", "처럼", "보다", "대로", "만큼", "이나", "이든", "이며", "이고",
    "밖에",
]
#: Single-syllable particles requiring a preceding 받침 (consonant-final stem).
_JOSA_AFTER_CONSONANT = {"이", "은", "을", "과"}
#: Single-syllable particles requiring NO preceding 받침 (vowel-final stem).
_JOSA_AFTER_VOWEL = {"가", "는", "를", "와", "로"}
#: Single-syllable particles valid after either — kept minimal; '의' excluded.
_JOSA_ANY = {"에", "도", "만"}

_EOMI = [
    "하겠습니다", "했습니다", "합니다만", "했는데요",
    "했습니까", "됩니다", "습니다", "합니다", "입니다",
    "했는데", "하면서", "하려고", "했지만", "하다가", "했어요", "하세요",
    "했다", "한다", "하는", "하게", "하고", "하지", "하면", "해서", "하여", "하기",
    "된다", "됐다", "되는", "되어", "돼서", "이다", "였다",
]
_EOMI_SORTED = sorted(_EOMI, key=len, reverse=True)
_JOSA_MULTI_SORTED = sorted(_JOSA_MULTI, key=len, reverse=True)


def strip_suffix(word: str) -> str:
    """Strip ONE trailing 조사/어미 under the guards above.

    Returns the word unchanged when no safe strip exists."""
    if len(word) < 3 or not is_hangul_syllable(word[-1]):
        return word
    # ① multi-syllable particles / restricted 어미 — longest match first.
    for table in (_JOSA_MULTI_SORTED, _EOMI_SORTED):
        for suf in table:
            if len(word) - len(suf) >= 2 and word.endswith(suf):
                return word[: -len(suf)]
    # ② single-syllable particles — 받침 agreement with the preceding syllable.
    last, prev = word[-1], word[-2]
    if not is_hangul_syllable(prev):
        return word
    tail = has_jongseong(prev)
    # 로 attaches after a vowel OR an ㄹ-final stem (서울로, 물로) — special case.
    rieul_ok = last == "로" and (ord(prev) - _BASE) % 28 == 8  # 8 = ㄹ jongseong
    if (last in _JOSA_ANY
            or (last in _JOSA_AFTER_CONSONANT and tail)
            or (last in _JOSA_AFTER_VOWEL and not tail)
            or rieul_ok):
        return word[:-1]
    return word
