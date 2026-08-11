"""Latin-script morphology — the counterpart to ``hangul.py``.

Korean got a guarded 조사/어미 stripper early because the evidence for it was
unambiguous. The Latin side never got one, and character trigrams over every
word ≥4 chars stood in for it: "browsing" and "browse" matched because they
share *brow, row, ows*, not because anything understood them. That worked, at
a price — measured on a production vault those trigrams were **72.8% of all
postings**, and the morphology they approximated still leaked: a query
re-inflected off its source note lost 22% of its MRR (0.551 → 0.429).

So: a real stemmer. Porter's algorithm, because it is the one every IR system
has agreed on for forty years, is deterministic, needs no dictionary, and its
aggressiveness is the right trade here — this stream is ADDITIVE. The surface
form is indexed too, so an over-eager conflation ("operate"/"operator" both →
"oper") adds recall without taking exact matching away. That is the same
contract the Korean stripper documents: a wrong strip only adds one noisy
term.

Reference: M.F. Porter, "An algorithm for suffix stripping" (1980).
"""

from __future__ import annotations

_VOWELS = frozenset("aeiou")

#: Below this length a stem is not worth having: three-letter words are
#: already their own stems, and stripping them produces collisions
#: ("ads"→"ad", "his"→"hi") that only add noise.
MIN_WORD_LEN = 4


def _is_consonant(word: str, i: int) -> bool:
    ch = word[i]
    if ch in _VOWELS:
        return False
    if ch == "y":
        # y is a consonant at the start and after a vowel ("toy"), a vowel
        # after a consonant ("happy").
        return i == 0 or not _is_consonant(word, i - 1)
    return True


def _measure(stem: str) -> int:
    """Porter's *m*: how many vowel-consonant sequences the stem contains.

    It is the algorithm's stand-in for "is there enough word left" — every
    rule below is conditioned on it, which is what stops ``-ate`` from
    eating half of ``plate``.
    """
    m = 0
    i = 0
    n = len(stem)
    # skip leading consonants
    while i < n and _is_consonant(stem, i):
        i += 1
    while i < n:
        while i < n and not _is_consonant(stem, i):
            i += 1
        if i >= n:
            break
        m += 1
        while i < n and _is_consonant(stem, i):
            i += 1
    return m


def _has_vowel(stem: str) -> bool:
    return any(not _is_consonant(stem, i) for i in range(len(stem)))


def _ends_double_consonant(stem: str) -> bool:
    return len(stem) >= 2 and stem[-1] == stem[-2] and _is_consonant(stem, len(stem) - 1)


def _ends_cvc(stem: str) -> bool:
    """consonant-vowel-consonant where the last is not w, x or y —
    the shape that wants an ``e`` back ("hop" → "hope")."""
    n = len(stem)
    if n < 3:
        return False
    return (
        _is_consonant(stem, n - 3)
        and not _is_consonant(stem, n - 2)
        and _is_consonant(stem, n - 1)
        and stem[-1] not in "wxy"
    )


_STEP2 = (
    ("ational", "ate"),
    ("tional", "tion"),
    ("enci", "ence"),
    ("anci", "ance"),
    ("izer", "ize"),
    ("abli", "able"),
    ("alli", "al"),
    ("entli", "ent"),
    ("eli", "e"),
    ("ousli", "ous"),
    ("ization", "ize"),
    ("ation", "ate"),
    ("ator", "ate"),
    ("alism", "al"),
    ("iveness", "ive"),
    ("fulness", "ful"),
    ("ousness", "ous"),
    ("aliti", "al"),
    ("iviti", "ive"),
    ("biliti", "ble"),
)

_STEP3 = (
    ("icate", "ic"),
    ("ative", ""),
    ("alize", "al"),
    ("iciti", "ic"),
    ("ical", "ic"),
    ("ful", ""),
    ("ness", ""),
)

_STEP4 = (
    "al",
    "ance",
    "ence",
    "er",
    "ic",
    "able",
    "ible",
    "ant",
    "ement",
    "ment",
    "ent",
    "ou",
    "ism",
    "ate",
    "iti",
    "ous",
    "ive",
    "ize",
)


def stem(word: str) -> str:
    """Porter stem of *word*. Returns it unchanged when nothing applies.

    Callers index BOTH the surface form and this, so a conflation that is
    too eager costs one extra posting, never a missed exact match.
    """
    w = word.lower()
    if len(w) < MIN_WORD_LEN or not w.isalpha():
        return w

    # ── 1a: plurals ──────────────────────────────────────────────────
    if w.endswith("sses"):
        w = w[:-2]
    elif w.endswith("ies"):
        w = w[:-2]
    elif w.endswith("ss"):
        pass
    elif w.endswith("s"):
        w = w[:-1]

    # ── 1b: -eed / -ed / -ing ────────────────────────────────────────
    step1b_hit = False
    if w.endswith("eed"):
        if _measure(w[:-3]) > 0:
            w = w[:-1]
    elif w.endswith("ed") and _has_vowel(w[:-2]):
        w = w[:-2]
        step1b_hit = True
    elif w.endswith("ing") and _has_vowel(w[:-3]):
        w = w[:-3]
        step1b_hit = True
    if step1b_hit:
        if w.endswith(("at", "bl", "iz")):
            w += "e"
        elif _ends_double_consonant(w) and not w.endswith(("l", "s", "z")):
            w = w[:-1]
        elif _measure(w) == 1 and _ends_cvc(w):
            w += "e"

    # ── 1c: terminal y → i ───────────────────────────────────────────
    if w.endswith("y") and _has_vowel(w[:-1]):
        w = w[:-1] + "i"

    # ── 2 / 3: derivational suffixes, longest match first ────────────
    for suffix, repl in sorted(_STEP2, key=lambda p: -len(p[0])):
        if w.endswith(suffix) and _measure(w[: len(w) - len(suffix)]) > 0:
            w = w[: len(w) - len(suffix)] + repl
            break
    for suffix, repl in sorted(_STEP3, key=lambda p: -len(p[0])):
        if w.endswith(suffix) and _measure(w[: len(w) - len(suffix)]) > 0:
            w = w[: len(w) - len(suffix)] + repl
            break

    # ── 4: strip the remainder when there is enough word left ────────
    for suffix in sorted(_STEP4, key=len, reverse=True):
        if w.endswith(suffix):
            base = w[: len(w) - len(suffix)]
            if _measure(base) > 1:
                if suffix != "ion" or base.endswith(("s", "t")):
                    w = base
            break
    if w.endswith("ion") and _measure(w[:-3]) > 1 and w[-4:-3] in ("s", "t"):
        w = w[:-3]

    # ── 5: tidy up ───────────────────────────────────────────────────
    if w.endswith("e"):
        m = _measure(w[:-1])
        if m > 1 or (m == 1 and not _ends_cvc(w[:-1])):
            w = w[:-1]
    if _measure(w) > 1 and _ends_double_consonant(w) and w.endswith("l"):
        w = w[:-1]

    return w
