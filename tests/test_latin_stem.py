"""Porter stemming — correctness first, then the property it buys.

The Latin side had no stemmer; character trigrams stood in for one. This
module replaces that guess with the algorithm every IR system agreed on, so
the first duty of these tests is that it IS Porter — checked against outputs
anyone can verify against the published algorithm — and the second is the
contract it shares with the Korean stripper: additive, so an over-eager
conflation costs a posting, never an exact match.
"""

from __future__ import annotations

import pytest

from xgen_agent_memory.latin import MIN_WORD_LEN, stem


# ── it is actually Porter ───────────────────────────────────────────


@pytest.mark.parametrize(
    "word,expected",
    [
        # 1a — plurals
        ("caresses", "caress"),
        ("ponies", "poni"),
        ("cats", "cat"),
        ("caress", "caress"),
        # 1b — -eed / -ed / -ing with the m-guard
        ("agreed", "agre"),
        ("plastered", "plaster"),
        ("motoring", "motor"),
        ("sing", "sing"),
        # 1b cleanup — at/bl/iz get an e, doubles collapse, cvc gets an e
        ("conflated", "conflat"),
        ("troubling", "troubl"),
        ("hopping", "hop"),
        ("falling", "fall"),
        ("filing", "file"),
        # 1c — terminal y
        ("happy", "happi"),
        ("sky", "sky"),
        # 2/3/4 — derivational
        ("relational", "relat"),
        ("conditional", "condit"),
        ("hopefulness", "hope"),
        ("formalize", "formal"),
        ("adjustment", "adjust"),
        ("dependent", "depend"),
    ],
)
def test_known_porter_outputs(word, expected):
    assert stem(word) == expected


# ── the property that matters for retrieval ─────────────────────────


@pytest.mark.parametrize(
    "a,b",
    [
        ("browsing", "browse"),
        ("browsed", "browse"),
        ("running", "run"),
        ("stopped", "stop"),
        ("files", "file"),
        ("connections", "connection"),
        ("configured", "configure"),
        ("observations", "observation"),
        ("indexing", "index"),
    ],
)
def test_inflections_of_one_word_share_a_stem(a, b):
    """THE point. A query re-inflected off its source note lost 22% of its
    MRR on the production vault because these did not meet."""
    assert stem(a) == stem(b), f"{a} → {stem(a)} vs {b} → {stem(b)}"


@pytest.mark.parametrize(
    "a,b",
    [
        ("cat", "dog"),
        ("index", "engine"),
        ("memory", "vector"),
        ("browse", "brownie"),
        ("stop", "storage"),
    ],
)
def test_unrelated_words_do_not_collide(a, b):
    assert stem(a) != stem(b)


# ── guards ──────────────────────────────────────────────────────────


def test_short_words_are_left_alone():
    """Three letters are already a stem; stripping them collides ("ads"→"ad",
    "his"→"hi") and buys nothing."""
    for w in ("ads", "his", "the", "cat", "run"):
        assert stem(w) == w.lower()
    assert MIN_WORD_LEN == 4


def test_non_alphabetic_tokens_are_untouched():
    """Identifiers, versions, hashes — stripping a trailing 's' off `v2s`
    would break exact matching on the thing most worth matching exactly."""
    for w in ("docker-compose", "v2.65.1", "a1b2c3", "__init__", "utf8"):
        assert stem(w) == w.lower()


def test_it_is_case_insensitive():
    assert stem("Browsing") == stem("browsing") == stem("BROWSING")


def test_the_pipeline_never_stems_a_stem():
    """Porter is not idempotent — `stem("brows")` is "brow" while
    `stem("browsing")` is "brows" — and that is fine here ONLY because both
    the index and the query stem the original surface form, never a previous
    stem. This test documents the invariant the callers must keep.
    """
    import inspect

    from xgen_agent_memory import tokenizer

    src = inspect.getsource(tokenizer.lexical_tokens)
    assert "latin_stem(word)" in src, (
        "the stemmer must be applied to the surface word; feeding it a "
        "previous stem would drift the index away from the query"
    )


def test_a_known_porter_quirk_is_recorded():
    """`deployment` → `deploy` but `deploy` → `deploi`: step 1c turns a
    terminal y into i, and a word that reaches step 4 has already passed it.
    Canonical Porter behaves this way; the pair is caught by the vector
    stream instead. Recorded so a future change does not "fix" it by
    accident and shift every other stem with it."""
    assert stem("deployment") == "deploy"
    assert stem("deploy") == "deploi"


def test_empty_and_tiny_input_is_safe():
    assert stem("") == ""
    assert stem("a") == "a"
