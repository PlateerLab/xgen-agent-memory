"""The two streams are deliberately different — effect-proving tests.

`tokenizer.py` has always described a split: the LEXICAL stream feeds BM25
and is precision-oriented; the EMBEDDING stream feeds the hash embedder and
is recall-oriented. Jamo n-grams were already on the recall side only,
"because they inflate postings ~3× and add vowel-noise to exact matching".

Latin character trigrams sat on the wrong side of that line. They were
generated for every non-Hangul word ≥4 chars and were 72.8% of a production
vault's postings — 59.6% of that from fifty boilerplate types whose IDF is
about zero. They are now where jamo is: embedding only.

What replaces them in BM25 is a real stemmer, and the measurements say that
is a better index, not just a smaller one: known-item MRR +4% verbatim and
+2% on re-inflected queries, with typo tolerance handed to the vector side.
"""

from __future__ import annotations

from xgen_agent_memory import SynapseConfig, SynapseMemory
from xgen_agent_memory.tokenizer import embed_tokens, lexical_tokens

SENTENCE = "The user is browsing an item trading interface"


def _latin_trigrams(tokens):
    """Character trigrams from Latin words — no marker prefix, length 3,
    and not a whole word in the sentence."""
    words = {w.lower() for w in SENTENCE.split()}
    return {t for t in tokens if len(t) == 3 and t.isalpha() and t.isascii() and t not in words}


# ── the split ───────────────────────────────────────────────────────


def test_bm25_stream_has_no_latin_trigrams_by_default():
    """THE change. This is where 72.8% of the postings came from."""
    assert _latin_trigrams(lexical_tokens(SENTENCE)) == set()


def test_the_embedding_stream_keeps_them():
    """Typo tolerance moves here rather than disappearing — the same place
    jamo already lives, and it costs no postings."""
    assert _latin_trigrams(embed_tokens(SENTENCE)), "the recall stream lost its fuzzy matching too"


def test_the_knob_can_bring_them_back():
    assert _latin_trigrams(lexical_tokens(SENTENCE, latin_ngram_min_len=4))


# ── what BM25 gets instead ──────────────────────────────────────────


def test_the_bm25_stream_carries_stems():
    tokens = lexical_tokens(SENTENCE)
    assert "brows" in tokens, "no stem — morphology would be unmatched"
    assert "browsing" in tokens, "the surface form must survive too"


def test_an_inflected_query_finds_its_note(tmp_path):
    """The property the stemmer is for. Without it the same query lost 16%
    of its MRR on the production vault."""
    mem = SynapseMemory(SynapseConfig(path=str(tmp_path / "a.db"), epsilon=0.0))
    mem.index("d1", "The user is browsing an item trading interface")
    mem.index("d2", "저녁 메뉴로 김치찌개를 끓였다")

    assert [h.id for h in mem.search("browse trade", top_k=2)][0] == "d1"


def test_korean_is_untouched(tmp_path):
    """Every measurement kept Korean flat (MRR 0.750–0.766); this pins the
    tokens rather than trusting that."""
    tokens = lexical_tokens("리듬게임 판정을 읽는다")
    assert "판정을" in tokens  # surface
    assert "판정" in tokens  # 조사 stripped
    assert any(len(t) == 2 and "가" <= t[0] <= "힣" for t in tokens)


# ── the digest must notice a tokenizer change ───────────────────────


def test_changing_the_tokenizer_invalidates_the_index(tmp_path):
    """Without this the vault silently becomes a mix of two tokenizations:
    the digest says "unchanged", nothing re-indexes, and the stored postings
    no longer agree with how queries are analysed."""
    path = str(tmp_path / "b.db")
    a = SynapseMemory(SynapseConfig(path=path, epsilon=0.0))
    a.index("d1", "The user is browsing an item trading interface")
    before = a.manifest()["d1"][1]
    a.close()

    b = SynapseMemory(SynapseConfig(path=path, epsilon=0.0, latin_ngram_min_len=4))
    out = b.index_many(
        [{"node_id": "d1", "text": "The user is browsing an item trading interface"}]
    )

    assert out["indexed"] == 1, "a tokenizer change did not trigger a re-index"
    assert b.manifest()["d1"][1] != before
    b.close()


def test_an_unrelated_config_change_does_not_reindex(tmp_path):
    """The digest must not be so broad that every restart rebuilds the vault."""
    path = str(tmp_path / "c.db")
    a = SynapseMemory(SynapseConfig(path=path, epsilon=0.0))
    a.index("d1", "The user is browsing an item")
    a.close()

    b = SynapseMemory(SynapseConfig(path=path, epsilon=0.0, top_k=99))
    out = b.index_many([{"node_id": "d1", "text": "The user is browsing an item"}])

    assert out["indexed"] == 0
    assert out["skipped"] == 1
    b.close()


def test_a_geometry_change_marks_every_row_stale(tmp_path):
    """Putting the geometry into `content_sha` is not enough on its own: the
    digest is only consulted for notes a HOST decides to offer, and a host
    that diffs on timestamps never offers an untouched note. Production
    upgraded to a new tokenizer and re-indexed nothing — "0 indexed" — while
    every signal said the vault was in sync.

    An empty digest is the contract that fixes it: "indexed, derived state
    unknown"."""
    path = str(tmp_path / "g.db")
    a = SynapseMemory(SynapseConfig(path=path, epsilon=0.0))
    a.index("d1", "The user is browsing an item")
    a.index("d2", "리듬게임 판정을 읽는다")
    assert all(sha for _ts, sha in a.manifest().values())
    a.close()

    b = SynapseMemory(SynapseConfig(path=path, epsilon=0.0, latin_ngram_min_len=4))
    assert all(sha == "" for _ts, sha in b.manifest().values()), (
        "a tokenizer change left the rows claiming to be up to date"
    )
    b.close()


def test_reopening_with_the_same_geometry_keeps_the_digests(tmp_path):
    """The flip side: an ordinary restart must not invalidate the vault."""
    path = str(tmp_path / "h.db")
    a = SynapseMemory(SynapseConfig(path=path, epsilon=0.0))
    a.index("d1", "The user is browsing an item")
    before = a.manifest()["d1"][1]
    a.close()

    b = SynapseMemory(SynapseConfig(path=path, epsilon=0.0))
    assert b.manifest()["d1"][1] == before
    b.close()


def test_re_indexing_after_a_geometry_change_restores_the_digest(tmp_path):
    path = str(tmp_path / "i.db")
    a = SynapseMemory(SynapseConfig(path=path, epsilon=0.0))
    a.index("d1", "The user is browsing an item")
    a.close()

    b = SynapseMemory(SynapseConfig(path=path, epsilon=0.0, latin_ngram_min_len=4))
    out = b.index_many([{"node_id": "d1", "text": "The user is browsing an item"}])
    assert out["indexed"] == 1
    assert b.manifest()["d1"][1] != ""
    b.close()


def test_a_vault_with_no_recorded_geometry_is_treated_as_stale(tmp_path):
    """ "Unknown" is not "fine". A vault written before the geometry was
    tracked was derived by SOME tokenization nobody can name; assuming it
    matches is how the production upgrade re-indexed nothing while every
    signal said it was in sync. Derived data is rebuildable; a wrong
    assumption of freshness is not."""
    path = str(tmp_path / "j.db")
    a = SynapseMemory(SynapseConfig(path=path, epsilon=0.0))
    a.index("d1", "The user is browsing an item")
    a.store._write(lambda c: c.execute("DELETE FROM params WHERE key='geometry'"))
    assert a.manifest()["d1"][1] != ""
    a.close()

    b = SynapseMemory(SynapseConfig(path=path, epsilon=0.0))
    assert b.manifest()["d1"][1] == "", "a vault of unknown provenance was assumed up to date"
    b.close()


def test_a_fresh_empty_vault_costs_nothing(tmp_path):
    """The flip side: recording the geometry for the first time on an empty
    file must not look like an invalidation."""
    mem = SynapseMemory(SynapseConfig(path=str(tmp_path / "k.db"), epsilon=0.0))
    mem.index("d1", "The user is browsing an item")
    assert mem.manifest()["d1"][1] != ""
    mem.close()


def test_the_geometry_version_is_part_of_the_fingerprint(tmp_path):
    """The config fields cannot express "the stemmer changed". Without an
    explicit version, 1.9.0 recorded the new geometry against rows derived
    by the old tokenizer and every later release saw a match — the vault was
    unfixable from inside."""
    from xgen_agent_memory import engine as eng

    mem = SynapseMemory(SynapseConfig(path=str(tmp_path / "v.db"), epsilon=0.0))
    assert mem._geometry().startswith(f"v{eng._GEOMETRY_VERSION}:")
    mem.close()


def test_bumping_the_version_invalidates_a_matching_config(tmp_path, monkeypatch):
    from xgen_agent_memory import engine as eng

    path = str(tmp_path / "w.db")
    a = SynapseMemory(SynapseConfig(path=path, epsilon=0.0))
    a.index("d1", "The user is browsing an item")
    a.close()

    monkeypatch.setattr(eng, "_GEOMETRY_VERSION", eng._GEOMETRY_VERSION + 1)
    b = SynapseMemory(SynapseConfig(path=path, epsilon=0.0))
    assert b.manifest()["d1"][1] == ""
    b.close()
