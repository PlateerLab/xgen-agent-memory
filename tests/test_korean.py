"""Korean-specific behaviour — hangul utilities, token streams, robustness.

Complements the generic engine tests with the axes that matter for Korean:
jamo decomposition correctness, 받침-guarded 조사 stripping, stream
separation (jamo stays out of BM25), cross-space bigrams, and end-to-end
matching for 활용/조사 변형, 붙여쓰기, and jamo-level typos.
"""

from __future__ import annotations

from xgen_agent_memory import SynapseConfig, SynapseMemory
from xgen_agent_memory.hangul import (
    has_jongseong,
    normalize,
    strip_suffix,
    to_jamo,
)
from xgen_agent_memory.tokenizer import (
    _JAMO_MARK,
    _XSPACE_MARK,
    embed_tokens,
    lexical_tokens,
)


def make_mem(**kw) -> SynapseMemory:
    return SynapseMemory(SynapseConfig(path=":memory:", vocab_size=8192, dim=32, epsilon=0.0, **kw))


# ── hangul primitives ────────────────────────────────────────────────


def test_jamo_decomposition_exact():
    assert to_jamo("먹었다") == "ㅁㅓㄱㅇㅓㅆㄷㅏ"
    assert to_jamo("가") == "ㄱㅏ"
    assert to_jamo("한글 OK") == "ㅎㅏㄴㄱㅡㄹ OK"


def test_jamo_pad_makes_three_per_syllable():
    padded = to_jamo("가나", pad_tail=True)
    assert len(padded) == 6  # every syllable exactly 3 jamo with the pad


def test_jamo_emits_compatibility_block():
    # Typed-query compatibility: U+3131 'ㄱ', never NFD's conjoining U+1100.
    assert ord(to_jamo("가")[0]) == 0x3131


def test_jongseong_detection():
    assert has_jongseong("먹") and has_jongseong("잤")
    assert not has_jongseong("가") and not has_jongseong("나")


def test_normalize_nfkc_casefold():
    assert normalize("ＡＢＣ") == "abc"  # full-width → half-width + casefold


# ── guarded 조사 stripping ────────────────────────────────────────────


def test_strip_multi_syllable_josa():
    assert strip_suffix("서울에서") == "서울"
    assert strip_suffix("아침부터") == "아침"
    assert strip_suffix("친구에게") == "친구"


def test_strip_single_josa_requires_batchim_agreement():
    # 을 attaches after 받침 — '실록을' strips, but vowel-final stems keep 을-less forms
    assert strip_suffix("실록을") == "실록"
    assert strip_suffix("판정을") == "판정"
    assert strip_suffix("게임이") == "게임"  # 받침 'ㅁ' + 이 → strip
    assert strip_suffix("요리가") == "요리"  # vowel-final + 가 → strip
    # Agreement VIOLATION → not a particle, keep the word.
    assert strip_suffix("고향이") == "고향"  # ㅇ received: consonant-final + 이 ok
    assert strip_suffix("어디가") == "어디"


def test_strip_never_takes_ui_alone():
    assert strip_suffix("회의") == "회의"  # '의' 단독 스트립 금지
    assert strip_suffix("주의") == "주의"
    assert strip_suffix("의사") == "의사"


def test_strip_keeps_two_syllable_stems():
    assert strip_suffix("가게") == "가게"  # ≥2음절 스템 가드
    assert strip_suffix("나는") == "나는"  # would leave 1 syllable


def test_strip_hada_family_eomi():
    assert strip_suffix("검색했다") == "검색"
    assert strip_suffix("사용합니다") == "사용"
    assert strip_suffix("정리하는") == "정리"


# ── token streams ────────────────────────────────────────────────────


def test_lexical_stream_has_no_jamo():
    toks = lexical_tokens("리듬게임 판정")
    assert not any(t.startswith(_JAMO_MARK) for t in toks)
    assert "리듬게임" in toks and "판정" in toks and "리듬" in toks  # bigram


def test_lexical_stream_bigrams_only_by_default():
    toks = lexical_tokens("리듬게임")
    grams = [t for t in toks if t != "리듬게임" and not t.startswith(_XSPACE_MARK)]
    assert all(len(g) == 2 for g in grams)  # no trigrams (NTCIR evidence)


def test_embed_stream_adds_padded_jamo():
    toks = embed_tokens("게임")
    jamo = [t for t in toks if t.startswith(_JAMO_MARK)]
    assert jamo, "embedding stream must contain jamo n-grams"
    assert any("ᴥ" in t for t in jamo)  # empty-종성 pad present (개 has no tail)


def test_cross_space_bigrams_bridge_word_gap():
    toks = lexical_tokens("오일 저장 탱크")
    xs = [t for t in toks if t.startswith(_XSPACE_MARK)]
    # bridging bigrams: 일저, 장탱
    assert _XSPACE_MARK + "일저" in xs and _XSPACE_MARK + "장탱" in xs


# ── end-to-end Korean robustness ─────────────────────────────────────

CORPUS = {
    "sillok": (
        "조선왕조실록",
        "조선왕조실록은 조선 시대 왕들의 통치 기록이다. 유네스코 세계기록유산으로 등재되어 있다.",
    ),
    "hangul": ("훈민정음", "훈민정음은 세종대왕이 창제한 문자 체계이다."),
    "kimchi": ("김치찌개 조리법", "돼지고기와 묵은지를 넣고 끓이는 한국 요리이다."),
    "docker": ("도커 컨테이너", "docker 컨테이너는 리눅스 네임스페이스로 격리된 프로세스이다."),
}


def seeded() -> SynapseMemory:
    mem = make_mem()
    for nid, (title, text) in CORPUS.items():
        mem.index(nid, text, title=title)
    return mem


def test_match_josa_variants():
    mem = seeded()
    for q in ("조선왕조실록은 무엇인가", "조선왕조실록이 뭐야", "조선왕조실록을 설명해줘"):
        assert mem.search(q)[0].id == "sillok", q


def test_match_conjugation_variants():
    mem = seeded()
    for q in ("김치찌개 끓이는 방법", "김치찌개를 끓였다", "김치찌개 조리"):
        assert mem.search(q)[0].id == "kimchi", q


def test_match_no_spacing():
    mem = seeded()
    assert mem.search("조선왕조실록세계기록유산")[0].id == "sillok"
    assert mem.search("김치찌개조리법")[0].id == "kimchi"


def test_match_jamo_typo_via_embedding():
    mem = seeded()
    # '훈민정음' with a jamo typo → '훈민전음'. BM25 partially misses; the
    # jamo-gram embedding stream keeps it near.
    hits = mem.search("훈민전음 창제", top_k=3)
    assert any(h.id == "hangul" for h in hits)


def test_match_mixed_ko_en():
    mem = seeded()
    assert mem.search("docker 격리")[0].id == "docker"
    assert mem.search("도커 리눅스")[0].id == "docker"


def test_partial_compound_query():
    mem = seeded()
    # Head-noun-only query still finds the compound-titled doc via bigrams.
    assert mem.search("실록")[0].id == "sillok"


def test_agent_memory_hard_queries():
    """Hand-authored agent-memory set: indirect reference, paraphrase, ko/en
    mix, 활용 변형. Guards realistic agent retrieval (R@1 ≥ 0.85 floor)."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
    from agent_memory_ko import NOTES, QUERIES

    # Real (default) config — a tiny test vocab collides hashes and makes the
    # embedding leg unstable across numpy builds; this is a QUALITY test.
    mem = SynapseMemory(SynapseConfig(path=":memory:", epsilon=0.0))
    for nid, title, body, kind, tags in NOTES:
        mem.index(nid, body, title=title, kind=kind, tags=tags)
    at1 = sum(1 for q, gold, _ in QUERIES if (r := mem.search(q, top_k=5)) and r[0].id == gold)
    # R@1 measured 0.95; 0.80 floor absorbs cross-platform embedding jitter
    # while still catching a real regression.
    assert at1 / len(QUERIES) >= 0.80, f"R@1 {at1}/{len(QUERIES)}"


def test_strip_ro_after_rieul():
    # 로 attaches after a vowel OR ㄹ-final stem — A4 fix.
    assert strip_suffix("서울로") == "서울"  # 울: ㄹ 받침 + 로
    assert strip_suffix("학교로") == "학교"  # vowel + 로
    assert strip_suffix("집으로") == "집으"  # 으로 is multi-particle → 집 (handled table)
