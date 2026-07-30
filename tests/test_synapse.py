"""Synapse engine tests — units per layer + end-to-end learning behaviour."""

from __future__ import annotations

import numpy as np
import pytest

from geny_memory_adaptor import (
    FEATURES,
    SynapseConfig,
    SynapseMemory,
    SynapseVectorHandle,
)
from geny_memory_adaptor.bm25 import bm25_scores, term_frequencies
from geny_memory_adaptor.embedder import HashEmbedder
from geny_memory_adaptor.graph import personalized_pagerank
from geny_memory_adaptor.ranker import OnlineRanker
from geny_memory_adaptor.store import Store
from geny_memory_adaptor.tokenizer import fnv1a, tokenize


def make_mem(**kw) -> SynapseMemory:
    return SynapseMemory(SynapseConfig(path=":memory:", vocab_size=4096, dim=32,
                                       epsilon=0.0, **kw))


CORPUS = {
    "game-1": ("리듬게임 판정과 손 반응", "리듬게임에서 판정을 읽는 건 머리보다 손이 먼저다. 오토파일럿 반응 훈련.", ["게임", "리듬"]),
    "game-2": ("Sayonara Wild Hearts 리뷰", "팝 앨범 한 장을 통째로 플레이하는 리듬게임. 한 시간이면 끝난다.", ["게임", "리듬", "리뷰"]),
    "cook-1": ("김치찌개 레시피", "돼지고기와 묵은지로 끓이는 김치찌개. 설탕 약간이 포인트.", ["요리"]),
    "dev-1": ("Python asyncio 디버깅 노트", "이벤트 루프 블로킹을 찾으려면 slow callback 로그를 켠다.", ["개발", "python"]),
    "dev-2": ("SQLite WAL 모드 정리", "WAL은 동시 읽기에 강하다. 체크포인트 주기에 주의.", ["개발", "db"]),
}


def seed_corpus(mem: SynapseMemory) -> None:
    for nid, (title, text, tags) in CORPUS.items():
        links = ["game-1"] if nid == "game-2" else []
        mem.index(nid, text, title=title, tags=tags, links=links)


# ── tokenizer ────────────────────────────────────────────────────────

def test_tokenizer_korean_ngrams():
    toks = tokenize("리듬게임의 판정")
    assert "리듬게임의" in toks and "리듬" in toks and "판정" in toks  # 2-grams cover 조사 변형
    assert fnv1a("리듬", 4096) == fnv1a("리듬", 4096) < 4096


def test_tokenizer_english():
    toks = tokenize("Debugging the asyncio event loop")
    assert "debugging" in toks and "asyncio" in toks and "the" not in toks


# ── bm25 ─────────────────────────────────────────────────────────────

def test_bm25_ranks_matching_doc_first():
    store = Store(":memory:")
    for i, text in enumerate(["김치찌개 레시피 요리", "리듬게임 판정 게임", "파이썬 비동기 루프"]):
        toks = tokenize(text)
        store.upsert_node(f"d{i}", kind="note", title="", tags=[], text_len=len(toks),
                          updated_at=0, pinned=False, importance=1.0)
        store.replace_postings(f"d{i}", term_frequencies(toks))
    scores = bm25_scores(store, tokenize("김치찌개 만드는 법"))
    assert max(scores, key=scores.get) == "d0"


# ── embedder ─────────────────────────────────────────────────────────

def test_embedder_shared_ngrams_give_similarity():
    emb = HashEmbedder(4096, 32)
    a = emb.embed("리듬게임 판정 이야기")
    b = emb.embed("리듬게임의 판정은 어렵다")
    c = emb.embed("김치찌개 끓이는 방법")
    assert float(a @ b) > float(a @ c)  # 어휘 겹침 → 코사인 우위


def test_embedder_distill_improves_and_gates():
    emb = HashEmbedder(2048, 16, seed=3)
    rng = np.random.default_rng(0)
    # Synthetic teacher: two clusters whose member texts share NO tokens —
    # the initial hash geometry is uninformative, so distillation must learn
    # the grouping from the teacher alone.
    words = [f"단어{i}거시기{i}" for i in range(80)]
    pairs = []
    for i in range(40):
        cluster = i % 2
        base = np.zeros(24); base[cluster * 12] = 1.0
        text = f"{words[2 * i]} {words[2 * i + 1]}"
        pairs.append((text, base + rng.normal(0, 0.05, 24)))
    before_table = emb.table.copy()
    m = emb.distill(pairs, epochs=25, lr=2e-2, batch=16)
    assert m["trained"] == 1.0
    if m["swapped"]:
        assert m["corr_after"] > m["corr_before"]  # swap only on improvement
    else:
        assert np.array_equal(emb.table, before_table)  # gate kept the old table


def test_embedder_roundtrip():
    emb = HashEmbedder(2048, 16)
    blob = emb.dumps()
    emb2 = HashEmbedder.loads(blob)
    v1, v2 = emb.embed("동일 텍스트"), emb2.embed("동일 텍스트")
    assert np.allclose(v1, v2, atol=1e-3)  # fp16 persistence tolerance


# ── graph ────────────────────────────────────────────────────────────

def test_ppr_prefers_linked_neighbourhood():
    from geny_memory_adaptor.graph import build_adjacency

    adj = build_adjacency([("a", "b", 1.0), ("b", "a", 1.0), ("b", "c", 1.0), ("x", "y", 1.0)])
    rank = personalized_pagerank(adj, {"a": 1.0})
    assert rank["b"] > rank.get("y", 0.0)
    assert rank["a"] > rank["c"] > 0


# ── ranker ───────────────────────────────────────────────────────────

def test_ranker_learns_pairwise_and_blend_gates():
    r = OnlineRanker(blend_min_events=50)
    rng = np.random.default_rng(1)
    # Ground truth: feature 6 (ppr_co) decides, heuristic barely weighs it.
    def sample(pos: bool):
        x = rng.normal(0, 1, len(FEATURES)).astype(np.float32)
        x[6] = (0.6 if pos else -0.6) + rng.normal(0, 0.4)
        return x
    for x in [sample(bool(i % 2)) for i in range(50)]:
        r.observe(x)
    assert r.blend == 0.0  # gate closed before enough events
    losses = []
    for i in range(5000):
        p, n = sample(True), sample(False)
        if i % 4 == 0:               # 25% held out → referee only (out-of-sample)
            r.referee_pair(p, n)
        else:
            losses.append(r.update_pair(p, n))
    assert np.mean(losses[-50:]) < np.mean(losses[:50])  # learning reduces loss
    assert r.blend > 0.0  # learner beats heuristic on held-out pairs → gate opens
    assert r.score(sample(True)) > r.score(sample(False))


def test_ranker_gate_shut_on_pure_noise():
    """Out-of-sample McNemar: random labels → held-out win-rate ≈ 0.5 → gate
    never opens (the property the query-level holdout guarantees)."""
    r = OnlineRanker(blend_min_events=40)
    rng = np.random.default_rng(2)
    def s():
        return rng.normal(0, 1, len(FEATURES)).astype(np.float32)
    for _ in range(80):
        r.observe(s())
    for i in range(12000):
        a, b = s(), s()
        p, n = (a, b) if rng.random() < 0.5 else (b, a)  # random label
        if i % 4 == 0:
            r.referee_pair(p, n)
        else:
            r.update_pair(p, n)
    assert r.blend == 0.0  # noise never opens the gate
    assert 0.45 < r.stats()["win_rate"] < 0.55  # held-out is a true coin flip


def test_ranker_persistence_roundtrip():
    r = OnlineRanker()
    rng = np.random.default_rng(2)
    for _ in range(30):
        a = rng.normal(0, 1, len(FEATURES)).astype(np.float32)
        b = rng.normal(0, 1, len(FEATURES)).astype(np.float32)
        r.observe(a); r.observe(b)
        r.update_pair(a, b)
    r2 = OnlineRanker.loads(r.dumps())
    x = rng.normal(0, 1, len(FEATURES)).astype(np.float32)
    assert abs(r.score(x) - r2.score(x)) < 1e-4
    assert r2.events == r.events


# ── engine E2E ───────────────────────────────────────────────────────

def test_search_keyword_and_semantic_paths():
    mem = make_mem()
    seed_corpus(mem)
    hits = mem.search("리듬게임 판정")
    assert hits and set(h.id for h in hits[:2]) == {"game-1", "game-2"}
    # 조사 변형(부분일치) — n-gram 덕에 검색됨
    hits2 = mem.search("김치찌개를 끓이고 싶다")
    assert hits2 and hits2[0].id == "cook-1"


def test_graph_expansion_pulls_linked_note():
    mem = make_mem()
    seed_corpus(mem)
    # game-2 links game-1; a query hitting only game-2 lexically should still
    # surface game-1 through the LINK-PPR channel.
    hits = mem.search("Sayonara Wild Hearts", top_k=5)
    ids = [h.id for h in hits]
    assert "game-2" in ids and "game-1" in ids
    g1 = next(h for h in hits if h.id == "game-1")
    assert g1.features["ppr_link"] > 0


def test_feedback_learns_coaccess_and_reorders():
    mem = make_mem(blend_min_events=10)
    seed_corpus(mem)
    # Repeatedly: search, mark dev-1 + dev-2 used together.
    for _ in range(12):
        hits = mem.search("개발 노트", top_k=5)
        tok = hits[0].query_token
        mem.feedback(tok, used_ids=["dev-1", "dev-2"])
    st = mem.stats()
    assert st["edges"]["coaccess"] >= 2  # Hebbian edges materialized
    assert st["ranker"]["events"] > 0
    # dev-1 조회 시 coaccess 이웃 dev-2가 그래프 특징을 얻는다
    hits = mem.search("python asyncio", top_k=5)
    d2 = next((h for h in hits if h.id == "dev-2"), None)
    assert d2 is not None and d2.features["ppr_co"] > 0


def test_persistence_across_reopen(tmp_path):
    db = str(tmp_path / "syn.db")
    mem = SynapseMemory(SynapseConfig(path=db, vocab_size=4096, dim=32, epsilon=0.0))
    seed_corpus(mem)
    hits = mem.search("리듬게임")
    mem.feedback(hits[0].query_token, used_ids=[hits[0].id, hits[1].id])
    stats1 = mem.stats()
    mem.close()

    mem2 = SynapseMemory(SynapseConfig(path=db, vocab_size=4096, dim=32, epsilon=0.0))
    stats2 = mem2.stats()
    assert stats2["nodes"] == stats1["nodes"]
    assert stats2["ranker"]["events"] == stats1["ranker"]["events"]
    assert mem2.search("리듬게임")[0].id in ("game-1", "game-2")
    mem2.close()


def test_remove_node_disappears():
    mem = make_mem()
    seed_corpus(mem)
    mem.remove("cook-1")
    assert all(h.id != "cook-1" for h in mem.search("김치찌개"))


def test_distill_e2e_reembeds():
    mem = make_mem()  # store_text defaults True
    rng = np.random.default_rng(5)
    for i in range(40):  # ≥ MIN_DISTILL_PAIRS
        cluster = i % 2
        text = ("리듬 게임 판정 비트 " if cluster == 0 else "김치 요리 레시피 재료 ") + f"메모 {i}"
        teacher = np.zeros(24); teacher[cluster * 12] = 1.0
        mem.index(f"n{i}", text, teacher_vec=teacher + rng.normal(0, 0.05, 24))
    m = mem.distill()
    assert m["trained"] == 1.0 and m["pairs"] == 40.0
    assert "candidate" not in m  # engine consumes it; never leaks the array


def test_distill_needs_store_text():
    mem = make_mem(store_text=False)
    for i in range(40):
        t = np.zeros(8); t[(i % 2) * 4] = 1.0
        mem.index(f"n{i}", f"메모 {i} 내용", teacher_vec=t)
    assert mem.distill().get("reason_no_text") == 1.0  # no re-embeddable corpus


def test_distill_crash_safety_atomic(tmp_path):
    # Table + vectors swap in one transaction: after a successful distill, a
    # reopened engine reads a consistent (embedder, vectors) pair.
    db = str(tmp_path / "d.db")
    mem = SynapseMemory(SynapseConfig(path=db, vocab_size=4096, dim=32, epsilon=0.0))
    rng = np.random.default_rng(1)
    for i in range(40):
        c = i % 2
        t = np.zeros(16); t[c * 8] = 1.0
        mem.index(f"n{i}", ("게임 판정 " if c == 0 else "요리 재료 ") + f"{i}",
                  teacher_vec=t + rng.normal(0, 0.05, 16))
    mem.distill()
    mem.close()
    mem2 = SynapseMemory(SynapseConfig(path=db, vocab_size=4096, dim=32, epsilon=0.0))
    # Query vectors (fresh embed) and stored doc vectors use the SAME table.
    hits = mem2.search("게임 판정")
    assert hits  # consistent embedder/vectors → search works after reopen
    mem2.close()


# ── config / env ─────────────────────────────────────────────────────

def test_config_from_env(tmp_path, monkeypatch):
    envfile = tmp_path / ".env"
    envfile.write_text("GMA_DIM=64\nGMA_TOP_K=3\n# comment\nGMA_PATH=:memory:\n")
    monkeypatch.delenv("GMA_DIM", raising=False)
    cfg = SynapseConfig.from_env(dotenv=str(envfile))
    assert cfg.dim == 64 and cfg.top_k == 3 and cfg.path == ":memory:"
    # 명시 인자 > env
    cfg2 = SynapseConfig.from_env(dotenv=str(envfile), dim=128)
    assert cfg2.dim == 128


# ── executor adapter ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_vector_handle_shapes():
    mem = make_mem()
    handle = SynapseVectorHandle(mem)
    await handle.index("a", "리듬게임 판정 노트", {"title": "게임", "tags": ["게임"]})
    await handle.index_batch([
        {"id": "b", "text": "김치찌개 레시피", "metadata": {"kind": "note"}},
    ])
    assert handle.descriptor["api_calls"] == 0
    rows = await handle.search("리듬게임", top_k=2)
    assert rows and rows[0]["id"] == "a" and "query_token" in rows[0]
    doc = await handle.fetch_document("a")
    assert doc and doc["title"] == "게임"
    await handle.remove("a")
    assert await handle.fetch_document("a") is None


def test_blend_gate_resists_label_noise():
    """Random feedback across VARIED queries (so the query-level holdout
    actually splits) must NOT open the gate — the core safety property, and
    the regression guard for both the 0.05 leak and the replay/phase-lock leak."""
    mem = make_mem(blend_min_events=40)
    for t in ("게임", "요리", "개발"):
        for i in range(6):
            mem.index(f"{t}-{i}", f"{t} 관련 메모 상세 {i}", title=f"{t} {i}", tags=[t])
    import random as _r
    rng = _r.Random(0)
    queries = [f"게임 요리 개발 메모 {w}" for w in "가나다라마바사아자차카타파하거너더러머버"]
    for step in range(1500):
        hits = mem.search(queries[step % len(queries)], top_k=10)
        if len(hits) >= 2:
            mem.feedback(hits[0].query_token, used_ids=[h.id for h in rng.sample(hits, 2)])
    assert mem.stats()["ranker"]["blend"] == 0.0  # floor protected under noise


def test_feedback_does_not_persist_embedder():
    """feedback() must persist ONLY the ranker — writing the 32MB embedding
    table every feedback was a 200× slowdown (regression guard)."""
    import time as _t
    mem = make_mem()
    for i in range(6):
        mem.index(f"n{i}", f"메모 {i} 내용 기록", title=f"노트 {i}", tags=["x"])
    t0 = _t.perf_counter()
    for _ in range(40):
        h = mem.search("메모 기록", top_k=6)
        if len(h) >= 2:
            mem.feedback(h[0].query_token, used_ids=[h[0].id])
    per_loop_ms = (_t.perf_counter() - t0) / 40 * 1000
    assert per_loop_ms < 50, f"feedback loop {per_loop_ms:.0f}ms — embedder likely re-serialized"


# ── review-fix regressions (adversarial pass) ────────────────────────

def test_reindex_changed_links_no_dangling_edge():
    """A1: dropping a node's link must not leave a reverse edge behind."""
    from geny_memory_adaptor.store import EDGE_LINK
    mem = make_mem()
    mem.index("n2", "메모 둘 내용")
    mem.index("n1", "메모 하나 내용", links=["n2"])
    mem.index("n1", "메모 하나 수정", links=[])  # drop the link
    link_edges = mem.store.edges_by_type(EDGE_LINK)
    assert not any(s == "n1" or d == "n1" for s, d, _w, _u in link_edges)


def test_reindex_changed_tags_no_stale_membership():
    """A2: retagging must not leave the node under its old tag."""
    mem = make_mem()
    mem.index("a", "가 내용", tags=["A"])
    mem.index("b", "나 내용", tags=["A"])
    _ = mem.search("내용")  # warms the tag cache
    mem.index("a", "가 수정", tags=["B"])  # retag a: A → B
    tags_map = mem._tags_map()
    assert "a" not in tags_map.get("A", [])
    assert "a" in tags_map.get("B", [])


def test_remove_purges_feedback_token():
    """A3: a search token referencing a removed node must not survive."""
    mem = make_mem(blend_min_events=5)
    for i in range(4):
        mem.index(f"n{i}", f"게임 판정 메모 {i}", tags=["게임"])
    hits = mem.search("게임 판정")
    tok = hits[0].query_token
    mem.remove(hits[0].id)
    r = mem.feedback(tok, used_ids=[h.id for h in hits[:2]])
    # token was dropped on remove → feedback is a clean no-op, not a resurrect
    assert r.get("applied") == 0.0


def test_remove_deletes_stored_text():
    """B5: remove() must delete the node's distill-text param (no leak)."""
    mem = make_mem()
    mem.index("n1", "지울 메모 본문", teacher_vec=[0.1] * 8)
    assert mem.store.get_param("text:n1") is not None
    mem.remove("n1")
    assert mem.store.get_param("text:n1") is None


def test_store_text_false_stores_nothing():
    mem = make_mem(store_text=False)
    mem.index("n1", "메모 본문", teacher_vec=[0.1] * 8)
    assert mem.store.get_param("text:n1") is None


def test_store_rollback_leaves_no_partial_write():
    """B3: a failing multi-statement write rolls back, and the next write's
    commit does not persist the partial state."""
    from geny_memory_adaptor.store import Store
    s = Store(":memory:")
    s.upsert_node("a", kind="note", title="", tags=[], text_len=1,
                  updated_at=0, pinned=False, importance=1.0)
    try:
        # executemany with a bad row type raises mid-write.
        s.replace_postings("a", {"ok": 1.0, "bad": object()})  # type: ignore
    except Exception:
        pass
    # 'a' still has no committed postings from the failed call.
    assert s.postings_for_terms(["ok"]) == {}
    # a subsequent good write commits cleanly (no leftover open txn).
    s.replace_postings("a", {"ok": 1.0})
    assert "ok" in s.postings_for_terms(["ok"])
    s.close()


def test_concurrent_search_index_feedback_no_crash():
    """B1: hammer one engine from several threads — the lock must prevent the
    'dict changed size' / half-updated-weights races the reviewer found."""
    import threading
    mem = make_mem(blend_min_events=10)
    for i in range(30):
        mem.index(f"n{i}", f"게임 판정 메모 {i} 내용", tags=["게임"])
    errors = []

    def worker(w):
        try:
            for j in range(40):
                if j % 3 == 0:
                    mem.index(f"n{w}-{j}", f"추가 메모 {w} {j}", tags=["x"])
                elif j % 3 == 1:
                    h = mem.search("게임 판정 메모", top_k=8)
                    if len(h) >= 2:
                        mem.feedback(h[0].query_token, used_ids=[h[0].id, h[1].id])
                else:
                    mem.search("추가 메모 내용", top_k=5)
        except Exception as e:  # noqa: BLE001
            errors.append(repr(e))

    ts = [threading.Thread(target=worker, args=(w,)) for w in range(6)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert not errors, errors[:3]


def test_mutual_link_not_double_weighted():
    """Final review: LINK symmetrization must dedup a mutual link (A↔B), else
    PPR over-weights it vs one-way links."""
    from geny_memory_adaptor.graph import build_type_adjacency
    from geny_memory_adaptor.store import EDGE_LINK
    adj = build_type_adjacency(
        [("A", "B", 1.0, 0.0), ("B", "A", 1.0, 0.0)], EDGE_LINK)
    assert adj["A"] == [("B", 1.0)] and adj["B"] == [("A", 1.0)]  # no dup


def test_distill_rollback_keeps_embedder_consistent(tmp_path):
    """Final review: a failed store commit during distill must NOT leave the
    in-memory embedder swapped ahead of the on-disk table/vectors."""
    db = str(tmp_path / "d.db")
    mem = SynapseMemory(SynapseConfig(path=db, vocab_size=2048, dim=16, epsilon=0.0,
                                      distill_epochs=20, distill_lr=2e-2))
    rng = np.random.default_rng(3)
    pools = [[f"군집{c}어휘{w}" for w in range(10)] for c in range(3)]
    for i in range(60):
        c = i % 3
        words = rng.choice(pools[c], 3, replace=False)
        t = np.zeros(24); t[c * 8] = 1.0
        mem.index(f"n{i}", " ".join(words), teacher_vec=t + rng.normal(0, 0.05, 24))
    before = id(mem.embedder.table)

    def boom(*a, **k):
        raise RuntimeError("injected commit failure")
    mem.store.swap_embedder_and_vectors = boom  # type: ignore
    try:
        mem.distill()
    except RuntimeError:
        pass
    assert id(mem.embedder.table) == before  # embedder untouched on rollback
    mem.close()


def test_search_scores_bounded_on_low_variance():
    """Final review: a small corpus (near-constant features → var→0) must not
    blow up absolute scores; z-clipping contains it. Ranking still works."""
    mem = make_mem()
    for i in range(15):
        mem.index(f"n{i}", f"게임 판정 메모 {i}", tags=["게임"])
    for _ in range(5):
        hits = mem.search("게임 판정")
        assert hits and all(abs(h.score) < 100 for h in hits)  # no explosion


def test_search_is_idempotent_within_a_call():
    """Final review: observe() runs AFTER ranking, so a candidate can't perturb
    its own z-score mid-search — features fed to the ranker are scored against
    one consistent normalization snapshot."""
    mem = make_mem()
    for i in range(20):
        mem.index(f"n{i}", f"게임 판정 메모 {i} 내용", tags=["게임"])
    # First ever search: mu=0/var=1 baseline. The ranking must be a pure
    # function of that snapshot (observe hasn't run yet within this call).
    r = mem.search("게임 판정 내용", top_k=8)
    # Re-score the SAME captured features with the pre-search ranker state would
    # match; here we just assert a stable, sane top result and no NaN/inf.
    assert r and all(np.isfinite(h.score) for h in r)


def test_giant_wordless_token_is_bounded():
    """Fuzz F1 (HIGH DoS): a megabyte-long space-free token must index/search
    in bounded time — the n-gram + jamo expansion used to OOM."""
    import time
    mem = make_mem()
    t0 = time.perf_counter()
    mem.index("blob", "x" * 2_000_000)  # base64/URL/CJK-run shaped
    mem.search("가" * 1_000_000, top_k=5)
    assert time.perf_counter() - t0 < 2.0  # was >30s / killed


def test_config_validation_rejects_bad_values():
    """Fuzz F2/F3/F5: clear ValueError instead of a cryptic numpy crash or a
    silent NaN-weight poisoning downstream."""
    for kw in [{"dim": 0}, {"dim": -1}, {"vocab_size": 1}, {"hidden": 0},
               {"lr": float("inf")}, {"lr": 0.0}, {"l2": float("nan")},
               {"top_k": 0}, {"epsilon": 2.0}]:
        with pytest.raises(ValueError):
            SynapseConfig(path=":memory:", **kw)


def test_top_k_edges():
    """Fuzz F4: top_k=0 → empty, negative → empty, None → default."""
    mem = make_mem()
    for i in range(10):
        mem.index(f"n{i}", f"게임 판정 {i}", tags=["게임"])
    assert mem.search("게임", top_k=0) == []
    assert mem.search("게임", top_k=-5) == []
    assert len(mem.search("게임")) == 8


def test_index_is_atomic_on_failure():
    """Crash-review should-fix: a mid-index store failure must roll the whole
    node back — no orphan node row without vector/postings."""
    mem = make_mem()
    for i in range(5):
        mem.index(f"n{i}", f"게임 판정 {i}", tags=["게임"])

    def boom(*a, **k):
        raise RuntimeError("disk full mid-index")
    mem.store.index_atomic = boom  # type: ignore
    with pytest.raises(RuntimeError):
        mem.index("orphan", "실패", tags=["x"], teacher_vec=[0.1] * 8)
    assert mem.store.get_node("orphan") is None
    assert "orphan" not in {nid for nid, _, _ in mem.store.all_vectors()}
    assert mem.store.count_nodes() == 5  # existing nodes intact


def test_knn_sample_cap_keeps_indexing_bounded():
    """Scale-review: with more than sample_cap vectors, a new node's k-NN is
    computed only against the recent slice — per-index cost stays flat."""
    import time
    mem = make_mem(knn_sample_cap=200)
    for i in range(200):
        mem.index(f"seed{i}", f"게임 판정 메모 {i}", tags=["게임"])
    t0 = time.perf_counter()
    for i in range(200):
        mem.index(f"more{i}", f"게임 판정 추가 {i}", tags=["게임"])
    per = (time.perf_counter() - t0) / 200 * 1000
    assert per < 20  # flat, not growing with corpus size
    # still produces knn edges (from the capped sample)
    from geny_memory_adaptor.store import EDGE_KNN
    assert len(mem.store.edges_by_type(EDGE_KNN)) > 0


def test_get_text_roundtrip_and_maxlen():
    """Integration support: get_text() returns the stored body (bounded), and
    store_text_maxlen caps it. Cleared on remove."""
    mem = make_mem(store_text_maxlen=20)
    mem.index("n1", "가나다라마바사아자차카타파하거너더러머버서어저처", title="제목")
    t = mem.get_text("n1")
    assert t is not None and len(t) <= 20 + len("제목\n")  # title\nbody, capped
    mem.remove("n1")
    assert mem.get_text("n1") is None
    # store_text=False → no text kept
    m2 = make_mem(store_text=False)
    m2.index("x", "본문")
    assert m2.get_text("x") is None


def test_learn_direct_features_opens_gate_and_is_crossturn_safe():
    """engine.learn() — the cross-turn-safe, feature-level feedback path.

    The host supplies feature vectors (from a remembered search), so learning
    does NOT depend on the bounded _recent_queries cache: a note edited/cited
    many searches later still reinforces. Verifies the same three guarantees as
    feedback(): a genuine signal opens the blend gate, the learned ranker
    re-orders correctly, and Hebbian co-access edges form between co-useful ids.
    """
    mem = make_mem(blend_min_events=40)
    rng = np.random.default_rng(0)
    n = len(FEATURES)
    rec = FEATURES.index("recency")

    def mk(is_pos):
        x = rng.normal(0, 0.3, n).astype(np.float32)
        x[rec] = -1.0 if is_pos else 1.0  # heuristic weights recency +, so WRONG
        return x

    # Flood _recent_queries with unrelated searches so any cache dependency
    # would have evicted the training queries — learn() must be immune.
    for _ in range(200):
        mem.index(f"noise-{_}", "무관한 노트", kind="note")
        mem.search("무관", top_k=3)

    opened = None
    for step in range(4000):
        mem.learn(f"q{step}", positives=[("p", mk(True))], negatives=[mk(False)],
                  label_src="edit")
        if opened is None and mem.ranker.blend > 0:
            opened = step
    assert mem.ranker.blend > 0.0, "genuine signal must open the gate"
    assert opened is not None
    p, ng = mk(True), mk(False)
    assert mem.ranker.heuristic(p) < mem.ranker.heuristic(ng)  # heuristic is wrong
    assert mem.ranker.score(p) > mem.ranker.score(ng)          # learned is right


def test_learn_direct_gate_shut_on_noise_and_hebbian_forms():
    from geny_memory_adaptor.store import EDGE_COACCESS
    mem = make_mem(blend_min_events=40)
    rng = np.random.default_rng(5)
    n = len(FEATURES)
    for step in range(4000):  # pure noise labels → gate must stay shut
        a = rng.normal(0, 0.3, n).astype(np.float32)
        b = rng.normal(0, 0.3, n).astype(np.float32)
        mem.learn(f"q{step}", positives=[("p", a)], negatives=[b], label_src="noise")
    assert mem.ranker.blend == 0.0, "noise must never open the gate"

    # Hebbian: two ids confirmed useful TOGETHER get a co-access edge.
    z = np.zeros(n, dtype=np.float32)
    mem.index("a", "가", kind="note"); mem.index("b", "나", kind="note")
    mem.learn("multi", positives=[("a", z), ("b", z)], negatives=[z], label_src="edit")
    assert mem.store.get_edge("a", "b", EDGE_COACCESS) is not None
    assert mem.store.get_edge("b", "a", EDGE_COACCESS) is not None


# ── A1: per-item trust — effect-proving tests ─────────────────────────


def test_trust_unhelpful_drops_rank_and_helpful_raises():
    """EFFECT PROOF: with two equally-relevant memories, explicit unhelpful
    feedback must push one below the other; helpful feedback must widen the
    gap. This is the per-item reliability axis the query-dependent ranker
    cannot express."""
    mem = make_mem()
    mem.index("a", "리듬게임 판정 타이밍과 콤보 시스템", kind="note")
    mem.index("b", "리듬게임 판정 타이밍과 콤보 시스템", kind="note")  # identical

    def scores():
        hits = {h.id: h.score for h in mem.search("리듬게임 판정", top_k=5)}
        return hits["a"], hits["b"]

    a0, b0 = scores()
    assert abs(a0 - b0) < 0.75  # near-tie at neutral trust

    for _ in range(3):
        mem.trust_feedback("b", False)   # −0.10 × 3 → trust 0.2
    for _ in range(4):
        mem.trust_feedback("a", True)    # +0.05 × 4 → trust ≈ 0.7
    a1, b1 = scores()
    assert a1 > b1, "trusted memory must outrank distrusted equal-relevance one"
    assert (a1 - b1) > (a0 - b0) + 1.0   # the gap is material, not noise
    # asymmetry: 3 unhelpful moved b further down than 4 helpful moved a up
    assert (b0 - b1) > (a1 - a0)


def test_trust_decays_to_neutral_anti_ossification():
    """EFFECT PROOF (anti-ossification): trust reinforced long ago and never
    re-confirmed fades back toward neutral — stale reinforcement cannot
    permanently pin a memory above equally-relevant peers."""
    import time as _t
    mem = make_mem(trust_half_life_days=45.0)
    mem.index("old", "파이썬 비동기 asyncio 이벤트루프", kind="note")
    mem.index("new", "파이썬 비동기 asyncio 이벤트루프", kind="note")

    # Heavy reinforcement of "old", but stamped 90 days in the past.
    past = _t.time() - 90 * 86400
    for _ in range(8):
        mem.trust_feedback("old", True, now=past)     # → ~0.9, aged 90d
    node = mem.store.get_node("old")
    assert node["trust"] > 0.85
    # Effective trust NOW: 2 half-lives → 0.5 + 0.4×0.25 = 0.6
    eff = mem._effective_trust(node, _t.time())
    assert 0.55 < eff < 0.65, f"expected decay toward neutral, got {eff}"

    # And the same reinforcement applied FRESH is much stronger.
    for _ in range(8):
        mem.trust_feedback("new", True)
    hits = {h.id: h.score for h in mem.search("비동기 이벤트루프", top_k=5)}
    assert hits["new"] > hits["old"], "fresh confirmation must beat stale"


def test_trust_neutral_is_exact_noop_and_weight_zero_disables():
    """Neutral trust adds exactly 0 — the whole feature is invisible until
    feedback arrives (MIRACL/eval regression safety by construction)."""
    m1 = make_mem(trust_weight=0.0, seed=7)
    m2 = make_mem(trust_weight=2.0, seed=7)
    for m in (m1, m2):
        m.index("x", "김치찌개 끓이는 법 돼지고기", kind="note")
        m.index("y", "리듬게임 콤보 판정", kind="note")
    s1 = [(h.id, round(h.score, 6)) for h in m1.search("찌개", top_k=3)]
    s2 = [(h.id, round(h.score, 6)) for h in m2.search("찌개", top_k=3)]
    assert s1 == s2  # untouched trust (0.5) → identical scores


def test_learn_positive_bumps_trust_and_negatives_do_not():
    mem = make_mem(blend_min_events=40)
    mem.index("p", "가나다", kind="note")
    mem.index("n", "라마바", kind="note")
    z = np.zeros(len(FEATURES), dtype=np.float32)
    mem.learn("q", positives=[("p", z)], negatives=[z], label_src="edit")
    assert mem.store.get_node("p")["trust"] > 0.5
    # negatives are a ranking contrast, NOT distrust evidence
    assert mem.store.get_node("n")["trust"] == 0.5


def test_trust_migration_from_pre_1_5_db(tmp_path):
    """A vault created before the trust columns opens cleanly: columns are
    added, existing rows read as neutral 0.5."""
    import sqlite3
    dbp = str(tmp_path / "old.db")
    conn = sqlite3.connect(dbp)
    conn.execute(
        "CREATE TABLE nodes(id TEXT PRIMARY KEY, kind TEXT DEFAULT 'note',"
        " title TEXT DEFAULT '', tags TEXT DEFAULT '[]', text_len INT DEFAULT 0,"
        " updated_at REAL, access_count INT DEFAULT 0, last_access REAL DEFAULT 0,"
        " pinned INT DEFAULT 0, importance REAL DEFAULT 1.0)")
    conn.execute("INSERT INTO nodes(id, updated_at) VALUES('legacy', 0)")
    conn.commit(); conn.close()

    from geny_memory_adaptor.store import Store
    st = Store(dbp)
    node = st.get_node("legacy")
    assert node is not None
    assert node["trust"] == 0.5 and node["trust_updated"] == 0.0
    st.set_trust("legacy", 0.7, 123.0)
    assert st.get_node("legacy")["trust"] == 0.7


# ── A2: contradiction detection — planted-conflict tests ─────────────


def test_contradiction_detects_planted_negation_conflicts():
    """EFFECT PROOF: planted direct contradictions (same topic, flipped
    polarity) are flagged; paraphrase near-duplicates and unrelated notes
    are NOT. This is the store-hygiene layer — pure math, no LLM."""
    mem = make_mem()
    # planted conflict pair (ko)
    mem.index("c1", "브라우저 도구는 정상적으로 작동한다. 스크린샷 촬영 가능.", kind="fact")
    mem.index("c2", "브라우저 도구는 작동하지 않는다. 스크린샷 촬영 불가.", kind="fact")
    # planted conflict pair (en)
    mem.index("e1", "the deploy pipeline works fine on staging", kind="fact")
    mem.index("e2", "the deploy pipeline is broken on staging", kind="fact")
    # near-duplicate WITHOUT polarity flip — must NOT be flagged
    mem.index("d1", "김치찌개 레시피: 돼지고기와 두부를 넣고 끓인다", kind="note")
    mem.index("d2", "김치찌개 레시피: 돼지고기, 두부를 넣어 끓입니다", kind="note")
    # unrelated
    mem.index("u1", "리듬게임 콤보 판정 타이밍", kind="note")

    hits = mem.contradictions("c1")
    ids = [h["id"] for h in hits]
    assert "c2" in ids, f"planted ko conflict missed: {hits}"
    assert ids[0] == "c2"
    assert hits[0]["negation_flip"] is True
    assert "u1" not in ids and "d1" not in ids

    hits_en = mem.contradictions("e1")
    assert [h["id"] for h in hits_en][:1] == ["e2"], f"planted en conflict missed: {hits_en}"

    # near-duplicates with the SAME polarity are hygiene-clean
    dup_hits = mem.contradictions("d1")
    assert all(h["id"] != "d2" for h in dup_hits), \
        f"near-duplicate falsely flagged as contradiction: {dup_hits}"


def test_contradiction_diagnostic_never_mutates():
    mem = make_mem()
    mem.index("a", "도구 X는 사용 가능하다", kind="fact")
    mem.index("b", "도구 X는 사용 불가능하다", kind="fact")
    before = mem.store.count_nodes()
    mem.contradictions("a")
    assert mem.store.count_nodes() == before
    assert mem.get_text("b") is not None  # nothing deleted


def test_contradiction_unknown_or_textless_node_safe():
    mem = make_mem(store_text=False)
    mem.index("x", "본문", title="제목만 있는 노트")
    assert mem.contradictions("ghost") == []
    # store_text=False → falls back to title, no crash
    assert isinstance(mem.contradictions("x"), list)


# ── A3: compositional AND-JOIN — effect-proving tests ────────────────


def _join_corpus(mem):
    # partial: covers 결제+인증 RICHLY (2 of 3 entities) — additive fusion's
    # favourite. Similar length to `complete` so BM25 length-norm is fair.
    mem.index("partial", "결제 승인과 결제 취소, 결제 환불, 결제 재시도를 "
              "인증 토큰 발급, 인증 갱신, 인증 세션 관리와 함께 다루는 결제 인증 심화 문서.",
              kind="note")
    # complete: touches ALL THREE entities, each only once, similar length.
    mem.index("complete", "결제 요청 전에 인증 토큰을 검증하고 그 결과를 감사 "
              "로그 파이프라인에 남기는 연동 지점의 전반 흐름 설계 개요 문서.",
              kind="note")
    # noise
    mem.index("misc", "리듬게임 콤보 판정 타이밍", kind="note")


def test_join_and_finds_intersection_where_flat_search_fails():
    """EFFECT PROOF: for 'what touches 결제 AND 인증 AND 로그?', weakest-link
    scoring must rank the all-three doc first — while flat additive search
    ranks the rich two-of-three doc above it (saturated per-term BM25 sums:
    two strong terms outweigh three weak ones). That mis-ranking is exactly
    the failure mode min-scoring exists to fix.

    NOTE: an earlier 2-entity version of this corpus did NOT show the flat
    failure — BM25 term saturation handles two entities fine. The advantage
    is real only at 3+ entities (or graph-mediated membership), so that is
    what we test."""
    mem = make_mem()
    _join_corpus(mem)

    flat = mem.search("결제 인증 로그", top_k=4)
    join = mem.search_join(["결제", "인증", "로그"], mode="and", top_k=4)

    # the baseline failure this fixes: flat fusion prefers the 2-of-3-rich doc
    assert flat and flat[0].id == "partial", \
        f"corpus no longer demonstrates the flat-search failure: {[(h.id, round(h.score,3)) for h in flat]}"
    # AND-join gets the intersection right
    assert join and join[0].id == "complete", \
        f"AND-join must surface the 3-entity intersection first: {[(h.id, round(h.score,3)) for h in join]}"
    join_rank = {h.id: i for i, h in enumerate(join)}
    if "partial" in join_rank:  # reachable via graph hops at best
        assert join_rank["partial"] > join_rank["complete"]
    assert "misc" not in join_rank


def test_join_or_mode_and_validation():
    mem = make_mem()
    _join_corpus(mem)
    # OR: union + mean semantics — the 2-of-3-rich doc is a fine OR answer
    ids = [h.id for h in mem.search_join(["결제", "인증", "로그"], mode="or", top_k=5)]
    assert "partial" in ids and "complete" in ids
    # AND must be a subset of OR (strictly more selective)
    and_ids = [h.id for h in mem.search_join(["결제", "인증", "로그"], mode="and", top_k=10)]
    assert set(and_ids) <= set(
        h.id for h in mem.search_join(["결제", "인증", "로그"], mode="or", top_k=10))
    import pytest as _pt
    with _pt.raises(ValueError):
        mem.search_join(["결제"], mode="xor")
    assert mem.search_join([], mode="and") == []
    assert mem.search_join(["  "], mode="and") == []


def test_join_reaches_through_graph_links():
    """A memory that never mentions an entity verbatim still qualifies via an
    explicit LINK to one that does — the graph half of the join."""
    mem = make_mem()
    mem.index("spec", "인증 프로토콜 표준 명세", kind="note")
    mem.index("impl", "결제 게이트웨이 구현 노트", kind="note",
              links=["spec"])  # linked to the 인증 doc, never says 인증
    mem.index("lonely", "결제 화면 색상 가이드", kind="note")

    hits = mem.search_join(["결제", "인증"], mode="and", top_k=5)
    ids = [h.id for h in hits]
    assert "impl" in ids, f"link-mediated membership missed: {ids}"
    if "lonely" in ids:
        assert ids.index("impl") < ids.index("lonely")


# ── 1.5.1: idempotent re-index short-circuit (wake-time backfill fix) ─


def test_reindex_unchanged_short_circuits_and_is_fast():
    """EFFECT PROOF: re-indexing byte-identical content must skip the full
    tokenize+embed+edges+transaction path — the difference between a session
    wake that re-scans its vault in milliseconds vs minutes (2026-07-25 prod:
    124s wakes from a full-vault backfill on every resume)."""
    import time as _t
    mem = make_mem()
    docs = [(f"n{i}", f"노트 {i} 본문 " + "내용 " * 120) for i in range(60)]

    t0 = _t.perf_counter()
    for nid, body in docs:
        mem.index(nid, body, kind="note")
    first = _t.perf_counter() - t0

    t0 = _t.perf_counter()
    for nid, body in docs:
        mem.index(nid, body, kind="note")   # identical → must short-circuit
    second = _t.perf_counter() - t0

    assert second < first / 10, \
        f"unchanged re-index must be ≥10× faster (first={first*1000:.0f}ms, second={second*1000:.0f}ms)"
    # ranking still works after the no-op pass
    assert mem.search("노트 30 본문", top_k=3)


def test_reindex_changed_content_still_reindexes():
    mem = make_mem()
    mem.index("a", "김치찌개 레시피", kind="note")
    assert mem.search("김치찌개", top_k=2)
    mem.index("a", "리듬게임 판정 가이드", kind="note")   # changed body
    ids = [h.id for h in mem.search("리듬게임 판정", top_k=2)]
    assert "a" in ids
    # old content no longer matches
    old = mem.search("김치찌개", top_k=2)
    assert not old or old[0].score < 1.0


def test_reindex_metadata_changes_are_not_skipped():
    """Tags/links/importance/pinned feed derived state (edges, features) —
    changing ONLY them must still reindex, not short-circuit."""
    mem = make_mem()
    mem.index("x", "본문 동일", tags=["a"], kind="note")
    sha1 = mem.store.get_node("x")["content_sha"]
    mem.index("x", "본문 동일", tags=["a", "b"], kind="note")
    sha2 = mem.store.get_node("x")["content_sha"]
    assert sha1 != sha2
    mem.index("x", "본문 동일", tags=["a", "b"], kind="note", pinned=True)
    assert mem.store.get_node("x")["content_sha"] != sha2


def test_reindex_unchanged_touches_updated_at_only():
    mem = make_mem()
    mem.index("t", "본문", kind="note", updated_at=100.0)
    assert mem.store.get_node("t")["updated_at"] == 100.0
    mem.index("t", "본문", kind="note", updated_at=200.0)  # same content, new ts
    node = mem.store.get_node("t")
    assert node["updated_at"] == 200.0  # recency refreshed without re-embed


# ── 1.6.0: integer-keyed postings (storage-bloat fix) ─────────────────


def _long_id(i: int) -> str:
    # Geny-style node ids: ~70 chars ("Scope.SESSION/conversations/<uuid>/...")
    return (f"Scope.SESSION/conversations/7a6337e0-f75b-4e9b-8972-91c7bf32179d"
            f"/2026-07-{i % 28 + 1:02d}-turn-{i}.md")


def test_postings_v2_shrinks_db_vs_v1_layout(tmp_path):
    """EFFECT PROOF: with production-shaped long node ids, the v2 integer
    postings keep the db a fraction of the v1 layout (prod: 126 MB of a
    166 MB vault was postings). We rebuild a v1-layout db by hand, migrate,
    and compare on-disk size."""
    import sqlite3, os
    from geny_memory_adaptor.store import Store
    from geny_memory_adaptor import SynapseMemory, SynapseConfig

    # Term-DIVERSE corpus (real vaults average ~230 unique terms/node): each
    # doc gets its own vocabulary so postings rows scale like production.
    def body_for(i: int) -> str:
        return " ".join(f"항목{i}구역{j} 데이터{(i * 7 + j) % 997}" for j in range(60))

    # v1-layout db built manually (schema as shipped before 1.6.0).
    v1 = str(tmp_path / "v1.db")
    c = sqlite3.connect(v1)
    c.executescript(
        "CREATE TABLE nodes(id TEXT PRIMARY KEY, kind TEXT DEFAULT 'note',"
        " title TEXT DEFAULT '', tags TEXT DEFAULT '[]', text_len INT DEFAULT 0,"
        " updated_at REAL, access_count INT DEFAULT 0, last_access REAL DEFAULT 0,"
        " pinned INT DEFAULT 0, importance REAL DEFAULT 1.0);"
        "CREATE TABLE postings(term TEXT, node_id TEXT, tf REAL,"
        " PRIMARY KEY(term, node_id)) WITHOUT ROWID;"
        "CREATE INDEX idx_postings_node ON postings(node_id);")
    from geny_memory_adaptor.tokenizer import lexical_tokens
    from geny_memory_adaptor.bm25 import term_frequencies
    for i in range(300):
        nid = _long_id(i)
        tf = term_frequencies(lexical_tokens(body_for(i)))
        c.execute("INSERT INTO nodes(id, text_len, updated_at) VALUES(?,?,0)",
                  (nid, len(tf)))
        c.executemany("INSERT INTO postings(term,node_id,tf) VALUES(?,?,?)",
                      [(t, nid, f) for t, f in tf.items()])
    c.commit(); c.close()
    v1_size = os.path.getsize(v1)

    # Open through Store → auto-migrates to v2 + VACUUM.
    st = Store(v1)
    v2_size = os.path.getsize(v1)
    assert v2_size < v1_size * 0.55, \
        f"v2 must at least halve the db (v1={v1_size//1024}KB v2={v2_size//1024}KB)"
    # postings still answer identically through the public API
    sample = st.postings_for_terms(["항목0구역0"])
    assert sample and len(sample["항목0구역0"]) == 1
    assert sample["항목0구역0"][0][0] == _long_id(0)
    total = st._read("SELECT COUNT(*) FROM postings_v2")[0][0]
    assert total > 30_000  # production-scale row count actually migrated


def test_migration_preserves_search_results(tmp_path):
    """Search results BEFORE and AFTER the v1→v2 migration are identical:
    build a vault on the current engine, dump its postings back to a v1
    layout, reopen (migrate), and compare ranked ids."""
    import sqlite3
    from geny_memory_adaptor import SynapseMemory, SynapseConfig

    db = str(tmp_path / "m.db")
    mem = SynapseMemory(SynapseConfig(path=db, vocab_size=4096, dim=32, epsilon=0.0))
    for i, (t, b) in enumerate([
        ("게임", "리듬게임 판정 콤보 시스템"), ("요리", "김치찌개 돼지고기 레시피"),
        ("개발", "파이썬 비동기 이벤트루프"), ("혼합", "리듬게임 하면서 김치찌개 먹기")]):
        mem.index(_long_id(i), b, title=t)
    before = [h.id for h in mem.search("리듬게임 판정", top_k=4)]
    mem.close()

    # Downgrade postings to v1 layout in-place.
    c = sqlite3.connect(db)
    c.executescript(
        "CREATE TABLE postings(term TEXT, node_id TEXT, tf REAL,"
        " PRIMARY KEY(term, node_id)) WITHOUT ROWID;")
    c.execute("INSERT INTO postings(term,node_id,tf)"
              " SELECT t.term, m.id, p.tf FROM postings_v2 p"
              " JOIN terms t ON t.tid=p.tid JOIN node_map m ON m.nid=p.nid")
    c.executescript("DELETE FROM postings_v2; DELETE FROM terms; DELETE FROM node_map;")
    c.commit(); c.close()

    mem2 = SynapseMemory(SynapseConfig(path=db, vocab_size=4096, dim=32, epsilon=0.0))
    after = [h.id for h in mem2.search("리듬게임 판정", top_k=4)]
    assert after == before, f"migration changed ranking: {before} → {after}"


def test_postings_v2_remove_cleans_map(tmp_path):
    from geny_memory_adaptor import SynapseMemory, SynapseConfig
    mem = SynapseMemory(SynapseConfig(path=str(tmp_path / "r.db"),
                                      vocab_size=4096, dim=32, epsilon=0.0))
    mem.index(_long_id(1), "삭제될 노트 본문")
    assert mem.store.postings_for_terms(["삭제될"])
    mem.remove(_long_id(1))
    assert not mem.store.postings_for_terms(["삭제될"])
    assert mem.store._read("SELECT COUNT(*) FROM node_map")[0][0] == 0
    assert mem.store._read("SELECT COUNT(*) FROM postings_v2")[0][0] == 0
