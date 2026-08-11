"""Incremental indexing — effect-proving tests.

A host that re-derives everything on every boot is the failure this file
guards against. Three primitives make an incremental host possible, and each
is only useful if it holds a specific property:

  * ``manifest()`` — what is already indexed, cheaply enough to ask every
    boot. Without it "what changed?" can only be answered by re-reading every
    note and re-hashing it.
  * ``index_many()`` — many notes, ONE commit. The commit was measured at
    43.7 ms against 2-3 ms of real indexing work, so a per-note transaction
    makes a catch-up 97% fsync. It must produce byte-identical results to
    indexing one at a time, or the batch is a different feature.
  * ``remove_many()`` — deletions the host discovered in bulk. Reaping a
    vault that drifted for weeks presents thousands at once.
"""

from __future__ import annotations

import time

import pytest

from xgen_agent_memory import SynapseConfig, SynapseMemory


@pytest.fixture
def mem(tmp_path):
    return SynapseMemory(SynapseConfig(path=str(tmp_path / "s.db")))


def _items(n, prefix="n", body="리듬게임 판정 연습 기록"):
    return [{"node_id": f"{prefix}{i}", "text": f"{body} {i}", "kind": "note"} for i in range(n)]


def _commits(mem):
    """Count transactions.

    ``Store._write`` IS the transaction boundary — it runs the statements and
    commits — so counting it counts fsyncs. (``sqlite3.Connection.commit``
    itself is read-only and cannot be wrapped.)
    """
    calls = []
    real = mem.store._write

    def counted(fn):
        calls.append(1)
        return real(fn)

    mem.store._write = counted
    return calls


# ── manifest ────────────────────────────────────────────────────────


def test_manifest_reports_what_is_indexed(mem):
    mem.index("a", "본문 하나", kind="note")
    mem.index("b", "본문 둘", kind="note")

    man = mem.manifest()

    assert set(man) == {"a", "b"}
    for updated_at, sha in man.values():
        assert updated_at > 0
        assert sha, "no digest — a host cannot tell changed from unchanged"


def test_manifest_digest_tracks_content(mem):
    mem.index("a", "처음 본문")
    before = mem.manifest()["a"][1]
    mem.index("a", "바뀐 본문")
    after = mem.manifest()["a"][1]

    assert before != after, "digest did not follow the content"


def test_manifest_carries_no_bodies(mem):
    """It is called on every boot; it must stay metadata-only."""
    mem.index("a", "아주 긴 본문 " * 200)
    (updated_at, sha) = mem.manifest()["a"]
    assert isinstance(updated_at, float)
    assert len(sha) == 40  # sha1 hex, not a body


# ── index_many: same result, one commit ─────────────────────────────


def test_batch_and_one_by_one_produce_the_same_index(tmp_path):
    """THE property. A batch that indexes differently is not a batch, it is a
    second implementation that will drift."""
    a = SynapseMemory(SynapseConfig(path=str(tmp_path / "a.db")))
    b = SynapseMemory(SynapseConfig(path=str(tmp_path / "b.db")))
    items = _items(25)

    for it in items:
        a.index(it["node_id"], it["text"], kind=it["kind"])
    b.index_many(items)

    assert a.manifest().keys() == b.manifest().keys()
    for nid in a.manifest():
        assert a.manifest()[nid][1] == b.manifest()[nid][1], f"{nid} differs"
    # Derived rows too, not just the digest.
    for table, col in (("postings_v2", "nid"), ("edges", "src"), ("vectors", "node_id")):
        na = a.store._read(f"SELECT COUNT(*) FROM {table}")[0][0]
        nb = b.store._read(f"SELECT COUNT(*) FROM {table}")[0][0]
        assert na == nb, f"{table}: {na} vs {nb}"


def test_a_chunk_costs_one_commit(mem):
    calls = _commits(mem)
    mem.index_many(_items(50), chunk_size=50)
    assert len(calls) == 1, f"expected one commit for the chunk, got {len(calls)}"


def test_chunking_bounds_the_transaction(mem):
    calls = _commits(mem)
    mem.index_many(_items(50), chunk_size=10)
    assert len(calls) == 5, (
        f"chunk_size must bound how much a crash rolls back; got {len(calls)} commits"
    )


def test_one_by_one_is_what_the_batch_improves_on(mem):
    """Anchors the claim: N notes cost N commits without the batch."""
    calls = _commits(mem)
    for it in _items(10):
        mem.index(it["node_id"], it["text"])
    assert len(calls) >= 10


# ── the whole point: unchanged content costs nothing ────────────────


def test_reindexing_unchanged_content_writes_nothing(mem):
    items = _items(30)
    mem.index_many(items)

    calls = _commits(mem)
    out = mem.index_many(items)

    assert out["indexed"] == 0, "re-indexed content that had not changed"
    assert out["skipped"] == 30
    assert len(calls) == 0, "an all-unchanged batch still hit the disk"


def test_only_the_changed_note_is_reindexed(mem):
    items = _items(30)
    mem.index_many(items)
    items[7]["text"] = "완전히 다른 본문"

    out = mem.index_many(items)

    assert out["indexed"] == 1
    assert out["skipped"] == 29


def test_a_moved_clock_is_a_touch_not_a_reindex(mem):
    """Same bytes, newer mtime. Re-deriving postings and edges for that is
    the exact waste this whole path exists to avoid."""
    items = _items(10)
    mem.index_many(items)
    later = time.time() + 60
    for it in items:
        it["updated_at"] = later

    out = mem.index_many(items)

    assert out["indexed"] == 0
    assert out["touched"] == 10
    assert all(v[0] == pytest.approx(later) for v in mem.manifest().values())


def test_a_touch_only_batch_costs_one_commit(mem):
    items = _items(40)
    mem.index_many(items)
    for it in items:
        it["updated_at"] = time.time() + 120

    calls = _commits(mem)
    mem.index_many(items, chunk_size=40)

    assert len(calls) == 1, f"touches were not batched ({len(calls)} commits)"


# ── correctness of derived state after a batch ──────────────────────


def test_notes_indexed_in_a_batch_are_searchable(mem):
    mem.index_many(
        [
            {"node_id": "k1", "text": "리듬게임 판정 보정 방법", "kind": "note"},
            {"node_id": "k2", "text": "저녁 식사 메뉴 기록", "kind": "note"},
        ]
    )

    hits = mem.search("리듬게임 판정", top_k=5)

    assert any(h.id == "k1" for h in hits), "batch-indexed note is not findable"


def test_knn_inside_a_batch_sees_earlier_members(tmp_path):
    """Sequential indexing links each note to the ones already there. If a
    batch derived every note against a frozen snapshot, the graph it builds
    would be sparser than the same notes added one at a time — a silent
    difference in retrieval quality, not an error anyone would see.

    So the assertion is equality with the sequential graph, edge for edge."""
    body = "판정 보정 오프셋 설정 리듬게임"
    items = [{"node_id": f"s{i}", "text": f"{body} {i}"} for i in range(12)]

    seq = SynapseMemory(SynapseConfig(path=str(tmp_path / "seq.db")))
    for it in items:
        seq.index(it["node_id"], it["text"])
    bat = SynapseMemory(SynapseConfig(path=str(tmp_path / "bat.db")))
    bat.index_many(items)

    def knn(m):
        return {(r[0], r[1]) for r in m.store._read("SELECT src, dst FROM edges WHERE etype=2")}

    assert knn(seq), "the fixture produces no kNN edges — test proves nothing"
    assert knn(bat) == knn(seq), "batch built a different graph"


def test_the_cache_matches_the_database_after_a_batch(mem):
    mem.index_many(_items(20))
    mem.search("리듬게임", top_k=3)  # populates caches
    mem.index_many([{"node_id": "late", "text": "새로 들어온 본문 리듬게임"}])

    hits = mem.search("새로 들어온", top_k=5)
    assert any(h.id == "late" for h in hits), "cache did not learn the batch"


# ── remove_many ─────────────────────────────────────────────────────


def test_remove_many_deletes_every_node_in_one_commit(mem):
    mem.index_many(_items(20))
    ids = list(mem.manifest())

    calls = _commits(mem)
    removed = mem.remove_many(ids)

    assert removed == 20
    assert mem.manifest() == {}
    assert len(calls) == 1, f"one fsync per deletion ({len(calls)} commits)"


def test_remove_many_clears_derived_rows(mem):
    """A node row deleted while its postings survive is worse than an orphan:
    the search still scores it and then cannot resolve it."""
    mem.index_many(_items(10))
    mem.remove_many(list(mem.manifest()))

    for table in ("postings_v2", "vectors", "edges", "node_map"):
        n = mem.store._read(f"SELECT COUNT(*) FROM {table}")[0][0]
        assert n == 0, f"{table} kept {n} rows for deleted nodes"


def test_remove_many_leaves_survivors_alone(mem):
    mem.index_many(_items(10))
    mem.remove_many(["n1", "n2", "n3"])

    assert set(mem.manifest()) == {"n0", "n4", "n5", "n6", "n7", "n8", "n9"}
    assert mem.search("리듬게임", top_k=5), "survivors became unsearchable"


def test_remove_many_on_an_empty_list_is_free(mem):
    calls = _commits(mem)
    assert mem.remove_many([]) == 0
    assert len(calls) == 0


# ── read/write concurrency ──────────────────────────────────────────


def test_searches_run_while_an_index_is_in_flight(tmp_path):
    """THE property behind splitting the lock.

    Under one mutex a search waited for every index — measured at 32.8 ms
    idle vs 165 ms during indexing on a production vault — and a write that
    wedged took every read with it, turning one stuck call into a total
    memory outage. Readers do not conflict; they must not queue.
    """
    import threading

    mem = SynapseMemory(SynapseConfig(path=str(tmp_path / "c.db")))
    mem.index_many(_items(30))

    searching = threading.Event()
    release_search = threading.Event()
    indexed = threading.Event()

    def searcher():
        with mem._lock.read():
            searching.set()
            release_search.wait(5)

    def indexer():
        mem.index("late", "새 본문 리듬게임")
        indexed.set()

    t1 = threading.Thread(target=searcher, daemon=True)
    t1.start()
    searching.wait(5)

    t2 = threading.Thread(target=indexer, daemon=True)
    t2.start()
    # A writer must WAIT for the in-flight reader — that half is unchanged.
    assert not indexed.wait(0.2)
    release_search.set()
    assert indexed.wait(5), "the write never completed after the read finished"
    t1.join(5)
    t2.join(5)


def test_two_searches_do_not_serialise(tmp_path):
    import threading

    mem = SynapseMemory(SynapseConfig(path=str(tmp_path / "d.db")))
    mem.index_many(_items(10))

    both = threading.Barrier(3, timeout=5)

    def searcher():
        with mem._lock.read():
            both.wait()  # unreachable if reads are exclusive

    threads = [threading.Thread(target=searcher, daemon=True) for _ in range(2)]
    for t in threads:
        t.start()
    both.wait()  # would time out under one mutex
    for t in threads:
        t.join(5)


def test_a_learning_step_does_not_deadlock_itself(tmp_path):
    """`feedback` and `learn` both update per-item trust while already
    holding the write lock. With a non-reentrant lock, calling the public
    `trust_feedback` there hangs the process — which is exactly what
    happened, on three tests at once, the moment the lock was split."""
    mem = SynapseMemory(SynapseConfig(path=str(tmp_path / "e.db"), epsilon=0.0))
    mem.index_many(_items(6))
    hits = mem.search("리듬게임", top_k=3)

    out = mem.feedback(hits[0].query_token, used_ids=[hits[0].id])

    assert out["applied"] >= 0.0
    assert mem.trust_feedback(hits[0].id, True) is not None


def test_reading_a_body_from_inside_a_read_does_not_deadlock(tmp_path):
    """`contradictions` runs under the read lock and needs several bodies."""
    mem = SynapseMemory(SynapseConfig(path=str(tmp_path / "f.db")))
    mem.index("a", "판정은 손이 먼저다")
    mem.index("b", "판정은 손이 먼저가 아니다")

    assert mem.contradictions("a", top_k=3) is not None
    assert mem.get_text("a")


def test_a_search_gives_up_on_a_wedged_write(tmp_path):
    """The blast-radius fix. Splitting the lock does not stop a stuck writer
    from blocking reads; the deadline does. A turn can answer without memory
    — it cannot answer while inheriting someone else's hang, which is how a
    single spinning matmul took conversations down for 27 hours."""
    import threading

    from xgen_agent_memory import MemoryBusy

    mem = SynapseMemory(SynapseConfig(path=str(tmp_path / "g.db")))
    mem.index_many(_items(5))

    holding = threading.Event()
    release = threading.Event()

    def wedged_writer():
        with mem._lock.write():
            holding.set()
            release.wait(10)

    t = threading.Thread(target=wedged_writer, daemon=True)
    t.start()
    holding.wait(5)

    with pytest.raises(MemoryBusy):
        mem.search("리듬게임", top_k=3, timeout=0.1)

    release.set()
    t.join(5)
    assert mem.search("리듬게임", top_k=3), "search never recovered"


def test_the_default_search_patience_is_sane(tmp_path):
    """Too short trips on a normal index chunk; too long is the hang."""
    mem = SynapseMemory(SynapseConfig(path=str(tmp_path / "h.db")))
    assert 5.0 <= mem.SEARCH_TIMEOUT_S <= 60.0


def test_a_caller_can_opt_out_of_the_deadline(tmp_path):
    """Callers that cannot degrade (a backfill verifying its own work) must
    be able to wait."""
    mem = SynapseMemory(SynapseConfig(path=str(tmp_path / "i.db")))
    mem.index_many(_items(3))
    assert mem.search("리듬게임", top_k=2, timeout=None) is not None
