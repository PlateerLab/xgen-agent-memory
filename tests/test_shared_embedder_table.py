"""One 64 MB table per process, not one per session.

Measured on a production host: an engine opened over an EMPTY vault
still cost 68 MB of RSS, and six live vaults held only TWO distinct
embedder tables between them. The table is `vocab_size × dim × 4B` —
64 MB at the defaults — so a host keeping ten sessions resident spent
640 MB on ten copies of identical numbers. That, not the stored
memories, was what made "keep every session awake" expensive.

Sharing is only safe because the table is never written in place: it is
read, and when distillation adopts a better one the whole embedder is
replaced by a scratch instance holding its own array. These tests pin
both halves of that — the sharing, and the immutability it rests on.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest

from xgen_agent_memory.embedder import (
    HashEmbedder,
    _TABLE_CACHE,
    _TABLE_CACHE_MAX,
    shared_table_stats,
)


@pytest.fixture(autouse=True)
def _clear_cache():
    _TABLE_CACHE.clear()
    yield
    _TABLE_CACHE.clear()


def test_same_parameters_share_one_array() -> None:
    """The whole point: two engines, one allocation."""
    a = HashEmbedder(4096, 32, seed=41)
    b = HashEmbedder(4096, 32, seed=41)
    assert a.table is b.table, (
        "two embedders with identical parameters each allocated their own "
        "table — this is the per-session 64 MB the change removes"
    )
    assert shared_table_stats()["tables"] == 1


def test_different_parameters_do_not_share() -> None:
    """Sharing must be keyed on what actually determines the table."""
    base = HashEmbedder(4096, 32, seed=41)
    assert HashEmbedder(4096, 32, seed=42).table is not base.table
    assert HashEmbedder(8192, 32, seed=41).table is not base.table
    assert HashEmbedder(4096, 64, seed=41).table is not base.table
    assert shared_table_stats()["tables"] == 4


def test_shared_table_is_read_only() -> None:
    """A shared array that anyone can write is a cross-session bug waiting
    to happen. Make the write fail loudly instead."""
    e = HashEmbedder(4096, 32, seed=41)
    with pytest.raises(ValueError):
        e.table[0, 0] = 1.0


def test_sharing_does_not_change_embeddings() -> None:
    """Behaviour must be byte-identical to the per-instance version."""
    a = HashEmbedder(4096, 32, seed=41)
    b = HashEmbedder(4096, 32, seed=41)
    for text in ("우주 전체 빛의 평균", "browsing files", "mixed 한글 and latin"):
        assert np.array_equal(a.embed(text), b.embed(text))
    # And still matches a table generated the old way, from the same seed.
    rng = np.random.default_rng(41)
    expected = (rng.standard_normal((4096, 32)) / np.sqrt(32)).astype(np.float32)
    assert np.array_equal(a.table, expected)


def test_persisted_tables_share_by_content() -> None:
    """Sessions that saved the same table decode it once between them."""
    src = HashEmbedder(4096, 32, seed=7)
    blob = src.dumps()
    x = HashEmbedder.loads(blob)
    y = HashEmbedder.loads(blob)
    assert x.table is y.table
    # A DIFFERENT saved table must not collide with it.
    other = HashEmbedder.loads(HashEmbedder(4096, 32, seed=8).dumps())
    assert other.table is not x.table


def test_adopting_a_distilled_table_leaves_the_shared_one_alone() -> None:
    """Copy-on-write in practice.

    `distill` builds its candidate on a scratch embedder and rebinds the
    attribute. The engine's other sessions must not see that.
    """
    shared = HashEmbedder(4096, 32, seed=41)
    peer = HashEmbedder(4096, 32, seed=41)
    original = np.array(shared.table, copy=True)

    scratch = HashEmbedder(4096, 32, seed=41)
    scratch.table = np.zeros((4096, 32), dtype=np.float32)  # what distill does

    assert peer.table is shared.table
    assert np.array_equal(shared.table, original), (
        "adopting a distilled table mutated the array other sessions read"
    )


def test_cache_is_bounded() -> None:
    """A dict keyed by content is a leak if content keeps changing."""
    for seed in range(_TABLE_CACHE_MAX + 4):
        HashEmbedder(256, 8, seed=seed)
    assert len(_TABLE_CACHE) <= _TABLE_CACHE_MAX


def test_stats_report_what_is_held() -> None:
    HashEmbedder(4096, 32, seed=41)
    s = shared_table_stats()
    assert s["tables"] == 1
    assert s["bytes"] == 4096 * 32 * 4
