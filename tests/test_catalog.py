"""Progressive browsing — effect-proving tests.

A vault browser asks three questions in order, and each one should cost what
it is worth:

  how much is there   → a count
  what days are there → counts per day
  what is on this day → that day's metadata
  what does this say  → one body

The host's own note store answers the first by materialising the whole vault
— 3.2 s and 4.8 MB of bodies held, measured on a 5,384-note production vault
— because its only listing primitive walks every note. The index already
holds the metadata these questions need; SQL can group and page it without
reading a body at all.
"""

from __future__ import annotations

import time

import pytest

from xgen_agent_memory import SynapseConfig, SynapseMemory

DAY_A = time.mktime((2026, 8, 1, 12, 0, 0, 0, 0, -1))
DAY_B = time.mktime((2026, 8, 2, 12, 0, 0, 0, 0, -1))


@pytest.fixture
def vault(tmp_path):
    mem = SynapseMemory(SynapseConfig(path=str(tmp_path / "c.db"), epsilon=0.0))
    mem.index_many(
        [
            {
                "node_id": f"obs/{i}",
                "text": f"관찰 기록 {i}",
                "kind": "observations",
                "updated_at": DAY_A,
            }
            for i in range(7)
        ]
        + [
            {
                "node_id": f"obs/b{i}",
                "text": f"관찰 기록 b{i}",
                "kind": "observations",
                "updated_at": DAY_B,
            }
            for i in range(3)
        ]
        + [
            {"node_id": "note/keep", "text": "사람이 쓴 노트", "kind": "note", "updated_at": DAY_B},
        ]
    )
    return mem


def _no_bodies_read(mem, monkeypatch):
    """Trip if anything reaches for a body while answering."""

    def _boom(*_a, **_kw):
        raise AssertionError("a body was read to answer a metadata question")

    monkeypatch.setattr(mem, "get_text", _boom)
    monkeypatch.setattr(mem, "_get_text_unlocked", _boom)


# ── level 1: how much is there ──────────────────────────────────────


def test_counts_by_kind(vault, monkeypatch):
    _no_bodies_read(vault, monkeypatch)
    assert dict(vault.catalog_counts(by="kind")) == {"observations": 10, "note": 1}


def test_counts_by_day(vault, monkeypatch):
    _no_bodies_read(vault, monkeypatch)
    assert dict(vault.catalog_counts(by="day")) == {
        "2026-08-01": 7,
        "2026-08-02": 4,
    }


def test_counts_can_be_scoped_to_one_kind(vault):
    assert dict(vault.catalog_counts(by="day", kind="note")) == {"2026-08-02": 1}


def test_days_come_back_newest_first(vault):
    assert [d for d, _n in vault.catalog_counts(by="day")] == ["2026-08-02", "2026-08-01"]


def test_an_empty_vault_counts_nothing(tmp_path):
    mem = SynapseMemory(SynapseConfig(path=str(tmp_path / "e.db")))
    assert mem.catalog_counts(by="kind") == []
    assert mem.catalog_counts(by="day") == []


def test_an_unknown_grouping_is_an_error(vault):
    with pytest.raises(ValueError):
        vault.catalog_counts(by="colour")


# ── level 2: what is on this day ────────────────────────────────────


def test_a_day_returns_only_that_day(vault, monkeypatch):
    _no_bodies_read(vault, monkeypatch)
    page = vault.catalog_page(day="2026-08-02")
    assert len(page) == 4
    assert {r["id"] for r in page} == {"obs/b0", "obs/b1", "obs/b2", "note/keep"}


def test_a_page_carries_metadata_not_content(vault):
    row = vault.catalog_page(day="2026-08-01", limit=1)[0]
    assert set(row) == {"id", "kind", "title", "updated_at", "text_len", "pinned", "importance"}
    assert "text" not in row and "body" not in row


def test_paging_is_done_in_sql(vault):
    """Not "fetch everything then slice" — that is the pattern being
    replaced, and it is why asking for one note cost the whole vault."""
    first = vault.catalog_page(day="2026-08-01", limit=3, offset=0)
    second = vault.catalog_page(day="2026-08-01", limit=3, offset=3)
    assert len(first) == 3 and len(second) == 3
    assert {r["id"] for r in first}.isdisjoint({r["id"] for r in second})


def test_a_day_with_nothing_is_empty_not_everything(vault):
    assert vault.catalog_page(day="2020-01-01") == []


def test_a_page_can_be_scoped_to_a_kind(vault):
    page = vault.catalog_page(kind="note")
    assert [r["id"] for r in page] == ["note/keep"]


# ── level 3: the graph, at a screen's worth ─────────────────────────


def test_a_neighbourhood_is_not_the_whole_vault(vault):
    """The production snapshot was 5,384 nodes and 4.3 MB of JSON for one
    screen. A graph view is read at a screen's worth of detail."""
    out = vault.neighbourhood(["obs/0"], depth=1)
    assert out["nodes"], "the seed itself came back empty"
    assert len(out["nodes"]) < 11


def test_the_seed_is_always_present(vault):
    out = vault.neighbourhood(["note/keep"], depth=1)
    assert "note/keep" in {n["id"] for n in out["nodes"]}


def test_node_caps_are_honoured_and_reported(vault):
    out = vault.neighbourhood(["obs/0"], depth=2, max_nodes=3)
    assert len(out["nodes"]) <= 3
    assert isinstance(out["truncated"], bool)


def test_an_empty_seed_returns_an_empty_graph(vault):
    out = vault.neighbourhood([], depth=2)
    assert out == {"nodes": [], "edges": [], "truncated": False}


def test_edges_name_their_type(vault):
    out = vault.neighbourhood(["obs/0"], depth=1)
    for e in out["edges"]:
        assert set(e) == {"src", "dst", "type", "w"}
        assert isinstance(e["type"], int)
