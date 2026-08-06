"""Real-vault deep test — index actual Geny prod agent notes into Synapse.

Parses Obsidian-style markdown (YAML frontmatter + body + [[wikilinks]] +
tags) exactly as Geny writes it, rebuilds the note graph, and runs a battery
of realistic agent-memory retrieval probes. Not a fixture — this is the
engine facing the messy Korean text a live agent actually produced (screen
observations, executions, digests) with inconsistent spacing, 조사, ko/en mix,
emoji, and long tails.

Usage: python eval/real_vault.py <notes_dir>
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from xgen_agent_memory import SynapseConfig, SynapseMemory  # noqa: E402

_FM = re.compile(r"^---\n(.*?)\n---\n(.*)$", re.DOTALL)
_LINK = re.compile(r"\[\[([^\]|]+)")


def parse_note(path: Path) -> dict:
    raw = path.read_text(encoding="utf-8", errors="replace")
    m = _FM.match(raw)
    title, tags, links, body = path.stem, [], [], raw
    if m:
        fm, body = m.group(1), m.group(2)
        for line in fm.splitlines():
            if line.startswith("title:"):
                title = line.split(":", 1)[1].strip().strip('"')
            elif line.startswith("tags:"):
                tags = re.findall(r"[\w가-힣]+", line.split(":", 1)[1])
    links = [x.strip() for x in _LINK.findall(raw)]
    return {"id": path.stem, "title": title, "tags": tags,
            "links": links, "text": body.strip()}


def load_notes(root: Path) -> List[dict]:
    return [parse_note(p) for p in sorted(root.rglob("*.md"))]


def build(notes: List[dict]) -> SynapseMemory:
    mem = SynapseMemory(SynapseConfig(path=":memory:", epsilon=0.0))
    ids = {n["id"] for n in notes}
    t0 = time.perf_counter()
    for n in notes:
        kind = ("digest" if n["id"].startswith("__digest") else
                "execution" if "execution" in n["id"] else "note")
        mem.index(n["id"], n["text"], title=n["title"], tags=n["tags"],
                  kind=kind, links=[x for x in n["links"] if x in ids])
    dt = time.perf_counter() - t0
    print(f"indexed {len(notes)} real notes in {dt:.1f}s ({1000*dt/len(notes):.1f} ms/note)")
    return mem


def show(mem: SynapseMemory, query: str, k: int = 5) -> None:
    t0 = time.perf_counter()
    hits = mem.search(query, top_k=k)
    ms = (time.perf_counter() - t0) * 1000
    print(f"\n[{ms:.1f}ms] Q: {query}")
    for i, h in enumerate(hits):
        title = h.title[:52].replace("\n", " ")
        print(f"  {i+1}. ({h.score:+.2f}) [{','.join(h.sources)}] {title}")


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "notes")
    notes = load_notes(root)
    mem = build(notes)
    st = mem.stats()
    print(f"graph: link={st['edges']['link']} tag={st['edges']['tag']} "
          f"knn={st['edges']['knn']}  | nodes={st['nodes']}")

    # Realistic agent-memory probes — natural Korean, varied phrasing.
    for q in [
        "화면에서 무슨 게임을 하고 있었어",
        "패스 오브 엑자일 아이템 가중치",
        "접두어 접미사 크래프팅 전략",
        "사용자가 코딩하던 내용",
        "카오스 특화 베이스",
        "리듬게임 판정",
        "메모리 검색 엔진 만들기",
    ]:
        show(mem, q)

    # Latency distribution over 100 varied queries built from note titles.
    import random
    rng = random.Random(1)
    sample_titles = [n["title"] for n in rng.sample(notes, min(100, len(notes)))]
    t0 = time.perf_counter()
    for t in sample_titles:
        mem.search(t[:40], top_k=8)
    print(f"\nlatency: {(time.perf_counter()-t0)/len(sample_titles)*1000:.1f} ms/query "
          f"over {len(sample_titles)} title-derived queries @ {st['nodes']} notes")
    mem.close()


if __name__ == "__main__":
    main()
