"""Learning simulation — does the online ranker learn, and stay safe?

Three scenarios with GROUND TRUTH:

1. Genuine learning (ambiguous query, feature-correlated preference).
   Corpus has game-판정 and legal-판정 notes that collide on the word 판정.
   The user ALWAYS uses the game notes. The preference correlates with
   observable features (the game notes share a tag / co-occur), so the ranker
   CAN learn it. Expect: blend gate opens, game notes rise.

2. Noise robustness. Random feedback → the gate must stay CLOSED (the learner
   never beats the heuristic on noise, so λ=0 protects the floor).

3. Co-access (Hebbian). Two specific notes are always used together → a
   CO-ACCESS edge forms and lifts one when the other is retrieved.

Run: python eval/learning_sim.py
"""

from __future__ import annotations

import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from xgen_agent_memory import SynapseConfig, SynapseMemory  # noqa: E402


def build_ambiguous(mem: SynapseMemory, n: int = 10) -> None:
    """game-* and legal-* both contain 판정; game notes share the 게임 tag."""
    for i in range(n):
        mem.index(f"game-{i}", f"리듬게임 판정 프레임 분석 콤보 {i} 기록", title=f"게임 판정 {i}",
                  tags=["게임", "리듬"])
        mem.index(f"legal-{i}", f"법원 판정 판례 요약 조항 {i} 정리", title=f"법률 판정 {i}",
                  tags=["법률", "판례"])


# Varied query phrasings so the query-level holdout actually splits ~25% out.
_QUERIES = [f"판정 기록 정리 {w}" for w in "가나다라마바사아자차카타파하거너더러머버서"]


def game_ranks(mem) -> float:
    hits = mem.search(_QUERIES[0], top_k=20)
    rs = [i for i, h in enumerate(hits) if h.id.startswith("game")]
    return sum(rs) / len(rs) if rs else 99.0


def scenario_genuine() -> None:
    print("── 1. genuine learning (ambiguous 판정, user prefers game) ──")
    mem = SynapseMemory(SynapseConfig(path=":memory:", epsilon=0.0, blend_min_events=40))
    build_ambiguous(mem)
    before = game_ranks(mem)
    opened = None
    for step in range(1, 601):
        hits = mem.search(_QUERIES[step % len(_QUERIES)], top_k=20)
        used = [h.id for h in hits if h.id.startswith("game")][:3]
        if used:
            r = mem.feedback(hits[0].query_token, used_ids=used)
            if opened is None and r.get("blend", 0) > 0:
                opened = step
    after = game_ranks(mem)
    st = mem.stats()["ranker"]
    print(f"  game-note mean rank: {before:.2f} → {after:.2f} (lower=better)")
    print(f"  blend gate opened at #{opened}, final blend={st['blend']:.2f}, "
          f"win_rate={st['win_rate']:.2f} (b={int(st['disc_b'])},c={int(st['disc_c'])}), "
          f"events={int(st['events'])}")
    top3 = [h.id for h in mem.search("판정 기록 정리", top_k=3)]
    print(f"  top-3 after learning: {top3}")
    mem.close()


def scenario_noise() -> None:
    print("\n── 2. noise robustness (random feedback must NOT open the gate) ──")
    mem = SynapseMemory(SynapseConfig(path=":memory:", epsilon=0.0, blend_min_events=40))
    build_ambiguous(mem)
    rng = random.Random(0)
    for step in range(800):
        hits = mem.search(_QUERIES[step % len(_QUERIES)], top_k=20)
        if len(hits) >= 2:
            mem.feedback(hits[0].query_token, used_ids=[h.id for h in rng.sample(hits, 2)])
    st = mem.stats()["ranker"]
    verdict = "SAFE (floor protected)" if st["blend"] == 0 else "LEAK — investigate"
    print(f"  after {int(st['events'])} random feedbacks: blend={st['blend']:.2f}, "
          f"win_rate={st['win_rate']:.2f} (b={int(st['disc_b'])},c={int(st['disc_c'])}) → {verdict}")
    mem.close()


def scenario_coaccess() -> None:
    print("\n── 3. co-access (Hebbian edge lifts a co-used memory) ──")
    mem = SynapseMemory(SynapseConfig(path=":memory:", epsilon=0.0))
    # Two topically-distinct notes the user always uses together.
    mem.index("api-key", "API 키 발급 방법과 보관 위치", title="API 키", tags=["설정"])
    mem.index("deploy", "배포 스크립트 실행 순서와 롤백", title="배포 절차", tags=["운영"])
    for i in range(8):
        mem.index(f"filler-{i}", f"기타 메모 {i} 내용", title=f"메모 {i}", tags=["기타"])
    # Before: does 'API 키' retrieval surface 'deploy'? (unrelated lexically)
    before = [h.id for h in mem.search("API 키 발급", top_k=5)]
    for _ in range(15):
        h = mem.search("API 키 발급", top_k=5)
        mem.feedback(h[0].query_token, used_ids=["api-key", "deploy"])
    after_hits = mem.search("API 키 발급", top_k=5)
    after = [h.id for h in after_hits]
    dep = next((h for h in after_hits if h.id == "deploy"), None)
    st = mem.stats()
    print(f"  coaccess edges formed: {st['edges']['coaccess']}")
    print(f"  'API 키 발급' before: {before}")
    print(f"  'API 키 발급' after:  {after}")
    if dep:
        print(f"  → 'deploy' now surfaces via ppr_co={dep.features['ppr_co']:.3f}")
    mem.close()


def scenario_persistence() -> None:
    print("\n── 4. persistence round-trip ──")
    with tempfile.TemporaryDirectory() as td:
        db = str(Path(td) / "s.db")
        m1 = SynapseMemory(SynapseConfig(path=db, epsilon=0.0, blend_min_events=30))
        build_ambiguous(m1)
        for step in range(240):
            h = m1.search(_QUERIES[step % len(_QUERIES)], top_k=20)
            u = [x.id for x in h if x.id.startswith("game")][:3]
            if u:
                m1.feedback(h[0].query_token, used_ids=u)
        e1, b1 = int(m1.stats()["ranker"]["events"]), m1.stats()["ranker"]["blend"]
        m1.close()
        m2 = SynapseMemory(SynapseConfig(path=db, epsilon=0.0, blend_min_events=30))
        e2, b2 = int(m2.stats()["ranker"]["events"]), m2.stats()["ranker"]["blend"]
        ok = e1 == e2 and abs(b1 - b2) < 1e-6
        print(f"  events {e1}→{e2}, blend {b1:.2f}→{b2:.2f}  {'OK' if ok else 'MISMATCH'}")
        m2.close()


if __name__ == "__main__":
    scenario_genuine()
    scenario_noise()
    scenario_coaccess()
    scenario_persistence()
