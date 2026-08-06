"""Micro-benchmark + learning-effect demo (not a pytest module).

Run: PYTHONPATH=src python tests/bench.py
"""

from __future__ import annotations

import random
import time

import numpy as np

from xgen_agent_memory import SynapseConfig, SynapseMemory

TOPICS = {
    "게임": ["리듬게임 판정 훈련 기록", "오토파일럿 손 반응 연습", "비트 정확도 통계 정리",
           "Sayonara Wild Hearts 한판 감상", "인터내셔널 도타 경기 메모"],
    "요리": ["김치찌개 돼지고기 레시피", "묵은지 활용 요리 아이디어", "설탕 간 맞추기 비법",
           "된장국 육수 내는 순서", "칼질 연습 일지"],
    "개발": ["asyncio 이벤트 루프 블로킹 디버깅", "SQLite WAL 체크포인트 튜닝",
           "파이썬 프로파일링 도구 비교", "웹소켓 재연결 백오프 설계", "도커 볼륨 권한 문제 해결"],
    "여행": ["오사카 야시장 방문 계획", "교토 사찰 코스 정리", "항공권 특가 알림 세팅",
           "숙소 취소 규정 비교", "환전 수수료 절약 팁"],
}


def build(n_docs: int) -> SynapseMemory:
    mem = SynapseMemory(SynapseConfig(path=":memory:", epsilon=0.0))
    rng = random.Random(7)
    t0 = time.perf_counter()
    for i in range(n_docs):
        topic = rng.choice(list(TOPICS))
        base = rng.choice(TOPICS[topic])
        mem.index(f"n{i}", f"{base} — 세부 메모 {i} {topic} 관련 내용",
                  title=f"{topic} 노트 {i}", tags=[topic])
    dt = time.perf_counter() - t0
    print(f"index: {n_docs} docs in {dt:.2f}s  ({1000 * dt / n_docs:.2f} ms/doc)")
    return mem


def bench_search(mem: SynapseMemory, queries: list[str], label: str) -> None:
    t0 = time.perf_counter()
    n = 50
    for i in range(n):
        mem.search(queries[i % len(queries)])
    dt = (time.perf_counter() - t0) / n * 1000
    print(f"search[{label}]: {dt:.2f} ms/query")


def learning_demo() -> None:
    """Ambiguous query, both groups shown; feedback should teach a preference."""
    mem = SynapseMemory(SynapseConfig(path=":memory:", epsilon=0.0, blend_min_events=30))
    # '판정' appears in game AND legal notes — lexically ambiguous.
    for i in range(6):
        mem.index(f"g{i}", f"리듬게임 판정 프레임 분석 {i}", title=f"게임판정 {i}", tags=["게임"])
        mem.index(f"l{i}", f"법원 판정 사례 요약 {i}", title=f"법률판정 {i}", tags=["법률"])

    def mean_game_rank(hits):
        ranks = [i for i, h in enumerate(hits) if h.id.startswith("g")]
        return sum(ranks) / len(ranks) if ranks else 99.0

    before_hits = mem.search("판정 기록", top_k=10)
    before = mean_game_rank(before_hits)
    # The user consistently USES the game notes for this ambiguous query.
    for _ in range(80):
        hits = mem.search("판정 기록", top_k=10)
        used = [h.id for h in hits if h.id.startswith("g")][:3]
        if used:
            mem.feedback(hits[0].query_token, used_ids=used)
    after_hits = mem.search("판정 기록", top_k=10)
    after = mean_game_rank(after_hits)
    st = mem.stats()
    print(f"learning: mean game rank {before:.1f} → {after:.1f} "
          f"(lower=better) blend={st['ranker']['blend']:.2f} "
          f"events={st['ranker']['events']:.0f} coaccess_edges={st['edges']['coaccess']}")
    print(f"learning: top3 after feedback = {[h.id for h in after_hits[:3]]}")


def distill_demo() -> None:
    mem = SynapseMemory(SynapseConfig(path=":memory:", epsilon=0.0,
                                      distill_epochs=12, distill_lr=5e-3))
    rng = np.random.default_rng(3)
    # 4 clusters, each with its own reusable vocabulary pool — tokens repeat
    # WITHIN a cluster (like real domain vocab), so the table can learn to
    # pull cluster vocab together purely from teacher geometry.
    pools = [[f"군집{c}어휘{w}" for w in range(12)] for c in range(4)]
    for i in range(120):
        c = i % 4
        words = rng.choice(pools[c], size=3, replace=False)
        base = np.zeros(32); base[c * 8] = 1.0
        mem.index(f"d{i}", " ".join(words), teacher_vec=base + rng.normal(0, 0.05, 32))
    t0 = time.perf_counter()
    m = mem.distill()
    print(f"distill: pairs={m.get('pairs'):.0f} swapped={m.get('swapped'):.0f} "
          f"corr {m.get('corr_before'):.3f}→{m.get('corr_after'):.3f} "
          f"in {time.perf_counter() - t0:.2f}s")


if __name__ == "__main__":
    for n in (500, 2000):
        mem = build(n)
        bench_search(mem, ["리듬게임 판정", "김치찌개 레시피", "asyncio 디버깅", "오사카 여행"],
                     label=f"{n} docs")
        mem.close()
    learning_demo()
    distill_demo()
