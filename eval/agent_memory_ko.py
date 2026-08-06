"""Agent-memory eval set (Korean) — hand-authored, realistic, labeled.

Wiki passages (MIRACL) test encyclopedic retrieval; an AGENT'S memory is
different: short notes, preferences, decisions, schedules, tech breadcrumbs,
mixed Korean/English, and queries full of pronouns, paraphrase, and indirect
reference. This 24-note / 20-query set mirrors that, with gold labels, and
exercises the HARD cases word-level Korean search fails on: 조사/활용 변형,
띄어쓰기, 동의어·상위어 패러프레이즈, 한영 혼용, 간접 지시.

Run: python eval/agent_memory_ko.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from xgen_agent_memory import SynapseConfig, SynapseMemory  # noqa: E402

# (id, title, body, kind, tags)
NOTES = [
    ("m01", "사용자 호칭", "사용자는 '하렴 사장님'으로 불러달라고 했다.", "fact", ["선호"]),
    ("m02", "말투 선호", "반말 섞인 친근한 말투를 선호함. 너무 딱딱한 존댓말은 싫어함.", "preference", ["선호"]),
    ("m03", "커피 취향", "아메리카노를 좋아하고 오후 3시 이후엔 디카페인을 마신다.", "preference", ["선호"]),
    ("m04", "블로그 리뉴얼 결정", "hr_blog 2.0을 Next.js로 갈아엎기로 결정. 기존 Jekyll은 유지보수 중단.", "decision", ["프로젝트"]),
    ("m05", "블로그 배포 서버", "블로그 프로덕션은 116.47.69.209 서버의 2222 포트로 SSH 접속해서 배포한다.", "fact", ["프로젝트", "인프라"]),
    ("m06", "도커 재빌드 절차", "edit2docs 반영은 no-cache 로 docker compose build 후 up -d 한다.", "fact", ["인프라"]),
    ("m07", "메모리 엔진 프로젝트", "가볍고 학습 가능한 그래프 메모리 검색 엔진을 만들고 있다. 임베딩+BM25 하이브리드.", "project", ["프로젝트", "개발"]),
    ("m08", "PyPI 배포 방식", "GitHub 릴리스 → OIDC trusted publisher → PyPI 자동 게시 파이프라인을 쓴다.", "fact", ["개발"]),
    ("m09", "회의 일정", "매주 화요일 오전 10시에 팀 스탠드업 회의가 있다.", "schedule", ["일정"]),
    ("m10", "마감 기한", "분기 보고서는 이번 달 말일까지 제출해야 한다.", "schedule", ["일정"]),
    ("m11", "좋아하는 게임", "패스 오브 엑자일을 즐겨 하고, 카오스 특화 크래프팅을 연구 중이다.", "preference", ["게임"]),
    ("m12", "리듬게임 취향", "Sayonara Wild Hearts 같은 팝 리듬게임을 새벽에 즐긴다.", "preference", ["게임"]),
    ("m13", "개발 스택", "백엔드는 Python FastAPI, 프론트는 Next.js와 TypeScript를 쓴다.", "fact", ["개발"]),
    ("m14", "데이터베이스 선택", "세션 저장에는 PostgreSQL, 벡터 검색에는 Qdrant를 쓴다.", "fact", ["개발", "인프라"]),
    ("m15", "아침 루틴", "보통 아침 7시에 일어나 30분 조깅을 하고 샤워한다.", "fact", ["생활"]),
    ("m16", "여행 계획", "가을에 교토로 사찰 투어를 갈 계획을 세우고 있다.", "project", ["여행"]),
    ("m17", "김치찌개 레시피", "돼지고기와 묵은지로 끓이고 설탕을 약간 넣는 것이 포인트.", "note", ["요리"]),
    ("m18", "MCP 도구 연동", "접속기가 로컬 MCP 서버를 호스팅하고 백엔드로 카탈로그를 브릿지한다.", "fact", ["개발", "프로젝트"]),
    ("m19", "아바타 오버레이", "데스크톱 아바타는 always-on-top으로 전체화면 위에도 떠 있어야 한다.", "decision", ["프로젝트"]),
    ("m20", "GitHub 계정", "코드 푸시는 CocoRoF 계정으로 한다.", "fact", ["개발"]),
    ("m21", "글쓰기 스타일", "보고서는 결론부터 쓰고, 불필요한 hedging 없이 단정적으로 쓴다.", "preference", ["선호"]),
    ("m22", "운동 목표", "주 3회 근력 운동과 주말 등산을 목표로 한다.", "schedule", ["생활"]),
    ("m23", "선호 모델", "AI 애플리케이션은 최신 Claude 모델을 기본으로 쓴다.", "preference", ["개발"]),
    ("m24", "저장소 위치", "작업 파일은 /home/workspace 아래 프로젝트별 폴더에 둔다.", "fact", ["인프라"]),
]

# (query, gold_id, note-on-difficulty)
QUERIES = [
    ("나를 뭐라고 부르라고 했지", "m01", "간접 지시 + 활용"),
    ("어떤 말투로 얘기하는 걸 좋아해", "m02", "패러프레이즈"),
    ("커피 뭐 마셔", "m03", "축약 구어체"),
    ("블로그를 어떤 걸로 다시 만들기로 했어", "m04", "활용 변형"),
    ("블로그 서버에 어떻게 접속해", "m05", "동의어(배포서버→서버)"),
    ("도커 이미지를 다시 빌드하려면", "m06", "조사/활용"),
    ("지금 무슨 프로젝트 하고 있어", "m07", "간접"),
    ("파이썬 패키지 어떻게 배포해", "m08", "한영 혼용(PyPI)"),
    ("이번 주 회의 언제야", "m09", "동의어(스탠드업)"),
    ("보고서 언제까지 내야 돼", "m10", "패러프레이즈"),
    ("요즘 무슨 게임 해", "m11", "구어체"),
    ("새벽에 하는 리듬게임", "m12", "부분 매칭"),
    ("백엔드 뭐로 만들었어", "m13", "한영 혼용"),
    ("벡터 검색은 뭘 써", "m14", "부분 매칭"),
    ("아침에 보통 뭐 해", "m15", "패러프레이즈"),
    ("가을 여행 어디 가", "m16", "직접"),
    ("김치찌개 어떻게 끓여", "m17", "활용"),
    ("로컬 MCP 어떻게 연동돼", "m18", "한영 혼용"),
    ("아바타가 다른 창 위에 떠야 하는 이유", "m19", "패러프레이즈(always-on-top)"),
    ("코드 푸시 계정 뭐야", "m20", "축약"),
]


def main() -> None:
    mem = SynapseMemory(SynapseConfig(path=":memory:", epsilon=0.0))
    for nid, title, body, kind, tags in NOTES:
        mem.index(nid, body, title=title, kind=kind, tags=tags)

    r1 = mrr = ndcg = 0.0
    misses = []
    for query, gold, note in QUERIES:
        ranked = [h.id for h in mem.search(query, top_k=10)]
        rank = ranked.index(gold) if gold in ranked else None
        r1 += 1.0 if ranked[:1] == [gold] else 0.0
        mrr += 1.0 / (rank + 1) if rank is not None else 0.0
        ndcg += 1.0 / math.log2(rank + 2) if rank is not None else 0.0
        if rank != 0:
            misses.append((query, gold, note, rank, ranked[:3]))

    n = len(QUERIES)
    print(f"agent-memory KO set: {n} labeled queries / {len(NOTES)} notes")
    print(f"  R@1={r1/n:.3f}  MRR@10={mrr/n:.3f}  nDCG@10={ndcg/n:.3f}")
    if misses:
        print(f"\n  {len(misses)} not-at-1 (difficulty · gold-rank · top3):")
        for q, gold, note, rank, top3 in misses:
            rk = "miss" if rank is None else f"#{rank+1}"
            print(f"    [{rk}] {q}  → gold={gold}({note})  top3={top3}")
    mem.close()


if __name__ == "__main__":
    main()
