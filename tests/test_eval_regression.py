"""Quality regression gate — MIRACL-ko fixture, hard thresholds.

Runs the committed Korean eval fixture (2,835 human-judged wiki passages,
Apache-2.0) with a query subset and asserts the engine never regresses below
the v1.0.0 measured floor. Full-corpus indexing dominates the runtime
(~15 s) — acceptable as the single slow test that guards retrieval quality.

Measured v1.0.0 (full 213 queries): nDCG@10 0.591 · MRR@10 0.595 · R@5 0.765.
Thresholds leave ~4pt slack for subset variance, NOT for regressions — a
change that eats the slack should be treated as a failure in review.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import pytest

from xgen_agent_memory import SynapseConfig, SynapseMemory

DATA = Path(__file__).resolve().parents[1] / "eval" / "data"

N_QUERIES = 80
NDCG_FLOOR = 0.55
MRR_FLOOR = 0.55
R5_FLOOR = 0.70


@pytest.mark.skipif(not (DATA / "corpus.jsonl").is_file(), reason="eval fixture not present")
def test_miracl_ko_quality_floor():
    docs = [json.loads(x) for x in (DATA / "corpus.jsonl").open(encoding="utf-8")]
    queries = [json.loads(x) for x in (DATA / "queries_clean.jsonl").open(encoding="utf-8")][
        :N_QUERIES
    ]
    positives = defaultdict(set)
    for line in (DATA / "qrels.tsv").open(encoding="utf-8"):
        qid, _, docid, grade = line.split("\t")
        if int(grade) > 0:
            positives[qid].add(docid)

    mem = SynapseMemory(SynapseConfig(path=":memory:", epsilon=0.0))
    for d in docs:
        mem.index(d["docid"], d["text"], title=d["title"])

    ndcg = mrr = r5 = n = 0.0
    for q in queries:
        gold = positives.get(q["qid"]) or set()
        if not gold:
            continue
        ranked = [h.id for h in mem.search(q["text"], top_k=10)]
        n += 1
        first = next((i for i, d in enumerate(ranked) if d in gold), None)
        mrr += 1.0 / (first + 1) if first is not None else 0.0
        r5 += 1.0 if any(d in gold for d in ranked[:5]) else 0.0
        dcg = sum(1.0 / math.log2(i + 2) for i, d in enumerate(ranked) if d in gold)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold), 10)))
        ndcg += dcg / idcg if idcg else 0.0
    mem.close()

    assert n >= 60
    assert ndcg / n >= NDCG_FLOOR, f"nDCG@10 {ndcg / n:.3f} < {NDCG_FLOOR}"
    assert mrr / n >= MRR_FLOOR, f"MRR@10 {mrr / n:.3f} < {MRR_FLOOR}"
    assert r5 / n >= R5_FLOOR, f"R@5 {r5 / n:.3f} < {R5_FLOOR}"
