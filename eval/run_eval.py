"""Korean retrieval eval harness — offline, engine-config comparison.

Usage:
    python eval/run_eval.py                 # all configs, clean + all axes
    python eval/run_eval.py --config hybrid --axes clean,jamo

Metrics: Recall@1/5/10, MRR@10, nDCG@10 on the clean set; per-axis deltas
(vs the SAME queries' clean scores) for the six Korean robustness axes.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from xgen_agent_memory import SynapseConfig, SynapseMemory  # noqa: E402

DATA = Path(__file__).parent / "data"

#: Engine configurations to compare. Keys become report rows.
CONFIGS: Dict[str, dict] = {
    "bm25-only": {"vector_seed_k": 0},
    "vector-only": {"bm25_seed_k": 0},
    "hybrid": {},
    "hybrid+graph": {},  # graph edges only exist via tags/links here; kept for parity
}


def load_jsonl(path: Path) -> List[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def build_engine(overrides: dict, docs: List[dict]) -> SynapseMemory:
    mem = SynapseMemory(SynapseConfig(path=":memory:", epsilon=0.0, **overrides))
    for d in docs:
        mem.index(d["docid"], d["text"], title=d["title"])
    return mem


def evaluate(mem: SynapseMemory, queries: List[dict],
             positives: Dict[str, set], k: int = 10) -> Dict[str, Dict[str, float]]:
    """Per-query metrics {qid: {r1, r5, r10, mrr, ndcg}}."""
    out: Dict[str, Dict[str, float]] = {}
    for q in queries:
        gold = positives.get(q["qid"]) or set()
        if not gold:
            continue
        hits = mem.search(q["text"], top_k=k)
        ranked = [h.id for h in hits]
        first_rel = next((i for i, d in enumerate(ranked) if d in gold), None)
        dcg = sum(1.0 / math.log2(i + 2) for i, d in enumerate(ranked) if d in gold)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold), k)))
        out[q["qid"]] = {
            "r1": 1.0 if ranked[:1] and ranked[0] in gold else 0.0,
            "r5": 1.0 if any(d in gold for d in ranked[:5]) else 0.0,
            "r10": 1.0 if any(d in gold for d in ranked[:10]) else 0.0,
            "mrr": 1.0 / (first_rel + 1) if first_rel is not None else 0.0,
            "ndcg": dcg / idcg if idcg > 0 else 0.0,
        }
    return out


def mean(rows: Dict[str, Dict[str, float]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(r[key] for r in rows.values()) / len(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="run one config only")
    ap.add_argument("--axes", default=None, help="comma list; default clean+all")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args()

    docs = load_jsonl(DATA / "corpus.jsonl")
    queries = load_jsonl(DATA / "queries_clean.jsonl")
    positives: Dict[str, set] = defaultdict(set)
    for line in (DATA / "qrels.tsv").open(encoding="utf-8"):
        qid, _, docid, grade = line.split("\t")
        if int(grade) > 0:
            positives[qid].add(docid)

    axis_files = sorted((DATA / "perturbations").glob("*.jsonl")) \
        if (DATA / "perturbations").is_dir() else []
    wanted_axes = args.axes.split(",") if args.axes else None

    report: Dict[str, dict] = {}
    for name, overrides in CONFIGS.items():
        if args.config and name != args.config:
            continue
        t0 = time.perf_counter()
        mem = build_engine(overrides, docs)
        index_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        clean = evaluate(mem, queries, positives)
        search_ms = (time.perf_counter() - t0) / max(1, len(clean)) * 1000

        row = {
            "R@1": mean(clean, "r1"), "R@5": mean(clean, "r5"),
            "R@10": mean(clean, "r10"), "MRR@10": mean(clean, "mrr"),
            "nDCG@10": mean(clean, "ndcg"),
            "index_s": index_s, "ms/q": search_ms, "queries": len(clean),
        }

        for pf in axis_files:
            axis = pf.stem
            if wanted_axes and axis not in wanted_axes:
                continue
            variants = load_jsonl(pf)
            pert = evaluate(mem, variants, positives)
            base = {qid: clean[qid] for qid in pert if qid in clean}
            row[f"Δ{axis}"] = mean(pert, "r5") - mean(base, "r5")
        report[name] = row
        mem.close()

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=1))
        return
    cols = ["R@1", "R@5", "R@10", "MRR@10", "nDCG@10", "ms/q"]
    axis_cols = sorted({c for r in report.values() for c in r if c.startswith("Δ")})
    header = f"{'config':<14}" + "".join(f"{c:>9}" for c in cols + axis_cols)
    print(header)
    for name, row in report.items():
        line = f"{name:<14}"
        for c in cols + axis_cols:
            v = row.get(c)
            line += f"{v:>9.3f}" if isinstance(v, float) else f"{'—':>9}"
        print(line)


if __name__ == "__main__":
    main()
