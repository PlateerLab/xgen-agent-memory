# xgen-agent-memory

**Synapse** — a learnable, lightweight graph-traversal memory engine for AI agents.

Semantic search (local static embeddings) **+** keyword search (BM25) **+** graph
traversal (typed-edge Personalized PageRank), fused by a tiny **online-learned
ranker** that adapts to how your agent actually uses its memories. Everything
lives in **one SQLite file**. `numpy` is the only dependency. **Zero API calls,
zero servers, zero idle cost** — every operation is CPU milliseconds.

```
pip install xgen-agent-memory
```

## Why

Typical agent memory stacks pay an embedding API call per write AND per query,
need a vector server, and never learn: every ranking weight is a hard-coded
constant. Synapse inverts all three:

| | typical stack | Synapse |
|---|---|---|
| query cost | 1 embedding API call + vector store op | **0 API calls, ~3–9 ms CPU** (500–2,000 docs) |
| write cost | 1 embedding API call + index rebuild | **~0.3–0.7 ms**, incremental SQLite upserts |
| infrastructure | vector DB / server | **one .db file** (+ a 32 MB fp16 table) |
| learning | none | **online, per-event, microseconds** |

## Quick start

```python
from xgen_agent_memory import SynapseMemory

mem = SynapseMemory.open("vault/synapse.db")          # or SynapseMemory.from_env()

# Write — idempotent by id. Links/tags feed the graph.
mem.index("note-1", "리듬게임에서 판정을 읽는 건 손이 먼저다",
          title="리듬게임 판정", tags=["게임"], links=["note-0"])

# Read — BM25 ∪ cosine seeds → per-type PageRank expansion → learned ranking.
hits = mem.search("리듬게임 판정", top_k=8)
for h in hits:
    print(h.id, round(h.score, 3), h.sources)   # e.g. ['bm25', 'vector', 'graph']

# Learn — tell it which shown memories were actually used.
mem.feedback(hits[0].query_token, used_ids=["note-1"])

# Optionally distill: if you already have better embeddings lying around
# (e.g. previously paid-for API vectors), hand them in as teachers…
mem.index("note-2", "text …", teacher_vec=my_stored_api_embedding)
mem.distill()   # bounded batch job; swaps the table ONLY if geometry improves

mem.close()
```

### Configuration

Everything is a constructor argument, an environment variable (`GMA_*`), or a
`.env` file — in that precedence order:

```python
from xgen_agent_memory import SynapseConfig, SynapseMemory
mem = SynapseMemory(SynapseConfig(path="synapse.db", dim=256, top_k=8))
# or
mem = SynapseMemory.from_env(dotenv=".env")   # GMA_PATH, GMA_DIM, GMA_TOP_K, …
```

## How it works

```
write  ─ tokenize (words + char 2/3-grams; Korean-friendly, no morphology dep)
       ─ BM25 postings + local hash embedding (65,536 × 256 fp16 table)
       ─ typed edges: LINK (explicit) · TAG (IDF-weighted) · KNN (semantic)

query  ─ ① seeds: BM25 top-k ∪ cosine top-k, RRF-fused
       ─ ② expansion: one Personalized PageRank per edge type
            (LINK / TAG / KNN / CO-ACCESS) → 4 graph features
       ─ ③ ranking: 14 features → Linear(14→16) → GELU → Linear(16→1)

learn  ─ Hebbian: memories retrieved together AND both used strengthen a
         CO-ACCESS edge (lazy decay, no maintenance jobs)
       ─ ranker: pairwise logistic SGD on used-vs-ignored, with a safety
         blend gate — the learned score only participates once it BEATS the
         built-in heuristic on live online AUC (performance floor guarantee)
       ─ distill: fit the embedding table to caller-provided teacher vectors
         (closed feedback loop with a geometry-improvement gate)
```

Design notes:

- **The heuristic floor.** Scoring starts as a fixed hand-tuned linear
  combination. The MLP's influence (λ) stays 0 until it has ≥100 labelled
  events *and* a better pairwise win-rate — learning can help, never regress.
- **Per-type PPR instead of a GNN.** Each edge type gets its own PageRank
  feature; the ranker learns the type weights implicitly. No differentiable
  graph machinery, fully interpretable, milliseconds on thousands of nodes.
- **Derived data only.** The SQLite file is an index, not a store of record —
  delete it any time and re-`index()` from your source of truth.

## geny-executor integration

`xgen_agent_memory.executor_adapter` ships duck-typed handles matching
geny-executor's memory Protocols (no import of geny-executor):

```python
from xgen_agent_memory import SynapseMemory, SynapseVectorHandle
handle = SynapseVectorHandle(SynapseMemory.open("vault/synapse.db"))
# → FileMemoryProvider(vector_store=handle)  # drop-in local vector layer
```

## Korean retrieval (v1.0.0 — the headline feature)

Korean is a first-class citizen, engineered from NTCIR/ACL evidence and
measured on a committed [MIRACL-ko](https://huggingface.co/datasets/miracl/miracl)
harness (213 human-judged queries / 2,835 wiki passages, Apache-2.0):

- **No morphological analyzer needed**: overlapping syllable **bigrams**
  (the NTCIR-validated recipe; trigrams measured −2.5 nDCG and are excluded),
  a **guarded 조사/어미 stripper** (받침-agreement gate, '의' protected,
  하다-family only — +1.6 nDCG), **cross-space bigrams** for 붙여쓰기, and
  padded **jamo 3/5-grams in the embedding stream only** (+1.8 nDCG; keeping
  jamo out of BM25 avoids a measured 3× slowdown).
- **Hybrid nDCG@10 0.593 · MRR@10 0.597 · R@10 0.911** on the harness —
  near analyzer-based BM25 (nori ≈ 0.64 on comparable data), pure Python.
- Robustness axes measured (Δ R@5 vs clean): 붙여쓰기 ±0.00 · 조사 변형
  +0.02 · 자모 오타 −0.02 — the failure modes that break word-level Korean
  search simply don't apply.
- CI regression gate pins the quality floor (nDCG ≥ 0.55 on the fixture).

Run it yourself: `python eval/run_eval.py` (offline) · `python eval/ablation.py`.

## Benchmarks (single CPU core, this repo's `tests/bench.py`)

```
index : 500 docs 0.32 ms/doc · 2,000 docs 0.71 ms/doc
search: ~3–16 ms/query at 500–2,800 docs (jamo embedding stream included)
learn : ambiguous-query demo — mean target rank 7.0 → 2.8 after 80 feedbacks
distill: 120 pairs, geometry corr 0.975 → 0.983, gate-approved swap, 8.5 s
```

## License

Apache-2.0
