"""geny-memory-adaptor — Synapse: a learnable, lightweight graph memory engine.

Semantic (local static embeddings) + keyword (BM25) + graph traversal
(typed-edge Personalized PageRank) fused by a tiny online-learned ranker.
One SQLite file, numpy-only, zero API calls, microsecond learning updates.

    from geny_memory_adaptor import SynapseMemory

    mem = SynapseMemory.open("vault/synapse.db")     # or .from_env()
    mem.index("id-1", "text …", title="…", tags=["…"], links=["id-0"])
    hits = mem.search("query", top_k=8)
    mem.feedback(hits[0].query_token, used_ids=["id-1"])   # online learning
    mem.distill()                                          # optional, w/ teachers
"""

from .config import SynapseConfig, load_dotenv
from .engine import SearchHit, SynapseMemory
from .executor_adapter import SynapseRetriever, SynapseVectorHandle
from .ranker import FEATURES

__version__ = "1.5.1"

__all__ = [
    "SynapseMemory",
    "SynapseConfig",
    "SearchHit",
    "SynapseVectorHandle",
    "SynapseRetriever",
    "load_dotenv",
    "FEATURES",
    "__version__",
]
