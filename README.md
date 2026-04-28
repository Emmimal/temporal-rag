# temporal-rag
A post-retrieval temporal layer for RAG systems — validity filtering, time decay, document kind classification, and hybrid reranking in one pipeline.

# temporal-rag

A post-retrieval temporal layer for RAG systems — validity filtering, time decay, document kind classification, and hybrid reranking in one pipeline.

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Most RAG tutorials stop at: retrieve documents, stuff them into a prompt, call the model. But vector search has no concept of time. A policy document from two years ago scores identically to the one published last week — if it uses more of the same vocabulary, it wins.

This library adds the temporal layer that sits between your vector retriever and the LLM. No retraining, no re-indexing, no new infrastructure. It works on top of any vector store.

Read the full write-up on Towards Data Science → [RAG Has No Memory of Time — I Built a Temporal Layer to Fix It in Production](https://towardsdatascience.com/)

---

## What It Does

```
Query
  ↓
Retriever (vector similarity, top-20)         ← unchanged
  ↓
Temporal Layer                                ← this library
   ├── Validity filter     (hard-remove EXPIRED facts)
   ├── Kind classification (STATIC / VERSIONED / EVENT)
   ├── Time decay score    (0.5 ^ age / half_life)
   ├── Recency score       (normalized position in pool)
   ├── EVENT relevance gate (raw cosine floor)
   └── Hybrid reranker     (vector + temporal combined)
  ↓
Re-ranked context → LLM
```

Five components, one `retrieve()` call:

| Component | Job |
|---|---|
| Validity filter | Hard-remove EXPIRED documents before ranking begins |
| Kind classifier | STATIC / VERSIONED / EVENT — three problems, three fixes |
| Time decay | `0.5 ^ (age / half_life)` — exponential decay by document age |
| Recency scorer | Normalized freshness position within the candidate pool |
| EVENT gate | Raw cosine floor prevents fresh-but-irrelevant events from hijacking results |

---

## Installation

```bash
git clone https://github.com/Emmimal/temporal-rag.git
cd temporal-rag
pip install numpy
```

No other dependencies. All core functionality runs on the Python standard library + numpy. The built-in embedder uses a deterministic TF-IDF approach — no API key required to run the demo.

To use dense embeddings (OpenAI, sentence-transformers, Cohere), swap in your own `EmbeddingModel` subclass. See [Plugging In Your Embedder](#plugging-in-your-embedder).

---

## Quick Start

```python
from datetime import datetime, timedelta
from temporal_rag import Document, DocumentKind, EmbeddingModel, TemporalRAG, TemporalConfig

now = datetime.now()

docs = [
    Document(
        id="policy_v1",
        content="API rate limits are set to 100 requests per minute.",
        created_at=now - timedelta(days=540),
        valid_until=now - timedelta(days=180),
        kind=DocumentKind.VERSIONED,
    ),
    Document(
        id="policy_v2",
        content="API rate limits have been updated to 1000 requests per minute.",
        created_at=now - timedelta(days=175),
        valid_from=now - timedelta(days=180),
        kind=DocumentKind.VERSIONED,
        supersedes_id="policy_v1",
    ),
    Document(
        id="announcement_today",
        content="Rate limiting is suspended for the next 48 hours due to infrastructure upgrades.",
        created_at=now - timedelta(hours=6),
        valid_until=now + timedelta(hours=42),
        kind=DocumentKind.EVENT,
    ),
]

rag = TemporalRAG(
    embedding_model=EmbeddingModel(),
    temporal_config=TemporalConfig(
        decay_half_life_days=60,
        temporal_weight=0.40,
        enforce_validity=True,
    )
)

rag.index(docs)
results = rag.retrieve("What are the current API rate limits?", top_k=3)

for r in results:
    print(r.explain())
```

---

## Running the Demos

### Before/After comparison (the article scenarios)

```bash
python demo.py
```

Four queries. Same 10-document corpus. Naive RAG vs temporal RAG side by side. Output includes per-document scoring breakdown and a diff of what changed.

### Advanced patterns

```bash
python advanced.py
```

Covers domain-specific decay profiles, temporal query parsing, sequence-aware deduplication, and the freshness report API.

---

## Configuration Reference

```python
TemporalRAG(
    embedding_model=EmbeddingModel(),   # swap for OpenAI, sentence-transformers, etc.
    temporal_config=TemporalConfig(
        decay_half_life_days=30.0,      # score halves every N days
        temporal_weight=0.35,           # 0.0 = pure vector | 1.0 = pure recency
        enforce_validity=True,          # hard-remove documents past valid_until
        validity_boost=1.2,             # boost multiplier for active EVENT documents
        min_vector_score=0.15,          # normalized relevance floor (0.0 to disable)
        event_min_raw_vector_score=0.20 # raw cosine floor for EVENT boost (see note)
    ),
    candidate_pool_size=20,             # how many candidates the retriever returns
)
```

Tuning `hybrid_alpha` between vector and temporal:

| Content type | Suggested `temporal_weight` |
|---|---|
| Breaking news / outages | 0.65 – 0.70 |
| Policies / announcements | 0.40 – 0.50 |
| Research / documentation | 0.25 – 0.35 |
| Legal / reference | 0.10 – 0.25 |
| Mathematics / definitions | 0.01 – 0.10 |

> **Calibration note:** `event_min_raw_vector_score=0.20` is tuned for TF-IDF sparse embeddings. Dense models (e.g. `text-embedding-3-small`, `all-MiniLM-L6-v2`) produce higher absolute similarity scores — recalibrate to **0.35–0.50** for dense embeddings in production.

---

## Project Structure

```
temporal-rag/
├── temporal_rag.py     # Core: Document, ValidityState, DocumentKind,
│                       #       TemporalLayer, TemporalRAG, NaiveRAG
├── demo.py             # Before/after comparison — 4 scenarios, same corpus
├── advanced.py         # Decay profiles, query parsing, sequence dedup,
│                       #       freshness report API
└── README.md
```

---

## Performance (CPU only, 10-doc knowledge base)

| Operation | Latency |
|---|---|
| TF-IDF retrieval | ~2 ms |
| Temporal reranking (20 candidates) | ~15–30 ms |
| Full `retrieve()` call | ~20–35 ms |

Temporal reranking is negligible compared to LLM inference time. For larger corpora, the bottleneck will be your vector store, not this layer.

---

## Plugging In Your Embedder

```python
import numpy as np
from temporal_rag import EmbeddingModel

class OpenAIEmbedder(EmbeddingModel):
    def __init__(self, client, model="text-embedding-3-small"):
        self.client = client
        self.model = model

    def encode(self, text: str) -> np.ndarray:
        response = self.client.embeddings.create(input=text, model=self.model)
        vec = np.array(response.data[0].embedding)
        return vec / np.linalg.norm(vec)

    def encode_corpus(self, texts: list[str]) -> list[np.ndarray]:
        return [self.encode(t) for t in texts]

rag = TemporalRAG(embedding_model=OpenAIEmbedder(client))
```

---

## When to Use This

Worth it when you have:
- A knowledge base where documents get updated, superseded, or expire
- Time-sensitive operational content (announcements, outages, policy changes)
- Users asking present-tense questions against a corpus with mixed document ages

Skip it when you have:
- A fully static corpus that never changes
- Single-turn queries where all documents were ingested at the same time
- Hard latency requirements where even 20ms is unacceptable

---

## Known Limitations

**Implicit expiration.** The system handles documents with explicit temporal metadata. Most real-world documents don't have this — you need to assign it at index time. LLM-based metadata extraction at ingestion or rule-based heuristics by document source are both viable approaches.

**Conflicting sources.** The temporal layer ensures you get the freshest, most relevant documents. It does not resolve conflicts between them.

**Embedding model calibration.** The `event_min_raw_vector_score` threshold is embedding-model-specific. Calibrate against your actual model on representative queries before going to production.

**Half-life calibration.** The values in `DECAY_PROFILES` are reasonable starting points, not universal constants. Tune empirically per domain.

---

## License

MIT
