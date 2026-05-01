# temporal-rag
A post-retrieval temporal layer for RAG systems — validity filtering, time decay, document kind classification, and hybrid reranking in one pipeline.

# temporal-rag

A pure-Python temporal awareness layer for RAG systems — validity filtering,
time decay scoring, event boosting, and freshness-aware reranking in one
post-retrieval pipeline.

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Most RAG systems retrieve by semantic similarity alone. A document about API
rate limits from two years ago is just as "similar" to a rate limit query as
one from last week. If the older version has more matching tokens, it wins.
This library adds the temporal layer that sits between the vector retriever
and the LLM — deciding not just what is relevant, but what is **still true**.

Read the full write-up on Towards Data Science →
**[RAG Is Blind to Time — I Built a Temporal Layer to Fix It in Production](https://towardsdatascience.com/)**

---

## What It Does

```
Query → Retriever (vector similarity) → Temporal Layer → Re-ranked Context → LLM
                                              ↑
                        Validity filtering · Time decay · Event boosting
```

The temporal layer applies eight steps before any document reaches the model:

| Step | What It Does |
|---|---|
| Validity filter | Hard-removes EXPIRED documents before scoring begins |
| Kind classification | Labels each document STATIC / VERSIONED / EVENT |
| Time decay | Exponential decay: `0.5 ^ (age / half_life)` |
| Recency score | Normalized freshness within the candidate pool |
| Validity multiplier | Boosts active EVENT signals (×1.2), zeroes EXPIRED |
| EVENT relevance gate | Raw cosine floor prevents fresh-but-irrelevant events from hijacking results |
| Semantic penalty | Penalizes low-similarity docs regardless of freshness |
| Hybrid rerank | `(1−w) × vector_score + w × temporal_component` |

---

## The Core Idea: Two Orthogonal Axes

Time in a knowledge base is not one problem — it is three. Most systems
collapse all three into "stale documents" and apply the same treatment.
That is a heuristic, not a model.

**Axis 1 — Validity state (3 states)**

```
EXPIRED  → was true, is no longer true. Hard remove before ranking.
VALID    → true with no active time constraint. Normal scoring.
TEMPORAL → true within a currently active time window. Boost.
```

**Axis 2 — Document kind (3 types)**

```
STATIC    → timeless fact (definitions, math, reference material)
VERSIONED → replaced by newer information (policies, tutorials, specs)
EVENT     → true only within a time window (announcements, outages)
```

Only `DocumentKind.EVENT` documents can reach `ValidityState.TEMPORAL`.
A versioned policy with a `valid_from` date is still `VALID` — not `TEMPORAL`.
This distinction is what prevents policy_v2 from being mislabeled as a
time-bounded event.

---

## Before / After

Same 10-document corpus. Same query. Same embedder.

```
QUERY: What are the API rate limits? Will I get a 429 error?

NAIVE RAG
  1. [policy_v1]          age=540d | EXPIRED | sim=0.447
  2. [announcement_today] age=0d   | valid   | sim=0.329
  3. [tutorial_old]       age=600d | EXPIRED | sim=0.303

TEMPORAL RAG
  [announcement_today]  EVENT     ⚡ temporal  score=1.079
    reason: Active EVENT signal (42h remaining) — overrides static sources
  [policy_v2]           VERSIONED ✓ valid      score=0.573
    reason: Latest version — supersedes policy_v1
  [news_recent]         STATIC    ✓ valid      score=0.509
    reason: Fresh, open-ended fact — high confidence

  removed  : policy_v1, tutorial_old
  surfaced : policy_v2, news_recent
```

The expired document ranked first in naive RAG. It would tell a user they
hit 429 errors at 100 req/min when the actual limit is 1,000. Temporal RAG
leads with the live suspension announcement and follows with the current policy.

---

## Installation

```bash
git clone https://github.com/Emmimal/temporal-rag.git
cd temporal-rag
pip install numpy
```

No other dependencies. The demo runs without any API key using a deterministic
TF-IDF embedder — every output in the article is fully reproducible.

---

## Quick Start

```python
from temporal_rag import Document, DocumentKind, TemporalRAG, TemporalConfig
from datetime import datetime, timedelta

now = datetime.now()

docs = [
    Document(
        id="policy_v2",
        content="API rate limits updated to 1000 requests per minute.",
        created_at=now - timedelta(days=175),
        valid_from=now - timedelta(days=180),
        doc_type="policy",
        kind=DocumentKind.VERSIONED,
        supersedes_id="policy_v1",
    ),
    Document(
        id="announcement_today",
        content="Rate limiting suspended for 48 hours due to infrastructure upgrades.",
        created_at=now - timedelta(hours=6),
        valid_until=now + timedelta(hours=42),
        doc_type="announcement",
        kind=DocumentKind.EVENT,
    ),
]

config = TemporalConfig(
    decay_half_life_days=60,
    temporal_weight=0.40,
    enforce_validity=True,
    event_min_raw_vector_score=0.20,
)

rag = TemporalRAG(temporal_config=config)
rag.index(docs)

results = rag.retrieve("What are the current API rate limits?", top_k=3)
for r in results:
    print(r.explain())
```

---

## Running the Demos

```bash
# Before / after comparison — the four scenarios from the article
python demo.py

# Advanced patterns: decay profiles, adaptive weighting,
# freshness report, sequence-aware deduplication
python advanced.py
```

### demo.py — Before / After (4 scenarios)

| Scenario | What It Shows |
|---|---|
| 1 | Expired rate limit policy outranks current one in naive RAG |
| 2 | Old research finding (7B plateau) outranks the paper that overturns it |
| 3 | Stale layoff story surfaces without recovery news; EVENT gate blocks unrelated announcement |
| 4 | Live suspension announcement buried at position 3 behind expired policy |

### advanced.py — Extended Patterns (8 improvements)

| Improvement | What It Shows |
|---|---|
| 1 | PAIR execution — weak docs retrieve alongside a fresher partner |
| 2 | Confidence scoring with margin and conflict adjustments |
| 3 | Failure logging with per-query tracking |
| 4 | Adaptive conflict boost and confidence penalty |
| 5 | Time-range query parsing and filtering |
| 6 | Adaptive temporal weighting from query language signals |
| 7 | Freshness report API — kind-aware grades and recommendations |
| 8 | Sequence-aware deduplication — only policy_v3 reaches the LLM |

---

## Domain-Specific Decay Profiles

One half-life does not fit all content. `advanced.py` ships seven profiles:

```python
DECAY_PROFILES = {
    "breaking_news": TemporalConfig(decay_half_life_days=1,     temporal_weight=0.70),
    "news":          TemporalConfig(decay_half_life_days=7,     temporal_weight=0.55),
    "policy":        TemporalConfig(decay_half_life_days=90,    temporal_weight=0.45),
    "research":      TemporalConfig(decay_half_life_days=180,   temporal_weight=0.35),
    "legal":         TemporalConfig(decay_half_life_days=365,   temporal_weight=0.25),
    "reference":     TemporalConfig(decay_half_life_days=1825,  temporal_weight=0.10),
    "mathematics":   TemporalConfig(decay_half_life_days=36500, temporal_weight=0.01),
}
```

A proof from 1950 is as valid as one from last week.
A breaking news item from yesterday may already be wrong.

---

## Freshness Report API

```python
from advanced import freshness_report

print(freshness_report(doc))
```

```
current_policy [VERSIONED]
  age            : 45.0 days
  decay score    : 0.3536
  validity state : VALID
  grade          : D — stale
  recommendation : Check for a newer version — VERSIONED document may have been replaced.
```

Recommendations are kind-aware, not just score-aware. A STATIC document at
near-zero decay gets a supersession warning. A VERSIONED document gets a
version-check warning. An active EVENT gets a window-expiry flag.

---

## Adaptive Weighting from Query Language

```python
from advanced import adaptive_retrieve

config, weight, signal = adaptive_retrieve("What is the current rate limit?")
# weight=0.70 — "current" triggers recency-heavy scoring

config, weight, signal = adaptive_retrieve("How does cosine similarity work?")
# weight=0.20 — no signal, baseline applies
```

```python
TEMPORAL_SIGNALS = [
    (r"\b(current|latest|now|today|right now)\b",  0.70),
    (r"\b(this week|this month|recently)\b",        0.55),
    (r"\b(still|anymore|yet|has .+ changed)\b",     0.50),
    (r"\b(new|updated|changed|revised)\b",          0.40),
    (r"\b(best|recommend|should I)\b",              0.35),
]
```

---

## Project Structure

```
temporal-rag/
├── temporal_rag.py   # Core: Document, DocumentKind, ValidityState,
│                     #       TemporalConfig, TemporalLayer, TemporalRAG, NaiveRAG
├── demo.py           # Four before/after scenarios from the article
└── advanced.py       # Eight extended patterns: decay profiles, adaptive
                      # weighting, freshness report, sequence deduplication
```

---

## Scoring Formula

```
final_score = semantic_penalty
            × [ (1 − w) × vector_score
                + w × (decay_score × recency_score
                       × validity_multiplier × event_relevance_multiplier) ]
```

| Component | Role |
|---|---|
| `vector_score` | Cosine similarity, normalized to [0, 1] within the pool |
| `decay_score` | `0.5 ^ (age / half_life)` — approaches 0 for old documents |
| `recency_score` | Normalized position: 1.0 for newest, 0.0 for oldest |
| `validity_multiplier` | EXPIRED=0.0 · VALID=1.0 · TEMPORAL=1.2 |
| `event_relevance_multiplier` | Raw cosine floor for EVENT documents (0.5× if below) |
| `semantic_penalty` | 0.3× if normalized score below relevance threshold |
| `w` | `temporal_weight` — balance between vector and temporal signal |

---

## Calibration Notes

**event_min_raw_vector_score** — The default `0.20` is tuned for TF-IDF
sparse embeddings as used in the demo. Dense models (text-embedding-3-small,
all-MiniLM-L6-v2) produce higher absolute similarity scores. Recalibrate to
`0.35–0.50` for dense embeddings in production.

**temporal_weight** — Defaults to `0.35`. The demo uses `0.40`. Higher values
favour recency; lower values favour semantic similarity. Tune per deployment.

**decay_half_life_days** — The values in `DECAY_PROFILES` are starting points,
not universal constants. Tune empirically per domain.

---

## Implementation Cost

| Concern | Detail |
|---|---|
| Latency | ~15–30ms reranking overhead on a 20-candidate pool |
| Retriever changes | Zero — the layer sits downstream of any vector search |
| Infrastructure | No new services — pure Python post-processing |
| Required metadata | `created_at` minimum; `valid_from`, `valid_until`, `kind` unlock the full benefit |

Documents without temporal metadata are treated as `STATIC` and `VALID` —
degrades gracefully to standard time-decay scoring.

---

## Known Limitations

- **Implicit expiration.** The system handles documents with explicit temporal metadata. Assign `kind` and expiry windows at index time. LLM-based metadata extraction at ingestion is one practical approach; rule-based heuristics by document source is another.
- **Conflicting sources.** The temporal layer surfaces the freshest, most relevant documents. It does not resolve semantic conflicts between them.
- **Embedding model calibration.** The `event_min_raw_vector_score` threshold is embedding-model-specific. Calibrate before going to production.
- **Half-life calibration.** `DECAY_PROFILES` values are empirically reasonable starting points. Tune per domain.

---

## License

MIT
