# temporal-rag
A post-retrieval temporal layer for RAG systems — validity filtering, time decay, and freshness tracking that runs downstream of any vector search system.

![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Dependencies](https://img.shields.io/badge/dependencies-numpy%20only-brightgreen)

Most RAG tutorials stop at: retrieve documents, rank by cosine similarity, send to the model. This library handles what comes next — deciding whether a retrieved document is still true, whether it has been superseded, and whether a fresh signal is actually relevant to the query being asked.

Read the full write-up on Towards Data Science → **RAG Is Blind to Time — I Built a Temporal Layer to Fix It in Production**

---

## What It Does

```
Query → Vector Retriever → Temporal Layer → Re-ranked Context → LLM
                                ↑
              validity filter · kind classifier · decay scorer
              recency scorer · event relevance gate · hybrid reranker
```

Three files, one `retrieve()` call:

| Component | Job |
|---|---|
| **Validity filter** | Hard-removes EXPIRED documents before any scoring |
| **Kind classifier** | Labels every document STATIC / VERSIONED / EVENT |
| **Decay scorer** | Exponential decay: `0.5 ^ (age / half_life)` |
| **Recency scorer** | Normalised freshness position within the candidate pool |
| **EVENT relevance gate** | Raw cosine floor — freshness cannot override relevance |
| **Hybrid reranker** | Combines vector similarity with all temporal signals |

Advanced patterns (in `advanced.py`):

| Pattern | Job |
|---|---|
| **PAIR executor** | Weak documents retrieved only alongside a fresher partner |
| **Confidence tiers** | HIGH / MEDIUM / LOW based on score margin and conflict |
| **Failure logging** | Rejection codes keyed by `query_id` for full auditability |
| **Conflict detection** | Severity-aware boost and confidence penalty when facts change |
| **Time-range filter** | Parses date windows from query text and applies a hard filter |
| **Adaptive weighting** | Adjusts `temporal_weight` based on recency signals in the query |
| **Freshness report** | Kind-aware grade and recommendation per document |
| **Sequence deduplication** | Collapses version chains before candidates reach the LLM |

---

## Installation

```bash
git clone https://github.com/Emmimal/temporal-rag.git
cd temporal-rag
pip install numpy
```

No other dependencies. Everything runs on the Python standard library and numpy.
No API key required — the demo uses a deterministic TF-IDF embedder so all output is reproducible.

---

## Quick Start

```python
from temporal_rag import Document, DocumentKind, EmbeddingModel, TemporalRAG, TemporalConfig
from datetime import datetime, timedelta

now = datetime.now()

docs = [
    Document(
        id="policy_v1",
        content="API rate limits are set to 100 requests per minute.",
        created_at=now - timedelta(days=540),
        valid_until=now - timedelta(days=180),   # expired
        doc_type="policy",
        kind=DocumentKind.VERSIONED,
    ),
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

**Output:**
```
[announcement_today]
  kind         : EVENT
  state        : ⚡ temporal (active)
  window       : 42h remaining
  reason       : Active EVENT signal (42h remaining) — overrides static sources
  FINAL SCORE  : 1.079

[policy_v2]
  kind         : VERSIONED
  state        : ✓ valid
  reason       : Latest version — supersedes policy_v1
  FINAL SCORE  : 0.573
```

`policy_v1` never reaches the model. It was expired before ranking began.

---

## Running the Demos

Two runnable scripts covering the full system:

```bash
python demo.py       # four before/after scenarios: naive RAG vs temporal RAG
python advanced.py   # eight production patterns
```

### demo.py — Before / After Comparison

| Scenario | What it shows |
|---|---|
| 1 — API rate limits | Expired policy outranks current one in naive RAG |
| 2 — LLM scaling research | Old finding outranks the newer one that overturns it |
| 3 — Company health | Only old news surfaces; recovery story never appears |
| 4 — Live outages | Active announcement buried at position 3 behind expired policy |

Sample output — Scenario 1:

```
QUERY: What are the API rate limits? Will I get a 429 error?

❌  NAIVE RAG
  1. [policy_v1]          age=540d | EXPIRED | sim=0.447
  2. [announcement_today] age=0d   | valid   | sim=0.329
  3. [tutorial_old]       age=600d | EXPIRED | sim=0.303

✅  TEMPORAL RAG
  [announcement_today]  EVENT    temporal   FINAL SCORE: 1.079
    reason: Active EVENT signal (42h remaining) — overrides static sources

  [policy_v2]           VERSIONED valid      FINAL SCORE: 0.573
    reason: Latest version — supersedes policy_v1

  removed  : ['policy_v1', 'tutorial_old']
  surfaced : ['policy_v2', 'news_recent']
```

### advanced.py — Eight Production Patterns

```
── Improvement 1: PAIR execution ──
  [Invalid] research_old     decay=0.100  → DO NOT RETRIEVE
  [Weak]    research_weak    decay=0.351  → PAIR WITH research_fresh (gain=+0.540)
  [Good]    research_fresh   decay=0.891  → RETRIEVE

── Improvement 2: Confidence tiers ──
  policy_v3 — clear winner    confidence 0.7485 → HIGH
  policy_v3 — with conflict   confidence 0.4727 → LOW
  math_theorem                confidence 0.6992 → MEDIUM

── Improvement 3: Failure logging ──
  EXPIRED_VERSIONED_DOC   × 1   doc=expired_policy
  STALE_STATIC_DOC        × 1   doc=stale_reference
  BELOW_RELEVANCE_GATE    × 1   doc=fresh_irrelevant

── Improvement 4: Conflict severity ──
  '100' → '5000'   severity=0.980   boost=+0.196   conf_pen=-0.098  (50× — severe)
  '1000' → '500'   severity=0.500   boost=+0.100   conf_pen=-0.050

── Improvement 5: Time-range filter ──
  'Show me research from 2021-2023'  → kept: research_2022
  'What were the findings in 2019?'  → kept: research_2019
  'Latest embeddings research'       → no filter, all docs pass

── Improvement 6: Adaptive weighting ──
  'What is the current rate limit?'  → temporal_weight: 0.70
  'Has the rate limit changed recently?' → temporal_weight: 0.55
  'How does cosine similarity work?' → temporal_weight: 0.20 (baseline)

── Improvement 7: Freshness report ──
  fresh_event    [EVENT]     grade: A → Verify before serving, window closes soon
  current_policy [VERSIONED] grade: D → Check for a newer version
  math_theorem   [STATIC]    grade: F → May have been superseded

── Improvement 8: Sequence deduplication ──
  Input : policy_v1 (v1), policy_v2 (v2), policy_v3 (v3)
  policy_v1 — EXPIRED → removed
  policy_v2 — superseded by v3 → removed
  policy_v3 — kept ✓
  Result: ['policy_v3']
```

---

## The Two Classification Axes

The core design separates temporal classification into two independent axes.

### Axis 1 — Validity State

| State | Meaning | Action |
|---|---|---|
| `EXPIRED` | Was true, is no longer | Hard removed before ranking |
| `VALID` | True with no active time constraint | Normal scoring |
| `TEMPORAL` | True within a currently active window | Boosted (×1.2) |

Only `EVENT` documents can reach `TEMPORAL` state. A versioned policy with a `valid_from` date is still `VALID` — its ranking is handled by time decay, not a validity window boost. Without this distinction, `policy_v2` looks identical to a time-bounded announcement and gets mislabeled.

### Axis 2 — Document Kind

| Kind | Meaning | Decay behaviour |
|---|---|---|
| `STATIC` | Timeless fact (definitions, math, reference) | Very slow |
| `VERSIONED` | Replaced by a newer document in a chain | Moderate — time decay handles ranking |
| `EVENT` | True only within a time window (announcements, outages) | N/A — window-based, not age-based |

---

## The Scoring Formula

```
final_score = semantic_penalty
            × [(1 − w) × vector_score
               + w × (decay_score × recency_score
                      × validity_multiplier × event_relevance_multiplier)]
```

| Component | What it does |
|---|---|
| `vector_score` | Cosine similarity, normalised to [0, 1] within the pool |
| `decay_score` | `0.5 ^ (age_in_days / half_life_days)` |
| `recency_score` | Normalised position: 1.0 = newest, 0.0 = oldest in pool |
| `validity_multiplier` | EXPIRED=0.0 · VALID=1.0 · TEMPORAL=1.2 |
| `event_relevance_multiplier` | Raw cosine floor for EVENT docs — halved if below threshold |
| `semantic_penalty` | 0.3× if normalised score below minimum threshold |
| `w` | `temporal_weight` — balance between vector and temporal signals |

---

## Configuration Reference

```python
TemporalConfig(
    decay_half_life_days=30.0,         # Score halves every N days
    temporal_weight=0.35,              # 0.0 = pure vector | 1.0 = pure recency
    max_age_days=None,                 # Hard age cutoff (None = disabled)
    enforce_validity=True,             # Hard-remove EXPIRED documents
    validity_boost=1.2,                # Multiplier for active EVENT documents
    min_vector_score=0.15,             # Normalised relevance floor (all kinds)
    event_min_raw_vector_score=0.20,   # Raw cosine floor for EVENT boost
)
```

**Tuning `temporal_weight`:**

| Query type | Suggested weight |
|---|---|
| "What is the **current** rate limit?" | 0.70 |
| "Has the policy changed **recently**?" | 0.55 |
| "How does cosine similarity work?" | 0.20 (baseline) |

**Tuning `event_min_raw_vector_score`:**

| Embedding type | Suggested floor |
|---|---|
| TF-IDF / sparse | 0.20 |
| Dense (text-embedding-3-small, all-MiniLM-L6-v2) | 0.35 – 0.50 |

---

## Domain-Specific Decay Profiles

One half-life does not fit all content types. `advanced.py` includes pre-configured profiles:

```python
DECAY_PROFILES = {
    "breaking_news": half_life=1d,     temporal_weight=0.70,
    "news":          half_life=7d,     temporal_weight=0.55,
    "policy":        half_life=90d,    temporal_weight=0.45,
    "research":      half_life=180d,   temporal_weight=0.35,
    "legal":         half_life=365d,   temporal_weight=0.25,
    "reference":     half_life=1825d,  temporal_weight=0.10,
    "mathematics":   half_life=36500d, temporal_weight=0.01,
}
```

Decay floors prevent timeless content from being penalised purely on age:

```python
DECAY_FLOORS = {
    ("mathematics", STATIC):    0.95,   # A 1954 theorem never decays to near-zero
    ("reference",   STATIC):    0.70,
    ("research",    STATIC):    0.10,
    ("legal",       STATIC):    0.20,
    ("policy",      VERSIONED): 0.05,
    ("tutorial",    VERSIONED): 0.05,
}
```

---

## Project Structure

```
temporal-rag/
├── temporal_rag.py      # Core: Document, DocumentKind, ValidityState,
│                        #       TemporalConfig, TemporalLayer, TemporalRAG, NaiveRAG
├── advanced.py          # Production patterns: PAIR, confidence, failure logging,
│                        #   conflict detection, time-range filter, adaptive weighting,
│                        #   freshness report, sequence deduplication
└── demo.py              # Four before/after scenarios: naive RAG vs temporal RAG
```

---

## Performance

Measured on Python 3.12, CPU only, 20-candidate pool:

| Operation | Latency |
|---|---|
| Temporal reranking (20 docs) | 15 – 30 ms |
| Validity filter | < 1 ms |
| Decay + recency scoring | < 1 ms |
| Full `retrieve()` call | ~20 ms |
| LLM inference (for reference) | 1,000 – 4,000 ms |

The temporal layer adds 15 – 30ms to a pipeline where the LLM itself takes 1 – 4 seconds. No retriever changes. No re-indexing. No new infrastructure. Pure Python, downstream of whatever vector search you are already running.

---

## When to Use This

**Worth it when you have:**
- A knowledge base you update regularly — tutorials, docs, policy pages, anything with versions
- Time-bounded signals like outages, announcements, or breaking changes that need to surface first
- A system where stale answers have real consequences (wrong rate limits, deprecated endpoints)
- Multi-version documents where the LLM should never see conflicting versions simultaneously

**Skip it when you have:**
- A static knowledge base that never changes
- A retriever that already handles temporal filtering
- Content that never goes stale

---

## Known Limitations

**Implicit expiration.** Documents without explicit `valid_until` dates cannot be hard-removed automatically. Rule-based heuristics by content type cover the obvious cases; edge cases require manual tagging.

**Conflicting sources.** The temporal layer surfaces the freshest and most relevant documents. Resolving disagreements between two current documents is the LLM's problem, not the retriever's.

**Embedding calibration.** `event_min_raw_vector_score=0.20` is tuned for TF-IDF sparse embeddings. Dense models produce higher absolute similarity scores — recalibrate to 0.35 – 0.50 before going to production.

**Half-life values are starting points.** The profiles in `DECAY_PROFILES` are reasonable defaults, not universal constants. Tune against real queries from your domain before deploying.

**Memory is in-process only.** No persistence across sessions. The `SequenceAwareRetriever` groups by `sequence_id` at query time from whatever is currently indexed.

---

## License

MIT
