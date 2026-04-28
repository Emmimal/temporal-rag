"""
demo.py — The Before/After Comparison
======================================
This is the viral moment of the article.

Run this and you see exactly what naive RAG gets wrong
and what the temporal layer fixes — on the same query,
with the same documents, side by side.
"""

from datetime import datetime, timedelta
from temporal_rag import (
    Document, DocumentKind, EmbeddingModel,
    NaiveRAG, TemporalRAG, TemporalConfig,
)


# ─────────────────────────────────────────────
# SHARED CORPUS
# A realistic knowledge base where time matters.
# Same documents fed to both naive and temporal RAG.
# ─────────────────────────────────────────────

def build_corpus() -> list[Document]:
    now = datetime.now()

    return [
        # Outdated policy — superseded but still semantically relevant
        # Kind: VERSIONED (replaced by policy_v2, not a time-bounded event)
        Document(
            id="policy_v1",
            content="API rate limits are set to 100 requests per minute per user. "
                    "Exceeding this limit will result in a 429 error. "
                    "Enterprise accounts are not exempt from this limit.",
            created_at=now - timedelta(days=540),
            valid_until=now - timedelta(days=180),   # expired 6 months ago
            doc_type="policy",
            kind=DocumentKind.VERSIONED,
        ),

        # Current policy — what users actually need
        # Kind: VERSIONED (it replaced policy_v1; valid_from marks when it took effect)
        # Not TEMPORAL — it's not a time-bounded event, it's the current standing policy.
        Document(
            id="policy_v2",
            content="API rate limits have been updated to 1000 requests per minute per user. "
                    "Enterprise accounts receive a 10x multiplier: 10,000 requests per minute. "
                    "Rate limit headers are now included in every response.",
            created_at=now - timedelta(days=175),
            valid_from=now - timedelta(days=180),
            doc_type="policy",
            kind=DocumentKind.VERSIONED,
            supersedes_id="policy_v1",
        ),

        # Old research finding
        # Kind: STATIC (a published finding — it doesn't expire, it may be superseded)
        Document(
            id="research_2022",
            content="Studies show transformer models plateau at 7B parameters for "
                    "most downstream tasks. Scaling beyond this yields diminishing returns "
                    "on standard benchmarks including MMLU and HellaSwag.",
            created_at=now - timedelta(days=730),
            doc_type="research",
            kind=DocumentKind.STATIC,
        ),

        # More recent research that overturns the old finding
        # Kind: STATIC (a published finding)
        Document(
            id="research_2024",
            content="Recent scaling studies demonstrate consistent improvement beyond 70B "
                    "parameters, with emergent capabilities appearing at 100B+ scale. "
                    "The earlier 7B plateau finding did not account for instruction tuning.",
            created_at=now - timedelta(days=120),
            doc_type="research",
            kind=DocumentKind.STATIC,
        ),

        # Stale news — relevant topic but old event
        # Kind: STATIC (a news report is a historical record — it doesn't expire)
        Document(
            id="news_old",
            content="The company announced layoffs affecting 10% of its workforce. "
                    "The engineering team was particularly impacted, with the infrastructure "
                    "division reduced by 40%. Stock dropped 8% on the announcement.",
            created_at=now - timedelta(days=400),
            doc_type="news",
            kind=DocumentKind.STATIC,
        ),

        # Recent news on same company
        # Kind: STATIC (a news report)
        Document(
            id="news_recent",
            content="The company has fully recovered from last year's restructuring, "
                    "hiring 2,000 new engineers in Q1. The infrastructure division "
                    "has expanded to three times its previous size. Stock at all-time high.",
            created_at=now - timedelta(days=30),
            doc_type="news",
            kind=DocumentKind.STATIC,
        ),

        # Tutorial — deprecated endpoint, content expired
        # Kind: VERSIONED (tutorial for v1 API, superseded by tutorial_new)
        Document(
            id="tutorial_old",
            content="To authenticate with the API, pass your API key in the Authorization "
                    "header as a Bearer token. Use POST /v1/completions with your prompt "
                    "in the request body.",
            created_at=now - timedelta(days=600),
            valid_until=now - timedelta(days=90),   # deprecated endpoint
            doc_type="tutorial",
            kind=DocumentKind.VERSIONED,
        ),

        # Updated tutorial
        # Kind: VERSIONED (replaces tutorial_old)
        Document(
            id="tutorial_new",
            content="Authentication uses Bearer tokens in the Authorization header. "
                    "The completions endpoint is deprecated — use POST /v1/messages instead. "
                    "The new endpoint supports multi-turn conversations natively.",
            created_at=now - timedelta(days=85),
            doc_type="tutorial",
            kind=DocumentKind.VERSIONED,
            supersedes_id="tutorial_old",
        ),

        # Evergreen reference — time matters less here
        # Kind: STATIC (mathematical definition — does not decay)
        Document(
            id="reference_core",
            content="Cosine similarity measures the angle between two vectors in embedding space. "
                    "It ranges from -1 to 1, where 1 means identical direction, "
                    "0 means orthogonal, and -1 means opposite. Used in semantic search.",
            created_at=now - timedelta(days=800),
            doc_type="reference",
            kind=DocumentKind.STATIC,
        ),

        # Very recent announcement — time-bounded operational event
        # Kind: EVENT (true only within a 48-hour window → correctly classified as TEMPORAL)
        Document(
            id="announcement_today",
            content="Rate limiting is temporarily suspended for all API tiers for the next 48 hours "
                    "due to infrastructure upgrades. No 429 errors will be returned during this window. "
                    "Normal limits resume Monday.",
            created_at=now - timedelta(hours=6),
            valid_until=now + timedelta(hours=42),
            doc_type="announcement",
            kind=DocumentKind.EVENT,
        ),
    ]


# ─────────────────────────────────────────────
# COMPARISON ENGINE
# ─────────────────────────────────────────────

def run_comparison(query: str, corpus: list[Document]):
    print("\n" + "═" * 70)
    print(f"QUERY: {query}")
    print("═" * 70)

    embedder = EmbeddingModel()

    # ── Naive RAG ─────────────────────────────
    naive = NaiveRAG(embedding_model=embedder)
    naive.index(corpus)
    naive_results = naive.retrieve(query, top_k=3)

    print("\n❌  NAIVE RAG (no temporal awareness)")
    print("─" * 50)
    for i, (doc, score) in enumerate(naive_results, 1):
        age = doc.age_in_days(datetime.now())
        expired = doc.valid_until and datetime.now() > doc.valid_until
        status = "EXPIRED" if expired else "valid"
        print(f"  {i}. [{doc.id}] age={age:.0f}d | {status} | sim={score:.3f}")
        print(f"     {doc.content[:100]}...")

    naive_context = naive.build_context(naive_results)

    # ── Temporal RAG ──────────────────────────
    config = TemporalConfig(
        decay_half_life_days=60,
        temporal_weight=0.40,
        enforce_validity=True,
        validity_boost=1.2,
        min_vector_score=0.15,          # general relevance gate (normalized)
        event_min_raw_vector_score=0.20, # EVENT-specific gate (raw cosine floor)
    )
    temporal = TemporalRAG(embedding_model=embedder, temporal_config=config)
    temporal.index(corpus)
    temporal_results = temporal.retrieve(query, top_k=3)

    print("\n✅  TEMPORAL RAG (with temporal layer)")
    print("─" * 50)
    for result in temporal_results:
        print(result.explain())

    temporal_context = temporal.build_context(temporal_results)

    # ── Delta ─────────────────────────────────
    naive_ids    = [doc.id for doc, _ in naive_results]
    temporal_ids = [r.document.id for r in temporal_results]
    promoted = [i for i in temporal_ids if i not in naive_ids]
    demoted  = [i for i in naive_ids if i not in temporal_ids]

    if promoted or demoted:
        print("  WHAT THE TEMPORAL LAYER CHANGED:")
        if demoted:
            print(f"    removed  : {demoted}")
        if promoted:
            print(f"    surfaced : {promoted}")


# ─────────────────────────────────────────────
# DEMO SCENARIOS
# ─────────────────────────────────────────────

def main():
    print("\n")
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║       RAG Has No Memory of Time — I Built a Temporal Layer That Does ║")
    print("║                    Before / After Comparison                         ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")

    corpus = build_corpus()

    # Scenario 1: Policy question — expired answer is dangerous
    run_comparison(
        query="What are the API rate limits? Will I get a 429 error?",
        corpus=corpus,
    )

    # Scenario 2: Research question — old finding overturned
    run_comparison(
        query="Do larger language models keep improving with scale?",
        corpus=corpus,
    )

    # Scenario 3: Company status — old news vs current state
    run_comparison(
        query="What is the current state of the engineering team and company health?",
        corpus=corpus,
    )

    # Scenario 4: Time-sensitive operational query
    run_comparison(
        query="Are there any current API outages or limit suspensions I should know about?",
        corpus=corpus,
    )

    print("\n" + "═" * 70)
    print("KEY INSIGHT")
    print("═" * 70)
    print("""
  Naive RAG ranks by semantic similarity alone.
  It has no concept of:
    - expired facts (rate limit policy v1 was superseded 6 months ago)
    - overturned findings (the 7B plateau finding no longer holds)
    - stale news (layoffs reversed, company fully recovered)
    - time-sensitive announcements (48hr rate limit suspension)

  The temporal layer adds three signals:
    1. Validity filtering  — hard-remove expired facts before ranking
    2. Time decay          — score = 0.5^(age / half_life)
    3. Recency normalization — relative freshness within the candidate pool

  The fix: two orthogonal axes.

  AXIS 1 — Validity state (3 states)
    EXPIRED  → hard removed before scoring
    VALID    → scored normally
    TEMPORAL → boosted (active time-bounded signals are high-value)

  AXIS 2 — Document kind (3 types)
    STATIC    → timeless fact (math, definitions). Decays slowly.
    VERSIONED → replaced by newer information (policies, tutorials).
                Time decay handles ranking; no validity window needed.
    EVENT     → true only within a time window (announcements, outages).
                The only kind that reaches TEMPORAL state.

  Without the kind axis, a versioned policy (policy_v2) looks identical
  to a time-bounded event — and gets mislabeled TEMPORAL.
  That's not a model — that's a heuristic.

  The semantic threshold closes the last gap:
    Fresh-but-irrelevant documents can dominate ranking when temporal
    scores are high. A minimum vector score guard prevents recency from
    overriding relevance entirely.

  Not all fresh information is useful.
  Production RAG must know when to ignore even active signals.

  Naive RAG retrieves what is similar.
  Temporal RAG retrieves what is still true.
  Production RAG must also know what is still relevant.
""")


if __name__ == "__main__":
    main()
