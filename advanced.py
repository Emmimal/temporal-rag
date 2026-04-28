"""
advanced.py — Advanced Temporal RAG Patterns
=============================================
Patterns for the article's deep-dive section:
  - Domain-specific decay profiles
  - Temporal query parsing
  - Sequence-aware retrieval
  - Freshness scoring API
"""

from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Optional
import math
import re

from temporal_rag import Document, DocumentKind, TemporalConfig, TemporalRAG, EmbeddingModel


# ─────────────────────────────────────────────
# PATTERN 1: DOMAIN-SPECIFIC DECAY PROFILES
# Different content types decay at different rates.
# News decays fast. Legal docs decay slow. Math never decays.
# ─────────────────────────────────────────────

DECAY_PROFILES = {
    "breaking_news":    TemporalConfig(decay_half_life_days=1,    temporal_weight=0.70),
    "news":             TemporalConfig(decay_half_life_days=7,    temporal_weight=0.55),
    "policy":           TemporalConfig(decay_half_life_days=90,   temporal_weight=0.45),
    "research":         TemporalConfig(decay_half_life_days=180,  temporal_weight=0.35),
    "legal":            TemporalConfig(decay_half_life_days=365,  temporal_weight=0.25),
    "reference":        TemporalConfig(decay_half_life_days=1825, temporal_weight=0.10),
    "mathematics":      TemporalConfig(decay_half_life_days=36500,temporal_weight=0.01),
}

def get_profile(doc_type: str) -> TemporalConfig:
    """Select the right decay profile for a document type."""
    return DECAY_PROFILES.get(doc_type, DECAY_PROFILES["reference"])


# ─────────────────────────────────────────────
# PATTERN 2: TEMPORAL QUERY PARSING
# Detect when a user's query is time-sensitive
# and automatically adjust temporal weight.
# ─────────────────────────────────────────────

TEMPORAL_SIGNALS = [
    # High urgency — strong recency bias
    (r"\b(current|latest|now|today|right now|at this moment)\b", 0.70),
    (r"\b(this week|this month|recent|recently)\b",              0.55),
    (r"\b(still|anymore|yet|has .+ changed)\b",                  0.50),
    # Lower urgency — mild recency preference
    (r"\b(new|updated|changed|revised)\b",                       0.40),
    (r"\b(best|recommend|should I)\b",                           0.35),
]

def parse_temporal_intent(query: str) -> tuple[float, str]:
    """
    Returns (temporal_weight, reason) based on query language.
    Default weight is 0.20 (slight recency preference).
    """
    query_lower = query.lower()
    for pattern, weight in TEMPORAL_SIGNALS:
        if re.search(pattern, query_lower):
            matched = re.search(pattern, query_lower).group()
            return weight, f"detected temporal signal: '{matched}'"
    return 0.20, "no temporal signal — using baseline weight"

def adaptive_retrieve(rag: TemporalRAG, query: str, top_k: int = 5):
    """Retrieve with automatically adjusted temporal weight."""
    weight, reason = parse_temporal_intent(query)
    rag.temporal.config.temporal_weight = weight
    print(f"  Temporal weight: {weight:.2f} ({reason})")
    return rag.retrieve(query, top_k=top_k)


# ─────────────────────────────────────────────
# PATTERN 3: SEQUENCE-AWARE RETRIEVAL
# When documents form a chain (v1 → v2 → v3),
# retrieve the latest in the chain, not the most
# semantically similar version.
# ─────────────────────────────────────────────

@dataclass
class VersionedDocument(Document):
    """A document that belongs to a versioned sequence."""
    sequence_id: str = ""    # groups documents in the same chain
    version: int = 1         # higher = newer


class SequenceAwareRetriever:
    """
    Extends temporal RAG with sequence awareness.
    For each sequence_id, keeps only the latest valid version.
    Prevents RAG from mixing v1 and v3 of the same policy.
    """

    def deduplicate_sequences(
        self,
        docs: list[Document],
        query_time: datetime,
    ) -> list[Document]:
        """
        For each sequence group, keep only the latest valid document.
        Unversioned documents are passed through unchanged.
        """
        sequences: dict[str, list[VersionedDocument]] = {}
        unversioned = []

        for doc in docs:
            if isinstance(doc, VersionedDocument) and doc.sequence_id:
                sequences.setdefault(doc.sequence_id, []).append(doc)
            else:
                unversioned.append(doc)

        deduped = list(unversioned)
        for seq_id, versions in sequences.items():
            # Filter to valid-at-query-time, then take highest version
            valid = [v for v in versions
                     if v.validity_state(query_time).name != "EXPIRED"]
            if valid:
                latest = max(valid, key=lambda v: v.version)
                deduped.append(latest)

        return deduped


# ─────────────────────────────────────────────
# PATTERN 4: FRESHNESS SCORING API
# A clean interface to inspect temporal scores
# for any document — useful for debugging and
# for showing readers "what the layer sees."
# ─────────────────────────────────────────────

def freshness_report(doc: Document, reference: datetime = None) -> dict:
    """
    Returns a full temporal breakdown for a document.
    Includes document kind so the scoring rationale is transparent.
    """
    if reference is None:
        reference = datetime.now()

    config = TemporalConfig()
    age = doc.age_in_days(reference)
    decay = math.pow(0.5, age / config.decay_half_life_days)
    state = doc.validity_state(reference)
    is_valid = state.name != "EXPIRED"
    days_to_expiry = (
        (doc.valid_until - reference).days
        if doc.valid_until else None
    )

    return {
        "doc_id":           doc.id,
        "kind":             doc.kind.value,
        "age_days":         round(age, 1),
        "decay_score":      round(decay, 4),
        "validity_state":   state.value,
        "is_valid":         is_valid,
        "days_to_expiry":   days_to_expiry,
        "freshness_grade":  _grade(decay, is_valid),
        "recommendation":   _recommend(decay, is_valid, days_to_expiry, doc.kind),
    }

def _grade(decay: float, is_valid: bool) -> str:
    if not is_valid:         return "F — expired"
    if decay >= 0.90:        return "A — very fresh"
    if decay >= 0.70:        return "B — fresh"
    if decay >= 0.50:        return "C — aging"
    if decay >= 0.25:        return "D — stale"
    return                          "F — very stale"

def _recommend(decay: float, is_valid: bool, days_to_expiry: Optional[int], kind: DocumentKind) -> str:
    if not is_valid:
        return "Do not retrieve — fact is no longer valid."
    if days_to_expiry is not None and days_to_expiry < 7:
        return "Use with caution — EVENT window closes soon. Verify before serving."
    if kind == DocumentKind.STATIC and decay < 0.25:
        return "Use with caution — STATIC document is very stale. May have been superseded."
    if kind == DocumentKind.VERSIONED and decay < 0.50:
        return "Check for a newer version — VERSIONED document may have been replaced."
    if decay >= 0.70:
        return "Safe to retrieve."
    return "Acceptable — consider pairing with a more recent source."


# ─────────────────────────────────────────────
# DEMO
# ─────────────────────────────────────────────

if __name__ == "__main__":
    now = datetime.now()

    print("\n── Pattern 1: Decay profiles ──")
    for name, cfg in DECAY_PROFILES.items():
        print(f"  {name:<20} half_life={cfg.decay_half_life_days:>6}d  "
              f"temporal_weight={cfg.temporal_weight:.2f}")

    print("\n── Pattern 2: Temporal query parsing ──")
    queries = [
        "What are the current API rate limits?",
        "How does cosine similarity work?",
        "Has the authentication method changed recently?",
        "What should I use for embeddings?",
    ]
    for q in queries:
        weight, reason = parse_temporal_intent(q)
        print(f"  [{weight:.2f}] {q}")
        print(f"         → {reason}")

    print("\n── Pattern 4: Freshness report (with kind axis) ──")
    docs = [
        Document("fresh_event",   "content", created_at=now - timedelta(hours=2),
                 valid_until=now + timedelta(hours=46), kind=DocumentKind.EVENT),
        Document("current_policy","content", created_at=now - timedelta(days=45),
                 valid_from=now - timedelta(days=50),   kind=DocumentKind.VERSIONED,
                 supersedes_id="old_policy"),
        Document("old_research",  "content", created_at=now - timedelta(days=200),
                 kind=DocumentKind.STATIC),
        Document("expired_policy","content", created_at=now - timedelta(days=400),
                 valid_until=now - timedelta(days=30),  kind=DocumentKind.VERSIONED),
        Document("math_theorem",  "content", created_at=now - timedelta(days=800),
                 kind=DocumentKind.STATIC),
    ]
    for doc in docs:
        report = freshness_report(doc)
        print(f"\n  {report['doc_id']} [{report['kind']}]")
        print(f"    age            : {report['age_days']} days")
        print(f"    decay score    : {report['decay_score']}")
        print(f"    validity state : {report['validity_state']}")
        print(f"    grade          : {report['freshness_grade']}")
        print(f"    recommendation : {report['recommendation']}")
