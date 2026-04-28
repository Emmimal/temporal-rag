"""
RAG Has No Memory of Time — I Built a Temporal Layer That Does
=============================================================
A from-scratch implementation of temporal-aware RAG with:
  - Recency weighting
  - Time decay scoring
  - 3-state validity classification (VALID / TEMPORAL / EXPIRED)
  - Document type classification (STATIC / VERSIONED / EVENT)
  - Hybrid reranking (vector similarity + temporal score)
  - Semantic relevance threshold (prevents fresh-but-irrelevant promotion)
"""

import math
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import numpy as np


# ─────────────────────────────────────────────
# 1. DATA STRUCTURES
# ─────────────────────────────────────────────

class ValidityState(Enum):
    """
    Three states of temporal truth — not two.

    VALID    — permanently or openly true. No expiry constraint.
    TEMPORAL — true only within a bounded time window.
               Example: "rate limits suspended for 48 hours"
               Active now → boost it. Window closed → hard remove.
    EXPIRED  — was true, no longer is. Hard remove before ranking.

    Most RAG systems collapse TEMPORAL into EXPIRED.
    That's the bug this layer fixes.
    """
    VALID    = "VALID"
    TEMPORAL = "TEMPORAL"
    EXPIRED  = "EXPIRED"


class DocumentKind(Enum):
    """
    Three kinds of truth — not one.

    Time in RAG isn't one problem — it's three:
      expiration, temporality, and versioning.

    STATIC    — fact that doesn't change over time.
                Example: cosine similarity definition, math theorem.
                Decay: very slow. No versioning concern.

    VERSIONED — fact that gets replaced by newer versions.
                Example: API policy v1 → v2, tutorial for deprecated endpoint.
                Decay: moderate. The latest version wins; old versions are
                still "valid" in the VALID/EXPIRED sense but semantically
                superseded. Use supersedes_id to track the chain.

    EVENT     — fact that is true only within a time window.
                Example: "rate limits suspended for 48 hours", breaking news.
                This is where TEMPORAL validity applies.
                Boost when active; hard-remove when window closes.

    Without this distinction, a versioned policy (policy_v2) looks the same
    as a time-bounded event — and gets mis-labeled as TEMPORAL.
    That's the credibility bug this enum fixes.
    """
    STATIC    = "STATIC"
    VERSIONED = "VERSIONED"
    EVENT     = "EVENT"


@dataclass
class Document:
    """A document with temporal metadata."""
    id: str
    content: str
    embedding: Optional[np.ndarray] = None

    # Temporal metadata — the fields naive RAG ignores
    created_at: datetime = field(default_factory=datetime.now)
    valid_from: Optional[datetime] = None   # when the fact became true
    valid_until: Optional[datetime] = None  # when the fact expires (None = open-ended)
    doc_type: str = "general"

    # ── New: document kind ────────────────────
    # STATIC    → timeless fact (cosine similarity, math)
    # VERSIONED → replaced by a newer document in a chain
    # EVENT     → true only within a bounded time window
    kind: DocumentKind = DocumentKind.STATIC

    # For VERSIONED docs: the id this document supersedes (optional)
    supersedes_id: Optional[str] = None

    def validity_state(self, query_time: datetime) -> ValidityState:
        """
        Classify this document's temporal state at query time.

        TEMPORAL only applies to EVENT-kind documents.
        VERSIONED documents are VALID (not TEMPORAL) even when they have
        valid_from set — their replacement story is tracked via supersedes_id,
        not via validity windows.

        The distinction that matters:
          EVENT + active window  → TEMPORAL (boost)
          EVENT + closed window  → EXPIRED  (hard remove)
          VERSIONED              → VALID    (normal scoring, recency handles the rest)
          STATIC                 → VALID    (normal scoring)
        """
        has_window = self.valid_from is not None or self.valid_until is not None

        if not has_window:
            return ValidityState.VALID

        before_window = self.valid_from and query_time < self.valid_from
        after_window  = self.valid_until and query_time > self.valid_until

        if after_window or before_window:
            return ValidityState.EXPIRED

        # Has a window and we're inside it.
        # Only EVENT documents are classified as TEMPORAL.
        # VERSIONED documents with valid_from are still just VALID.
        if self.kind == DocumentKind.EVENT:
            return ValidityState.TEMPORAL

        return ValidityState.VALID

    def window_label(self, query_time: datetime) -> str:
        """Human-readable window description for explain output."""
        state = self.validity_state(query_time)
        if state == ValidityState.EXPIRED:
            days_ago = (query_time - self.valid_until).days if self.valid_until else "?"
            return f"expired {days_ago}d ago"
        if state == ValidityState.TEMPORAL:
            if self.valid_until:
                hours_left = (self.valid_until - query_time).total_seconds() / 3600
                return f"{hours_left:.0f}h remaining"
            return "active"
        if self.kind == DocumentKind.VERSIONED and self.supersedes_id:
            return f"supersedes {self.supersedes_id}"
        return "open-ended"

    def age_in_days(self, reference: datetime) -> float:
        delta = reference - self.created_at
        return max(0.0, delta.total_seconds() / 86400)


@dataclass
class ScoredDocument:
    """A retrieved document with its full scoring breakdown."""
    document: Document
    vector_score: float
    recency_score: float
    decay_score: float
    validity_state: ValidityState
    validity_multiplier: float    # 0.0 (expired) | 1.0 (valid) | 1.2 (temporal active)
    final_score: float
    reason: str                   # one-line reasoning trace

    def explain(self) -> str:
        now = datetime.now()
        age = self.document.age_in_days(now)
        state = self.validity_state
        kind = self.document.kind
        window = self.document.window_label(now)

        state_icon = {
            ValidityState.VALID:    "✓ valid",
            ValidityState.TEMPORAL: "⚡ temporal (active)",
            ValidityState.EXPIRED:  "✗ expired",
        }[state]

        kind_label = {
            DocumentKind.STATIC:    "STATIC",
            DocumentKind.VERSIONED: "VERSIONED",
            DocumentKind.EVENT:     "EVENT",
        }[kind]

        return (
            f"  [{self.document.id}]\n"
            f"    content      : {self.document.content[:80]}...\n"
            f"    age          : {age:.1f} days\n"
            f"    kind         : {kind_label}\n"
            f"    vector score : {self.vector_score:.3f}\n"
            f"    recency score: {self.recency_score:.3f}\n"
            f"    decay score  : {self.decay_score:.3f}\n"
            f"    state        : {state_icon}\n"
            f"    window       : {window}\n"
            f"    reason       : {self.reason}\n"
            f"    FINAL SCORE  : {self.final_score:.3f}\n"
        )


# ─────────────────────────────────────────────
# 2. EMBEDDING MODEL (pluggable)
# ─────────────────────────────────────────────

class EmbeddingModel:
    """
    Drop-in interface — swap for OpenAI, Cohere, sentence-transformers, etc.
    For demo purposes we use a deterministic TF-IDF-style sparse embedding
    so the article runs without any API key.
    """

    def __init__(self):
        self.vocab: dict[str, int] = {}
        self.dim = 256

    def _tokenize(self, text: str) -> list[str]:
        return text.lower().split()

    def _build_vocab(self, texts: list[str]):
        words = set()
        for t in texts:
            words.update(self._tokenize(t))
        self.vocab = {w: i % self.dim for i, w in enumerate(sorted(words))}

    def encode(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim)
        tokens = self._tokenize(text)
        for tok in tokens:
            if tok in self.vocab:
                vec[self.vocab[tok]] += 1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def encode_corpus(self, texts: list[str]) -> list[np.ndarray]:
        self._build_vocab(texts)
        return [self.encode(t) for t in texts]


# ─────────────────────────────────────────────
# 3. VECTOR STORE (minimal, from scratch)
# ─────────────────────────────────────────────

class VectorStore:
    """Simple in-memory vector store with cosine similarity search."""

    def __init__(self):
        self.documents: list[Document] = []

    def add(self, docs: list[Document]):
        self.documents.extend(docs)

    def search(self, query_embedding: np.ndarray, top_k: int = 20) -> list[tuple[Document, float]]:
        """
        Returns top_k documents by cosine similarity.
        Deliberately returns MORE than needed — the temporal layer reranks.
        """
        results = []
        for doc in self.documents:
            if doc.embedding is None:
                continue
            score = float(np.dot(query_embedding, doc.embedding))
            results.append((doc, score))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]


# ─────────────────────────────────────────────
# 4. TEMPORAL LAYER — the core innovation
# ─────────────────────────────────────────────

class TemporalConfig:
    """
    Controls how the temporal layer weights time against vector similarity.
    Tune these per use-case.
    """
    def __init__(
        self,
        # Half-life in days: score halves every N days
        # News: 7 days | Research: 180 days | Policy: 365 days
        decay_half_life_days: float = 30.0,

        # How much recency matters vs vector similarity
        # 0.0 = pure vector | 1.0 = pure recency
        temporal_weight: float = 0.35,

        # Hard filter: reject documents older than N days (None = no filter)
        max_age_days: Optional[float] = None,

        # Hard filter: reject documents whose valid_until has passed
        enforce_validity: bool = True,

        # Boost multiplier for EVENT documents still within their active window
        validity_boost: float = 1.2,

        # Minimum NORMALIZED vector similarity for general relevance gate.
        # Documents below this threshold have their final score penalized by 0.3×.
        # Applied to the normalized (0–1) score within the candidate pool.
        # Set to 0.0 to disable.
        min_vector_score: float = 0.15,

            # Minimum RAW cosine similarity for EVENT documents to receive their boost.
            # This is an absolute floor, independent of the candidate pool composition.
            # CALIBRATION NOTE: 0.20 is tuned for TF-IDF sparse embeddings (as used in
            # this demo). Dense models (e.g. text-embedding-3-small, all-MiniLM-L6-v2)
            # produce higher absolute similarity scores — recalibrate to 0.35–0.50 for
            # dense embeddings in production before going live.
            # Set to 0.0 to disable.
            event_min_raw_vector_score: float = 0.20,
    ):
        self.decay_half_life_days = decay_half_life_days
        self.temporal_weight = temporal_weight
        self.max_age_days = max_age_days
        self.enforce_validity = enforce_validity
        self.validity_boost = validity_boost
        self.min_vector_score = min_vector_score
        self.event_min_raw_vector_score = event_min_raw_vector_score


class TemporalLayer:
    """
    The layer naive RAG is missing.

    Takes vector-retrieved candidates and applies:
      1. 3-state validity classification — VALID / TEMPORAL / EXPIRED
      2. Document kind classification    — STATIC / VERSIONED / EVENT
      3. Hard removal of EXPIRED documents
      4. Time decay scoring  — exponential decay by document age
      5. Recency scoring     — normalized position in time
      6. Hybrid reranking    — combines vector + temporal scores
      7. Semantic threshold  — penalizes fresh-but-irrelevant docs

    The key insight:
      Naive RAG treats all documents the same along the time axis.
      This layer distinguishes three separate time problems:
        - Expiration  (was true, no longer is)
        - Temporality (true only within a window — EVENT kind)
        - Versioning  (replaced by a newer document — VERSIONED kind)
    """

    def __init__(self, config: TemporalConfig = None):
        self.config = config or TemporalConfig()

    # ── 4.1 3-state validity filter ──────────

    def _classify_and_filter(
        self,
        candidates: list[tuple[Document, float]],
        query_time: datetime,
    ) -> list[tuple[Document, float, ValidityState]]:
        """
        Classify each document into VALID / TEMPORAL / EXPIRED.
        Hard-remove EXPIRED documents entirely.
        TEMPORAL (EVENT, active window) documents get a boost later.
        VERSIONED documents are classified as VALID — their ranking
        is handled by time decay, not validity windows.
        """
        result = []
        for doc, score in candidates:
            state = doc.validity_state(query_time)
            if state == ValidityState.EXPIRED and self.config.enforce_validity:
                continue  # hard remove — expired truth has no place in context
            result.append((doc, score, state))
        return result

    def _filter_too_old(
        self,
        candidates: list[tuple[Document, float, ValidityState]],
        query_time: datetime,
    ) -> list[tuple[Document, float, ValidityState]]:
        """Hard-remove documents older than max_age_days."""
        if self.config.max_age_days is None:
            return candidates
        return [
            (doc, score, state) for doc, score, state in candidates
            if doc.age_in_days(query_time) <= self.config.max_age_days
        ]

    # ── 4.2 Time decay score ──────────────────

    def _decay_score(self, doc: Document, query_time: datetime) -> float:
        """
        Exponential decay: score = 0.5 ^ (age / half_life)

        A document at half-life age gets score 0.5.
        A brand-new document gets score ~1.0.
        A very old document asymptotically approaches 0.
        """
        age = doc.age_in_days(query_time)
        half_life = self.config.decay_half_life_days
        return math.pow(0.5, age / half_life)

    # ── 4.3 Recency score ─────────────────────

    def _recency_score(
        self,
        doc: Document,
        all_docs: list[Document],
        query_time: datetime,
    ) -> float:
        """
        Normalized recency: 1.0 for newest, 0.0 for oldest in the candidate set.
        """
        ages = [d.age_in_days(query_time) for d in all_docs]
        min_age, max_age = min(ages), max(ages)
        if max_age == min_age:
            return 1.0
        doc_age = doc.age_in_days(query_time)
        return 1.0 - (doc_age - min_age) / (max_age - min_age)

    # ── 4.4 Validity multiplier ───────────────

    def _validity_multiplier(self, state: ValidityState) -> float:
        """
        Score multiplier based on 3-state classification.

        EXPIRED  → 0.0  (should already be filtered, safety net)
        VALID    → 1.0  (normal scoring)
        TEMPORAL → 1.2  (boost: active EVENT signal is high-value)

        Note: only EVENT-kind documents reach TEMPORAL state.
        VERSIONED documents are always VALID here — their recency
        is handled by time decay, which naturally ranks policy_v2
        above policy_v1 without mislabeling it as time-bounded.
        """
        return {
            ValidityState.EXPIRED:  0.0,
            ValidityState.VALID:    1.0,
            ValidityState.TEMPORAL: self.config.validity_boost,
        }[state]

    # ── 4.5 Semantic relevance threshold ──────

    def _semantic_penalty(self, normalized_vector_score: float, min_score: float) -> float:
        """
        General relevance gate — applied to ALL document kinds.

        Uses the NORMALIZED vector score (0–1 within the candidate pool).
        This ensures the most semantically similar document in any pool always
        passes, regardless of how low the absolute similarities are.

        Below threshold → 0.3× penalty on final score.
        Above threshold → no penalty (1.0×).
        """
        if min_score <= 0.0:
            return 1.0
        return 1.0 if normalized_vector_score >= min_score else 0.3

    # ── 4.6 EVENT-specific relevance gate ─────

    def _event_relevance_multiplier(
        self,
        doc: Document,
        state: ValidityState,
        raw_vector_score: float,
    ) -> float:
        """
        Query-intent gate for EVENT documents — applied to the temporal component only.

        Problem: a fresh EVENT (announcement_today) can rank #1 on any query
        because its recency/decay scores are near-perfect. But freshness only
        earns a boost when the event is actually relevant to the query.

        This gate uses the RAW cosine similarity as an absolute floor.
        Unlike the normalized threshold, raw similarity is independent of the
        candidate pool — it tells us whether the query and document share
        vocabulary, regardless of what else is in the pool.

        Rule:
          EVENT + TEMPORAL + raw_score < floor → temporal component × 0.5
          Everything else                      → temporal component × 1.0

        Why only TEMPORAL EVENTs?
          An EVENT past its window is already EXPIRED and hard-removed.
          An EVENT below the floor is still retrieved — it just doesn't
          override genuinely relevant documents.
        """
        if doc.kind != DocumentKind.EVENT:
            return 1.0
        if state != ValidityState.TEMPORAL:
            return 1.0
        floor = self.config.event_min_raw_vector_score
        if floor <= 0.0:
            return 1.0
        return 1.0 if raw_vector_score >= floor else 0.5

    def _build_reason(
        self,
        doc: Document,
        state: ValidityState,
        decay: float,
        normalized_vector_score: float,
        raw_vector_score: float,
        query_time: datetime,
    ) -> str:
        """One-line reasoning trace — makes the system feel like it thinks."""
        min_score = self.config.min_vector_score
        event_floor = self.config.event_min_raw_vector_score
        if min_score > 0.0 and normalized_vector_score < min_score:
            return f"Penalized: normalized vector score {normalized_vector_score:.3f} below relevance threshold {min_score}"
        if state == ValidityState.TEMPORAL:
            if event_floor > 0.0 and raw_vector_score < event_floor:
                return (
                    f"EVENT signal present but low query relevance "
                    f"(raw sim {raw_vector_score:.3f} < {event_floor}) — temporal boost halved"
                )
            window = doc.window_label(query_time)
            return f"Active EVENT signal ({window}) — overrides static sources"
        if doc.kind == DocumentKind.VERSIONED:
            if doc.supersedes_id:
                return f"Latest version — supersedes {doc.supersedes_id}"
            return "Latest version in sequence"
        if state == ValidityState.VALID and decay > 0.7:
            return "Fresh, open-ended fact — high confidence"
        if state == ValidityState.VALID and decay > 0.3:
            return "Aging but valid — use alongside fresher sources"
        if state == ValidityState.VALID and decay <= 0.3:
            return "Stale — semantically relevant but low freshness weight"
        return "Included by vector similarity"

    # ── 4.7 Hybrid reranker ───────────────────

    def rerank(
        self,
        candidates: list[tuple[Document, float]],
        query_time: datetime,
        top_k: int = 5,
    ) -> list[ScoredDocument]:
        """
        Full temporal reranking pipeline.

        final_score = semantic_penalty
                    × [ (1 - w) × vector_score
                        + w × (decay_score × recency_score
                               × validity_multiplier × event_relevance_multiplier) ]

        validity_multiplier:
          EXPIRED  → 0.0  (hard zeroed, already filtered — safety net)
          VALID    → 1.0  (normal)
          TEMPORAL → 1.2  (boost: active EVENT signals are high-value)

        event_relevance_multiplier (EVENT + TEMPORAL only):
          raw cosine ≥ event_min_raw_vector_score → 1.0 (full boost)
          raw cosine <  event_min_raw_vector_score → 0.5 (present but irrelevant)

        semantic_penalty (all kinds, normalized score):
          normalized score ≥ min_vector_score → 1.0 (no penalty)
          normalized score <  min_vector_score → 0.3 (fresh-but-irrelevant guard)

        Not all fresh information is useful.
        Production RAG must know when to ignore even active EVENT signals.

        where w = temporal_weight from config.
        """
        # Step 1: classify into VALID/TEMPORAL/EXPIRED, hard-remove EXPIRED
        classified = self._classify_and_filter(candidates, query_time)
        classified = self._filter_too_old(classified, query_time)

        if not classified:
            return []

        docs_only = [doc for doc, _, _ in classified]

        # Step 2: normalize vector scores to [0, 1]
        raw_scores = [s for _, s, _ in classified]
        min_s, max_s = min(raw_scores), max(raw_scores)
        def norm(s):
            return (s - min_s) / (max_s - min_s) if max_s > min_s else 1.0

        w = self.config.temporal_weight
        scored = []

        for doc, raw_vector_score, state in classified:
            vs      = norm(raw_vector_score)
            ds      = self._decay_score(doc, query_time)
            rs      = self._recency_score(doc, docs_only, query_time)
            vm      = self._validity_multiplier(state)
            erm     = self._event_relevance_multiplier(doc, state, raw_vector_score)
            penalty = self._semantic_penalty(vs, self.config.min_vector_score)
            reason  = self._build_reason(doc, state, ds, vs, raw_vector_score, query_time)

            temporal_component = ds * rs * vm * erm
            final = penalty * ((1 - w) * vs + w * temporal_component)

            scored.append(ScoredDocument(
                document=doc,
                vector_score=vs,
                recency_score=rs,
                decay_score=ds,
                validity_state=state,
                validity_multiplier=vm,
                final_score=final,
                reason=reason,
            ))

        scored.sort(key=lambda x: x.final_score, reverse=True)
        return scored[:top_k]


# ─────────────────────────────────────────────
# 5. TEMPORAL RAG SYSTEM
# ─────────────────────────────────────────────

class TemporalRAG:
    """
    Full RAG pipeline with temporal awareness.

    Architecture:
        Query
          ↓
        Retriever (vector similarity)       ← broad candidate pool
          ↓
        Temporal Layer                      ← your core innovation
           - validity filtering (VALID / TEMPORAL / EXPIRED)
           - kind classification (STATIC / VERSIONED / EVENT)
           - time decay
           - recency scoring
           - semantic relevance threshold
           - hybrid reranking
          ↓
        Re-ranked context
          ↓
        LLM (pluggable)
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel = None,
        temporal_config: TemporalConfig = None,
        candidate_pool_size: int = 20,
    ):
        self.embedder = embedding_model or EmbeddingModel()
        self.store = VectorStore()
        self.temporal = TemporalLayer(temporal_config)
        self.candidate_pool_size = candidate_pool_size

    def index(self, documents: list[Document]):
        """Embed and store documents."""
        texts = [doc.content for doc in documents]
        embeddings = self.embedder.encode_corpus(texts)
        for doc, emb in zip(documents, embeddings):
            doc.embedding = emb
        self.store.add(documents)
        print(f"Indexed {len(documents)} documents.")

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        query_time: datetime = None,
    ) -> list[ScoredDocument]:
        """
        Retrieve and temporally rerank documents for a query.
        query_time defaults to now — pass a specific time to simulate
        queries at different points in history.
        """
        if query_time is None:
            query_time = datetime.now()

        query_emb = self.embedder.encode(query)
        candidates = self.store.search(query_emb, top_k=self.candidate_pool_size)
        return self.temporal.rerank(candidates, query_time, top_k=top_k)

    def build_context(self, results: list[ScoredDocument]) -> str:
        """Format retrieved documents into LLM context."""
        parts = []
        for i, r in enumerate(results, 1):
            doc = r.document
            age = doc.age_in_days(datetime.now())
            parts.append(
                f"[Source {i} | {doc.doc_type} | {doc.kind.value} | {age:.0f} days ago | "
                f"freshness: {r.decay_score:.2f}]\n{doc.content}"
            )
        return "\n\n".join(parts)


# ─────────────────────────────────────────────
# 6. NAIVE RAG (baseline for comparison)
# ─────────────────────────────────────────────

class NaiveRAG:
    """
    Standard RAG — no temporal awareness.
    Retrieves purely by vector similarity, ignores all time metadata.
    Used to demonstrate what temporal RAG fixes.
    """

    def __init__(self, embedding_model: EmbeddingModel = None):
        self.embedder = embedding_model or EmbeddingModel()
        self.store = VectorStore()

    def index(self, documents: list[Document]):
        texts = [doc.content for doc in documents]
        embeddings = self.embedder.encode_corpus(texts)
        for doc, emb in zip(documents, embeddings):
            doc.embedding = emb
        self.store.add(documents)

    def retrieve(self, query: str, top_k: int = 5) -> list[tuple[Document, float]]:
        query_emb = self.embedder.encode(query)
        return self.store.search(query_emb, top_k=top_k)

    def build_context(self, results: list[tuple[Document, float]]) -> str:
        parts = []
        for i, (doc, score) in enumerate(results, 1):
            age = doc.age_in_days(datetime.now())
            parts.append(
                f"[Source {i} | {doc.doc_type} | {age:.0f} days ago | "
                f"similarity: {score:.2f}]\n{doc.content}"
            )
        return "\n\n".join(parts)
