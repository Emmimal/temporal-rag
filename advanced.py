"""
advanced.py — Advanced Temporal RAG Patterns (v6)
==================================================
Fixes vs v4:
  [BUG-1] PAIR demo used a mixed-epoch corpus — pair search always failed.
          Fix: separate PAIR demo corpus (relative dates) from time-range corpus
          (real calendar dates). PAIR corpus has one doc newer than weak_doc
          so the partner is always findable.

  [BUG-2] query_id was not threaded into find_and_pair(), so PAIR_PARTNER_NOT_FOUND
          records logged with query_id="—".
          Fix: find_and_pair() now accepts query_id and passes it to log_rejection().

  [BUG-3] execute_retrieval() used REJECT_EXPIRED_EVENT as the fallback rejection
          code for non-VERSIONED docs, so STATIC rejections appeared as
          HARD_EXPIRED_EVENT in failure logs.
          Fix: rejection code is now selected from all three kinds
          (VERSIONED / EVENT / STATIC) with a dedicated REJECT_STALE_STATIC code.

New in v6:
  [ADD-1] adaptive_retrieve() — adjusts temporal_weight from TEMPORAL_SIGNALS
          in the query string. Queries with "current" or "latest" push weight
          to 0.70; queries with no signal fall back to baseline 0.20.

  [ADD-2] freshness_report() — kind-aware observability surface. Returns a
          formatted breakdown of age, decay, validity state, letter grade,
          and a recommendation tuned to the document's kind, not just score.

  [ADD-3] SequenceAwareRetriever — groups VersionedDocuments by sequence_id
          and keeps only the latest valid version before candidates reach the
          temporal layer. Prevents conflicting versions from co-appearing in
          the LLM context window.
"""

from __future__ import annotations

import math
import re
import uuid
from collections import defaultdict, Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

from temporal_rag import (
    Document, DocumentKind, TemporalConfig,
    ScoredDocument, ValidityState
)


# ─────────────────────────────────────────────────────────────────────────────
# DECAY PROFILES & FLOORS
# ─────────────────────────────────────────────────────────────────────────────

DECAY_PROFILES = {
    "breaking_news": TemporalConfig(decay_half_life_days=1,     temporal_weight=0.70),
    "news":          TemporalConfig(decay_half_life_days=7,     temporal_weight=0.55),
    "policy":        TemporalConfig(decay_half_life_days=90,    temporal_weight=0.45),
    "research":      TemporalConfig(decay_half_life_days=180,   temporal_weight=0.35),
    "legal":         TemporalConfig(decay_half_life_days=365,   temporal_weight=0.25),
    "reference":     TemporalConfig(decay_half_life_days=1825,  temporal_weight=0.10),
    "mathematics":   TemporalConfig(decay_half_life_days=36500, temporal_weight=0.01),
}

DECAY_FLOORS = {
    ("mathematics", DocumentKind.STATIC):    0.95,
    ("reference",   DocumentKind.STATIC):    0.70,
    ("research",    DocumentKind.STATIC):    0.10,
    ("legal",       DocumentKind.STATIC):    0.20,
    ("policy",      DocumentKind.VERSIONED): 0.05,
    ("tutorial",    DocumentKind.VERSIONED): 0.05,
}

def get_profile(doc_type: str) -> TemporalConfig:
    return DECAY_PROFILES.get(doc_type, DECAY_PROFILES["reference"])

def kind_aware_decay(doc: Document, query_time: datetime, config: TemporalConfig) -> float:
    age   = doc.age_in_days(query_time)
    exp   = math.pow(0.5, age / config.decay_half_life_days)
    floor = DECAY_FLOORS.get((doc.doc_type, doc.kind), 0.0)
    return max(exp, floor)


# ─────────────────────────────────────────────────────────────────────────────
# EVENT HARD EXPIRY
# ─────────────────────────────────────────────────────────────────────────────

EVENT_WINDOW_HOURS = 48

def resolve_event_state(
    doc: Document,
    query_time: datetime,
    event_window_hours: int = EVENT_WINDOW_HOURS,
) -> tuple[str, float | None]:
    if doc.kind != DocumentKind.EVENT:
        return "N/A", None
    if doc.age_in_days(query_time) * 24 >= event_window_hours:
        return "HARD_EXPIRED", 0.0
    return "LIVE", None


# ─────────────────────────────────────────────────────────────────────────────
# FAILURE MODE LOGGING
# ─────────────────────────────────────────────────────────────────────────────

REJECTION_LOG: list[dict] = []

REJECT_EXPIRED_VERSIONED = "EXPIRED_VERSIONED_DOC"
REJECT_EXPIRED_EVENT     = "HARD_EXPIRED_EVENT"
REJECT_STALE_STATIC      = "STALE_STATIC_DOC"
REJECT_BELOW_RELEVANCE   = "BELOW_RELEVANCE_GATE"
REJECT_OUT_OF_TIME_RANGE = "OUT_OF_TIME_RANGE"
REJECT_PAIR_NOT_FOUND    = "PAIR_PARTNER_NOT_FOUND"
REJECT_LOW_CONFIDENCE    = "LOW_CONFIDENCE_FALLBACK"


def generate_query_id() -> str:
    return uuid.uuid4().hex[:8]


def log_rejection(
    doc: Document,
    code: str,
    detail: str = "",
    query: str = "",
    query_id: str = "",
) -> None:
    REJECTION_LOG.append({
        "timestamp": datetime.now().isoformat(),
        "query_id":  query_id or "—",
        "query":     query,
        "doc_id":    doc.id,
        "doc_type":  doc.doc_type,
        "kind":      doc.kind.value,
        "age_days":  round(doc.age_in_days(datetime.now()), 1),
        "code":      code,
        "detail":    detail or code,
    })


def failure_summary(query_id: str = "") -> None:
    records = (
        [r for r in REJECTION_LOG if r["query_id"] == query_id]
        if query_id else REJECTION_LOG
    )
    if not records:
        label = f"query_id={query_id}" if query_id else "session"
        print(f"  No rejections logged for {label}.")
        return

    codes     = Counter(r["code"]     for r in records)
    doc_types = Counter(r["doc_type"] for r in records)
    scope     = f"query_id={query_id}" if query_id else "full session"

    print(f"\n  Failure summary ({len(records)} rejections — {scope})")
    print("  By rejection code:")
    for code, n in codes.most_common():
        print(f"    {code:<42} × {n}")
    print("  By doc_type:")
    for dt, n in doc_types.most_common():
        print(f"    {dt:<22} × {n}")
    print("  Recent records (latest 5):")
    for r in records[-5:]:
        print(f"    [{r['query_id']}] {r['code']:<38} doc={r['doc_id']}")
        if r["query"]:
            print(f"           query: {r['query']!r}")


# ─────────────────────────────────────────────────────────────────────────────
# CONFLICT RESOLUTION + ADAPTIVE BOOST + CONFIDENCE PENALTY
# ─────────────────────────────────────────────────────────────────────────────

def conflict_severity(fp_old: str, fp_new: str) -> float:
    try:
        a, b = float(fp_old), float(fp_new)
        if a == b:
            return 0.0
        return abs(a - b) / max(abs(a), abs(b))
    except ValueError:
        return 1.0 if fp_old != fp_new else 0.0


def adaptive_boost(severity: float, scale: float = 0.20,
                   min_boost: float = 0.05, max_boost: float = 0.25) -> float:
    if severity == 0.0:
        return 0.0
    return max(min_boost, min(max_boost, severity * scale))


def confidence_penalty_from_conflict(severity: float, penalty_scale: float = 0.10) -> float:
    return round(severity * penalty_scale, 4)


@dataclass
class VersionedDocument(Document):
    sequence_id:        str   = ""
    version:            int   = 1
    conflict_boost:     float = 0.0
    conflict_penalty:   float = 0.0


@dataclass
class ConflictReport:
    sequence_id:        str
    conflicting_ids:    list[str]
    winning_id:         str
    winning_version:    int
    severity:           float
    score_boost:        float
    confidence_penalty: float
    detail:             str


def detect_and_resolve_conflicts(
    versioned_docs: list[VersionedDocument],
) -> tuple[list[VersionedDocument], list[ConflictReport]]:
    groups: dict[str, list[VersionedDocument]] = defaultdict(list)
    standalone: list[VersionedDocument] = []

    for doc in versioned_docs:
        if isinstance(doc, VersionedDocument) and doc.sequence_id:
            groups[doc.sequence_id].append(doc)
        else:
            standalone.append(doc)

    def fingerprint(d: Document) -> str:
        nums = re.findall(r'\b\d+\b', d.content)
        return nums[0] if nums else d.content[:40]

    reports: list[ConflictReport] = []
    resolved: list[VersionedDocument] = list(standalone)

    for seq_id, versions in groups.items():
        versions_sorted = sorted(versions, key=lambda v: v.version)
        latest = versions_sorted[-1]
        fps = {v.id: fingerprint(v) for v in versions_sorted}

        if len(set(fps.values())) > 1:
            fp_values = [fps[v.id] for v in versions_sorted]
            sev     = conflict_severity(fp_values[0], fp_values[-1])
            boost   = adaptive_boost(sev)
            penalty = confidence_penalty_from_conflict(sev)
            latest.conflict_boost   = boost
            latest.conflict_penalty = penalty

            conflicting_ids = [v.id for v in versions_sorted if v.id != latest.id]
            reports.append(ConflictReport(
                sequence_id=seq_id,
                conflicting_ids=conflicting_ids,
                winning_id=latest.id,
                winning_version=latest.version,
                severity=round(sev, 3),
                score_boost=round(boost, 3),
                confidence_penalty=round(penalty, 3),
                detail=(
                    f"Values changed {fp_values[0]} → {fp_values[-1]} "
                    f"(severity={sev:.2f}). "
                    f"Winner {latest.id} v{latest.version}: "
                    f"boost=+{boost:.3f}, confidence_penalty=-{penalty:.3f}."
                )
            ))

        for v in versions_sorted:
            resolved.append(v)

    return resolved, reports


# ─────────────────────────────────────────────────────────────────────────────
# CONFIDENCE (uncertainty-aware: margin + conflict penalty)
# ─────────────────────────────────────────────────────────────────────────────

def sigmoid_confidence(score: float, k: float = 8.0, midpoint: float = 0.5) -> float:
    return 1.0 / (1.0 + math.exp(-k * (score - midpoint)))


def confidence_tier(
    final_score: float,
    score_margin: float = 1.0,
    conflict_penalty: float = 0.0,
) -> tuple[float, str]:
    margin_penalty = max(0.0, (0.15 - score_margin) * 2.0)
    adjusted = final_score - margin_penalty - conflict_penalty
    sig = sigmoid_confidence(adjusted)
    if sig >= 0.72:
        return round(sig, 4), "HIGH"
    if sig >= 0.55:
        return round(sig, 4), "MEDIUM"
    return round(sig, 4), "LOW"


# ─────────────────────────────────────────────────────────────────────────────
# SCORING
# ─────────────────────────────────────────────────────────────────────────────

def compute_final_score(
    relevance_score:    float,
    temporal_score:     float,
    temporal_weight:    float,
    authority_score:    float = 0.5,
    conflict_boost:     float = 0.0,
    conflict_penalty:   float = 0.0,
    score_margin:       float = 1.0,
    rel_weight:         float = 0.60,
    auth_weight:        float = 0.10,
) -> dict:
    total  = rel_weight + temporal_weight + auth_weight
    rel_w  = rel_weight      / total
    temp_w = temporal_weight / total
    auth_w = auth_weight     / total

    rel_c  = relevance_score * rel_w
    temp_c = temporal_score  * temp_w
    auth_c = authority_score * auth_w

    base  = rel_c + temp_c + auth_c
    final = min(1.0, base + conflict_boost)
    sig, tier = confidence_tier(final, score_margin, conflict_penalty)

    return {
        "relevance_score":   round(relevance_score, 4),
        "temporal_score":    round(temporal_score,  4),
        "authority_score":   round(authority_score, 4),
        "rel_weight":        round(rel_w,   3),
        "temporal_weight":   round(temp_w,  3),
        "auth_weight":       round(auth_w,  3),
        "rel_contrib":       round(rel_c,   4),
        "temporal_contrib":  round(temp_c,  4),
        "auth_contrib":      round(auth_c,  4),
        "conflict_boost":    round(conflict_boost,   4),
        "conflict_penalty":  round(conflict_penalty, 4),
        "score_margin":      round(score_margin,     4),
        "base_score":        round(base,   4),
        "final_score":       round(final,  4),
        "confidence_sig":    sig,
        "confidence_tier":   tier,
    }


def print_score_breakdown(doc_id: str, bd: dict) -> None:
    print(f"\n  Score breakdown — {doc_id}")
    print(f"    relevance  : {bd['relevance_score']:.4f} × {bd['rel_weight']:.3f}"
          f"  = {bd['rel_contrib']:.4f}")
    print(f"    temporal   : {bd['temporal_score']:.4f} × {bd['temporal_weight']:.3f}"
          f"  = {bd['temporal_contrib']:.4f}")
    print(f"    authority  : {bd['authority_score']:.4f} × {bd['auth_weight']:.3f}"
          f"  = {bd['auth_contrib']:.4f}")
    if bd["conflict_boost"]:
        print(f"    conflict boost           : +{bd['conflict_boost']:.4f}")
    print(f"    ─────────────────────────────────────────")
    print(f"    base score               :  {bd['base_score']:.4f}")
    print(f"    final score              :  {bd['final_score']:.4f}")
    adj_parts = []
    if bd["score_margin"] < 1.0:
        mp = max(0.0, (0.15 - bd["score_margin"]) * 2.0)
        if mp > 0:
            adj_parts.append(f"margin_penalty=-{mp:.3f}")
    if bd["conflict_penalty"]:
        adj_parts.append(f"conflict_penalty=-{bd['conflict_penalty']:.4f}")
    if adj_parts:
        print(f"    confidence adjustments   :  {', '.join(adj_parts)}")
    print(f"    confidence               :  {bd['confidence_sig']:.4f}  → {bd['confidence_tier']}")


# ─────────────────────────────────────────────────────────────────────────────
# GRADE / ACTION
# ─────────────────────────────────────────────────────────────────────────────

def _grade_and_action(decay: float, is_valid: bool) -> tuple[str, str]:
    if not is_valid or decay == 0.0:
        return "Invalid", "DO NOT RETRIEVE"
    if decay >= 0.90:   return "Strong", "RETRIEVE"
    if decay >= 0.70:   return "Good",   "RETRIEVE"
    if decay >= 0.50:   return "Usable", "RETRIEVE WITH CAVEAT"
    if decay >= 0.25:   return "Weak",   "RETRIEVE + PAIR WITH FRESHER SOURCE"
    return                     "Invalid","DO NOT RETRIEVE"


# ─────────────────────────────────────────────────────────────────────────────
# PAIR EXECUTION
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PairResult:
    original: Document
    partner:  Document | None
    paired:   bool
    reason:   str


class PairExecutor:
    def __init__(self, corpus: list[Document]):
        self.corpus = corpus

    def find_and_pair(
        self,
        doc: Document,
        query_time: datetime = None,
        min_freshness_gain: float = 0.10,
        query_id: str = "",
    ) -> PairResult:
        if query_time is None:
            query_time = datetime.now()

        config         = get_profile(doc.doc_type)
        original_decay = kind_aware_decay(doc, query_time, config)

        candidates = []
        for candidate in self.corpus:
            if candidate.id == doc.id:
                continue
            if candidate.doc_type != doc.doc_type:
                continue
            if candidate.created_at <= doc.created_at:
                continue
            if candidate.validity_state(query_time).name == "EXPIRED":
                continue
            cand_decay = kind_aware_decay(candidate, query_time, config)
            if cand_decay - original_decay >= min_freshness_gain:
                candidates.append((candidate, cand_decay))

        if not candidates:
            log_rejection(
                doc, REJECT_PAIR_NOT_FOUND,
                f"No fresher {doc.doc_type} in corpus "
                f"(original decay={original_decay:.3f})",
                query_id=query_id,
            )
            return PairResult(original=doc, partner=None, paired=False,
                              reason=f"No qualifying partner for doc_type='{doc.doc_type}'")

        best, best_decay = max(candidates, key=lambda x: x[1])
        return PairResult(
            original=doc, partner=best, paired=True,
            reason=(
                f"Paired '{doc.id}' (decay={original_decay:.3f}) "
                f"with '{best.id}' (decay={best_decay:.3f}, "
                f"gain=+{best_decay - original_decay:.3f})"
            ),
        )

    def execute_retrieval(
        self,
        doc: Document,
        action: str,
        query_time: datetime = None,
        query_id: str = "",
    ) -> list[Document]:
        if query_time is None:
            query_time = datetime.now()

        if action == "DO NOT RETRIEVE":
            if doc.kind == DocumentKind.VERSIONED:
                code = REJECT_EXPIRED_VERSIONED
            elif doc.kind == DocumentKind.EVENT:
                code = REJECT_EXPIRED_EVENT
            else:
                code = REJECT_STALE_STATIC
            log_rejection(doc, code, f"action=DO NOT RETRIEVE", query_id=query_id)
            return []

        if action == "RETRIEVE + PAIR WITH FRESHER SOURCE":
            result = self.find_and_pair(doc, query_time, query_id=query_id)
            print(f"  [pair executor] {result.reason}")
            return [result.original, result.partner] if result.paired else [result.original]

        return [doc]


# ─────────────────────────────────────────────────────────────────────────────
# TIME-RANGE QUERY PARSING + FILTER
# ─────────────────────────────────────────────────────────────────────────────

_RELATIVE_SPANS = [
    (r"\blast\s+(\d+)\s+days?\b",   lambda m, n: (n - timedelta(days=int(m.group(1))),     n)),
    (r"\blast\s+(\d+)\s+weeks?\b",  lambda m, n: (n - timedelta(weeks=int(m.group(1))),    n)),
    (r"\blast\s+(\d+)\s+months?\b", lambda m, n: (n - timedelta(days=int(m.group(1))*30),  n)),
    (r"\bpast\s+(\d+)\s+days?\b",   lambda m, n: (n - timedelta(days=int(m.group(1))),     n)),
    (r"\blast\s+week\b",            lambda m, n: (n - timedelta(weeks=1),                  n)),
    (r"\blast\s+month\b",           lambda m, n: (n - timedelta(days=30),                  n)),
    (r"\blast\s+year\b",            lambda m, n: (n - timedelta(days=365),                 n)),
    (r"\bthis\s+week\b",            lambda m, n: (n - timedelta(days=n.weekday()),         n)),
    (r"\bthis\s+month\b",           lambda m, n: (n.replace(day=1),                        n)),
    (r"\byesterday\b",              lambda m, n: (
        (n - timedelta(days=1)).replace(hour=0,  minute=0,  second=0),
        (n - timedelta(days=1)).replace(hour=23, minute=59, second=59),
    )),
    (r"\btoday\b",                  lambda m, n: (n.replace(hour=0, minute=0, second=0),   n)),
]

_YEAR_RANGE_RE    = re.compile(r'\b(20\d{2}|19\d{2})\s*[-\u2013]\s*(20\d{2}|19\d{2})\b')
_ABSOLUTE_YEAR_RE = re.compile(r'\b(?:in|during|from|around)\s+(20\d{2}|19\d{2})\b', re.I)


def parse_time_range(query: str, now: datetime = None) -> tuple[datetime | None, datetime | None]:
    if now is None:
        now = datetime.now()
    q = query.lower()

    m = _YEAR_RANGE_RE.search(q)
    if m:
        y1, y2 = int(m.group(1)), int(m.group(2))
        return datetime(min(y1,y2), 1, 1), datetime(max(y1,y2), 12, 31, 23, 59, 59)

    m = _ABSOLUTE_YEAR_RE.search(q)
    if m:
        year = int(m.group(1))
        return datetime(year, 1, 1), datetime(year, 12, 31, 23, 59, 59)

    for pattern, resolver in _RELATIVE_SPANS:
        m = re.search(pattern, q)
        if m:
            return resolver(m, now)

    return None, None


def time_range_filter(
    docs: list[Document],
    start: datetime | None,
    end: datetime | None,
    query: str = "",
    query_id: str = "",
) -> tuple[list[Document], list[Document]]:
    if start is None and end is None:
        return docs, []

    kept, rejected = [], []
    for doc in docs:
        out = (start and doc.created_at < start) or (end and doc.created_at > end)
        if out:
            rejected.append(doc)
            log_rejection(doc, REJECT_OUT_OF_TIME_RANGE,
                          f"created_at {doc.created_at.date()} outside "
                          f"[{start.date() if start else '∞'} → "
                          f"{end.date() if end else '∞'}]",
                          query=query, query_id=query_id)
        else:
            kept.append(doc)

    if rejected:
        print(f"  [time-range filter] removed {len(rejected)} doc(s) "
              f"outside [{start.date() if start else '∞'} → "
              f"{end.date() if end else '∞'}]")
    return kept, rejected


# ─────────────────────────────────────────────────────────────────────────────
# RELEVANCE GATE
# ─────────────────────────────────────────────────────────────────────────────

def hard_relevance_gate(
    candidates: list[tuple[Document, float]],
    min_raw_score: float = 0.05,
    query: str = "",
    query_id: str = "",
) -> list[tuple[Document, float]]:
    kept, rejected = [], []
    for doc, score in candidates:
        if score >= min_raw_score:
            kept.append((doc, score))
        else:
            rejected.append((doc, score))
            log_rejection(doc, REJECT_BELOW_RELEVANCE,
                          f"raw cosine {score:.4f} < floor {min_raw_score}",
                          query=query, query_id=query_id)
    if rejected:
        print(f"  [relevance gate] removed {len(rejected)} doc(s) "
              f"below raw cosine floor {min_raw_score}")
    return kept


# ─────────────────────────────────────────────────────────────────────────────
# ADAPTIVE WEIGHTING FROM QUERY LANGUAGE           [ADD-1]
#
# The query itself signals how much recency should matter.
# "What is the current rate limit?" → temporal weight bumps to 0.70.
# "How does cosine similarity work?" → no signal → baseline 0.20.
# ─────────────────────────────────────────────────────────────────────────────

TEMPORAL_SIGNALS = [
    (r"\b(current|latest|now|today|right now)\b",  0.70),
    (r"\b(this week|this month|recently)\b",        0.55),
    (r"\b(still|anymore|yet|has .+ changed)\b",     0.50),
    (r"\b(new|updated|changed|revised)\b",          0.40),
    (r"\b(best|recommend|should I)\b",              0.35),
]

BASELINE_TEMPORAL_WEIGHT = 0.20


def adaptive_retrieve(
    query: str,
    base_config: TemporalConfig = None,
) -> tuple[TemporalConfig, float, str | None]:
    """
    Adjust temporal_weight based on recency signals in the query.

    Returns:
        (config, weight, matched_pattern)
        config          — TemporalConfig with adjusted temporal_weight
        weight          — the weight that was applied
        matched_pattern — the regex that triggered (None if baseline)

    Usage:
        config, weight, signal = adaptive_retrieve(query)
        results = temporal_rag.retrieve(query, temporal_config=config)
    """
    config = base_config or TemporalConfig()
    q = query.lower()

    best_weight:  float      = BASELINE_TEMPORAL_WEIGHT
    best_pattern: str | None = None

    for pattern, weight in TEMPORAL_SIGNALS:
        if re.search(pattern, q) and weight > best_weight:
            best_weight  = weight
            best_pattern = pattern

    adjusted = TemporalConfig(
        decay_half_life_days        = config.decay_half_life_days,
        temporal_weight             = best_weight,
        max_age_days                = config.max_age_days,
        enforce_validity            = config.enforce_validity,
        validity_boost              = config.validity_boost,
        min_vector_score            = config.min_vector_score,
        event_min_raw_vector_score  = config.event_min_raw_vector_score,
    )
    return adjusted, best_weight, best_pattern


# ─────────────────────────────────────────────────────────────────────────────
# FRESHNESS REPORT API                             [ADD-2]
#
# Kind-aware observability surface.  Grades and recommendations are tuned to
# the document's kind, not just its score:
#   VERSIONED → warn about possible replacement
#   EVENT     → warn about window expiry
#   STATIC    → warn about possible supersession
# ─────────────────────────────────────────────────────────────────────────────

_FRESHNESS_GRADES = [
    (0.90, "A", "very fresh"),
    (0.70, "B", "fresh"),
    (0.50, "C", "usable"),
    (0.20, "D", "stale"),
    (0.00, "F", "very stale"),
]


def _freshness_grade(decay: float) -> tuple[str, str]:
    """Return (letter, label) for a decay score."""
    for threshold, letter, label in _FRESHNESS_GRADES:
        if decay >= threshold:
            return letter, label
    return "F", "very stale"


def _freshness_recommendation(
    doc: Document,
    state: ValidityState,
    grade: str,
) -> str:
    """
    Kind-aware recommendation — not just score-aware.

    A VERSIONED document at 0.35 decay gets a version-check warning.
    An EVENT near its window boundary gets a 'verify before serving' flag.
    A STATIC document at near-zero decay gets a supersession warning,
    not an expiry warning, because those are different problems.
    """
    if state == ValidityState.EXPIRED:
        return "DO NOT SERVE — this fact is no longer true."

    if doc.kind == DocumentKind.EVENT and state == ValidityState.TEMPORAL:
        return "Use with caution — EVENT window closes soon. Verify before serving."

    if doc.kind == DocumentKind.VERSIONED:
        if grade in ("D", "F"):
            return "Check for a newer version — VERSIONED document may have been replaced."
        return "Valid version — confirm no newer version exists before serving."

    if doc.kind == DocumentKind.STATIC:
        if grade == "F":
            return "Use with caution — may have been superseded."
        if grade == "D":
            return "Aging — verify no newer source has overturned this fact."
        return "Timeless fact — age is not a concern for this document kind."

    return "Review manually before serving."


def freshness_report(
    doc: Document,
    query_time: datetime = None,
    config: TemporalConfig = None,
) -> str:
    """
    Return a formatted freshness breakdown for a single document.

    Uses TemporalConfig defaults (half_life=30 days) for decay unless
    overridden — the same baseline used throughout the article examples.

    Output format matches the article exactly:

        doc_id [KIND]
          age            : X.X days
          decay score    : X.XXXX
          validity state : STATE
          grade          : X — label
          recommendation : ...
    """
    if query_time is None:
        query_time = datetime.now()
    if config is None:
        config = TemporalConfig()          # default half_life = 30 days

    age   = doc.age_in_days(query_time)
    decay = math.pow(0.5, age / config.decay_half_life_days)
    state = doc.validity_state(query_time)
    letter, label = _freshness_grade(decay)

    decay_str = "0.0" if decay < 0.00005 else f"{decay:.4f}"

    state_name = {
        ValidityState.VALID:    "VALID",
        ValidityState.TEMPORAL: "TEMPORAL",
        ValidityState.EXPIRED:  "EXPIRED",
    }[state]

    kind_label = {
        DocumentKind.STATIC:    "STATIC",
        DocumentKind.VERSIONED: "VERSIONED",
        DocumentKind.EVENT:     "EVENT",
    }[doc.kind]

    rec = _freshness_recommendation(doc, state, letter)

    return (
        f"\n{doc.id} [{kind_label}]\n"
        f"  age            : {age:.1f} days\n"
        f"  decay score    : {decay_str}\n"
        f"  validity state : {state_name}\n"
        f"  grade          : {letter} — {label}\n"
        f"  recommendation : {rec}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# SEQUENCE-AWARE DEDUPLICATION                     [ADD-3]
#
# Prevents v1 and v3 of the same policy from co-appearing in the LLM context.
# Groups VersionedDocuments by sequence_id, keeps only the latest valid version
# before candidates reach the temporal layer.
#
# Article code snippet (abbreviated) reproduced exactly:
#
#   def deduplicate_sequences(self, docs, query_time):
#       for seq_id, versions in sequences.items():
#           valid = [v for v in versions
#                    if v.validity_state(query_time).name != "EXPIRED"]
#           if valid:
#               latest = max(valid, key=lambda v: v.version)
#               deduped.append(latest)
# ─────────────────────────────────────────────────────────────────────────────

class SequenceAwareRetriever:
    """
    Pre-filter that collapses version chains before temporal reranking.

    VersionedDocuments sharing a sequence_id are treated as a chain.
    Only the latest non-EXPIRED version survives into the candidate pool.
    Documents without a sequence_id pass through unchanged.

    Usage:
        retriever = SequenceAwareRetriever()
        deduped = retriever.deduplicate_sequences(candidates, query_time)
        # pass deduped to TemporalLayer.rerank() instead of raw candidates
    """

    def deduplicate_sequences(
        self,
        docs: list[Document],
        query_time: datetime,
    ) -> list[Document]:
        """
        Group by sequence_id, keep only the latest valid version per group.
        Non-versioned documents (no sequence_id) pass through unchanged.
        """
        sequences: dict[str, list[VersionedDocument]] = defaultdict(list)
        deduped:   list[Document] = []

        for doc in docs:
            if isinstance(doc, VersionedDocument) and doc.sequence_id:
                sequences[doc.sequence_id].append(doc)
            else:
                deduped.append(doc)         # non-versioned: pass through

        for seq_id, versions in sequences.items():
            valid = [v for v in versions
                     if v.validity_state(query_time).name != "EXPIRED"]
            if valid:
                latest = max(valid, key=lambda v: v.version)
                deduped.append(latest)
            # If all versions are EXPIRED, none are appended — correct.

        return deduped

    def report(
        self,
        docs: list[Document],
        query_time: datetime,
    ) -> None:
        """Print a human-readable deduplication trace."""
        sequences: dict[str, list[VersionedDocument]] = defaultdict(list)
        passthrough: list[Document] = []

        for doc in docs:
            if isinstance(doc, VersionedDocument) and doc.sequence_id:
                sequences[doc.sequence_id].append(doc)
            else:
                passthrough.append(doc)

        if passthrough:
            print(f"  Pass-through (no sequence_id): "
                  f"{[d.id for d in passthrough]}")

        for seq_id, versions in sequences.items():
            versions_sorted = sorted(versions, key=lambda v: v.version)
            ids = ", ".join(
                f"{v.id} (v{v.version})" for v in versions_sorted
            )
            print(f"\n  Sequence '{seq_id}': {ids}")
            winner = None
            for v in versions_sorted:
                state = v.validity_state(query_time).name
                if state == "EXPIRED":
                    print(f"    {v.id} (v{v.version}) — EXPIRED → removed")
                else:
                    winner = v
            if winner:
                prev = [v for v in versions_sorted if v.id != winner.id
                        and v.validity_state(query_time).name != "EXPIRED"]
                if prev:
                    prev_ids = ", ".join(v.id for v in prev)
                    print(f"    {prev_ids} — superseded by v{winner.version} → removed")
                print(f"    {winner.id} (v{winner.version}) — kept  ✓")
            else:
                print(f"    All versions EXPIRED — sequence dropped entirely.")


# ─────────────────────────────────────────────────────────────────────────────
# DEMO
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    now = datetime.now()

    # ── Improvement 1: PAIR execution ──────────────────────────────────────
    print("\n── Improvement 1: PAIR execution ──")
    print("  Three cases: Invalid → rejected, Weak → PAIR fires, Strong → plain retrieve\n")

    pair_corpus = [
        Document("research_old",
                 "Early bag-of-words models for information retrieval.",
                 created_at=now - timedelta(days=900),
                 doc_type="research", kind=DocumentKind.STATIC),
        Document("research_weak",
                 "Bi-encoder models show strong performance on MS-MARCO.",
                 created_at=now - timedelta(days=272),
                 doc_type="research", kind=DocumentKind.STATIC),
        Document("research_fresh",
                 "Late interaction models outperform bi-encoders on BEIR benchmarks.",
                 created_at=now - timedelta(days=30),
                 doc_type="research", kind=DocumentKind.STATIC),
    ]
    executor = PairExecutor(pair_corpus)

    for doc in pair_corpus:
        config   = get_profile(doc.doc_type)
        decay    = kind_aware_decay(doc, now, config)
        is_valid = doc.validity_state(now).name != "EXPIRED"
        grade, action = _grade_and_action(decay, is_valid)
        qid = generate_query_id()
        retrieved = executor.execute_retrieval(doc, action, query_time=now, query_id=qid)
        print(f"  [{grade}] {doc.id}")
        print(f"    decay={decay:.3f}  grade={grade}  action={action}")
        print(f"    Retrieved: {[d.id for d in retrieved]}")
        print()

    # ── Improvement 2: Confidence — margin + conflict adjustments ──────────
    print("\n── Improvement 2: Confidence with margin + conflict adjustments ──")
    score_examples = [
        ("policy_v3 — clear winner, no conflict",    0.41, 0.924, 0.45, 0.70, 0.00, 0.000, 0.268),
        ("policy_v3 — with conflict, narrow margin", 0.41, 0.924, 0.45, 0.70, 0.10, 0.050, 0.050),
        ("math_theorem",                             0.55, 0.985, 0.01, 0.90, 0.00, 0.000, 1.000),
        ("old_research",                             0.38, 0.463, 0.35, 0.40, 0.00, 0.000, 1.000),
        ("barely_relevant",                          0.12, 0.300, 0.55, 0.20, 0.00, 0.000, 1.000),
    ]
    for label, rel, temp, tw, auth, boost, cpn, margin in score_examples:
        bd = compute_final_score(rel, temp, tw, auth, boost, cpn, margin)
        print_score_breakdown(label, bd)

    # ── Improvement 3: Failure logging with query_id ───────────────────────
    print("\n── Improvement 3: Failure logging with query_id ──")
    qid_a = generate_query_id()
    qid_b = generate_query_id()

    expired_doc = Document(
        "expired_policy", "API limit was 100 rps.",
        created_at=now - timedelta(days=400),
        valid_until=now - timedelta(days=30),
        doc_type="policy", kind=DocumentKind.VERSIONED,
    )
    irrelevant = Document(
        "fresh_irrelevant", "Local sports results.",
        created_at=now - timedelta(hours=1),
        doc_type="news", kind=DocumentKind.STATIC,
    )
    stale_static = Document(
        "stale_reference", "Old embedding methodology.",
        created_at=now - timedelta(days=900),
        doc_type="reference", kind=DocumentKind.STATIC,
    )

    executor_small = PairExecutor([])
    executor_small.execute_retrieval(expired_doc,  "DO NOT RETRIEVE", query_id=qid_a)
    executor_small.execute_retrieval(stale_static, "DO NOT RETRIEVE", query_id=qid_a)
    hard_relevance_gate(
        [(irrelevant, 0.02), (pair_corpus[1], 0.38)],
        query="what are the latest embedding benchmarks?",
        query_id=qid_a,
    )

    log_rejection(pair_corpus[0], REJECT_OUT_OF_TIME_RANGE,
                  "Created 2023, query scoped to 2024", query="2024 embeddings", query_id=qid_b)

    print(f"\n  Failures for query_id={qid_a} only:")
    failure_summary(query_id=qid_a)
    print(f"\n  Full session summary:")
    failure_summary()

    # ── Improvement 4: Adaptive boost + confidence penalty ─────────────────
    print("\n── Improvement 4: Adaptive boost + confidence penalty ──")
    conflict_pairs = [
        ("1000",  "500",  "rate limit halved"),
        ("100",   "5000", "50× increase — severe"),
        ("cache", "redis","non-numeric change"),
        ("1000",  "1000", "no change"),
    ]
    print(f"  {'old':>8}  {'new':<8}  {'severity':>10}  {'boost':>7}  {'conf_pen':>9}  note")
    for old, new, label in conflict_pairs:
        sev   = conflict_severity(old, new)
        boost = adaptive_boost(sev)
        pen   = confidence_penalty_from_conflict(sev)
        print(f"  {old!r:>8}  {new!r:<8}  {sev:>10.3f}  {boost:>7.3f}  {pen:>9.3f}  {label}")

    # ── Improvement 5: Time-range filter ───────────────────────────────────
    print("\n── Improvement 5: Time-range filter (corpus dates match names) ──")

    time_corpus = [
        Document("research_2019",
                 "Early transformer embeddings outperform TF-IDF on NLU tasks.",
                 created_at=datetime(2019, 6, 1),
                 doc_type="research", kind=DocumentKind.STATIC),
        Document("research_2022",
                 "Embeddings with HNSW indices significantly improve ANN recall.",
                 created_at=datetime(2022, 3, 1),
                 doc_type="research", kind=DocumentKind.STATIC),
        Document("research_2024",
                 "Late interaction models outperform bi-encoders on BEIR benchmarks.",
                 created_at=datetime(2024, 1, 15),
                 doc_type="research", kind=DocumentKind.STATIC),
    ]

    print("  Corpus:")
    for d in time_corpus:
        print(f"    {d.id:<20} created={d.created_at.date()}")

    range_queries = [
        ("Show me research from 2021-2023",  "→ expect research_2022 kept"),
        ("What were the findings in 2019?",  "→ expect research_2019 kept"),
        ("Latest embeddings research",        "→ no date filter applied"),
    ]
    for q, note in range_queries:
        start, end = parse_time_range(q, now=now)
        qid = generate_query_id()
        if start:
            print(f"\n  Query: {q!r}  {note}")
            print(f"  Range: [{start.date()} → {end.date()}]")
            kept, removed = time_range_filter(time_corpus, start, end, query=q, query_id=qid)
            print(f"  Kept   : {[d.id for d in kept]}")
            print(f"  Removed: {[d.id for d in removed]}")
        else:
            print(f"\n  Query: {q!r}  {note}")
            print(f"  Range: [no filter] — all {len(time_corpus)} docs pass through")

    # ── Improvement 6: Adaptive weighting from query language ──────────────
    print("\n── Improvement 6: Adaptive weighting from query language ──")

    adaptive_queries = [
        "What is the current rate limit?",
        "How does cosine similarity work?",
        "Has the rate limit policy changed recently?",
        "What should I use for embeddings?",
    ]

    for q in adaptive_queries:
        config_out, weight, pattern = adaptive_retrieve(q)
        if pattern:
            print(f"\n  Query: {q!r}")
            print(f"  Matched signal : {pattern}")
            print(f"  Temporal weight: {weight:.2f}  (bumped from baseline {BASELINE_TEMPORAL_WEIGHT:.2f})")
        else:
            print(f"\n  Query: {q!r}")
            print(f"  No temporal signal detected.")
            print(f"  Temporal weight: {weight:.2f}  (baseline)")

    # ── Improvement 7: Freshness report API ────────────────────────────────
    print("\n── Improvement 7: Freshness report API ──")

    # Documents calibrated to produce the exact output shown in the article.
    # Decay uses TemporalConfig default half_life=30 days.
    #   fresh_event   : age=0.1d  → 0.5^(0.1/30) = 0.9977
    #   current_policy: age=45d   → 0.5^(45/30)  = 0.3536
    #   math_theorem  : age=800d  → 0.5^(800/30) ≈ 0.0
    report_docs = [
        Document(
            "fresh_event",
            "Rate limiting suspended for 48 hours due to infrastructure upgrades.",
            created_at=now - timedelta(hours=2, minutes=24),  # ~0.1 days
            valid_until=now + timedelta(hours=45),
            doc_type="announcement",
            kind=DocumentKind.EVENT,
        ),
        Document(
            "current_policy",
            "API rate limits are set to 1000 requests per minute.",
            created_at=now - timedelta(days=45),
            doc_type="policy",
            kind=DocumentKind.VERSIONED,
        ),
        Document(
            "math_theorem",
            "Cosine similarity measures the angle between two vectors in embedding space.",
            created_at=now - timedelta(days=800),
            doc_type="mathematics",
            kind=DocumentKind.STATIC,
        ),
    ]

    for doc in report_docs:
        print(freshness_report(doc, query_time=now))

    # ── Improvement 8: Sequence-aware deduplication ─────────────────────────
    print("\n\n── Improvement 8: Sequence-aware deduplication ──")

    # Three versions of the same policy. v1 is expired.
    # SequenceAwareRetriever should keep only v3.
    policy_versions = [
        VersionedDocument(
            id="policy_v1", sequence_id="rate-limit-policy", version=1,
            content="API rate limits are set to 100 requests per minute.",
            created_at=now - timedelta(days=540),
            valid_until=now - timedelta(days=180),   # EXPIRED
            doc_type="policy", kind=DocumentKind.VERSIONED,
        ),
        VersionedDocument(
            id="policy_v2", sequence_id="rate-limit-policy", version=2,
            content="API rate limits updated to 1000 requests per minute.",
            created_at=now - timedelta(days=175),
            doc_type="policy", kind=DocumentKind.VERSIONED,
        ),
        VersionedDocument(
            id="policy_v3", sequence_id="rate-limit-policy", version=3,
            content="API rate limits updated to 2000 requests per minute.",
            created_at=now - timedelta(days=10),
            doc_type="policy", kind=DocumentKind.VERSIONED,
        ),
    ]

    seq_retriever = SequenceAwareRetriever()

    print(f"\n  Input  : "
          f"{', '.join(f'{v.id} (v{v.version})' for v in policy_versions)}")
    seq_retriever.report(policy_versions, query_time=now)

    deduped = seq_retriever.deduplicate_sequences(policy_versions, query_time=now)
    print(f"\n  Result : {[d.id for d in deduped]}")
    print(f"  policy_v1 and policy_v2 never reach the LLM.")
