"""Synthetic Python demonstration for healthcare document intake, PII/PHI redaction,
and citation-grounded SOP retrieval with explicit confidence-abstention routing.

Created as a synthetic demonstration for Upwork proposal review; uses illustrative
in-memory fixtures only and is not a previous client or clinical deployment.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import re
from typing import Any


PHI_PATTERNS = {
    "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "MRN": re.compile(r"\bMRN[-:\s]*\d{5,10}\b", re.IGNORECASE),
    "PHONE": re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    "EMAIL": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
}


@dataclass
class RedactionResult:
    sanitized_text: str
    redacted_counts: dict[str, int]
    total_redactions: int


def redact_phi_text(raw_text: str) -> RedactionResult:
    """Deterministic regex redaction gate before any external LLM or embedding call."""
    text = raw_text
    counts: dict[str, int] = {}
    for label, pat in PHI_PATTERNS.items():
        matches = list(pat.finditer(text))
        counts[label] = len(matches)
        if matches:
            text = pat.sub(f"[REDACTED_{label}]", text)
    return RedactionResult(
        sanitized_text=text,
        redacted_counts=counts,
        total_redactions=sum(counts.values()),
    )


@dataclass
class ExtractionDecision:
    status: str  # "auto_commit" or "needs_human_review"
    validated_fields: dict[str, Any]
    flagged_fields: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)


REQUIRED_INTAKE_FIELDS = ("clinic_suite_id", "service_category", "requested_date", "insurance_plan_code")


def evaluate_intake_extraction(
    extracted_fields: dict[str, Any],
    field_confidences: dict[str, float],
    confidence_threshold: float = 0.85,
) -> ExtractionDecision:
    """Route extracted PDF/form fields to auto_commit only when all required fields
    are present and meet the calibrated confidence threshold; otherwise abstain to staff review.
    """
    flagged: list[str] = []
    reasons: list[str] = []
    clean_fields: dict[str, Any] = {}

    for req in REQUIRED_INTAKE_FIELDS:
        val = extracted_fields.get(req)
        conf = float(field_confidences.get(req, 0.0))
        if val is None or str(val).strip() == "":
            flagged.append(req)
            reasons.append(f"Missing required field: {req}")
            continue
        if conf < confidence_threshold:
            flagged.append(req)
            reasons.append(f"Low confidence on {req} ({conf:.2f} < {confidence_threshold:.2f})")
            continue
        clean_fields[req] = str(val).strip()

    if "requested_date" in clean_fields and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", clean_fields["requested_date"]):
        flagged.append("requested_date")
        reasons.append("Invalid ISO date format for requested_date")

    status = "auto_commit" if not flagged else "needs_human_review"
    return ExtractionDecision(
        status=status,
        validated_fields=clean_fields,
        flagged_fields=sorted(set(flagged)),
        reasons=reasons,
    )


@dataclass(frozen=True)
class SopChunk:
    chunk_id: str
    section: str
    text: str


@dataclass
class GroundedAnswer:
    abstained: bool
    top_score: float
    cited_chunk_ids: list[str]
    response_text: str


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def retrieve_and_ground_sop(
    query: str,
    chunks: list[SopChunk],
    min_overlap_score: float = 0.28,
) -> GroundedAnswer:
    """Deterministic lexical cosine similarity retrieval with strict abstention
    when the query is outside approved operational SOP chunks.
    """
    q_tokens = set(_tokenize(query))
    if not q_tokens or not chunks:
        return GroundedAnswer(
            abstained=True,
            top_score=0.0,
            cited_chunk_ids=[],
            response_text="ABSTAIN_TO_HUMAN: Insufficient query or empty SOP knowledge base.",
        )

    scored: list[tuple[float, SopChunk]] = []
    for ch in chunks:
        c_tokens = set(_tokenize(ch.text + " " + ch.section))
        if not c_tokens:
            continue
        inter = len(q_tokens & c_tokens)
        score = inter / math.sqrt(len(q_tokens) * len(c_tokens))
        scored.append((score, ch))

    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best_chunk = scored[0]
    if best_score < min_overlap_score:
        return GroundedAnswer(
            abstained=True,
            top_score=round(best_score, 4),
            cited_chunk_ids=[],
            response_text="ABSTAIN_TO_HUMAN: Query falls outside approved SOP threshold.",
        )

    cited = [ch.chunk_id for sc, ch in scored[:2] if sc >= min_overlap_score]
    return GroundedAnswer(
        abstained=False,
        top_score=round(best_score, 4),
        cited_chunk_ids=cited,
        response_text=f"[{best_chunk.chunk_id}] ({best_chunk.section}): {best_chunk.text}",
    )
