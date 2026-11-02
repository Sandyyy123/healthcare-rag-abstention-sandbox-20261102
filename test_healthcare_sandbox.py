"""Unit tests for the synthetic healthcare intake, redaction, and SOP RAG sandbox."""
from __future__ import annotations

from healthcare_sandbox import (
    SopChunk,
    evaluate_intake_extraction,
    redact_phi_text,
    retrieve_and_ground_sop,
)


SAMPLE_CHUNKS = [
    SopChunk(
        chunk_id="SOP-101",
        section="Infusion Scheduling & Prior Authorization",
        text="Prior authorization codes and insurance_plan_code must be confirmed 48 hours before infusion suite booking.",
    ),
    SopChunk(
        chunk_id="SOP-204",
        section="Document Intake Escalation",
        text="Scanned PDF forms with field confidence below 0.85 are routed to the staff review queue and never auto-committed.",
    ),
    SopChunk(
        chunk_id="SOP-309",
        section="Portal Role Permissions",
        text="Clinic suite availability updates require an authenticated manager session and audit log entry.",
    ),
]


def test_phi_redaction_strips_ssn_mrn_phone_and_email():
    raw = "Patient MRN-44981203 (SSN 123-45-6789) called 305-555-0199 and emailed jane.doe@example.org for suite booking."
    res = redact_phi_text(raw)
    assert res.total_redactions == 4
    assert "123-45-6789" not in res.sanitized_text
    assert "MRN-44981203" not in res.sanitized_text
    assert "305-555-0199" not in res.sanitized_text
    assert "jane.doe@example.org" not in res.sanitized_text
    assert "[REDACTED_SSN]" in res.sanitized_text


def test_clean_text_has_zero_false_redactions():
    raw = "Suite 12B infusion chair schedule updated for 2026-11-15 under standard operational protocol."
    res = redact_phi_text(raw)
    assert res.total_redactions == 0
    assert res.sanitized_text == raw


def test_high_confidence_intake_auto_commits():
    fields = {
        "clinic_suite_id": "SUITE-04",
        "service_category": "IV_INFUSION",
        "requested_date": "2026-11-18",
        "insurance_plan_code": "FL-PPO-882",
    }
    confs = {k: 0.94 for k in fields}
    dec = evaluate_intake_extraction(fields, confs, confidence_threshold=0.85)
    assert dec.status == "auto_commit"
    assert dec.flagged_fields == []
    assert dec.validated_fields["clinic_suite_id"] == "SUITE-04"


def test_low_confidence_field_routes_to_human_review():
    fields = {
        "clinic_suite_id": "SUITE-04",
        "service_category": "IV_INFUSION",
        "requested_date": "2026-11-18",
        "insurance_plan_code": "FL-PPO-??",
    }
    confs = {
        "clinic_suite_id": 0.96,
        "service_category": 0.92,
        "requested_date": 0.91,
        "insurance_plan_code": 0.62,
    }
    dec = evaluate_intake_extraction(fields, confs, confidence_threshold=0.85)
    assert dec.status == "needs_human_review"
    assert "insurance_plan_code" in dec.flagged_fields


def test_missing_required_field_routes_to_human_review():
    fields = {
        "clinic_suite_id": "SUITE-04",
        "service_category": "IV_INFUSION",
        "requested_date": "2026-11-18",
    }
    confs = {k: 0.95 for k in fields}
    dec = evaluate_intake_extraction(fields, confs, confidence_threshold=0.85)
    assert dec.status == "needs_human_review"
    assert "insurance_plan_code" in dec.flagged_fields


def test_malformed_date_rejected_by_schema_guard():
    fields = {
        "clinic_suite_id": "SUITE-04",
        "service_category": "IV_INFUSION",
        "requested_date": "18/11/2026",
        "insurance_plan_code": "FL-PPO-882",
    }
    confs = {k: 0.95 for k in fields}
    dec = evaluate_intake_extraction(fields, confs, confidence_threshold=0.85)
    assert dec.status == "needs_human_review"
    assert "requested_date" in dec.flagged_fields


def test_in_scope_sop_query_returns_inline_chunk_citation():
    ans = retrieve_and_ground_sop(
        "When must prior authorization and insurance_plan_code be confirmed before infusion suite booking?",
        SAMPLE_CHUNKS,
    )
    assert ans.abstained is False
    assert "SOP-101" in ans.cited_chunk_ids
    assert "[SOP-101]" in ans.response_text


def test_out_of_scope_query_abstains_cleanly():
    ans = retrieve_and_ground_sop(
        "What is the stock ticker price forecast for international shipping containers?",
        SAMPLE_CHUNKS,
    )
    assert ans.abstained is True
    assert ans.cited_chunk_ids == []
    assert ans.response_text.startswith("ABSTAIN_TO_HUMAN:")
