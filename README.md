# Healthcare Intake, PHI Redaction & Abstention RAG Sandbox (Synthetic Demo)

> **Important Disclosure:** This repository is a **synthetic Python demonstration** created specifically to illustrate deterministic guardrails for an Upwork proposal. It uses synthetic in-memory fixtures only and is **not** a prior commercial client project or clinical deployment.

## What This Synthetic Sandbox Demonstrates

1. **Deterministic PII/PHI Redaction (`redact_phi_text`)**: Strips SSN, MRN, phone, and email patterns prior to external LLM or embedding calls and records redaction counts.
2. **Confidence-Gated Document Intake (`evaluate_intake_extraction`)**: Validates required healthcare intake fields and routes any field with confidence `< 0.85` or invalid ISO date formatting to `needs_human_review` instead of silently committing uncertain extractions.
3. **Citation-Grounded SOP Retrieval With Abstention (`retrieve_and_ground_sop`)**: Returns inline `[SOP-xxx]` chunk citations when lexical/cosine overlap meets threshold and abstains (`ABSTAIN_TO_HUMAN`) on out-of-scope questions.

## Run Contract Tests

```bash
pytest -v
```
