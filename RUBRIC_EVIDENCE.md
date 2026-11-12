# Track 2 Rubric Evidence

This file maps the current repository to the MIST 5400 Track 2 product
requirements. It distinguishes completed evidence from work that still requires
people outside the team.

## Working prototype

- Public Streamlit deployment linked from the README.
- End-to-end role selection, record loading, question answering, evidence
  display, checklist generation, and proactive audit.
- Real Gemini calls through `continuity/llm.py`; no canned answer path.
- PDF, DOCX, TXT, and CSV parsing with clear errors.

## Product and business fit

- User: incoming student-organization officers.
- Problem: fragmented transition records and institutional knowledge loss.
- Action: complete cited transition tasks and verify high-impact deadlines.
- Plausible payer: university student-affairs or campus-engagement department.
- Illustrative ROI assumptions and an interactive calculator are in the Product
  Evidence tab. Estimates are explicitly labeled as unmeasured.

## Evaluation

- Three success metrics are defined in `PRODUCT_WRITEUP.md`.
- `scripts/evaluate.py` runs 11 realistic questions and four role checklists
  with live Gemini calls.
- `evaluation_results.csv` contains the latest committed scenario results.
- `python -m pytest -q` covers parsing, retrieval, provider errors, citation behavior,
  audit behavior, and sample-corpus quality.
- `scripts/validate_sample_data.py` verifies the ten-file synthetic corpus
  without an API call.
- External-user task timing is not yet claimed; the planned protocol appears in
  `PRODUCT_WRITEUP.md`.

## Deployment, monitoring, ROI, and ethics

- Streamlit Community Cloud deployment instructions are documented.
- Monitoring targets include failures, latency, citation/fallback rate, cost,
  and user-reported errors.
- Guardrails include visible evidence, exact fallback behavior, citation
  validation, prompt-injection instructions, in-memory processing, and an
  upload authorization step.
- Production gaps are documented: authentication, storage, access control,
  retention, audit logs, and external-user validation.

## GenAI usage

The product write-up identifies what GenAI helped build, what required human
debugging, and what the team learned about directing and validating generated
work.

## Presentation items outside the repository

The team must still prepare and rehearse the required proposal and final demo
decks. Delivery timing, participation by every team member, and answers to live
questions cannot be established by source code.
