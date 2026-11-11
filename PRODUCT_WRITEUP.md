# Continuity AI Product Write-up

## Product-track fit

Continuity AI is a Track 2 product prototype: a student-organization officer
can upload records, ask a grounded question, inspect evidence, and generate a
role-specific first-30-days checklist. The live flow uses the Gemini API and
local TF-IDF retrieval; it is not a canned demo.

## User and business problem

The primary user is an incoming President, Treasurer, Event Chair, or other
student-organization officer. Their operational problem is that responsibilities,
deadlines, past decisions, and vendor details are scattered across handoff
records. The business action is to complete the right transition tasks on time
with less searching and fewer repeated questions to outgoing officers or
advisers.

## Core flow and decision logic

1. Select the incoming role.
2. Upload PDF, DOCX, TXT, or CSV records.
3. Continuity AI extracts and chunks the records.
4. TF-IDF retrieves relevant passages for a question or checklist request.
5. Gemini summarizes only those passages and must cite a source.
6. The app rejects uncited answers and uses the exact fallback when evidence is
   unavailable.
7. After upload, the automatic handoff audit independently scans all records for
   priority actions, deadlines, conflicts, and missing information.
8. The officer can download the audit or checklist for transition use.

## AI and agent workflow

The agentic behavior is the post-upload handoff audit. It runs without a
question, reviews all records together, identifies actionable risks, and
produces a prioritized cited report. The system also automatically selects
evidence, validates citations against uploaded filenames, retries a closed
Gemini client, and surfaces exact retrieved passages for verification.

## Success metrics and evaluation

The project evaluates:

- **Grounded answer accuracy:** required facts appear in answers to realistic
  questions.
- **Citation accuracy:** at least one cited filename is a correct supporting
  source.
- **Fallback precision:** unavailable questions return the exact fallback.
- **Checklist usability:** each role receives cited tasks and available dates.
- **Task-completion time:** compare manual record search time with assisted time
  in an external user test.

Run the automated realistic-scenario evaluation with:

```text
python scripts/evaluate.py
```

It records per-case pass/fail, citation correctness, source files, and response
time in `evaluation_results.csv`. The script includes 11 questions, four role
checklists, parsing robustness, missing-key behavior, and rate-limit simulation.

## Business case and deployment

The prototype can be deployed on Streamlit Community Cloud. An organization
would gain a reusable transition workspace without changing its existing record
keeping. The main operating cost is hosted Gemini usage; TF-IDF retrieval is
local and does not require embeddings. A production version could add
organization workspaces, access controls, persistent storage, Google Drive
connectors, calendar reminders, and usage-based cost monitoring.

## Ethics, risks, and mitigations

- **Hallucinated policy:** restrict answers to uploaded records and require
  citations; show the exact retrieved evidence.
- **Missed or conflicting deadlines:** include a cross-document audit and label
  conflicts or missing information instead of silently resolving them.
- **Sensitive student data:** process uploads in memory, do not commit `.env`,
  and avoid storing user records in the MVP.
- **Over-reliance on automation:** present the assistant as a research aid and
  require officers to verify high-impact deadlines with the organization or
  university.
- **Provider availability and cost:** show configuration/rate-limit errors and
  keep retrieval local.

## GenAI usage statement

GenAI was used to help implement and test the prototype, including parser
scaffolding, prompt design, provider migration, and test generation. The team
had to inspect failures, fix citation-format variability, handle a closed
Gemini client, validate corrupted PDFs, and enforce secret hygiene. The main
lesson was that an LLM response is not enough: retrieval, source validation,
fallback behavior, evaluation, and explicit error handling are required for a
trustworthy product.
