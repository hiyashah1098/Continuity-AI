# Continuity AI Product Write-up

## Product-track fit and industry

Continuity AI is a Track 2 product prototype for higher education, campus
engagement, and student-organization administration. An incoming officer can
upload records, ask a grounded question, inspect evidence, generate a
role-specific first-30-days checklist, and receive a proactive handoff audit.
The live flow uses the Gemini API and local TF-IDF retrieval; it is not a
clickable mockup or canned demonstration.

## User, problem, and business action

The primary user is an incoming President, Treasurer, Event Chair, or other
student-organization officer. Responsibilities, deadlines, approvals, past
decisions, and vendor details are often scattered across meeting notes, shared
drives, budgets, and transition documents. The resulting risks are missed
deadlines, duplicated work, repeated questions to advisers, and institutional
knowledge loss when officers graduate.

The business action is concrete: the incoming officer uses cited answers and a
first-30-days plan to complete the right transition work on time. A university
student-affairs or campus-engagement department is the most plausible payer
because it supports many organizations and bears the advising and compliance
cost of poor transitions.

## Core flow and decision logic

1. Select the incoming role.
2. Load the synthetic demo corpus or upload authorized PDF, DOCX, TXT, or CSV
   records.
3. Continuity AI extracts and chunks the records locally.
4. TF-IDF retrieves relevant passages for a question or checklist request.
5. Gemini summarizes only those passages and must cite an uploaded filename.
6. The app rejects outputs with missing or unknown citations and uses an exact
   fallback when evidence is unavailable.
7. After records are loaded, the automatic handoff audit scans across them for
   actions, deadlines, conflicts, and missing information.
8. The officer verifies the displayed excerpts and downloads the audit or plan.

The output-to-action rule is: supported evidence becomes a proposed task;
unsupported questions receive a fallback; conflicts or high-impact deadlines
must be verified with the organization or university before action.

## AI and bounded agentic workflow

The agentic behavior is the post-upload handoff audit. It triggers proactively,
reviews multiple records together, selects evidence, identifies risks, validates
citations, recovers from a closed Gemini client, and produces a prioritized
report. It is deliberately bounded: it does not send messages, edit calendars,
approve events, or perform consequential actions for an officer.

Uploaded records are treated as untrusted evidence rather than instructions to
the model. The interface requires authorization before user-uploaded text is
sent to Gemini and recommends the synthetic records for demonstration.

## Minimum working version and stretch scope

The committed MVP includes multi-format ingestion, grounded questions,
citations and evidence, role-specific checklists, the automatic audit, sample
data, downloads, error handling, and a public deployment. Stretch features for
a real product are organization workspaces, authentication and access control,
persistent storage, Google Drive and calendar integrations, notifications,
administrator analytics, and usage-based cost controls.

## Success metrics and evaluation

The three product success metrics are:

1. **Grounded task accuracy:** expected facts appear in realistic answers and
   applicable role checklists.
2. **Citation and fallback correctness:** supported outputs cite an appropriate
   source, while unsupported questions return the exact fallback.
3. **Task efficiency:** response time is recorded now; manual-versus-assisted
   completion time will be measured in an external-user study.

The committed live-Gemini evaluation covers 11 realistic questions and four
role checklists. It also checks parsing failures, zero-overlap retrieval,
missing-key behavior, and simulated rate limiting. The committed run reports
15 of 15 scenario outputs passing their fact and citation/fallback checks.
This is automated realistic-scenario testing, not a claim that outside users
have completed a usability study.

### What testing changed

| Observed risk or failure | Product change | Verification |
|---|---|---|
| Models varied citation bracket formatting | Normalize supported citation brackets and reject unknown filenames | Unit tests for valid, absent, and unknown citations |
| A reused hosted client could be closed | Recreate the Gemini client once and retry | Mocked closed-client test |
| Corrupt or empty uploads produced unclear failures | Validate file type, PDF header, and readable text | Parser robustness tests |
| Unsupported questions could invite guesses | Enforce a single exact fallback | Three out-of-record evaluation questions |
| Automatic audits could repeat on Streamlit reruns | Hash the upload set and run once per session state | Upload-signature logic |
| Demo records lacked explicit provenance and years | Add fictional/synthetic notices, full dates, corroboration, and deterministic corpus validation | `scripts/validate_sample_data.py` |

## Business case and illustrative ROI

Continuity AI improves manual shared-drive searching and repeated handoff
questions. It does not replace advisers; it reduces low-value record retrieval
so advisers and officers can focus on judgment and relationship-building.

An illustrative campus scenario - not a measured result - uses these assumptions:

- 20 supported organizations;
- three officer handoffs per organization each year;
- two hours saved per handoff;
- $20 of staff/student-leader time value per hour; and
- $600 in estimated annual hosting, model, and maintenance cost.

That scenario produces $2,400 in annual time value, $1,800 in net value, and
300% illustrative ROI: `($2,400 - $600) / $600`. The live Product Evidence tab
lets a reviewer change every assumption. A production decision would require
measured task-time savings, actual provider usage, security costs, and a pilot
with campus organizations.

## Deployment and monitoring

The prototype is deployed on Streamlit Community Cloud from the GitHub
repository. Production monitoring should track scenario/fallback rate,
invalid-citation rate, upload and parsing failures, Gemini errors, latency,
estimated cost per session, user-reported incorrect answers, and high-impact
deadlines that users flag for correction. Alerts should be configured for
sudden increases in failures, latency, or spend.

## Ethics, risks, and mitigations

- **Hallucinated policy:** restrict answers to retrieved records, reject missing
  or unknown citations, and display exact evidence.
- **Prompt injection in records:** explicitly treat uploaded instructions as
  untrusted data and never as directions to the model.
- **Missed or conflicting deadlines:** audit across documents and label
  uncertainty instead of silently resolving it.
- **Sensitive student data:** process files in memory, require processing
  authorization, protect secrets, and recommend synthetic data for demos.
- **Over-reliance:** position the product as a research aid and require human
  verification of compliance, money, and deadline decisions.
- **Access and retention:** the MVP intentionally does not persist uploads; a
  production version requires authentication, least-privilege access, retention
  controls, and audit logs.
- **Provider availability and cost:** provide clear configuration/rate-limit
  errors, keep retrieval local, and monitor usage.

## GenAI usage statement

GenAI helped with parser scaffolding, prompt design, provider integration,
interface iteration, documentation, and test generation. The team still had to
inspect failures and correct citation variability, closed-client behavior,
corrupt-PDF handling, secret hygiene, demo-data consistency, and overly broad
trust in generated outputs. The main lesson was that an LLM response alone is
not a product: reliable retrieval, source validation, transparent evidence,
evaluation, privacy boundaries, and explicit error handling are all required.

## Remaining validation

Before presenting measured task-time savings or production ROI as facts, the
team should have three to five people outside the team complete the same search
tasks manually and with Continuity AI. Record completion rate, time, incorrect
answers, confidence, and usability feedback, then document the changes made.
