"""Live end-to-end evaluation; run explicitly with `python scripts/evaluate.py`."""

from __future__ import annotations

import csv
import os
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from dotenv import load_dotenv
from google.genai.errors import ClientError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from continuity.chunking import chunk_documents
from continuity.llm import (
    LLMConfigurationError,
    LLMRateLimitError,
    _complete,
    answer_question,
    generate_checklist,
)
from continuity.models import Chunk
from continuity.parsers import DocumentParseError, parse_uploaded_file
from continuity.retrieval import retrieve_chunks, retrieve_context
from continuity.suggestions import SUGGESTED_QUESTIONS

FALLBACK = "I could not find that information in the available records."
RESULTS_PATH = ROOT / "evaluation_results.csv"


def load_records() -> list[Chunk]:
    documents = [
        parse_uploaded_file(path.name, path.read_bytes())
        for path in sorted((ROOT / "sample_data").iterdir())
        if path.is_file()
    ]
    if len(documents) != 10:
        raise AssertionError(f"Expected 10 sample documents, found {len(documents)}")
    return chunk_documents(documents)


QUESTIONS = [
    ("President", "What must happen before an event is publicly advertised?", ["written approval"], ["01_officer_responsibilities.txt", "05_event_checklist.csv", "09_event_approval_guide.txt"]),
    ("Treasurer", "When is the fall budget proposal due?", ["September 15"], ["02_fall_meeting_minutes.txt", "06_annual_budget.csv"]),
    ("Event Chair", "What is the attendance target and location for Community Skills Night?", ["80", "Learning Commons"], ["02_fall_meeting_minutes.txt", "04_event_plan_community_skills.txt"]),
    ("Treasurer", "Which vendor requires a deposit, how much is it, and when is it due?", ["Northstar Catering", "25%", "October 4"], ["07_vendor_information.txt"]),
    ("President", "What naming format and archive location should officers use for final records?", ["YYYY-MM-DD_topic", "Handoff Archive"], ["01_officer_responsibilities.txt", "08_transition_notes.txt"]),
    ("Event Chair", "What must be attached to the event approval packet?", ["accessibility plan", "vendor information"], ["09_event_approval_guide.txt"]),
    ("Event Chair", "When is the final catering headcount due for Northstar Catering?", ["October 11"], ["07_vendor_information.txt"]),
    ("President", "What is the event date, the approval deadline, and the promotion rule?", ["October 18", "September 18", "written approval"], ["04_event_plan_community_skills.txt", "09_event_approval_guide.txt"]),
    ("Other", "What is the organization's mascot?", None, []),
    ("President", "Who won the most recent national election?", None, []),
    ("Treasurer", "What does university policy require for federal tax filing?", None, []),
]

SUGGESTED_SCENARIOS = [
    ("President", "What deadlines should I know about?", ["September"], ["01_officer_responsibilities.txt", "02_fall_meeting_minutes.txt", "04_event_plan_community_skills.txt", "05_event_checklist.csv", "06_annual_budget.csv", "09_event_approval_guide.txt"]),
    ("President", "What must happen before an event is promoted?", ["approval"], ["01_officer_responsibilities.txt", "02_fall_meeting_minutes.txt", "05_event_checklist.csv", "09_event_approval_guide.txt", "10_communications_calendar.txt"]),
    ("President", "What did outgoing officers recommend?", ["approval guide"], ["08_transition_notes.txt"]),
    ("Treasurer", "When is the budget proposal due?", ["September 15"], ["01_officer_responsibilities.txt", "02_fall_meeting_minutes.txt", "06_annual_budget.csv"]),
    ("Treasurer", "Which vendors require deposits?", ["Northstar Catering", "25%"], ["07_vendor_information.txt"]),
    ("Treasurer", "What financial tasks happen monthly?", ["reconcile"], ["01_officer_responsibilities.txt"]),
    ("Event Chair", "What steps are required before an event?", ["approval"], ["01_officer_responsibilities.txt", "04_event_plan_community_skills.txt", "05_event_checklist.csv", "09_event_approval_guide.txt"]),
    ("Event Chair", "What is the attendance target?", ["80"], ["02_fall_meeting_minutes.txt", "04_event_plan_community_skills.txt"]),
    ("Event Chair", "What accessibility requirements apply?", ["accessible"], ["02_fall_meeting_minutes.txt", "04_event_plan_community_skills.txt", "05_event_checklist.csv", "09_event_approval_guide.txt"]),
    ("Other", "What are the next important deadlines?", ["September"], ["01_officer_responsibilities.txt", "02_fall_meeting_minutes.txt", "04_event_plan_community_skills.txt", "05_event_checklist.csv", "06_annual_budget.csv", "09_event_approval_guide.txt"]),
    ("Other", "What decisions were made recently?", ["October 18"], ["02_fall_meeting_minutes.txt"]),
    ("Other", "What records should I review first?", [], ["01_officer_responsibilities.txt", "08_transition_notes.txt", "09_event_approval_guide.txt"]),
]

assert {
    (role, question)
    for role, questions in SUGGESTED_QUESTIONS.items()
    for question in questions
} == {(role, question) for role, question, _facts, _sources in SUGGESTED_SCENARIOS}


def cited_files(response: str) -> set[str]:
    return {
        part.split(",", 1)[0].strip()
        for part in response.split("[Source: ")[1:]
        if "]" in part
        for part in [part.split("]", 1)[0]]
    }


def run_questions(
    chunks: list[Chunk],
    scenarios: list[tuple[str, str, list[str] | None, list[str]]] = QUESTIONS,
    scenario_type: str = "question",
) -> list[dict[str, object]]:
    rows = []
    for role, question, facts, sources in scenarios:
        started = time.perf_counter()
        error = ""
        try:
            relevant = retrieve_context(question, chunks)
            response = FALLBACK if not relevant else answer_question(question, role, relevant)
        except Exception as exc:
            response = FALLBACK
            error = f"{type(exc).__name__}: {exc}"
        elapsed = time.perf_counter() - started
        cited = cited_files(response)
        normalized_response = re.sub(
            r"\s+",
            " ",
            response.replace("\u202f", " ")
            .replace("\u2011", "-")
            .replace("\u2013", "-")
            .replace("\u2014", "-"),
        )
        passed = (
            not error and response == FALLBACK
            if facts is None
            else not error
            and response != FALLBACK
            and all(fact.lower() in normalized_response.lower() for fact in facts)
            and bool(cited.intersection(sources))
        )
        rows.append(
            {
                "type": scenario_type,
                "role": role,
                "question": question,
                "response_time_seconds": f"{elapsed:.3f}",
                "pass": passed,
                "citation_correct": bool(cited.intersection(sources)) if sources else response == FALLBACK,
                "cited_files": ";".join(sorted(cited)),
                "error": error,
            }
        )
    return rows


def run_checklists(chunks: list[Chunk]) -> list[dict[str, object]]:
    rows = []
    outputs: dict[str, str] = {}
    for role in ["President", "Treasurer", "Event Chair", "Other"]:
        query = f"{role} responsibilities deadlines first 30 days transition tasks"
        relevant = [chunk for chunk, _score in retrieve_chunks(query, chunks, top_k=8)]
        started = time.perf_counter()
        error = ""
        try:
            response = generate_checklist(role, relevant)
        except Exception as exc:
            response = FALLBACK
            error = f"{type(exc).__name__}: {exc}"
        elapsed = time.perf_counter() - started
        outputs[role] = response
        items = [
            line
            for line in response.splitlines()
            if "[Source:" in line
        ]
        has_citations = bool(items) and all("[Source:" in item for item in items)
        has_deadline = bool(
            re.search(
                r"\b(?:Aug(?:ust)?|Sept(?:ember)?|Oct(?:ober)?)\D{0,4}\d{1,2}\b",
                response,
                re.IGNORECASE,
            )
        )
        passed = not error and response != FALLBACK and has_citations and (role == "Other" or has_deadline)
        rows.append(
            {
                "type": "checklist",
                "role": role,
                "question": "first-30-days checklist",
                "response_time_seconds": f"{elapsed:.3f}",
                "pass": passed,
                "citation_correct": has_citations,
                "cited_files": ";".join(sorted(cited_files(response))),
                "error": error,
            }
        )
    if len({output.strip() for output in outputs.values() if output != FALLBACK}) < 3:
        for row in rows:
            row["pass"] = False
            row["error"] = row["error"] or "Role checklists did not differ meaningfully"
    return rows


def run_robustness(chunks: list[Chunk]) -> None:
    cases = [
        ("empty.txt", b"", "empty"),
        ("unsupported.png", b"not an image", "Unsupported file type"),
        ("corrupt.pdf", b"not a PDF", "Could not read"),
    ]
    for filename, data, expected in cases:
        try:
            parse_uploaded_file(filename, data)
        except DocumentParseError as exc:
            if expected.lower() not in str(exc).lower():
                raise AssertionError(f"Unexpected error for {filename}: {exc}")
        else:
            raise AssertionError(f"Expected parsing error for {filename}")
    assert retrieve_chunks("?", chunks) == []


def run_configuration_checks(chunks: list[Chunk]) -> None:
    original = os.environ.pop("GEMINI_API_KEY", None)
    try:
        try:
            answer_question("What is the mascot?", "Other", chunks)
        except LLMConfigurationError as exc:
            assert "GEMINI_API_KEY is missing" in str(exc)
        else:
            raise AssertionError("Missing-key check did not raise")
    finally:
        if original is not None:
            os.environ["GEMINI_API_KEY"] = original

    client = Mock()
    client.models.generate_content.side_effect = ClientError(
        429, {"error": {"message": "rate limited"}}
    )
    with patch("continuity.llm._client", return_value=client):
        try:
            _complete("system", "records")
        except LLMRateLimitError as exc:
            assert "try again" in str(exc)
        else:
            raise AssertionError("Rate-limit check did not raise")


def write_results(rows: list[dict[str, object]]) -> None:
    fields = ["type", "role", "question", "response_time_seconds", "pass", "citation_correct", "cited_files", "error"]
    with RESULTS_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    chunks = load_records()
    rows = run_questions(chunks)
    rows.extend(run_questions(chunks, SUGGESTED_SCENARIOS, "suggested_question"))
    rows.extend(run_checklists(chunks))
    run_robustness(chunks)
    run_configuration_checks(chunks)
    write_results(rows)
    question_rows = [row for row in rows if row["type"] in {"question", "suggested_question"}]
    citation_rows = [row for row in rows if row["citation_correct"]]
    correct_fallbacks = sum(
        row["question"] in {question for _role, question, _facts, _sources in QUESTIONS if _facts is None}
        and row["pass"]
        for row in question_rows
    )
    average_time = sum(float(row["response_time_seconds"]) for row in rows) / len(rows)
    failed_rows = [row for row in rows if not row["pass"]]
    print(f"EVALUATION={'pass' if not failed_rows else 'fail'} questions={len(question_rows)} checklists=4")
    print(f"Accuracy: {sum(bool(row['pass']) for row in question_rows) / len(question_rows) * 100:.1f}%")
    print(f"Citation accuracy: {len(citation_rows) / len(rows) * 100:.1f}%")
    print(f"Average response time: {average_time:.3f}s")
    print(f"Correct fallback responses: {correct_fallbacks}")
    print(f"Results: {RESULTS_PATH.name}")
    if failed_rows:
        failed_names = ", ".join(f"{row['role']}: {row['question']}" for row in failed_rows)
        raise AssertionError(f"{len(failed_rows)} evaluation scenario(s) failed: {failed_names}")


if __name__ == "__main__":
    main()
