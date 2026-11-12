"""Deterministic quality checks for the bundled synthetic demonstration records."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .parsers import parse_uploaded_file


EXPECTED_FILES = {
    "01_officer_responsibilities.txt",
    "02_fall_meeting_minutes.txt",
    "03_spring_meeting_minutes.txt",
    "04_event_plan_community_skills.txt",
    "05_event_checklist.csv",
    "06_annual_budget.csv",
    "07_vendor_information.txt",
    "08_transition_notes.txt",
    "09_event_approval_guide.txt",
    "10_communications_calendar.txt",
}

REQUIRED_FACTS = {
    "event_date": "October 18, 2026",
    "approval_deadline": "September 18, 2026",
    "budget_deadline": "September 15, 2026",
    "deposit_deadline": "October 4, 2026",
    "headcount_deadline": "October 11, 2026",
    "attendance_target": "80 attendees",
    "archive_location": "Handoff Archive",
}


@dataclass(frozen=True)
class SampleValidationReport:
    file_count: int
    parsed_files: tuple[str, ...]
    verified_facts: tuple[str, ...]


def validate_sample_directory(sample_dir: Path) -> SampleValidationReport:
    """Raise ``ValueError`` when the demonstration corpus is incomplete or inconsistent."""
    paths = sorted(path for path in sample_dir.iterdir() if path.is_file())
    filenames = {path.name for path in paths}
    if filenames != EXPECTED_FILES:
        missing = sorted(EXPECTED_FILES - filenames)
        unexpected = sorted(filenames - EXPECTED_FILES)
        raise ValueError(f"Sample file mismatch. Missing={missing}; unexpected={unexpected}")

    documents = [parse_uploaded_file(path.name, path.read_bytes()) for path in paths]
    corpus = "\n".join(document.text for document in documents)

    missing_facts = [name for name, fact in REQUIRED_FACTS.items() if fact not in corpus]
    if missing_facts:
        raise ValueError(f"Required sample facts are missing: {missing_facts}")

    for fact_name in ("event_date", "approval_deadline", "budget_deadline"):
        fact = REQUIRED_FACTS[fact_name]
        if corpus.count(fact) < 2:
            raise ValueError(f"'{fact}' must be corroborated in at least two sample records")

    if "written approval" not in corpus.lower() or "do not" not in corpus.lower():
        raise ValueError("The sample corpus must preserve the written-approval promotion rule")

    if "synthetic" not in corpus.lower() or "fictional" not in corpus.lower():
        raise ValueError("The sample corpus must clearly identify itself as synthetic and fictional")

    if re.search(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b", corpus):
        raise ValueError("Synthetic sample data must not contain email addresses")
    if re.search(r"\b(?:\+?1[-. ]?)?\(?\d{3}\)?[-. ]\d{3}[-. ]\d{4}\b", corpus):
        raise ValueError("Synthetic sample data must not contain phone numbers")

    return SampleValidationReport(
        file_count=len(documents),
        parsed_files=tuple(document.filename for document in documents),
        verified_facts=tuple(REQUIRED_FACTS),
    )
