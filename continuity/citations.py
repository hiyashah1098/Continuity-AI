"""Parse model citations into UI-friendly source references."""

from __future__ import annotations

import re
from dataclasses import dataclass


CITATION_PATTERN = re.compile(r"\[Source:\s*([^\]]+)\]")


@dataclass(frozen=True)
class SourceReference:
    filename: str
    location: str = ""


def extract_source_references(
    text: str, available_filenames: set[str]
) -> list[SourceReference]:
    """Return ordered, unique source references from one model-output line."""
    references: list[SourceReference] = []
    for match in CITATION_PATTERN.finditer(text):
        parts = [part.strip() for part in match.group(1).split(",") if part.strip()]
        current_filename = ""
        for part in parts:
            if part in available_filenames:
                current_filename = part
                reference = SourceReference(filename=part)
                if reference not in references:
                    references.append(reference)
            elif current_filename:
                located_reference = SourceReference(
                    filename=current_filename,
                    location=part,
                )
                references = [
                    reference
                    for reference in references
                    if reference.filename != current_filename
                    or reference.location
                ]
                if located_reference not in references:
                    references.append(located_reference)
    return references


def strip_source_citations(text: str) -> str:
    """Remove bracketed citations while preserving readable Markdown."""
    cleaned = CITATION_PATTERN.sub("", text)
    cleaned = re.sub(r"[ \t]+([.,;:])", r"\1", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned.rstrip()
