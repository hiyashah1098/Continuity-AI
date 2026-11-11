"""Hosted LLM integration with prompts that enforce document-only answers."""

from __future__ import annotations

import os
import re

from groq import Groq
from groq import RateLimitError

from .models import Chunk
from .prompts import (
    ANSWER_SYSTEM_PROMPT,
    AUDIT_SYSTEM_PROMPT,
    CHECKLIST_SYSTEM_PROMPT,
    FALLBACK,
)


class LLMConfigurationError(RuntimeError):
    """Raised when the hosted model is not configured."""


class LLMRateLimitError(RuntimeError):
    """Raised when Groq temporarily rate-limits a request."""


def _client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "your_key_here":
        raise LLMConfigurationError(
            "GROQ_API_KEY is missing. Add it to a local .env file. "
            "Get a key at https://console.groq.com/keys."
        )
    return Groq(api_key=api_key)


def _records(chunks: list[Chunk]) -> str:
    return "\n\n".join(f"[Source: {chunk.label}]\n{chunk.text}" for chunk in chunks)


def _normalize_citations(answer: str) -> str:
    """Normalize citation brackets used by some chat models to the app format."""
    return re.sub(r"【Source: ([^】]+)】", r"[Source: \1]", answer)


def _has_valid_citation(answer: str, chunks: list[Chunk]) -> bool:
    cited_filenames = {
        match.group(1).strip()
        for match in re.finditer(r"\[Source: ([^,\]]+)", answer)
    }
    available_filenames = {chunk.filename for chunk in chunks}
    return bool(cited_filenames.intersection(available_filenames))


def _complete(system_prompt: str, user_prompt: str) -> str:
    try:
        response = _client().chat.completions.create(
            model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
    except RateLimitError as exc:
        raise LLMRateLimitError(
            "Groq is temporarily rate-limiting requests. Please wait a moment and try again."
        ) from exc
    answer = response.choices[0].message.content
    if not answer:
        raise RuntimeError("The hosted model returned an empty response.")
    return answer.strip()


def answer_question(question: str, role: str, chunks: list[Chunk]) -> str:
    answer = _normalize_citations(_complete(
        ANSWER_SYSTEM_PROMPT,
        f"Officer role: {role}\nQuestion: {question}\n\nRECORDS:\n{_records(chunks)}",
    ))
    if answer == FALLBACK:
        return answer
    if not _has_valid_citation(answer, chunks):
        return FALLBACK
    return answer


def generate_checklist(role: str, chunks: list[Chunk]) -> str:
    prompt = f"Officer role: {role}\n\nRECORDS:\n{_records(chunks)}"
    focused_chunks = [
        chunk
        for chunk in chunks
        if role.lower() in chunk.text.lower() or "all officers" in chunk.text.lower()
    ]
    focused_prompt = (
        f"Officer role: {role}\n\nROLE-RELEVANT RECORDS:\n"
        f"{_records(focused_chunks or chunks)}"
        "\n\nCreate a checklist from every applicable responsibility and deadline above."
    )
    checklist = FALLBACK
    prompts = [
        prompt,
        prompt
        + "\n\nThe records include role responsibilities and deadlines. "
        "Create the checklist from those records instead of returning the fallback.",
        focused_prompt,
    ]
    for attempt_prompt in prompts:
        extra_instruction = (
            ""
            if attempt_prompt == focused_prompt
            else "\n\nDo not return the fallback when the records contain an applicable task."
        )
        checklist = _normalize_citations(
            _complete(CHECKLIST_SYSTEM_PROMPT, attempt_prompt + extra_instruction)
        )
        if checklist != FALLBACK:
            break
    if checklist == FALLBACK:
        return checklist
    if not _has_valid_citation(checklist, chunks):
        return FALLBACK
    return checklist


def audit_handoff(chunks: list[Chunk]) -> str:
    """Review all uploaded records and return a cited, prioritized audit."""
    if not chunks:
        return FALLBACK
    audit = _normalize_citations(
        _complete(
            AUDIT_SYSTEM_PROMPT,
            "RECORDS TO AUDIT:\n" + _records(chunks),
        )
    )
    if not _has_valid_citation(audit, chunks):
        raise RuntimeError("The audit response did not include valid document citations.")
    return audit
