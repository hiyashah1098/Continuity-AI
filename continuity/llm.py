"""Hosted LLM integration with prompts that enforce document-only answers."""

from __future__ import annotations

import os
import re
from threading import Lock

from google import genai
from google.genai import types
from google.genai.errors import ClientError

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
    """Raised when Gemini temporarily rate-limits a request."""


_client_instance: genai.Client | None = None
_client_key: str | None = None
_client_lock = Lock()


def _client() -> genai.Client:
    global _client_instance, _client_key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your_key_here":
        raise LLMConfigurationError(
            "GEMINI_API_KEY is missing. Add it to a local .env file. "
            "Get a key at https://aistudio.google.com/apikey."
        )
    with _client_lock:
        if _client_instance is None or _client_key != api_key:
            _client_instance = genai.Client(api_key=api_key)
            _client_key = api_key
        return _client_instance


def _reset_client() -> None:
    global _client_instance, _client_key
    with _client_lock:
        _client_instance = None
        _client_key = None


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
    return bool(cited_filenames) and cited_filenames.issubset(available_filenames)


def _complete(system_prompt: str, user_prompt: str) -> str:
    for attempt in range(2):
        try:
            response = _client().models.generate_content(
                model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0,
                ),
            )
            break
        except ClientError as exc:
            if getattr(exc, "code", None) == 429:
                raise LLMRateLimitError(
                    "Gemini is temporarily rate-limiting requests. Please wait a moment and try again."
                ) from exc
            raise RuntimeError(f"Gemini request failed: {exc}") from exc
        except LLMConfigurationError:
            raise
        except RuntimeError as exc:
            if attempt == 0 and "client has been closed" in str(exc).lower():
                _reset_client()
                continue
            raise RuntimeError(f"Gemini request failed: {exc}") from exc
    else:
        raise RuntimeError("Gemini request failed after recreating its client.")
    answer = response.text
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
