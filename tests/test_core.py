import os
from types import SimpleNamespace
from unittest.mock import Mock, patch

from continuity.chunking import chunk_documents
from continuity.llm import (
    LLMConfigurationError,
    LLMRateLimitError,
    _complete,
    audit_handoff,
    answer_question,
    generate_checklist,
)
from continuity.models import Document
from continuity.parsers import DocumentParseError, parse_uploaded_file
from continuity.retrieval import retrieve_chunks


def test_txt_parser_and_retrieval():
    document = parse_uploaded_file("minutes.txt", b"The budget proposal is due September 15.")
    chunks = chunk_documents([document])
    results = retrieve_chunks("When is the budget proposal due?", chunks)
    assert results
    assert "September 15" in results[0][0].text


def test_csv_parser():
    document = parse_uploaded_file("budget.csv", b"Item,Amount\nRoom,200")
    assert "Item | Amount" in document.text
    assert "Room | 200" in document.text


def test_unsupported_and_empty_files_are_clear():
    try:
        parse_uploaded_file("image.png", b"data")
    except DocumentParseError as exc:
        assert "Unsupported file type" in str(exc)
    else:
        raise AssertionError("Expected unsupported-file error")

    try:
        parse_uploaded_file("empty.txt", b"")
    except DocumentParseError as exc:
        assert "empty" in str(exc)
    else:
        raise AssertionError("Expected empty-file error")


def test_llm_uses_groq_configuration_without_live_call():
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="Answer [Source: minutes.txt, words 1-8]"))]
    )
    client = Mock()
    client.chat.completions.create.return_value = response
    with patch.dict(
        os.environ,
        {"GROQ_API_KEY": "test-key", "GROQ_MODEL": "test-model"},
        clear=False,
    ), patch("continuity.llm._client", return_value=client):
        result = _complete("system", "records")

    assert result.startswith("Answer")
    client.chat.completions.create.assert_called_once()
    assert client.chat.completions.create.call_args.kwargs["model"] == "test-model"


def test_grounded_answer_and_checklist_require_citations():
    chunk = chunk_documents(
        [Document(filename="minutes.txt", text="The budget proposal is due September 15.")]
    )
    with patch("continuity.llm._complete", return_value="Due September 15 [Source: minutes.txt, words 1-8]"):
        assert "[Source:" in answer_question("When?", "Treasurer", chunk)
    with patch("continuity.llm._complete", return_value="Review budget [Source: minutes.txt, words 1-8]"):
        assert "[Source:" in generate_checklist("Treasurer", chunk)
    with patch("continuity.llm._complete", return_value="Unsupported answer"):
        assert answer_question("When?", "Treasurer", chunk) == (
            "I could not find that information in the available records."
        )


def test_handoff_audit_requires_citations_and_uses_all_records():
    chunks = chunk_documents(
        [
            Document(filename="minutes.txt", text="The budget proposal is due September 15."),
            Document(filename="guide.txt", text="Written approval is required before promotion."),
        ]
    )
    with patch(
        "continuity.llm._complete",
        return_value=(
            "## Priority actions\nSubmit the budget [Source: minutes.txt, words 1-8]\n"
            "## Deadlines\nSeptember 15 [Source: minutes.txt, words 1-8]\n"
            "## Conflicts or inconsistencies\nNone identified in the available records.\n"
            "## Missing information and risks\nApproval is required [Source: guide.txt, words 1-8]"
        ),
    ) as complete:
        result = audit_handoff(chunks)

    assert "## Priority actions" in result
    assert "[Source: minutes.txt" in result
    assert "[Source: guide.txt" in result
    assert "RECORDS TO AUDIT" in complete.call_args.args[1]


def test_missing_key_and_rate_limit_errors_are_explicit():
    with patch.dict(os.environ, {}, clear=True):
        try:
            _complete("system", "records")
        except LLMConfigurationError as exc:
            assert "GROQ_API_KEY" in str(exc)
        else:
            raise AssertionError("Expected missing-key error")

    from groq import RateLimitError

    client = Mock()
    client.chat.completions.create.side_effect = RateLimitError(
        message="rate limited", response=Mock(status_code=429), body=None
    )
    with patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}, clear=False), patch(
        "continuity.llm._client", return_value=client
    ):
        try:
            _complete("system", "records")
        except LLMRateLimitError as exc:
            assert "try again" in str(exc)
        else:
            raise AssertionError("Expected rate-limit error")
