"""Streamlit entry point for Continuity AI."""

from __future__ import annotations

import csv
import hashlib
import html
import re
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from continuity.citations import (
    SourceReference,
    extract_source_references,
    strip_source_citations,
)
from continuity.chunking import chunk_documents
from continuity.llm import (
    LLMConfigurationError,
    LLMRateLimitError,
    audit_handoff,
    answer_question,
    generate_checklist,
)
from continuity.models import Chunk, Document
from continuity.parsers import DocumentParseError, parse_uploaded_file
from continuity.retrieval import retrieve_chunks

load_dotenv()

FALLBACK = "I could not find that information in the available records."
ROLES = {
    "President": ("🧭", "Leads the organization"),
    "Treasurer": ("💰", "Owns the budget"),
    "Event Chair": ("🎟️", "Runs events"),
    "Other": ("✨", "Explore all records"),
}
SUGGESTED_QUESTIONS = {
    "President": [
        "What deadlines should I know about?",
        "What must happen before an event is promoted?",
        "What did outgoing officers recommend?",
    ],
    "Treasurer": [
        "When is the budget proposal due?",
        "Which vendors require deposits?",
        "What financial tasks happen monthly?",
    ],
    "Event Chair": [
        "What steps are required before an event?",
        "What is the attendance target?",
        "What accessibility requirements apply?",
    ],
    "Other": [
        "What are the next important deadlines?",
        "What decisions were made recently?",
        "What records should I review first?",
    ],
}


def _load_styles() -> None:
    css_path = Path(__file__).parent / "continuity" / "styles.css"
    st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def _hero() -> None:
    st.markdown(
        """
        <section class="continuity-hero">
          <div class="sparkles">✦ ❄ ✧</div>
          <h1>Continuity AI</h1>
          <p>Your organization's memory, ready for the next officer.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def _step(number: int, title: str, hint: str = "") -> None:
    st.markdown(f'<div class="step-label">Step {number} · {title}</div>', unsafe_allow_html=True)
    if hint:
        st.markdown(f'<div class="muted">{hint}</div>', unsafe_allow_html=True)


def _role_picker() -> str:
    _step(1, "Choose your role", "Your role helps prioritize the most useful handoff records.")
    labels = [f"{icon}  {role} — {description}" for role, (icon, description) in ROLES.items()]
    current = st.session_state.get("selected_role", "President")
    current_label = next(label for label in labels if current in label)
    selected_label = st.radio(
        "Incoming officer role",
        labels,
        index=labels.index(current_label),
        horizontal=True,
        label_visibility="collapsed",
    )
    selected = next(role for role in ROLES if role in selected_label)
    st.session_state["selected_role"] = selected
    return selected


def _sample_records() -> list[tuple[str, bytes]]:
    return [
        (path.name, path.read_bytes())
        for path in sorted((Path(__file__).parent / "sample_data").iterdir())
        if path.is_file()
    ]


def _records_section() -> tuple[list[Document], list[Chunk], bool]:
    _step(
        2,
        "Upload records",
        "Files stay in memory for this session. AI features send retrieved text to Gemini.",
    )
    st.caption(
        "Use the fictional sample records for the demo. Only upload real records when your "
        "organization permits them to be processed by a hosted AI provider."
    )
    if st.button("Load sample records", use_container_width=False):
        st.session_state["sample_records"] = _sample_records()
        st.session_state.pop("audit_signature", None)
        st.rerun()

    uploads = st.file_uploader(
        "Drop organization records here",
        type=["pdf", "docx", "txt", "csv"],
        accept_multiple_files=True,
        help="Upload meeting notes, budgets, event plans, and transition records.",
    )
    sources: list[tuple[str, bytes]] = (
        [(upload.name, upload.getvalue()) for upload in uploads]
        if uploads
        else st.session_state.get("sample_records", [])
    )
    using_uploaded_records = bool(uploads)
    ai_allowed = not using_uploaded_records
    if using_uploaded_records:
        ai_allowed = st.checkbox(
            "I am authorized to process these records and understand that relevant text "
            "will be sent to Google Gemini.",
            key="upload_processing_consent",
        )
    documents: list[Document] = []
    signatures: list[tuple[str, str]] = []
    for filename, data in sources:
        try:
            documents.append(parse_uploaded_file(filename, data))
            signatures.append((filename, hashlib.sha256(data).hexdigest()))
        except DocumentParseError as exc:
            st.error(str(exc))

    if not documents:
        st.info("Upload some records to get started, or load the synthetic sample records.")
        return [], [], False

    st.success(f"{len(documents)} record(s) ready · indexed for grounded search")
    for document in documents:
        st.markdown(
            f'<div class="record-row">📄 <strong>{html.escape(document.filename)}</strong></div>',
            unsafe_allow_html=True,
        )

    chunks = chunk_documents(documents)
    if not ai_allowed:
        st.warning("Local parsing is complete. Confirm authorization above to enable Gemini features.")
        return documents, chunks, False
    upload_signature = tuple(sorted(signatures))
    if st.session_state.get("audit_signature") != upload_signature:
        st.session_state.pop("audit_result", None)
        st.session_state.pop("audit_error", None)
        with st.spinner("❄ Reviewing your records for priorities, deadlines, and risks..."):
            try:
                st.session_state["audit_result"] = audit_handoff(chunks)
                st.session_state["audit_signature"] = upload_signature
            except LLMRateLimitError as exc:
                st.session_state["audit_error"] = str(exc)
            except (LLMConfigurationError, RuntimeError) as exc:
                st.session_state["audit_error"] = str(exc)
    return documents, chunks, True


def _source_popover(
    references: list[SourceReference],
    chunks: list[Chunk],
    statement: str,
    key: str,
) -> None:
    with st.popover(
        "Source",
        icon=":material/source:",
        type="secondary",
        use_container_width=True,
        key=key,
    ):
        st.caption("Evidence from the loaded records")
        for reference in references:
            st.markdown(f"**{reference.filename}**")
            if reference.location:
                st.caption(reference.location)
            candidates = [
                chunk
                for chunk in chunks
                if chunk.filename == reference.filename
                and (not reference.location or chunk.location == reference.location)
            ]
            if not candidates:
                candidates = [chunk for chunk in chunks if chunk.filename == reference.filename]
            if not reference.location and len(candidates) > 1:
                ranked = retrieve_chunks(statement, candidates, top_k=1)
                candidates = [ranked[0][0]] if ranked else candidates[:1]
            for chunk in candidates[:1]:
                st.caption(chunk.label)
                st.write(chunk.text)


def _render_cited_output(content: str, chunks: list[Chunk], key_prefix: str) -> None:
    """Render readable model output with citations hidden behind source controls."""
    available_filenames = {chunk.filename for chunk in chunks}
    for index, line in enumerate(content.splitlines()):
        if line.strip().lower().startswith("## supporting excerpts"):
            break
        references = extract_source_references(line, available_filenames)
        clean_line = strip_source_citations(line)
        if not clean_line.strip():
            continue
        if not references or clean_line.lstrip().startswith("#"):
            st.markdown(clean_line)
            continue
        text_column, source_column = st.columns(
            [0.86, 0.14],
            gap="small",
            vertical_alignment="center",
        )
        text_column.markdown(clean_line)
        with source_column:
            _source_popover(
                references,
                chunks,
                statement=clean_line,
                key=f"{key_prefix}_source_{index}",
            )


def _render_audit(documents: list[Document], chunks: list[Chunk]) -> None:
    if not documents:
        return
    st.markdown("### Automatic handoff audit")
    st.caption("Continuity proactively scans your records for actions, deadlines, conflicts, and risks.")
    if st.button("Re-run handoff audit"):
        st.session_state.pop("audit_signature", None)
        st.rerun()
    if st.session_state.get("audit_error"):
        st.warning(st.session_state["audit_error"])
    elif st.session_state.get("audit_result"):
        with st.container(border=True):
            _render_cited_output(st.session_state["audit_result"], chunks, "audit")
        st.download_button(
            "Download handoff audit",
            data=st.session_state["audit_result"],
            file_name="continuity_handoff_audit.md",
            mime="text/markdown",
        )


def _render_fallback() -> None:
    st.markdown(f'<div class="fallback-card">🌸 {FALLBACK}</div>', unsafe_allow_html=True)


def _ask_tab(role: str, chunks: list[Chunk], ai_allowed: bool) -> None:
    _step(3, "Ask a question", "Answers are generated only from your uploaded records.")
    question_key = "question_text"
    if question_key not in st.session_state:
        st.session_state[question_key] = ""
    st.markdown("**Try a suggested question**")
    for index, suggestion in enumerate(SUGGESTED_QUESTIONS[role]):
        if st.button(suggestion, key=f"suggestion_{role}_{index}"):
            st.session_state[question_key] = suggestion
            st.rerun()
    question = st.text_input(
        "Question",
        key=question_key,
        placeholder="What deadlines should I know about?",
    )
    if chunks and not ai_allowed:
        st.info("Confirm record-processing authorization above to ask Gemini a question.")
    if st.button(
        "Ask question",
        type="primary",
        disabled=not chunks or not question.strip() or not ai_allowed,
    ):
        relevant = [chunk for chunk, _score in retrieve_chunks(question, chunks)]
        if not relevant:
            _render_fallback()
            return
        try:
            with st.spinner("Searching your records and composing a grounded answer..."):
                answer = answer_question(question, role, relevant)
            _step(4, "Your answer")
            if answer == FALLBACK:
                _render_fallback()
            else:
                with st.container(border=True):
                    _render_cited_output(answer, relevant, "answer")
        except LLMRateLimitError as exc:
            st.warning(str(exc))
        except (LLMConfigurationError, RuntimeError) as exc:
            st.error(str(exc))


def _checklist_items(checklist: str) -> list[str]:
    return [
        line.strip().lstrip("*- ").strip()
        for line in checklist.splitlines()
        if "[Source:" in line and not line.lstrip().startswith("#")
    ]


def _render_checklist(role: str, checklist: str, chunks: list[Chunk]) -> None:
    if checklist == FALLBACK:
        _render_fallback()
        return
    items = _checklist_items(checklist)
    if not items:
        st.markdown(checklist)
        return
    st.session_state.setdefault("checklist_checks", {})
    groups = [items[index:index + 3] for index in range(0, len(items), 3)]
    for week_index, week_items in enumerate(groups, start=1):
        st.markdown(f"#### Week {week_index}")
        for item_index, item in enumerate(week_items):
            references = extract_source_references(
                item,
                {chunk.filename for chunk in chunks},
            )
            clean_item = strip_source_citations(item).strip()
            key = f"{role}_check_{week_index}_{item_index}"
            task_column, source_column = st.columns(
                [0.86, 0.14],
                gap="small",
                vertical_alignment="center",
            )
            checked = task_column.checkbox(clean_item, key=key)
            if references:
                with source_column:
                    _source_popover(
                        references,
                        chunks,
                        statement=clean_item,
                        key=f"{key}_source",
                    )
            st.session_state["checklist_checks"][key] = checked
    completed = sum(st.session_state["checklist_checks"].get(
        f"{role}_check_{week_index}_{item_index}", False
    ) for week_index, week_items in enumerate(groups, start=1) for item_index, _ in enumerate(week_items))
    total = len(items)
    st.progress(completed / total, text=f"{completed} of {total} tasks complete")
    st.download_button(
        "Download checklist",
        data=checklist,
        file_name=f"continuity_{role.lower().replace(' ', '_')}_checklist.md",
        mime="text/markdown",
    )


def _plan_tab(role: str, chunks: list[Chunk], ai_allowed: bool) -> None:
    _step(5, "Build your first 30 days", "Turn grounded responsibilities into a plan you can work through.")
    if st.button("Generate first-30-days checklist", disabled=not chunks or not ai_allowed):
        relevant = [
            chunk
            for chunk, _score in retrieve_chunks(
                f"{role} responsibilities deadlines first 30 days transition tasks",
                chunks,
                top_k=8,
            )
        ]
        if not relevant:
            _render_fallback()
            return
        try:
            with st.spinner("Building your grounded first-30-days plan..."):
                checklist = generate_checklist(role, relevant)
            st.session_state["latest_checklist"] = checklist
            st.snow()
        except LLMRateLimitError as exc:
            st.warning(str(exc))
        except (LLMConfigurationError, RuntimeError) as exc:
            st.error(str(exc))
    if st.session_state.get("latest_checklist"):
        _render_checklist(role, st.session_state["latest_checklist"], chunks)


def _evaluation_rows() -> list[dict[str, str]]:
    results_path = Path(__file__).parent / "evaluation_results.csv"
    if not results_path.exists():
        return []
    with results_path.open(encoding="utf-8", newline="") as results_file:
        return list(csv.DictReader(results_file))


def _project_evidence_tab() -> None:
    """Show transparent Track 2 evidence without presenting estimates as measured facts."""
    st.markdown("### Product evidence")
    st.caption("What was built, how it was tested, and what remains to validate with people.")
    rows = _evaluation_rows()
    if rows:
        passed = sum(row.get("pass", "").lower() == "true" for row in rows)
        cited = sum(row.get("citation_correct", "").lower() == "true" for row in rows)
        response_times = [float(row["response_time_seconds"]) for row in rows]
        metric_columns = st.columns(3)
        metric_columns[0].metric("Scenario pass rate", f"{passed / len(rows):.0%}", f"{passed}/{len(rows)}")
        metric_columns[1].metric("Citation / fallback checks", f"{cited / len(rows):.0%}")
        metric_columns[2].metric("Average response time", f"{sum(response_times) / len(rows):.1f}s")
        st.caption(
            "Latest committed live-Gemini evaluation: 11 realistic questions and four role "
            "checklists. These are automated scenario results, not an external-user study."
        )

    st.markdown("#### Bounded agentic workflow")
    st.markdown(
        "**Upload → parse → retrieve evidence → proactive audit → validate citations → "
        "surface actions and risks.** The system is intentionally bounded: it does not send "
        "messages, change calendars, or take consequential actions for the officer."
    )

    st.markdown("#### Illustrative business case")
    st.caption("Adjust the assumptions below. The result is a planning estimate, not measured ROI.")
    inputs = st.columns(4)
    organizations = inputs[0].number_input("Organizations supported", 1, 500, 20)
    handoffs = inputs[1].number_input("Handoffs per organization", 1, 20, 3)
    hours_saved = inputs[2].number_input("Hours saved per handoff", 0.0, 20.0, 2.0, 0.5)
    hourly_value = inputs[3].number_input("Value per hour ($)", 1.0, 200.0, 20.0, 1.0)
    annual_cost = st.number_input("Estimated annual operating cost ($)", 1.0, 100000.0, 600.0, 100.0)
    annual_value = organizations * handoffs * hours_saved * hourly_value
    net_value = annual_value - annual_cost
    roi = net_value / annual_cost
    result_columns = st.columns(3)
    result_columns[0].metric("Estimated annual value", f"${annual_value:,.0f}")
    result_columns[1].metric("Estimated net value", f"${net_value:,.0f}")
    result_columns[2].metric("Illustrative ROI", f"{roi:.0%}")
    st.markdown(
        "A university student-affairs or campus-engagement team is the most plausible payer. "
        "The product improves manual shared-drive searching and repeated handoff questions."
    )

    st.markdown("#### Guardrails and limitations")
    st.markdown(
        "- Answers are limited to retrieved records and must cite uploaded filenames.\n"
        "- Retrieved excerpts remain visible for human verification.\n"
        "- Missing information returns an explicit fallback instead of a guess.\n"
        "- Sample records are synthetic; uploaded records require processing authorization.\n"
        "- TF-IDF retrieval can miss unusual synonyms, and citations do not replace policy verification.\n"
        "- External-user task timing and usability testing remain future validation work."
    )


st.set_page_config(page_title="Continuity AI", page_icon="❄️", layout="wide")
_load_styles()
_hero()
role = _role_picker()
documents, chunks, ai_allowed = _records_section()
tabs = st.tabs(["💬 Ask", "🗓️ 30-Day Plan", "📚 Records", "📊 Product Evidence"])
with tabs[0]:
    _ask_tab(role, chunks, ai_allowed)
with tabs[1]:
    _plan_tab(role, chunks, ai_allowed)
with tabs[2]:
    _render_audit(documents, chunks)
with tabs[3]:
    _project_evidence_tab()
