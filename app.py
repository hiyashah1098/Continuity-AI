"""Streamlit entry point for Continuity AI."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

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


def _records_section() -> tuple[list[Document], list[Chunk]]:
    _step(2, "Upload records", "PDF, DOCX, TXT, and CSV files stay in memory while this page is open.")
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
        return [], []

    st.success(f"{len(documents)} record(s) ready · indexed for grounded search")
    for document in documents:
        st.markdown(
            f'<div class="record-row">📄 <strong>{document.filename}</strong></div>',
            unsafe_allow_html=True,
        )

    chunks = chunk_documents(documents)
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
    return documents, chunks


def _render_audit(documents: list[Document]) -> None:
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
        st.markdown('<div class="audit-card">', unsafe_allow_html=True)
        st.markdown(st.session_state["audit_result"])
        st.markdown("</div>", unsafe_allow_html=True)
        st.download_button(
            "Download handoff audit",
            data=st.session_state["audit_result"],
            file_name="continuity_handoff_audit.md",
            mime="text/markdown",
        )


def _render_fallback() -> None:
    st.markdown(f'<div class="fallback-card">🌸 {FALLBACK}</div>', unsafe_allow_html=True)


def _render_sources(chunks: list[Chunk]) -> None:
    names = sorted({chunk.filename for chunk in chunks})
    st.markdown("**Sources used**")
    st.markdown(" ".join(f'<span class="source-pill">📄 {name}</span>' for name in names), unsafe_allow_html=True)
    with st.expander("View retrieved evidence"):
        for chunk in chunks:
            st.markdown(f"**{chunk.label}**")
            st.write(chunk.text)


def _ask_tab(role: str, chunks: list[Chunk]) -> None:
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
    if st.button("Ask question", type="primary", disabled=not chunks or not question.strip()):
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
                st.markdown(f'<div class="answer-card">{answer}</div>', unsafe_allow_html=True)
                _render_sources(relevant)
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


def _render_checklist(role: str, checklist: str) -> None:
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
            source_match = re.search(r"\[Source:\s*([^,\]]+)", item)
            source = source_match.group(1).strip() if source_match else "Uploaded records"
            clean_item = re.sub(r"\s*\[Source:.*$", "", item).strip()
            key = f"{role}_check_{week_index}_{item_index}"
            checked = st.checkbox(clean_item, key=key)
            st.caption(f"📄 {source}")
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


def _plan_tab(role: str, chunks: list[Chunk]) -> None:
    _step(5, "Build your first 30 days", "Turn grounded responsibilities into a plan you can work through.")
    if st.button("Generate first-30-days checklist", disabled=not chunks):
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
        _render_checklist(role, st.session_state["latest_checklist"])


st.set_page_config(page_title="Continuity AI", page_icon="❄️", layout="wide")
_load_styles()
_hero()
role = _role_picker()
documents, chunks = _records_section()
tabs = st.tabs(["💬 Ask", "🗓️ 30-Day Plan", "📚 Records"])
with tabs[0]:
    _ask_tab(role, chunks)
with tabs[1]:
    _plan_tab(role, chunks)
with tabs[2]:
    _render_audit(documents)
