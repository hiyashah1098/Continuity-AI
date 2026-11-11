"""Streamlit entry point for Continuity AI."""

from __future__ import annotations

import hashlib

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
from continuity.models import Document
from continuity.parsers import DocumentParseError, parse_uploaded_file
from continuity.retrieval import retrieve_chunks

load_dotenv()

st.set_page_config(page_title="Continuity AI", page_icon="📚", layout="wide")
st.title("Continuity AI")
st.caption("A grounded handoff assistant for student-organization officers")

role = st.selectbox("Select your incoming role", ["President", "Treasurer", "Event Chair", "Other"])
uploads = st.file_uploader(
    "Upload organization records",
    type=["pdf", "docx", "txt", "csv"],
    accept_multiple_files=True,
    help="Records stay in memory while this page is open.",
)

documents: list[Document] = []
upload_signature_parts: list[tuple[str, str]] = []
if uploads:
    for upload in uploads:
        upload_data = upload.getvalue()
        try:
            documents.append(parse_uploaded_file(upload.name, upload_data))
            upload_signature_parts.append(
                (upload.name, hashlib.sha256(upload_data).hexdigest())
            )
        except DocumentParseError as exc:
            st.error(str(exc))

if documents:
    chunks = chunk_documents(documents)
    st.success(f"Loaded {len(documents)} document(s) and indexed {len(chunks)} chunk(s).")
    upload_signature = tuple(sorted(upload_signature_parts))
    if st.session_state.get("audit_signature") != upload_signature:
        st.session_state.pop("audit_result", None)
        st.session_state.pop("audit_error", None)
        with st.spinner("Auditing the uploaded records for priorities, deadlines, and risks..."):
            try:
                st.session_state["audit_result"] = audit_handoff(chunks)
                st.session_state["audit_signature"] = upload_signature
                st.session_state.pop("audit_error", None)
            except LLMRateLimitError as exc:
                st.session_state["audit_error"] = str(exc)
            except (LLMConfigurationError, RuntimeError) as exc:
                st.session_state["audit_error"] = str(exc)
else:
    chunks = []
    st.info("Upload records to begin. Synthetic examples are available in sample_data/.")

st.divider()
st.subheader("Automatic handoff audit")
st.write(
    "After a new upload set is detected, Continuity AI scans all records once "
    "and identifies prioritized actions, deadlines, conflicts, and missing information."
)
if documents:
    if st.button("Re-run handoff audit"):
        st.session_state.pop("audit_signature", None)
        st.rerun()
    if st.session_state.get("audit_error"):
        st.warning(st.session_state["audit_error"])
    elif st.session_state.get("audit_result"):
        st.markdown(st.session_state["audit_result"])
        st.download_button(
            "Download handoff audit",
            data=st.session_state["audit_result"],
            file_name="continuity_handoff_audit.md",
            mime="text/markdown",
        )

st.divider()
st.subheader("Ask the records")
question = st.text_input("Question", placeholder="What deadlines should I know about?")
if st.button("Ask question", type="primary", disabled=not chunks or not question.strip()):
    relevant = [chunk for chunk, _score in retrieve_chunks(question, chunks)]
    if not relevant:
        st.info("I could not find that information in the available records.")
    else:
        try:
            with st.spinner("Searching records and composing a grounded answer..."):
                answer = answer_question(question, role, relevant)
                st.markdown(answer)
                if answer != "I could not find that information in the available records.":
                    with st.expander("Retrieved evidence used"):
                        for chunk in relevant:
                            st.markdown(f"**{chunk.label}**")
                            st.write(chunk.text)
        except LLMRateLimitError as exc:
            st.warning(str(exc))
        except (LLMConfigurationError, RuntimeError) as exc:
            st.error(str(exc))

st.divider()
st.subheader("First 30 days")
st.write("Generate a role-specific checklist grounded in the uploaded records.")
if st.button("Generate first-30-days checklist", disabled=not chunks):
    relevant = [chunk for chunk, _score in retrieve_chunks(
        f"{role} responsibilities deadlines first 30 days transition tasks", chunks, top_k=8
    )]
    if not relevant:
        st.info("I could not find that information in the available records.")
    else:
        try:
            with st.spinner("Building a grounded checklist..."):
                checklist = generate_checklist(role, relevant)
                st.markdown(checklist)
                if checklist != "I could not find that information in the available records.":
                    st.download_button(
                        "Download checklist",
                        data=checklist,
                        file_name=f"continuity_{role.lower().replace(' ', '_')}_checklist.md",
                        mime="text/markdown",
                    )
        except LLMRateLimitError as exc:
            st.warning(str(exc))
        except (LLMConfigurationError, RuntimeError) as exc:
            st.error(str(exc))
