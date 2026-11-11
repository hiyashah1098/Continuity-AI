"""Streamlit entry point for Continuity AI."""

from __future__ import annotations

import streamlit as st
from dotenv import load_dotenv

from continuity.chunking import chunk_documents
from continuity.llm import (
    LLMConfigurationError,
    LLMRateLimitError,
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
if uploads:
    for upload in uploads:
        try:
            documents.append(parse_uploaded_file(upload.name, upload.getvalue()))
        except DocumentParseError as exc:
            st.error(str(exc))

if documents:
    chunks = chunk_documents(documents)
    st.success(f"Loaded {len(documents)} document(s) and indexed {len(chunks)} chunk(s).")
else:
    chunks = []
    st.info("Upload records to begin. Synthetic examples are available in sample_data/.")

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
                st.markdown(answer_question(question, role, relevant))
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
                st.markdown(generate_checklist(role, relevant))
        except LLMRateLimitError as exc:
            st.warning(str(exc))
        except (LLMConfigurationError, RuntimeError) as exc:
            st.error(str(exc))
