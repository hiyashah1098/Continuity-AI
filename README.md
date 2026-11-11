# Continuity AI

Continuity AI is a locally runnable Streamlit MVP for student-organization officer
handoffs. It answers questions and creates first-30-days checklists using only the
uploaded records, with source excerpts shown for every result.

## Setup

1. Create and activate a Python 3.10+ virtual environment.
2. Install dependencies:

   ```text
   pip install -r requirements.txt
   ```

3. Copy `.env.example` to `.env` and add a Groq API key. Create one at
   [console.groq.com/keys](https://console.groq.com/keys). The default model is
   `openai/gpt-oss-120b`; set `GROQ_MODEL` in `.env` to choose another
   currently supported Groq model.
4. Start the app:

   ```text
   streamlit run app.py
   ```

Run the automated tests with:

```text
pytest -q
```

Run the live Groq smoke check and end-to-end evaluation explicitly:

```text
python scripts/smoke_test.py
python scripts/evaluate.py
```

The evaluation makes real Groq calls, checks document parsing, retrieval,
citations, fallback behavior, role checklists, missing-key and rate-limit
handling, and writes metrics to `evaluation_results.csv`.

The app also includes synthetic records in `sample_data/`. Upload those files to
try the complete flow.

## Project structure

- `app.py` - Streamlit interface and application flow.
- `continuity/parsers.py` - PDF, DOCX, TXT, and CSV extraction.
- `continuity/chunking.py` - Small overlapping text chunks.
- `continuity/retrieval.py` - TF-IDF retrieval. This is intentionally transparent
  and dependency-light; it ranks document chunks by vocabulary overlap with the
  question.
- `continuity/llm.py` - Hosted LLM call and grounded prompts.
- `sample_data/` - Fictional organization records with no real people or data.
- `tests/` - Focused unit tests and manual question checklist.

## Grounding behavior

Retrieved excerpts are passed to the model with filename and page/row metadata.
The prompt requires the model to answer only from those excerpts and include
citations. The application rejects an answer without citations and uses the exact
fallback sentence when the model cannot find supporting information.

## Known limitations

- Retrieval is lexical TF-IDF rather than semantic embeddings, so unusual
  synonyms may not rank highly.
- The app uses Groq's hosted chat-completions API configured through
  `GROQ_API_KEY` and `GROQ_MODEL`.
- Uploaded files are processed in memory and are not persisted.
- Citation quality depends on the model following the grounded output format;
  the UI still displays the retrieved source excerpts as a verification aid.

## Manual test questions

After uploading all files in `sample_data/`, select the indicated role:

| Role | Question | Expected result |
|---|---|---|
| President | What must happen before an event is publicly advertised? | Approval guide says submit the event form 30 days before the event; cite the approval guide. |
| Treasurer | When is the fall budget proposal due? | September 15; cite the meeting minutes and budget. |
| Event Chair | What is the attendance target for the Community Skills Night? | 80 attendees; cite the event plan. |
| Treasurer | Which vendor requires a deposit and when? | Northstar Catering requires a 25% deposit two weeks before the event; cite vendor info. |
| President | What did outgoing officers recommend about shared-drive naming? | Use `YYYY-MM-DD_topic` naming; cite transition notes. |
| Event Chair | What accessibility step is required for every event? | Include an accessible room route and accommodation contact in the event checklist. |
| Other | What is the organization’s mascot? | `I could not find that information in the available records.` |
| President | What was the final vote count for a national election? | `I could not find that information in the available records.` |

Checklist testing: choose each role and click **Generate first-30-days checklist**.
Every generated item should include a source filename and should include a deadline
when the records provide one.