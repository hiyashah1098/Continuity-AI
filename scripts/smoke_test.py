"""Make one real Groq request to verify local configuration and model availability."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from continuity.llm import _complete


load_dotenv(ROOT / ".env")

if not os.getenv("GROQ_API_KEY"):
    raise SystemExit("GROQ_API_KEY is missing")

model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
response = _complete(
    "Reply with a short confirmation only.",
    "Return the word READY.",
)
if not response:
    raise SystemExit("Groq returned an empty response")

print(f"GROQ_SMOKE=pass model={model} response_length={len(response)}")
