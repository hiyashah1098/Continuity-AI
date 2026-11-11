"""Make one real Gemini request to verify local configuration and model availability."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from continuity.llm import _complete


load_dotenv(ROOT / ".env")

if not os.getenv("GEMINI_API_KEY"):
    raise SystemExit("GEMINI_API_KEY is missing")

model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
response = _complete(
    "Reply with a short confirmation only.",
    "Return the word READY.",
)
if not response:
    raise SystemExit("Gemini returned an empty response")

print(f"GEMINI_SMOKE=pass model={model} response_length={len(response)}")
