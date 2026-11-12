"""Validate the bundled synthetic records without making an external API call."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from continuity.sample_validation import validate_sample_directory


def main() -> None:
    report = validate_sample_directory(ROOT / "sample_data")
    print(
        "SAMPLE_DATA=pass "
        f"files={report.file_count} verified_facts={len(report.verified_facts)}"
    )


if __name__ == "__main__":
    main()
