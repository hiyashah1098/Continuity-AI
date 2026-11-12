from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest


def test_generated_answer_uses_clickable_source_instead_of_brackets():
    grounded_answer = (
        "Written approval is required before public promotion "
        "[Source: 09_event_approval_guide.txt, words 1-60]"
    )
    with patch("continuity.llm._complete", return_value=grounded_answer):
        app_path = Path(__file__).resolve().parents[1] / "app.py"
        app = AppTest.from_file(app_path).run()
        app.button[0].click().run()
        app.text_input[0].input(
            "What must happen before an event is publicly advertised?"
        ).run()
        next(button for button in app.button if button.label == "Ask question").click().run()

    markdown_values = [element.value for element in app.markdown]
    assert "Written approval is required before public promotion" in markdown_values
    assert all("[Source:" not in value for value in markdown_values)
    assert any(
        "09_event_approval_guide.txt" in value
        for value in markdown_values
    )
