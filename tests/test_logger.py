from chuvashia_rag.logger import preview


def test_preview_keeps_short_text():
    assert preview("короткий текст") == "короткий текст"


def test_preview_truncates_and_reports_length():
    result = preview("а" * 200, max_len=10)

    assert result == "аааааааааа… [всего 200 симв.]"


def test_preview_replaces_newlines_and_handles_none():
    assert preview("a\nb") == "a ⏎ b"
    assert preview(None) == "None"
