"""Text normalization (docs/design.md, "Text normalization").

Full golden tests for cache keys come with the namespace/key step; these cover the
normalization rules the embedder depends on.
"""

from cachebench.text import normalize_text, preprocess_for_embedding


def test_nfc() -> None:
    assert normalize_text("café") == "café"


def test_crlf_to_lf() -> None:
    assert normalize_text("a\r\nb\r\nc") == "a\nb\nc"


def test_strips_leading_and_trailing_whitespace() -> None:
    assert normalize_text(" \t\n hello \r\n ") == "hello"


def test_internal_whitespace_unchanged() -> None:
    code = "def f():\n    return 1\n\n\nx  =  2"
    assert normalize_text(code) == code
    assert normalize_text("a\n    b") != normalize_text("a\n  b")
    assert normalize_text("a\nb") != normalize_text("a\n\nb")


def test_no_case_folding() -> None:
    assert normalize_text("Hello WORLD") == "Hello WORLD"


def test_idempotent() -> None:
    text = "  café\r\n  x  "
    assert normalize_text(normalize_text(text)) == normalize_text(text)


def test_embedding_preprocessing_is_the_shared_normalization() -> None:
    # No query instruction prefix (design.md, open item 6).
    for text in ["  What is X?\r\n", "café", ""]:
        assert preprocess_for_embedding(text) == normalize_text(text)
