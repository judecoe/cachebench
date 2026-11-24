"""Text normalization shared by the gateway and the tuning tool.

See docs/design.md, "Text normalization". The same function produces the text used for
the exact-match key and the text the embedder sees. Bump PREPROCESSING_VERSION whenever
the output of either function changes; it is part of the cache namespace.
"""

import unicodedata

PREPROCESSING_VERSION = 1


def normalize_text(text: str) -> str:
    """Unicode NFC, CRLF -> LF, strip leading and trailing whitespace.

    Internal whitespace is never changed and there is no case folding.
    """
    return unicodedata.normalize("NFC", text).replace("\r\n", "\n").strip()


def preprocess_for_embedding(text: str) -> str:
    """The exact text passed to the embedding model.

    No query instruction prefix: this is query-to-query similarity (design.md, open item 6).
    """
    return normalize_text(text)
