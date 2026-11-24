"""Sentence embedder: bge-small at the pinned revision, on CPU, unit-length float32 vectors.

Every input goes through `preprocess_for_embedding` (the shared normalization), so the
gateway and the tuning tool embed identical text. Inputs longer than the model's 512-token
window are rejected rather than silently truncated: two prompts sharing their first 512
tokens would otherwise embed identically (design.md, open item 5). Callers check `fits()`
and fall back to the exact cache only.

Thread count is not set here; torch threading is process-wide and belongs to the caller.
"""

from collections.abc import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from cachebench.model_pin import EMBEDDING_DIM, MAX_SEQ_TOKENS, MODEL_NAME, MODEL_REVISION
from cachebench.text import preprocess_for_embedding


class TextTooLongError(ValueError):
    def __init__(self, n_tokens: int, limit: int = MAX_SEQ_TOKENS) -> None:
        super().__init__(f"text is {n_tokens} tokens; the embedding limit is {limit}")
        self.n_tokens = n_tokens
        self.limit = limit


class Embedder:
    def __init__(self, device: str = "cpu") -> None:
        self._model = SentenceTransformer(MODEL_NAME, revision=MODEL_REVISION, device=device)
        self._tokenizer = self._model.tokenizer
        dim = self._model.get_embedding_dimension()
        if dim != EMBEDDING_DIM or self._model.max_seq_length != MAX_SEQ_TOKENS:
            raise RuntimeError(
                f"{MODEL_NAME}@{MODEL_REVISION}: expected dim {EMBEDDING_DIM} and "
                f"max_seq_length {MAX_SEQ_TOKENS}, got {dim} and {self._model.max_seq_length}"
            )

    def _token_counts(self, prepared: Sequence[str]) -> list[int]:
        encoded = self._tokenizer(
            list(prepared), add_special_tokens=True, truncation=False, verbose=False
        )
        return [len(ids) for ids in encoded["input_ids"]]

    def token_count(self, text: str) -> int:
        """Tokens the model would see, including [CLS] and [SEP], before any truncation."""
        return self._token_counts([preprocess_for_embedding(text)])[0]

    def fits(self, text: str) -> bool:
        return self.token_count(text) <= MAX_SEQ_TOKENS

    def embed(self, text: str) -> np.ndarray:
        """Embed one text. Returns a unit-length float32 vector of shape (384,)."""
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        """Embed several texts. Returns unit-length float32 vectors of shape (n, 384)."""
        prepared = [preprocess_for_embedding(t) for t in texts]
        for n_tokens in self._token_counts(prepared):
            if n_tokens > MAX_SEQ_TOKENS:
                raise TextTooLongError(n_tokens)
        vecs = self._model.encode(
            prepared,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        ).astype(np.float32, copy=False)
        # Normalize here rather than rely on the model's Normalize module: inner product
        # in the index equals cosine similarity only if this holds.
        return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
