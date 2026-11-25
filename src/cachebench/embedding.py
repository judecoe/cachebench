"""Sentence embedder: bge-small at the pinned revision, on CPU, unit-length float32 vectors.

Every input goes through `preprocess_for_embedding` (the shared normalization), so the
gateway and the tuning tool embed identical text. Inputs longer than the model's 512-token
window are rejected rather than silently truncated: two prompts sharing their first 512
tokens would otherwise embed identically (design.md, open item 5). Callers check `fits()`
and fall back to the exact cache only.

Two runtimes load the same pinned revision: "torch" (sentence-transformers) and "onnx"
(onnxruntime on the repo's ONNX export). The gateway and the tuning tool both use
`EMBEDDING_RUNTIME`; pass `runtime` only to compare runtimes.

Threads: the onnx runtime takes `threads` (intra-op threads for its session). torch
threading is process-wide, so the torch runtime ignores `threads` and leaves
`torch.set_num_threads` to the caller.
"""

from collections.abc import Sequence
from typing import Literal

import numpy as np
from transformers import AutoTokenizer

from cachebench.model_pin import (
    EMBEDDING_DIM,
    EMBEDDING_RUNTIME,
    MAX_SEQ_TOKENS,
    MODEL_NAME,
    MODEL_REVISION,
    ONNX_FILE,
)
from cachebench.text import preprocess_for_embedding

Runtime = Literal["torch", "onnx"]


class TextTooLongError(ValueError):
    def __init__(self, n_tokens: int, limit: int = MAX_SEQ_TOKENS) -> None:
        super().__init__(f"text is {n_tokens} tokens; the embedding limit is {limit}")
        self.n_tokens = n_tokens
        self.limit = limit


class Embedder:
    def __init__(self, runtime: Runtime = EMBEDDING_RUNTIME, threads: int = 1) -> None:
        self.runtime = runtime
        self.threads = threads
        self._tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
        if runtime == "torch":
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(MODEL_NAME, revision=MODEL_REVISION, device="cpu")
            dim = self._model.get_embedding_dimension()
            if dim != EMBEDDING_DIM or self._model.max_seq_length != MAX_SEQ_TOKENS:
                raise RuntimeError(
                    f"{MODEL_NAME}@{MODEL_REVISION}: expected dim {EMBEDDING_DIM} and "
                    f"max_seq_length {MAX_SEQ_TOKENS}, "
                    f"got {dim} and {self._model.max_seq_length}"
                )
        elif runtime == "onnx":
            import onnxruntime as ort
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(MODEL_NAME, ONNX_FILE, revision=MODEL_REVISION)
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = threads
            opts.inter_op_num_threads = 1
            opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self._session = ort.InferenceSession(
                path, sess_options=opts, providers=["CPUExecutionProvider"]
            )
            self._pad_id = self._tokenizer.pad_token_id
        else:
            raise ValueError(f"unknown runtime {runtime!r}")

    def _token_ids(self, prepared: Sequence[str]) -> list[list[int]]:
        # Same settings on every call (no padding, no truncation): fast tokenizers keep
        # these as mutable state, and flipping them from several threads is unsafe.
        encoded = self._tokenizer(
            list(prepared), add_special_tokens=True, truncation=False, verbose=False
        )
        return encoded["input_ids"]

    def token_count(self, text: str) -> int:
        """Tokens the model would see, including [CLS] and [SEP], before any truncation."""
        return len(self._token_ids([preprocess_for_embedding(text)])[0])

    def fits(self, text: str) -> bool:
        return self.token_count(text) <= MAX_SEQ_TOKENS

    def embed(self, text: str) -> np.ndarray:
        """Embed one text. Returns a unit-length float32 vector of shape (384,)."""
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        """Embed several texts. Returns unit-length float32 vectors of shape (n, 384)."""
        prepared = [preprocess_for_embedding(t) for t in texts]
        ids = self._token_ids(prepared)
        for row in ids:
            if len(row) > MAX_SEQ_TOKENS:
                raise TextTooLongError(len(row))
        if self.runtime == "torch":
            vecs = self._model.encode(
                prepared,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=False,
                show_progress_bar=False,
            )
        else:
            vecs = np.concatenate(
                [self._onnx_cls(ids[i : i + batch_size]) for i in range(0, len(ids), batch_size)]
            )
        vecs = vecs.astype(np.float32, copy=False)
        # Normalize here rather than rely on the model: inner product in the index equals
        # cosine similarity only if this holds.
        return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)

    def _onnx_cls(self, ids: list[list[int]]) -> np.ndarray:
        """CLS-token hidden state (bge's pooling) for one batch of token id lists."""
        width = max(len(row) for row in ids)
        input_ids = np.full((len(ids), width), self._pad_id, dtype=np.int64)
        attention_mask = np.zeros((len(ids), width), dtype=np.int64)
        for i, row in enumerate(ids):
            input_ids[i, : len(row)] = row
            attention_mask[i, : len(row)] = 1
        (hidden,) = self._session.run(
            ["last_hidden_state"],
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": np.zeros_like(input_ids),
            },
        )
        return hidden[:, 0, :]
