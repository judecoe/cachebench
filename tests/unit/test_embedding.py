"""Embedder: bge-small at the pinned revision. Loads the real model (CPU), hence `slow`."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from cachebench.embedding import Embedder, TextTooLongError
from cachebench.model_pin import EMBEDDING_DIM, MAX_SEQ_TOKENS

pytestmark = pytest.mark.slow

TEXTS = [
    "How do I reverse a list in Python?",
    "What is the capital of France?",
    "Enable two-factor authentication on my account",
    "x",
    "",
]


@pytest.fixture(scope="module")
def embedder() -> Embedder:
    return Embedder()


def test_limits_match_design() -> None:
    assert EMBEDDING_DIM == 384
    assert MAX_SEQ_TOKENS == 512


def test_embedding_is_384_dim_float32(embedder: Embedder) -> None:
    vec = embedder.embed("How do I reverse a list in Python?")
    assert vec.shape == (384,)
    assert vec.dtype == np.float32


@pytest.mark.parametrize("text", TEXTS)
def test_embedding_is_unit_norm(embedder: Embedder, text: str) -> None:
    assert abs(float(np.linalg.norm(embedder.embed(text))) - 1.0) < 1e-5


def test_embedding_is_deterministic(embedder: Embedder) -> None:
    first = embedder.embed_batch(TEXTS)
    second = embedder.embed_batch(TEXTS)
    assert np.array_equal(first, second)


def test_embedding_is_deterministic_across_instances(embedder: Embedder) -> None:
    assert np.array_equal(Embedder().embed_batch(TEXTS), embedder.embed_batch(TEXTS))


def test_batch_matches_single(embedder: Embedder) -> None:
    # The tuning tool embeds in batches; the gateway embeds one query at a time.
    batch = embedder.embed_batch(TEXTS)
    singles = np.stack([embedder.embed(t) for t in TEXTS])
    assert batch.shape == (len(TEXTS), 384)
    np.testing.assert_allclose(batch, singles, atol=1e-5)


def test_concurrent_embedding_matches_sequential(embedder: Embedder) -> None:
    # The gateway embeds from a thread pool.
    texts = [f"question number {i} about topic {i % 7}" for i in range(64)]
    sequential = np.stack([embedder.embed(t) for t in texts])
    with ThreadPoolExecutor(max_workers=16) as pool:
        concurrent = np.stack(list(pool.map(embedder.embed, texts)))
    np.testing.assert_allclose(concurrent, sequential, atol=1e-6)


def test_embedder_sees_normalized_text(embedder: Embedder) -> None:
    assert np.array_equal(embedder.embed("  café\r\nmenu \n"), embedder.embed("café\nmenu"))


def test_token_count_includes_special_tokens(embedder: Embedder) -> None:
    assert embedder.token_count("hello world") == 4  # [CLS] hello world [SEP]


def test_text_at_limit_fits(embedder: Embedder) -> None:
    text = "hello " * (MAX_SEQ_TOKENS - 2)
    assert embedder.token_count(text) == MAX_SEQ_TOKENS
    assert embedder.fits(text)
    assert embedder.embed(text).shape == (384,)


def test_over_limit_check_fires(embedder: Embedder) -> None:
    text = "hello " * (MAX_SEQ_TOKENS - 1)
    assert embedder.token_count(text) == MAX_SEQ_TOKENS + 1
    assert not embedder.fits(text)
    with pytest.raises(TextTooLongError) as excinfo:
        embedder.embed(text)
    assert excinfo.value.n_tokens == MAX_SEQ_TOKENS + 1
    assert excinfo.value.limit == MAX_SEQ_TOKENS


def test_over_limit_in_batch_raises(embedder: Embedder) -> None:
    with pytest.raises(TextTooLongError):
        embedder.embed_batch(["short", "hello " * MAX_SEQ_TOKENS])
