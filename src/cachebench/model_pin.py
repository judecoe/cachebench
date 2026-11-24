"""Pinned embedding model. Single source of truth for the build-time download and runtime load.

Deliberately imports nothing from cachebench and nothing heavy at module level: the
Dockerfile runs this file as a script before the project is installed, so the ~130 MB
download stays in its own cached layer.

The revision is a full commit SHA (verified on Hugging Face on 2026-11-23; it is also the
`main` head, unchanged since 2024-02-22). It is part of the cache namespace, so changing it
invalidates every semantic entry.
"""

MODEL_NAME = "BAAI/bge-small-en-v1.5"
MODEL_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
EMBEDDING_DIM = 384
MAX_SEQ_TOKENS = 512  # includes [CLS] and [SEP]

# Only the safetensors weights are needed; skip the duplicate PyTorch and ONNX weights.
_IGNORE_PATTERNS = ["pytorch_model.bin", "onnx/*", "*.onnx", "openvino/*"]


def download() -> str:
    """Download the pinned snapshot into the Hugging Face cache and return its path."""
    from huggingface_hub import snapshot_download

    return snapshot_download(MODEL_NAME, revision=MODEL_REVISION, ignore_patterns=_IGNORE_PATTERNS)


if __name__ == "__main__":
    print(download())
