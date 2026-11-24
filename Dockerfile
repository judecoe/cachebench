# TODO: pin both base images by @sha256 digest.
FROM ghcr.io/astral-sh/uv:0.13.9 AS uv

FROM python:3.12-slim
COPY --from=uv /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/hf \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Dependencies first (cached layer), then the model, then the project itself.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

# Embedding model at the pinned revision (src/cachebench/model_pin.py). Copied alone so
# source changes don't invalidate the download layer.
COPY src/cachebench/model_pin.py /tmp/model_pin.py
RUN python /tmp/model_pin.py && rm /tmp/model_pin.py && chmod -R a+rX /opt/hf

COPY src ./src
RUN uv sync --frozen --no-dev

# From here on the Hub is never contacted. Fail the build if the model can't load offline.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1
RUN python -c "from cachebench.embedding import Embedder; Embedder().embed('build check')"

RUN useradd --create-home --uid 10001 app
USER app

EXPOSE 8000
# Must stay at one worker: the semantic index and coalescing map are in process memory.
CMD ["uvicorn", "cachebench.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
