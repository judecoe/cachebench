# TODO: pin both base images by @sha256 digest.
FROM ghcr.io/astral-sh/uv:0.13.9 AS uv

FROM python:3.12-slim
COPY --from=uv /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Dependencies first (cached layer), then the project itself.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev

# Step 1 adds: download BAAI/bge-small-en-v1.5 at the pinned revision into the image.

RUN useradd --create-home --uid 10001 app
USER app
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000
# Must stay at one worker: the semantic index and coalescing map are in process memory.
CMD ["uvicorn", "cachebench.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
