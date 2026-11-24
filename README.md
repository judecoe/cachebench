# CacheBench

LLM gateway with exact-match and semantic caching, plus tooling to choose and justify the
similarity threshold. Design: [docs/design.md](docs/design.md).

## Development

Requires [uv](https://docs.astral.sh/uv/) and Docker.

```sh
uv sync                 # create .venv with Python 3.12 and pinned dependencies
uv run pytest -q        # tests
docker compose up       # gateway on :8000 (fake backend) + Redis
curl localhost:8000/healthz
```
