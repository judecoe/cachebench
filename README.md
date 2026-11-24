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

## Scripts

```sh
scripts/measure_embed_latency.sh                        # embedding p50/p95 in Docker; JSON to results/embed_latency/
uv run python scripts/score_pairs.py pairs.jsonl        # cosine similarity per pair, grouped by category
```

`pairs.jsonl` has one object per line: `{"a": "...", "b": "...", "category": "..."}`, with
optional `id` and `note`.
