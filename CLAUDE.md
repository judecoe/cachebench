# CacheBench

Design: @docs/design.md (source of truth; v1 scope is fixed)

## Stack

Python 3.12, FastAPI, Redis, FAISS (IndexIDMap(IndexFlatIP)), sentence-transformers
(BAAI/bge-small-en-v1.5, pinned revision), pytest, Docker Compose

## Commands

- Test: pytest -q
- Run: docker compose up
  (update as scripts are added)

## Rules

- Fail open: any cache-path error or timeout → skip cache, call backend, log it
- FAISS inserts and removals go through the single writer only; searches run off the event loop
- Only stateless, single-turn, temperature-0 requests are cacheable
- Fixed seeds for all splits, workloads, and benchmarks
- Local backend: never report dollar savings
- Tests use a fake backend; never call a paid API in tests
- Out-of-scope ideas go in TODO.md, not code
- Never edit tests/adversarial/ to make tests pass
