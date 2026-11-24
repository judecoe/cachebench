# Design Doc: CacheBench (working name)

**Author:** Jude Gilligan · **Status:** v4 (v1 decisions locked) · **Last updated:** November 23, 2026

## Goal

Build a production-style LLM gateway with a semantic cache that is safe to run in production, plus the tooling to choose and justify its similarity threshold.

## Problem

Teams calling LLM APIs pay for, and wait on, many requests that are paraphrases of ones already answered. A semantic cache can reuse those answers, but a threshold set too loosely silently serves wrong answers, and a cache that adds latency or fails badly makes the service worse. Shipping one safely requires solving both the systems problems and the threshold problem.

## Requirements

**Functional**
- OpenAI-compatible chat completions endpoint
- Exact-match and semantic caching, with clear rules for which requests are cacheable
- Two interchangeable model backends (one local, one paid API)
- A threshold tuning tool that recommends a threshold from labeled data

**Non-functional (set November 2026, before building)**
- Correctness: the upper bound of the 95% Wilson confidence interval on the held-out false-hit rate is at most 2%, and zero adversarial pairs hit the cache at the shipped threshold (hard gate)
- Overhead: on a cache miss, gateway overhead (gateway latency minus direct backend latency) is at most 10 ms at p50 and 25 ms at p95 at concurrency 16; the full concurrency ladder is reported
- Availability: cache failures never fail a request (fail open)
- Memory: the semantic index holds at most 100K entries across all namespaces and never grows unbounded

Targets are fixed before benchmarking. If a target is missed, the miss and any fix are reported; the target is not moved.

## What we need to know before shipping

1. Which threshold meets the correctness target, and what hit rate and savings remain at it?
2. How much latency does the gateway add on hits and misses, and how does that change under load?
3. Which kinds of requests produce false hits (negation, opposite intent, version or number changes)?
4. Does semantic caching save enough over exact-match caching to justify its overhead under realistic repetition?

A "no" on question 4 is an acceptable outcome, and the write-up reports it honestly.

## Scope

| Stage | When | In scope |
|---|---|---|
| v1 | Winter break | Gateway endpoint; two backends; request boundary rules; exact-match cache; semantic cache with bounded memory, eviction, and TTL; request coalescing; fail-open behavior; threshold tuning tool; latency and load benchmarks; adversarial regression tests; structured logs; Docker Compose |
| v1.5 | January | Streaming responses with caching; reliability features (upstream timeouts, HTTP 429 and 5xx handling, retries with backoff, circuit breaker); Prometheus metrics; automated answer grader validated against human labels |
| v2 | Spring 2027 | Simple two-model routing evaluated against fixed baselines; metrics dashboard; optional shared cache state across gateway instances |

**Out of scope:** request batching, adaptive or learned routing, multi-tenant auth, Kubernetes, distributed vector databases.

## Request and cache boundary

- **Cacheable (v1):** stateless, single-turn requests with temperature 0. Requests with conversation history, tool calls, or nonzero temperature bypass the cache.
- **Namespace:** model and version, a hash of the system prompt, generation settings, and response format. Semantic lookups only compare within a namespace.
- **Expiration:** every entry has a configurable TTL.

## Architecture

**Request lifecycle:** boundary check (cacheable or bypass) → exact-match lookup (Redis) → coalescing check → semantic lookup (embed, nearest-neighbor search within namespace, threshold comparison) → on a miss, call the backend → write to both caches → return.

**Components:** FastAPI gateway; Redis exact-match cache; in-process FAISS semantic index (one `IndexIDMap(IndexFlatIP)` per namespace); local embedding model `BAAI/bge-small-en-v1.5` at a pinned revision; backend abstraction over a local model and a paid API; a shared cache interface (get, put, invalidate) implemented separately by each cache.

Every request logs its outcome (bypass, exact hit, semantic hit, coalesced, or miss), similarity score, per-stage latency, and token usage.

## Key engineering problems

**Request coalescing.** When several identical requests miss at the same moment, only one goes to the backend; the rest wait for its result. If the backend call fails or times out, every waiter receives the error and nothing is cached. v1 coalesces on the exact key only. Coalescing near-identical in-flight requests is harder (it would need a semantic check on requests that have no answer yet) and is a non-goal for v1.

**Bounded memory and eviction.** Each namespace has its own flat index; vectors are L2-normalized, so inner product equals cosine similarity. A global LRU across namespaces enforces the 100K-entry limit. Flat search is exact and takes a few milliseconds at this size; HNSW is not used because it does not support removal.

Deletion uses tombstones plus batched removal:
1. When an entry expires or is evicted, it is marked dead in the metadata store immediately.
2. Searches request the top 5 neighbors and return the first live one.
3. When dead entries exceed 10% of an index, the single writer removes them in one `remove_ids` call.

A microbenchmark in winter break week 1 confirms batched removal cost at full capacity.

**Concurrent writes.** FAISS isn't safe for concurrent writes, so all index inserts and batched removals go through a single writer, and searches run off the event loop so they don't block the server. Before inserting, the writer checks for an existing near-duplicate in the same namespace; if one exists, the first entry wins and the new one is skipped.

**Failure handling.** If Redis, the embedding model, or FAISS errors out or exceeds a time limit, the request skips caching, goes straight to the backend, and the failure is logged. Fault-injection tests verify this.

**Streaming (v1.5).** A streamed response is cached only after it completes successfully. A stream that fails partway is never cached, and a stream can't be retried once bytes have reached the client.

## Threshold tuning tool

A command (for example, `cachebench tune`) that takes labeled query pairs and a correctness target (default: 2% false-hit rate at the 95% Wilson upper bound). It splits the data with a fixed seed into a tuning set and a held-out set, sweeps thresholds on the tuning set, picks the loosest threshold that meets the target, verifies it on the held-out set, and outputs the threshold with its false-hit rate, confidence interval, and expected hit rate as gateway config. It also outputs the full threshold curve so readers can apply a different target.

It is demonstrated on Quora Question Pairs (QQP), a public dataset of question pairs labeled as duplicates or not. Any team could run it on its own labeled traffic.

## Evaluation

**Data**
- **QQP:** input to the tuning tool and source for generated workloads. Duplicate labels are a proxy for "the cached answer is acceptable." QQP labels are noisy, which sets a floor on measurable false-hit rate; this is why the target is 2% rather than 1%.
- **Training overlap:** many embedding models were trained on data that includes Quora duplicates. The bge model card is checked for this, and any overlap is reported as a limitation, since it would make QQP results optimistic.
- **Hand-checked sample:** about 100 QQP pairs sampled across similarity scores, judged by whether an answer to one question acceptably answers the other. This measures how far the proxy diverges from answer acceptability.
- **Adversarial regression tests:** a few dozen hand-built near-miss pairs (negation, enable versus disable, version or number changes, swapped comparisons). These live in the test suite and must not hit the cache at the shipped threshold. They are unaffected by training overlap.
- **Workloads:** generated query streams with fixed seeds: exact repeats, paraphrases, and low, medium, and high repetition with a skewed (Zipf) popularity distribution.

**Metrics**

| Metric | What it shows |
|---|---|
| False-hit rate (wrong hits ÷ all semantic hits, with 95% Wilson interval) and incorrect-served rate (wrong hits ÷ all requests) | Correctness, always reported next to hit rate |
| Hit rate (exact and semantic) | Coverage |
| API cost per 1,000 requests (actual tokens × documented pricing) | Real savings, paid API only; local backend reported as resource use, never dollars |
| p50 and p95 latency: exact hit, semantic hit, miss, direct backend, gateway overhead alone | Where time goes |
| Throughput and overhead at concurrency 1, 4, 16, 32, and 64 | Cost of the gateway itself |

**Baselines:** no cache, and exact-match cache only.

## Testing

Unit tests for boundary rules, namespacing, eviction, TTL, tombstone filtering, and coalescing; integration tests through the full request lifecycle; adversarial regression tests; fault-injection tests for fail-open behavior; load tests on the fixed concurrency ladder.

## Risks

| Risk | Mitigation |
|---|---|
| Scope creep during winter break | Stage table is fixed; anything new moves to v1.5 or v2 |
| Batched removal is slower than expected at full capacity | Lower the tombstone threshold, or rebuild the affected namespace's index; document the tradeoff |
| CPU embedding throughput breaks the overhead budget at higher concurrency | Try one fix (thread pool or ONNX runtime); report results either way |
| QQP label noise or training overlap distorts results | Report both; lean on the hand-checked sample and adversarial set |
| Small hand-checked sample gives wide intervals | Report the intervals; don't claim more than the data supports |
| API cost during benchmarks | Local model for most runs; paid API only for final runs |
| Semantic caching saves little over exact-match | Report it; the system and tooling are still the deliverable |

## Deliverables

Public GitHub repo with Docker Compose setup; the tuning tool; benchmark scripts with fixed seeds and raw results; architecture diagram; a short engineering write-up covering design decisions, tradeoffs, results, and limitations; one resume bullet based only on measured results.

## Milestones

- **November (1–2 hours/week):** Inspect QQP and confirm its license; check the bge model card for training overlap; draft adversarial pairs; design the workload generator.
- **December (as finals allow):** API skeleton, boundary rules, backend abstraction, cache interface, tests; measure single-request embedding latency on target hardware.
- **Winter break, weeks 1–2:** Both caches end to end; coalescing; bounded index with LRU, TTL, and tombstones; batched-removal microbenchmark; fail-open behavior with fault-injection tests.
- **Winter break, weeks 3–4:** Tuning tool; hand-checked sample; latency and load benchmarks; draft write-up.
- **January:** Choose the paid API; v1.5 features; publish the repo and write-up.

## Decisions made

- Embedding model: `BAAI/bge-small-en-v1.5`, pinned revision. A second model is added only if it would change the threshold recommendation.
- Correctness target: 95% Wilson upper bound on held-out false-hit rate of at most 2%; zero adversarial hits.
- Overhead budget: miss overhead at most 10 ms p50 and 25 ms p95 at concurrency 16.
- Index: one `IndexIDMap(IndexFlatIP)` per namespace with normalized vectors; global LRU with 100K-entry capacity; no HNSW.
- Deletion: tombstones filtered at search time; batched `remove_ids` through the single writer when dead entries exceed 10%.
- Coalescing on exact keys only in v1.
- Local inference reported as resource use, never dollar savings.
- Automated grader deferred to v1.5.

## Open questions

1. Which paid API and model for final runs? Decide in January: a small, cheap model with documented pricing and token usage in responses. A provider whose API isn't OpenAI-compatible makes the backend abstraction do real work but adds effort to v1.5 streaming.
