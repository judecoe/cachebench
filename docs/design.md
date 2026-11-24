# Design Doc: CacheBench (working name)

**Author:** Jude Gilligan · **Status:** v5.1 (v1 decisions locked; remaining items listed under "Open before winter break") · **Last updated:** November 23, 2026

**Changes in v5.1:** text normalization no longer touches internal whitespace; absent and null parameters are omitted from the namespace; a semantic hit writes nothing to either cache; the threshold sweep requires at least 400 semantic hits.

**Changes in v5:** paid backend and cost metric moved to v1.5; local backend is Ollama on the host; single-process gateway; per-namespace readers-writer lock; explicit cacheable allowlist; namespace fields and text normalization defined; no index writes on semantic hits and cache writes after the response; tuning split by connected component; new section "Open before winter break."

## Goal

Build a production-style LLM gateway with a semantic cache that is safe to run in production, plus the tooling to choose and justify its similarity threshold.

## Problem

Teams calling LLM APIs pay for, and wait on, many requests that are paraphrases of ones already answered. A semantic cache can reuse those answers, but a threshold set too loosely silently serves wrong answers, and a cache that adds latency or fails badly makes the service worse. Shipping one safely requires solving both the systems problems and the threshold problem.

## Requirements

**Functional**
- OpenAI-compatible chat completions endpoint
- Exact-match and semantic caching, with clear rules for which requests are cacheable
- Two interchangeable model backends (one local, one paid API); the paid API backend lands in v1.5
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
| v1 | Winter break | Gateway endpoint; OpenAI-compatible backend (local model via Ollama) and fake backend; request boundary rules; exact-match cache; semantic cache with bounded memory, eviction, and TTL; request coalescing; fail-open behavior; threshold tuning tool; latency and load benchmarks; adversarial regression tests; structured logs; Docker Compose |
| v1.5 | January | Paid API backend and API cost metric; streaming responses with caching; reliability features (upstream timeouts, HTTP 429 and 5xx handling, retries with backoff, circuit breaker); Prometheus metrics; automated answer grader validated against human labels |
| v2 | Spring 2027 | Simple two-model routing evaluated against fixed baselines; metrics dashboard; optional shared cache state across gateway instances |

**Out of scope:** request batching, adaptive or learned routing, multi-tenant auth, Kubernetes, distributed vector databases.

## Request and cache boundary

**Cacheable (v1):** a request is cacheable only if it matches this allowlist. Anything else bypasses the cache, and the bypass reason is logged.
- **Shape:** an optional single `system` message plus exactly one `user` message whose `content` is a string. No conversation history, no content-part arrays, no multiple system messages.
- **Parameters:** only `model`, `messages`, `temperature`, `max_tokens`, `top_p`, `stop`, `presence_penalty`, `frequency_penalty`, `response_format`, `n`, and `stream`.
- **Required values:** `temperature` must be present and exactly 0 (the OpenAI default is 1, so a missing temperature bypasses). `n`, if present, must be 1.
- Any other parameter (for example `tools`, `tool_choice`, `logprobs`, `seed`, `user`) bypasses.

**Streaming:** `stream: true` returns HTTP 400 for every request, cacheable or not, until v1.5.

**Namespace:** semantic lookups only compare within a namespace. The namespace is built from:
- the pinned model name from config (dated version or Ollama tag; never an alias)
- a hash of the system prompt
- generation settings: `max_tokens`, `top_p`, `stop`, `presence_penalty`, `frequency_penalty`
- `response_format`
- the embedding model revision
- the preprocessing version (bumped whenever normalization or embedding preprocessing changes)

A parameter that is absent from the request is omitted from the namespace, and an explicit `null` is treated as absent. No provider defaults are filled in, so a request that omits `top_p` and one that sends `top_p: 1` land in different namespaces. This splits the cache slightly but never assumes what a provider's default is.

**Text normalization:** each message's text (system and user) is normalized with Unicode NFC, CRLF converted to LF, and leading and trailing whitespace stripped. Internal whitespace is never changed: indentation and line breaks matter for code, tables, and verse. There is no case folding. The exact-match key is a hash of the namespace and the normalized user text. The same normalization function produces the text the embedder sees, in both the gateway and the tuning tool. Golden tests lock its output and the resulting keys, including that indentation-only and line-break-only differences produce different keys.

**Expiration:** every entry has a configurable TTL.

## Architecture

**Request lifecycle:** boundary check (cacheable or bypass) → exact-match lookup (Redis) → coalescing check → semantic lookup (embed, nearest-neighbor search within namespace, threshold comparison) → on a miss, call the backend → return the response → write to both caches.

Cache writes happen after the response is sent, so they stay out of request latency. A repeat that arrives between the response and the write may miss; this is accepted. A semantic hit writes nothing to either cache. It never inserts into the semantic index, because chaining near-matches (A≈B, B≈C) would let C receive A's answer; it also skips the exact cache, so a later exact hit is never an answer that was originally a semantic hit.

**Process model:** the gateway runs as a single process (`uvicorn --workers 1`). The FAISS index and the coalescing map live in process memory; with several workers, each would hold its own index and coalescing map, splitting the cache and duplicating backend calls without any error. Sharing cache state across instances is a v2 item.

**Components:** FastAPI gateway; Redis exact-match cache; in-process FAISS semantic index (one `IndexIDMap(IndexFlatIP)` per namespace); local embedding model `BAAI/bge-small-en-v1.5` at a pinned revision; a backend abstraction with an `OpenAICompatBackend` (the local model via Ollama on the host, reached at `host.docker.internal`; the paid API from v1.5) and a `FakeBackend` for tests and overhead benchmarks; a shared cache interface (get, put, invalidate) implemented separately by each cache.

Every request logs its outcome (bypass, exact hit, semantic hit, coalesced, or miss), similarity score, per-stage latency, and token usage.

## Key engineering problems

**Request coalescing.** When several identical requests miss at the same moment, only one goes to the backend; the rest wait for its result. If the backend call fails or times out, every waiter receives the error and nothing is cached. v1 coalesces on the exact key only. Coalescing near-identical in-flight requests is harder (it would need a semantic check on requests that have no answer yet) and is a non-goal for v1.

**Bounded memory and eviction.** Each namespace has its own flat index; vectors are L2-normalized, so inner product equals cosine similarity. A global LRU across namespaces enforces the 100K-entry limit. Flat search is exact and takes a few milliseconds at this size; HNSW is not used because it does not support removal.

Deletion uses tombstones plus batched removal:
1. When an entry expires or is evicted, it is marked dead in the metadata store immediately.
2. Searches request the top 5 neighbors and return the first live one.
3. When dead entries exceed 10% of an index, the single writer removes them in one `remove_ids` call.

A microbenchmark in winter break week 1 confirms batched removal cost at full capacity.

**Concurrent access.** FAISS is not safe for concurrent writes, and a search that runs during a write is also unsafe (an insert can reallocate the vector buffer mid-scan). All index inserts and batched removals go through a single writer. Each namespace index has a readers-writer lock: searches hold it shared, and the writer holds it exclusively. Searches run in a thread pool off the event loop so they don't block the server. FAISS's internal threading is set to one thread (`faiss.omp_set_num_threads(1)`) so it doesn't oversubscribe the CPU alongside the search threads. Before inserting, the writer checks for an existing near-duplicate in the same namespace; if one exists, the first entry wins and the new one is skipped.

**Failure handling.** If Redis, the embedding model, or FAISS errors out or exceeds a time limit, the request skips caching, goes straight to the backend, and the failure is logged. Fault-injection tests verify this.

**Streaming (v1.5).** A streamed response is cached only after it completes successfully. A stream that fails partway is never cached, and a stream can't be retried once bytes have reached the client.

## Threshold tuning tool

A command (for example, `cachebench tune`) that takes labeled query pairs and a correctness target (default: 2% false-hit rate at the 95% Wilson upper bound). It splits the data with a fixed seed into a tuning set and a held-out set, sweeps thresholds on the tuning set, picks the loosest threshold that meets the target, verifies it on the held-out set, and outputs the threshold with its false-hit rate, confidence interval, and expected hit rate as gateway config. It also outputs the full threshold curve so readers can apply a different target.

The split is by connected component: questions are nodes, labeled pairs are edges, and each component goes entirely to one side, so no question appears in both the tuning and held-out sets. The tool uses the gateway's normalization and embedding code.

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
| API cost per 1,000 requests (actual tokens × documented pricing) | Real savings, paid API only (v1.5); local backend reported as resource use, never dollars |
| p50 and p95 latency: exact hit, semantic hit, miss, direct backend, gateway overhead alone | Where time goes |
| Throughput and overhead at concurrency 1, 4, 16, 32, and 64 | Cost of the gateway itself |

Overhead benchmarks use the `FakeBackend`, so model latency variance doesn't swamp the overhead budget.

**Baselines:** no cache, and exact-match cache only.

## Testing

Unit tests for boundary rules, namespacing, eviction, TTL, tombstone filtering, and coalescing; golden tests for text normalization and cache keys (indentation-only and line-break-only differences give different keys; omitted `top_p` and `top_p: 1` give different keys); integration tests through the full request lifecycle; adversarial regression tests; fault-injection tests for fail-open behavior; load tests on the fixed concurrency ladder. Tests use the `FakeBackend` and never call a paid API.

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
- **December (as finals allow):** API skeleton, boundary rules, namespace and text normalization with golden tests, `OpenAICompatBackend` and `FakeBackend`, cache interface, tests; measure single-request embedding latency on target hardware.
- **Winter break, weeks 1–2:** Both caches end to end; coalescing; bounded index with LRU, TTL, and tombstones; batched-removal microbenchmark; fail-open behavior with fault-injection tests.
- **Winter break, weeks 3–4:** Tuning tool; hand-checked sample; latency and load benchmarks; draft write-up.
- **January:** Choose the paid API; paid backend and API cost metric; other v1.5 features; publish the repo and write-up.

## Decisions made

- Embedding model: `BAAI/bge-small-en-v1.5`, pinned revision. A second model is added only if it would change the threshold recommendation.
- Correctness target: 95% Wilson upper bound on held-out false-hit rate of at most 2%; zero adversarial hits.
- Overhead budget: miss overhead at most 10 ms p50 and 25 ms p95 at concurrency 16.
- Index: one `IndexIDMap(IndexFlatIP)` per namespace with normalized vectors; global LRU with 100K-entry capacity; no HNSW.
- Deletion: tombstones filtered at search time; batched `remove_ids` through the single writer when dead entries exceed 10%.
- Coalescing on exact keys only in v1.
- Local inference reported as resource use, never dollar savings.
- Automated grader deferred to v1.5.
- Backends: December builds `OpenAICompatBackend` and `FakeBackend`; the paid API backend and the API cost metric move to v1.5 (January).
- Local model: Ollama on the host, reached at `host.docker.internal`; a small instruct model at a pinned tag. Overhead benchmarks use `FakeBackend`.
- Process model: one gateway process (`--workers 1`), because the index and coalescing map are in process memory.
- Index concurrency: per-namespace readers-writer lock (searches shared, writer exclusive); `faiss.omp_set_num_threads(1)`.
- Cacheable requests: the allowlist in "Request and cache boundary"; `temperature` must be explicitly 0; `stream: true` returns 400 until v1.5.
- Namespace: pinned model names from config only (no aliases), system prompt hash, listed generation settings, response format, embedding model revision, and preprocessing version. Absent parameters are omitted; explicit `null` equals absent; no provider defaults are hard-coded.
- Text normalization: per message, NFC, CRLF→LF, strip leading and trailing whitespace; internal whitespace unchanged; no case folding. The same function feeds the exact key and the embedder; locked by golden tests.
- Writes: a semantic hit writes nothing to either cache; cache writes happen after the response is sent.
- Tuning split: by connected component of the question graph.
- Threshold sweep: a threshold needs at least 400 semantic hits on the tuning set to qualify. With zero false hits, the 95% Wilson upper bound is 3.7% at 100 hits, so 100 can never meet the 2% target; the minimum that can is 189. At 400 hits, up to 2 false hits pass (upper bound 1.8%) and 3 fail (2.2%).

## Open before winter break

Each item has a proposed default. Unless changed, the default is adopted when winter-break implementation reaches it.

1. **Where semantic answers are stored.** Proposed: semantic metadata (id → namespace, exact key, `expires_at`, `last_access`, dead flag) lives in process; the response body lives in Redis under the exact key. If the Redis key is missing, the entry is marked dead and the lookup is a miss. Consequence: a Redis outage also disables semantic hits, which fail-open covers.
2. **Redis memory bound.** The memory requirement covers only the semantic index. Proposed: Redis `maxmemory 512mb` with `allkeys-lru`, and persistence off, so a restart empties both caches and benchmarks start from the same state.
3. **Timeouts.** Coalescing assumes a backend timeout, but upstream timeouts are a v1.5 feature. Proposed: one fixed backend timeout in v1 with no retries; time limits for Redis, embedding, and search set from the December latency measurement; a bounded queue for the embedding and search thread pool that fails open when full, because `asyncio.wait_for` doesn't stop a thread that is already running.
4. **Coalescing details.** Proposed: the backend call is shielded so a leader's client disconnecting doesn't cancel it for followers; followers receive whatever result the leader got (exact, semantic, or backend) and are logged as `coalesced`; only cacheable requests coalesce.
5. **Long prompts.** bge-small reads at most 512 tokens, so two prompts with the same first 512 tokens embed identically. Proposed: prompts over the limit use the exact cache only.
6. **bge query prefix.** Proposed: no query instruction prefix, since this is query-to-query similarity; one preprocessing function shared by the tuning tool and the gateway.
7. **LRU and tombstone details.** Proposed:
   - An exact hit refreshes the LRU position of the semantic entry for the same key.
   - The 100K limit counts stored vectors (live and dead), so the memory requirement holds literally.
   - Expiry is checked lazily at search time, plus a periodic sweep by the writer.
   - If all top-5 neighbors are dead, the lookup is a miss.
   - Threshold comparison is `≥` in both the tuning tool and the gateway.
   - The near-duplicate check before insert uses the serving threshold.
8. **Pairwise versus served false-hit rate.** The tuning tool measures precision on labeled pairs (about 37% of QQP pairs are duplicates); the gateway returns the nearest of many cached entries under a different traffic mix. Proposed: state the 2% target as pairwise on held-out QQP, and report the workload false-hit rate separately.
9. **Labels for workload hits.** A workload query can hit a cached entry it has no QQP label with. Proposed: build workloads from duplicate clusters (connected components); a hit within the same cluster counts as correct, a cross-cluster hit counts as wrong (conservative), and a sample of cross-cluster hits is hand-checked.
10. **Threshold sweep rules.** The Wilson upper bound is wide when there are few hits and doesn't fall steadily as the threshold changes. The 400-hit minimum is decided (see "Decisions made"). Proposed: grid step 0.005; "loosest" means the lowest threshold at which it and every stricter qualifying threshold meet the target. If the held-out check fails, the miss is reported and the tool does not re-tune on held-out data.
11. **Adversarial gate versus tuning.** Proposed: the shipped threshold is the stricter of the QQP threshold and the loosest threshold that blocks every adversarial pair; both are reported. An adversarial "hit" means the pair's similarity reaches the threshold (stricter than testing through the gateway). The write-up notes that the adversarial set then helps set the threshold as well as check it.
12. **Question 4 criterion and workload parameters.** Proposed: semantic caching is worth its overhead if it adds at least 5 percentage points of hit rate over exact-only at medium repetition at the shipped threshold, reported next to the incorrect-served rate. Zipf exponents, number of intents, stream length, and paraphrase fraction are fixed before any workload is generated.
13. **Overhead measurement.** Proposed: the primary number is per-request gateway handler time minus backend call time; the end-to-end difference against direct backend calls is a cross-check. "Miss" means a full embed-and-search miss on all-unique queries, not a bypass. The load generator runs inside the Compose network, and the gateway container has a fixed CPU limit on documented hardware.
14. **QQP license and published results.** QQP is covered by Quora's terms, not an open license. Proposed: commit QQP IDs and a download script, never question text (including workloads and the hand-checked sample).
15. **Training overlap.** BGE's English training mix is reported to include Quora duplicates. Proposed: confirm in November; if confirmed, report QQP results as optimistic and give the hand-checked sample and adversarial set more weight.
16. **Cached response format.** Proposed: a cached response gets a new `id` and `created` and keeps its original `usage`; the gateway adds `X-Cache` and `X-Cache-Similarity` headers and logs `tokens_saved`.

## Open questions

1. Which paid API and model for final runs? Decide in January: a small, cheap model with documented pricing and token usage in responses. A provider whose API isn't OpenAI-compatible makes the backend abstraction do real work but adds effort to v1.5 streaming.
2. Which Ollama model tag for the local backend? Author's choice: a small instruct model, pinned in config before any benchmark run.
