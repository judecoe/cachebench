"""Single-query embedding latency for bge-small on CPU, inside Docker.

Run through scripts/measure_embed_latency.sh, which builds the image, starts this script in
the gateway container (same CPU and memory limits as the gateway), and passes host info in.

For each (threads, concurrency, think time) combination, this embeds 1,000 unique short
queries one per call, the way the gateway will: an asyncio event loop hands each call to a
thread pool (`concurrency` workers and clients). Latency is measured per call from submit
to result, so at concurrency > 1 it includes time queued behind other calls. A warmup of
separate queries runs in the same pool first.

Think time is how long each client waits after a call before sending the next. 0 ms gives
capacity (every client re-sends at once). 1,000 ms stands in for the FakeBackend's fixed
delay in the overhead benchmark (design.md, "Decisions made"), so it shows embed latency
at the load the miss path will actually see. Think time > 0 applies only at concurrency > 1.

--runtime picks torch or onnx (default: the pinned EMBEDDING_RUNTIME). Threads are torch
intra-op threads or the onnx session's intra-op threads.

Queries are generated from templates with a fixed seed. Raw latencies, hardware, and
library versions go to results/embed_latency/<UTC timestamp>.json.
"""

import argparse
import asyncio
import importlib.metadata
import json
import os
import platform
import random
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from cachebench.embedding import Embedder
from cachebench.model_pin import EMBEDDING_RUNTIME, MODEL_NAME, MODEL_REVISION

# Miss-path overhead budget (docs/design.md, "Non-functional"). Embedding is one component
# of the miss path, so it has to fit well inside this with room for search and Redis.
BUDGET = {"concurrency": 16, "p50_ms": 10.0, "p95_ms": 25.0}

TEMPLATES = [
    "how do I {verb} {obj} in {ctx}",
    "what is the best way to {verb} {obj}",
    "why does {verb} {obj} fail in {ctx}",
    "can I {verb} {obj} without {ctx}",
    "{verb} {obj} using {ctx}",
    "is it safe to {verb} {obj}",
    "how long does it take to {verb} {obj}",
]
VERBS = [
    "install", "remove", "update", "enable", "disable", "reset", "back up", "restore",
    "configure", "debug", "migrate", "export", "import", "sort", "merge", "split",
    "encrypt", "compress", "rename", "monitor",
]  # fmt: skip
OBJECTS = [
    "a python package", "my router", "a git branch", "two-factor authentication",
    "a docker image", "a postgres database", "my laptop battery", "a csv file",
    "an ssh key", "dark mode", "a kubernetes pod", "my email inbox", "a json array",
    "the bios", "a shared calendar", "my phone storage", "a linked list", "a cron job",
    "a vpn connection", "a spreadsheet column",
]  # fmt: skip
CONTEXTS = [
    "windows 11", "macos", "ubuntu", "python 3.12", "excel", "chrome", "a script",
    "the command line", "an iphone", "android", "admin rights", "the gui",
]  # fmt: skip


def make_queries(n: int, seed: int) -> list[str]:
    """n unique short queries, deterministic for a given seed."""
    rng = random.Random(seed)
    seen: set[str] = set()
    out: list[str] = []
    while len(out) < n:
        q = rng.choice(TEMPLATES).format(
            verb=rng.choice(VERBS), obj=rng.choice(OBJECTS), ctx=rng.choice(CONTEXTS)
        )
        if q not in seen:
            seen.add(q)
            out.append(q)
    return out


async def run_calls(
    embedder: Embedder,
    pool: ThreadPoolExecutor,
    queries: list[str],
    concurrency: int,
    think_ms: float,
) -> tuple[list[float], float]:
    """Embed every query from `concurrency` clients. Returns (latencies_ms, wall_s)."""
    loop = asyncio.get_running_loop()
    latencies = [0.0] * len(queries)
    pending = iter(enumerate(queries))

    async def client() -> None:
        for i, q in pending:  # shared iterator; safe on a single event loop
            t0 = time.perf_counter_ns()
            await loop.run_in_executor(pool, embedder.embed, q)
            latencies[i] = (time.perf_counter_ns() - t0) / 1e6
            if think_ms:
                await asyncio.sleep(think_ms / 1000)

    start = time.perf_counter()
    await asyncio.gather(*(client() for _ in range(concurrency)))
    return latencies, time.perf_counter() - start


def summarize(latencies: list[float], wall_s: float) -> dict[str, float]:
    arr = np.asarray(latencies)
    return {
        "n": len(latencies),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "mean_ms": float(arr.mean()),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
        "throughput_qps": len(latencies) / wall_s,
    }


async def measure(
    embedder: Embedder,
    concurrency: int,
    think_ms: float,
    warmup: list[str],
    queries: list[str],
) -> dict:
    threads = embedder.threads
    torch.set_num_threads(threads)
    with ThreadPoolExecutor(
        max_workers=concurrency, initializer=torch.set_num_threads, initargs=(threads,)
    ) as pool:
        loop = asyncio.get_running_loop()
        observed = await loop.run_in_executor(pool, torch.get_num_threads)
        await run_calls(embedder, pool, warmup, concurrency, think_ms)
        latencies, wall_s = await run_calls(embedder, pool, queries, concurrency, think_ms)
    return {
        "runtime": embedder.runtime,
        "threads": threads,
        "torch_threads_observed_in_worker": observed,
        "concurrency": concurrency,
        "think_ms": think_ms,
        "kind": "capacity" if think_ms == 0 and concurrency > 1 else "latency",
        "pool_workers": concurrency,
        **summarize(latencies, wall_s),
        "latencies_ms": [round(x, 4) for x in latencies],
    }


def read_text(path: str) -> str | None:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def hardware_info() -> dict:
    cpuinfo = read_text("/proc/cpuinfo") or ""
    cpu_fields = sorted(
        {
            line.strip()
            for line in cpuinfo.splitlines()
            if line.split(":")[0].strip()
            in {"model name", "CPU implementer", "CPU part", "Features", "BogoMIPS"}
        }
    )
    pkgs = [
        "torch",
        "onnxruntime",
        "sentence-transformers",
        "transformers",
        "tokenizers",
        "numpy",
    ]
    return {
        "container": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "os_cpu_count": os.cpu_count(),
            "sched_affinity_cpus": len(os.sched_getaffinity(0)),
            "cgroup_cpu_max": read_text("/sys/fs/cgroup/cpu.max"),
            "cgroup_memory_max": read_text("/sys/fs/cgroup/memory.max"),
            "cpuinfo": cpu_fields,
        },
        "host": {
            k.removeprefix("BENCH_").lower(): os.environ.get(k)
            for k in [
                "BENCH_HOST_CPU",
                "BENCH_HOST_MEMORY_BYTES",
                "BENCH_HOST_OS",
                "BENCH_DOCKER_SERVER_VERSION",
                "BENCH_DOCKER_VM_CPUS",
                "BENCH_DOCKER_VM_MEMORY_BYTES",
                "BENCH_GIT_COMMIT",
                "BENCH_GIT_DIRTY",
            ]
        },
        "versions": {p: importlib.metadata.version(p) for p in pkgs},
        "torch_parallel_info": torch.__config__.parallel_info(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 16])
    parser.add_argument("--think-ms", type=float, nargs="+", default=[0, 1000])
    parser.add_argument("--runtime", choices=["torch", "onnx"], default=EMBEDDING_RUNTIME)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default="results/embed_latency")
    args = parser.parse_args()

    if not Path("/.dockerenv").exists():
        sys.exit("runs in Docker only: use scripts/measure_embed_latency.sh")

    all_queries = make_queries(args.n + args.warmup, args.seed)
    warmup, queries = all_queries[: args.warmup], all_queries[args.warmup :]

    runs = []
    for threads in args.threads:
        embedder = Embedder(runtime=args.runtime, threads=threads)
        token_counts = [embedder.token_count(q) for q in queries]
        for concurrency in args.concurrency:
            for think_ms in args.think_ms:
                if think_ms and concurrency == 1:
                    continue
                run = asyncio.run(measure(embedder, concurrency, think_ms, warmup, queries))
                runs.append(run)
                print(
                    f"{args.runtime:<5} threads={threads:<2} concurrency={concurrency:<3} "
                    f"think={think_ms:>5.0f} ms  "
                    f"p50={run['p50_ms']:7.2f} ms  p95={run['p95_ms']:7.2f} ms  "
                    f"p99={run['p99_ms']:7.2f} ms  {run['throughput_qps']:7.1f} q/s",
                    flush=True,
                )

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    result = {
        "benchmark": "embed_latency",
        "timestamp_utc": timestamp,
        "model": {
            "name": MODEL_NAME,
            "revision": MODEL_REVISION,
            "runtime": args.runtime,
            "device": "cpu",
        },
        "workload": {
            "seed": args.seed,
            "n": args.n,
            "warmup": args.warmup,
            "unique_queries": len(set(all_queries)) == len(all_queries),
            "tokens_per_query": {
                "min": min(token_counts),
                "median": statistics.median(token_counts),
                "max": max(token_counts),
            },
            "example_queries": queries[:5],
        },
        "method": (
            "one embed() call per query (normalize, token-length check, encode, L2 normalize); "
            "asyncio loop -> ThreadPoolExecutor(pool_workers); latency from submit to result, "
            "including queueing; each client waits think_ms after each call (0 = capacity; "
            "1000 = FakeBackend delay); percentiles by numpy linear interpolation"
        ),
        "reference_budget_miss_overhead": BUDGET,
        "hardware": hardware_info(),
        "runs": runs,
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{timestamp}_{args.runtime}.json"
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
