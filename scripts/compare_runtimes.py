"""Compare torch and ONNX Runtime vectors for the pinned bge-small revision.

Embeds the same 1,000 fixed-seed queries as measure_embed_latency.py, one call per query
(as the gateway does), with each runtime, and reports:
- cosine similarity between the torch and onnx vector for each query (min is the gate);
- how much query-to-query similarity changes between runtimes, over consecutive query
  pairs, since that is what a threshold is compared against.

Run in Docker so the numbers come from the deployment runtime:
    docker compose run --rm --no-deps -v "$PWD/scripts:/app/scripts:ro" \
        -v "$PWD/results:/app/results" gateway python scripts/compare_runtimes.py
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from measure_embed_latency import hardware_info, make_queries

from cachebench.embedding import Embedder
from cachebench.model_pin import MODEL_NAME, MODEL_REVISION


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default="results/runtime_compare")
    args = parser.parse_args()

    # Same split as measure_embed_latency.py: the measured queries follow the warmup ones.
    queries = make_queries(args.n + args.warmup, args.seed)[args.warmup :]

    vecs = {}
    for runtime in ("torch", "onnx"):
        embedder = Embedder(runtime=runtime)
        vecs[runtime] = np.stack([embedder.embed(q) for q in queries])

    cos = np.sum(vecs["torch"] * vecs["onnx"], axis=1)
    sim_torch = np.sum(vecs["torch"][:-1] * vecs["torch"][1:], axis=1)
    sim_onnx = np.sum(vecs["onnx"][:-1] * vecs["onnx"][1:], axis=1)
    pair_delta = np.abs(sim_torch - sim_onnx)
    worst = np.argsort(cos)[:5]

    summary = {
        "n": len(queries),
        "cosine_torch_vs_onnx": {
            "min": float(cos.min()),
            "p01": float(np.percentile(cos, 1)),
            "median": float(np.median(cos)),
            "max": float(cos.max()),
        },
        "max_abs_element_diff": float(np.abs(vecs["torch"] - vecs["onnx"]).max()),
        "pair_similarity_abs_delta": {
            "n_pairs": len(pair_delta),
            "max": float(pair_delta.max()),
            "p99": float(np.percentile(pair_delta, 99)),
            "median": float(np.median(pair_delta)),
        },
        "lowest_cosine_queries": [{"query": queries[i], "cosine": float(cos[i])} for i in worst],
    }
    print(json.dumps(summary, indent=2))

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{timestamp}.json"
    result = {
        "benchmark": "runtime_compare",
        "timestamp_utc": timestamp,
        "model": {"name": MODEL_NAME, "revision": MODEL_REVISION},
        "workload": {"seed": args.seed, "n": args.n, "warmup_skipped": args.warmup},
        "summary": summary,
        "hardware": hardware_info(),
        "cosines": [round(float(c), 9) for c in cos],
    }
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
