"""Print the cosine similarity of each text pair in a JSONL file, grouped by category.

Input: one JSON object per line (blank lines are skipped):
    {"a": "Enable dark mode", "b": "Disable dark mode", "category": "negation"}
Optional fields: "id" and "note" (printed if present).

Each text is embedded on its own with the gateway's Embedder (shared normalization,
512-token check), exactly as the gateway would, and similarity is the inner product of the
unit vectors. With --threshold, pairs at or above it (>=, as in the gateway) are marked
HIT and the exit code is 1 if any pair hits.

Usage: uv run python scripts/score_pairs.py pairs.jsonl [--threshold 0.92]
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from cachebench.embedding import Embedder, TextTooLongError


def load_pairs(path: Path) -> list[dict]:
    pairs = []
    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            sys.exit(f"{path}:{lineno}: invalid JSON: {e}")
        missing = [k for k in ("a", "b", "category") if not isinstance(obj.get(k), str)]
        if missing:
            sys.exit(f"{path}:{lineno}: missing or non-string field(s): {', '.join(missing)}")
        pairs.append(obj)
    if not pairs:
        sys.exit(f"{path}: no pairs")
    return pairs


def one_line(text: str, width: int) -> str:
    flat = text.replace("\n", "\\n")
    return flat if len(flat) <= width else flat[: width - 1] + "…"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("pairs", type=Path)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--width", type=int, default=60, help="max characters shown per text")
    args = parser.parse_args()

    pairs = load_pairs(args.pairs)
    embedder = Embedder()

    by_category: dict[str, list[tuple[float | None, dict, str]]] = defaultdict(list)
    for pair in pairs:
        try:
            vec_a, vec_b = embedder.embed(pair["a"]), embedder.embed(pair["b"])
            score: float | None = float(np.dot(vec_a, vec_b))
            err = ""
        except TextTooLongError as e:
            score, err = None, f"too long ({e.n_tokens} tokens)"
        by_category[pair["category"]].append((score, pair, err))

    hits = 0
    for category, rows in by_category.items():
        scores = [s for s, _, _ in rows if s is not None]
        stats = (
            f"min {min(scores):.4f}  mean {np.mean(scores):.4f}  max {max(scores):.4f}"
            if scores
            else "no scores"
        )
        print(f"\n{category}  (n={len(rows)}; {stats})")
        rows.sort(key=lambda r: -1.0 if r[0] is None else r[0], reverse=True)
        for score, pair, err in rows:
            hit = args.threshold is not None and score is not None and score >= args.threshold
            hits += hit
            label = f"[{pair['id']}] " if "id" in pair else ""
            shown = f"{score:.4f}" if score is not None else err
            print(
                f"  {'HIT ' if hit else '    '}{shown}  {label}"
                f"{one_line(pair['a'], args.width)}  ||  {one_line(pair['b'], args.width)}"
            )
            if "note" in pair:
                print(f"              note: {pair['note']}")

    if args.threshold is not None:
        print(f"\n{hits} of {len(pairs)} pairs at or above threshold {args.threshold}")
        sys.exit(1 if hits else 0)


if __name__ == "__main__":
    main()
