#!/usr/bin/env python3
"""
Print routed model names for representative sample queries from the bundled
example dataset, so it is easy to see whether a router predicts anything other
than the dominant label.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

from llmrouter.cli.router_inference import load_router


DEFAULT_ROUTING_DATA = "data/example_data/routing_data/default_routing_train_data.jsonl"
DEFAULT_CONFIG = "configs/model_config_test/svmrouter.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check whether a query-dependent router predicts diverse model names."
    )
    parser.add_argument(
        "--router",
        default="svmrouter",
        help="Router name to load, for example svmrouter or knnrouter.",
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help="Router config to use for inference.",
    )
    parser.add_argument(
        "--routing-data",
        default=DEFAULT_ROUTING_DATA,
        help="JSONL routing dataset used to pick representative sample queries.",
    )
    parser.add_argument(
        "--per-model",
        type=int,
        default=2,
        help="Number of representative queries to test for each expected model.",
    )
    parser.add_argument(
        "--exclude-model",
        default="qwen2.5-7b-instruct",
        help="Skip sample queries whose best label matches this model.",
    )
    parser.add_argument(
        "--max-query-chars",
        type=int,
        default=220,
        help="Maximum number of query characters to print per sample.",
    )
    return parser.parse_args()


def load_best_rows(routing_data_path: Path) -> List[Dict[str, object]]:
    best_by_query: Dict[str, Dict[str, object]] = {}

    with routing_data_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            query = row["query"]
            current = best_by_query.get(query)
            if current is None or row["performance"] > current["performance"]:
                best_by_query[query] = row

    return list(best_by_query.values())


def truncate(text: str, limit: int) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def choose_samples(
    best_rows: List[Dict[str, object]],
    per_model: int,
    excluded_model: str,
) -> List[Dict[str, object]]:
    grouped: Dict[str, List[Dict[str, object]]] = {}

    for row in best_rows:
        model_name = str(row["model_name"])
        if model_name == excluded_model:
            continue
        grouped.setdefault(model_name, []).append(row)

    samples: List[Dict[str, object]] = []
    for model_name in sorted(grouped):
        for row in grouped[model_name][:per_model]:
            samples.append(row)
    return samples


def main() -> None:
    args = parse_args()
    routing_data_path = Path(args.routing_data)

    if not routing_data_path.exists():
        raise FileNotFoundError(f"Routing data not found: {routing_data_path}")

    best_rows = load_best_rows(routing_data_path)
    label_counts = Counter(str(row["model_name"]) for row in best_rows)
    samples = choose_samples(best_rows, args.per_model, args.exclude_model)

    print(f"Loading router={args.router} config={args.config}")
    router = load_router(args.router, args.config)

    print(f"Unique best-label queries: {len(best_rows)}")
    print("Best-label distribution:")
    for model_name, count in label_counts.most_common():
        print(f"  {model_name}: {count}")

    if not samples:
        print("No sample queries matched the requested filters.")
        return

    print("\nSample predictions:")
    predicted_counts: Counter[str] = Counter()
    for index, row in enumerate(samples, start=1):
        query = str(row["query"])
        expected = str(row["model_name"])
        prediction = router.route_single({"query": query})
        predicted = str(prediction.get("model_name"))
        predicted_counts[predicted] += 1

        print(f"\n[{index}] expected={expected} predicted={predicted}")
        print(truncate(query, args.max_query_chars))

    print("\nPredicted-label distribution across sampled queries:")
    for model_name, count in predicted_counts.most_common():
        print(f"  {model_name}: {count}")


if __name__ == "__main__":
    main()
