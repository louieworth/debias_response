#!/usr/bin/env python3
"""Validate and summarize all output-head ablation results."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from run_head_ablation import (
    DATASETS,
    DEFAULT_NAMESPACE,
    LEVEL_CONFIG,
    METHODS,
    PROJECT_ROOT,
    Task,
)


METRICS = {
    "MAE": "test_mae_model_original",
    "NAcc": "test_acc_model_mad",
    "HA": "test_acc_model_hard",
    "SA": "test_acc_model_soft",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--namespace",
        default=DEFAULT_NAMESPACE,
        help="Directory name under results/ used by the runner.",
    )
    parser.add_argument(
        "--levels",
        nargs="+",
        choices=sorted(LEVEL_CONFIG),
        default=list(LEVEL_CONFIG),
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=list(METHODS),
        default=list(METHODS),
    )
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args()


def load_rows(
    allow_partial: bool,
    namespace: str,
    levels: list[str],
    methods: list[str],
) -> pd.DataFrame:
    rows = []
    missing = []
    for level in levels:
        for dataset in DATASETS:
            for method in methods:
                for seed in LEVEL_CONFIG[level]["seeds"]:
                    task = Task(level, dataset, method, seed, namespace)
                    if not task.result_path.exists():
                        missing.append(str(task.result_path))
                        continue
                    frame = pd.read_csv(task.result_path)
                    if len(frame) != 1:
                        raise ValueError(
                            f"{task.result_path} must contain exactly one row"
                        )
                    row = frame.iloc[0].to_dict()
                    expected = {
                        "variant": METHODS[method]["variant"],
                        "model_type": "mlp",
                        "prediction_head": METHODS[method]["head"],
                    }
                    for key, value in expected.items():
                        if row.get(key) != value:
                            raise ValueError(
                                f"{task.result_path} has {key}={row.get(key)!r}, "
                                f"expected {value!r}"
                            )
                    output = {
                        "level": level,
                        "dataset": dataset,
                        "method": method,
                        "method_label": METHODS[method]["label"],
                        "seed": seed,
                        "queries_per_question": (
                            1
                            if method == "one"
                            else (50 if level == "population" else 8)
                        ),
                    }
                    for metric, column in METRICS.items():
                        output[metric] = float(row[column]) * 100.0
                    rows.append(output)

    if missing and not allow_partial:
        preview = "\n".join(missing[:10])
        raise FileNotFoundError(
            f"Missing {len(missing)} expected result files. First entries:\n{preview}"
        )
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    raw = load_rows(
        args.allow_partial,
        args.namespace,
        args.levels,
        args.methods,
    )
    if raw.empty:
        raise SystemExit("No result rows found")

    group_columns = [
        "level",
        "dataset",
        "method",
        "method_label",
        "queries_per_question",
    ]
    summary = (
        raw.groupby(group_columns, sort=False)[list(METRICS)]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary.columns = [
        "_".join(value for value in column if value)
        if isinstance(column, tuple)
        else column
        for column in summary.columns
    ]

    output_root = PROJECT_ROOT / "results" / args.namespace
    output_root.mkdir(parents=True, exist_ok=True)
    raw.to_csv(output_root / "seed_metrics.csv", index=False)
    summary.to_csv(output_root / "summary.csv", index=False)
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"Wrote {output_root / 'seed_metrics.csv'}")
    print(f"Wrote {output_root / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
