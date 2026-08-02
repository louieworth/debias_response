#!/usr/bin/env python3
"""Summarize five-seed Twin source-robustness results for the appendix table."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


SOURCE_FIELDS = {
    "claude-3.5-haiku": "claude-3.5-haiku_norm",
    "deepseek-v3": "deepseek-v3_norm",
    "gpt-3.5-turbo": "gpt-3.5-turbo_norm",
    "gpt-4o-mini": "gpt-4o-mini_norm",
    "gpt-4o": "gpt-4o_norm",
    "gpt-5-mini": "gpt-5-mini_norm",
    "llama-3.3-70B": "llama-3.3-70B-instruct-turbo_norm",
    "mistral-7B-v0.3": "mistral-7B-instruct-v0.3_norm",
}
VARIANTS = {
    "One": "x_one_llm",
    "Mean": "x_avg_llm",
    "Vector": "x_all_llm",
}
METRICS = {
    "MAE": (
        "test_mae_base_original",
        "test_mae_model_original",
        "min",
    ),
    "Acc": ("test_acc_base_mad", "test_acc_model_mad", "max"),
    "HA": ("test_acc_base_hard", "test_acc_model_hard", "max"),
    "SA": ("test_acc_base_soft", "test_acc_model_soft", "max"),
}


def load_seed_files(directory: Path, prefix: str) -> pd.DataFrame:
    frames = []
    for seed in range(5):
        path = directory / f"{prefix}_seed_{seed}.csv"
        frame = pd.read_csv(path)
        frame["seed"] = seed
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def metric_values(
    source_rows: pd.DataFrame,
    no_llm_rows: pd.DataFrame,
    metric: str,
) -> dict[str, tuple[float, float]]:
    base_column, model_column, _ = METRICS[metric]
    mean_rows = source_rows.loc[
        source_rows["variant"].eq("x_avg_llm")
    ].sort_values("seed")
    base_values = mean_rows[base_column].to_numpy(dtype=float) * 100.0
    if len(base_values) != 5 or not np.allclose(
        base_values, base_values[0], atol=1e-12
    ):
        raise ValueError(f"{metric}: Base must be fixed across five seeds")

    values = {"Base": (float(base_values[0]), 0.0)}
    no_llm_values = (
        no_llm_rows.sort_values("seed")[model_column].to_numpy(dtype=float)
        * 100.0
    )
    if len(no_llm_values) != 5:
        raise ValueError(f"{metric}: expected five w/o LLM rows")
    values["w/o"] = (
        float(no_llm_values.mean()),
        float(no_llm_values.std(ddof=1)),
    )
    for method, variant in VARIANTS.items():
        method_values = (
            source_rows.loc[
                source_rows["variant"].eq(variant)
            ].sort_values("seed")[model_column].to_numpy(dtype=float)
            * 100.0
        )
        if len(method_values) != 5:
            raise ValueError(
                f"{metric}/{method}: expected five rows, "
                f"found {len(method_values)}"
            )
        values[method] = (
            float(method_values.mean()),
            float(method_values.std(ddof=1)),
        )
    return values


def latex_cell(mean: float, sd: float, bold: bool) -> str:
    displayed_mean = f"{np.round(mean, 1):.1f}"
    displayed_sd = f"{np.round(sd, 1):.1f}"
    if bold:
        return (
            r"\metricstack{\textbf{"
            + displayed_mean
            + "}}{"
            + displayed_sd
            + "}"
        )
    return rf"\plaincell{{{displayed_mean}}}{{{displayed_sd}}}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--primary-dir",
        type=Path,
        default=Path(
            "results/group/twin/aggreated_279_source_robustness"
        ),
    )
    parser.add_argument(
        "--primary-prefix",
        default="aggreated_twin_279_source_robustness",
    )
    parser.add_argument(
        "--local-dir",
        type=Path,
        default=Path(
            "results/group/twin/"
            "aggreated_279_source_robustness_local_models"
        ),
    )
    parser.add_argument(
        "--local-prefix",
        default="aggreated_twin_279_source_robustness_local_models",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/group/twin/aggreated_279_source_robustness/"
            "table4_summary.csv"
        ),
    )
    args = parser.parse_args()

    primary = load_seed_files(args.primary_dir, args.primary_prefix)
    local = load_seed_files(args.local_dir, args.local_prefix)
    no_llm = primary.loc[primary["variant"].eq("x_only")]
    if no_llm.groupby("seed").size().to_dict() != {
        seed: 1 for seed in range(5)
    }:
        raise ValueError("Expected exactly one primary x_only row per seed")

    all_source_rows = pd.concat([primary, local], ignore_index=True)
    records = []
    latex_rows = []
    method_order = ["Base", "w/o", "One", "Mean", "Vector"]
    for source, field in SOURCE_FIELDS.items():
        source_rows = all_source_rows.loc[
            all_source_rows["llm_field"].eq(field)
        ]
        cells = []
        for metric in METRICS:
            values = metric_values(source_rows, no_llm, metric)
            displayed = {
                method: float(np.round(values[method][0], 1))
                for method in method_order
            }
            direction = METRICS[metric][2]
            best = (
                min(displayed.values())
                if direction == "min"
                else max(displayed.values())
            )
            for method in method_order:
                mean, sd = values[method]
                bold = displayed[method] == best
                records.append(
                    {
                        "source": source,
                        "metric": metric,
                        "method": method,
                        "mean": mean,
                        "sd": sd,
                        "display_mean": displayed[method],
                        "display_sd": float(np.round(sd, 1)),
                        "bold": bold,
                    }
                )
                cells.append(latex_cell(mean, sd, bold))
        latex_rows.append(
            rf"\texttt{{{source}}} & " + " & ".join(cells) + r" \\"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(args.output, index=False)
    print(f"wrote {args.output}")
    print("\n".join(latex_rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
