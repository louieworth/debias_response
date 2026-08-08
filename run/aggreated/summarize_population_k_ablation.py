#!/usr/bin/env python3
"""Summarize the current population K ablation for seeds 0--4."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("results/population_k_ablation_k1_8_25_50_100_200_seed0_4_precision17")
MAIN_ROOT = Path("results/population_one_significance_seed0_4_precision17")
PAPER_CSV = Path("paper/code_src/data/ablation_k.csv")
DATASETS = ("Twin-2K-500", "OpinionQA", "EEDI")
K_VALUES = (1, 8, 25, 50, 100, 200)
TRAINED_K_VALUES = (8, 25, 100, 200)
SEEDS = tuple(range(5))
METRICS = {
    "MAE": "test_mae_model_original",
    "Acc": "test_acc_model_mad",
    "HA": "test_acc_model_hard",
    "SA": "test_acc_model_soft",
}


def main_row(dataset: str, seed: int, k: int) -> pd.Series:
    path = MAIN_ROOT / dataset / f"population_{dataset}_seed_{seed}.csv"
    frame = pd.read_csv(path)
    variant = "x_one_llm" if k == 1 else "x_all_llm"
    rows = frame.loc[
        frame["variant"].eq(variant)
        & frame["model_type"].eq("mlp")
        & frame["llm_field"].eq("gpt-4o_norm")
    ]
    if len(rows) != 1:
        raise ValueError(f"{path}: expected one {variant} gpt-4o row, found {len(rows)}")
    if k == 50 and int(rows.iloc[0]["llm_responses_length"]) != 50:
        raise ValueError(f"{path}: main Vector is not K=50")
    return rows.iloc[0]


def trained_row(dataset: str, seed: int, k: int) -> pd.Series:
    path = ROOT / dataset / f"population_k{k}_{dataset}_seed_{seed}.csv"
    frame = pd.read_csv(path)
    if len(frame) != 1:
        raise ValueError(f"{path}: expected one row, found {len(frame)}")
    row = frame.iloc[0]
    uses_bootstrap = dataset in {"Twin-2K-500", "EEDI"} and k > 50
    expected = {
        "variant": "x_all_llm",
        "model_type": "mlp",
        "llm_field": "gpt-4o_norm",
        "llm_input_mode": (
            "population_k_ablation_bootstrap50"
            if uses_bootstrap
            else "population_k_ablation_native"
        ),
        "llm_input_name": (
            f"gpt-4o_K{k}_bootstrap50" if uses_bootstrap else f"gpt-4o_K{k}"
        ),
        "llm_vector_transform": "raw",
    }
    for column, value in expected.items():
        if row[column] != value:
            raise ValueError(f"{path}: expected {column}={value}, found {row[column]}")
    if int(row["llm_responses_length"]) != k:
        raise ValueError(f"{path}: expected K={k}")
    return row


def main() -> int:
    seed_records = []
    for dataset in DATASETS:
        for k in K_VALUES:
            for seed in SEEDS:
                if k in (1, 50):
                    row = main_row(dataset, seed, k)
                    provenance = "current main population table"
                else:
                    row = trained_row(dataset, seed, k)
                    provenance = (
                        "fixed replacement-bootstrap extension from 50 stored draws"
                        if dataset in {"Twin-2K-500", "EEDI"} and k > 50
                        else "current native-draw K-ablation rerun"
                    )
                record = {
                    "dataset": dataset,
                    "k": k,
                    "seed": seed,
                    "provenance": provenance,
                }
                for metric, column in METRICS.items():
                    value = float(row[column]) * 100.0
                    if not np.isfinite(value):
                        raise ValueError(f"Non-finite {dataset}/K={k}/seed={seed}/{metric}")
                    record[metric] = value
                seed_records.append(record)

    seed_frame = pd.DataFrame(seed_records)
    if len(seed_frame) != len(DATASETS) * len(K_VALUES) * len(SEEDS):
        raise ValueError("Unexpected number of seed rows")
    seed_frame.to_csv(ROOT / "population_k_seed_rows.csv", index=False)

    summary_records = []
    for (dataset, k), group in seed_frame.groupby(["dataset", "k"], sort=False):
        if sorted(group["seed"]) != list(SEEDS):
            raise ValueError(f"Seed mismatch for {dataset}/K={k}")
        record = {
            "dataset": dataset,
            "k": int(k),
            "provenance": group["provenance"].iloc[0],
        }
        for metric in METRICS:
            record[f"{metric}_mean"] = float(group[metric].mean())
            record[f"{metric}_sd"] = float(group[metric].std(ddof=1))
        summary_records.append(record)
    summary = pd.DataFrame(summary_records)
    summary.to_csv(ROOT / "population_k_summary.csv", index=False)

    plot_records = []
    for _, row in summary.iterrows():
        record = {
            "dataset": row["dataset"],
            "k": int(row["k"]),
            "source": row["provenance"],
        }
        for metric in METRICS:
            record[metric] = (
                f"{float(row[f'{metric}_mean']):.2f} ± "
                f"{float(row[f'{metric}_sd']):.2f}"
            )
        plot_records.append(record)
    plot_frame = pd.DataFrame(plot_records)[
        ["dataset", "k", "MAE", "Acc", "HA", "SA", "source"]
    ]
    plot_frame.to_csv(ROOT / "ablation_k.csv", index=False)
    plot_frame.to_csv(PAPER_CSV, index=False)

    # K=1 and K=50 must be copied exactly from the current main seed rows.
    for dataset in DATASETS:
        for k in (1, 50):
            for seed in SEEDS:
                copied = seed_frame.loc[
                    seed_frame["dataset"].eq(dataset)
                    & seed_frame["k"].eq(k)
                    & seed_frame["seed"].eq(seed)
                ].iloc[0]
                original = main_row(dataset, seed, k)
                for metric, column in METRICS.items():
                    if not np.isclose(
                        copied[metric], float(original[column]) * 100.0,
                        atol=1e-12, rtol=0.0,
                    ):
                        raise ValueError(f"Main-copy mismatch: {dataset}/K={k}/{metric}")

    print(summary.to_string(index=False))
    print(f"Wrote {ROOT / 'ablation_k.csv'}")
    print(f"Wrote {PAPER_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
