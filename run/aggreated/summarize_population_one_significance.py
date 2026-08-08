#!/usr/bin/env python3
"""Summarize the seeds 0--4 population rerun and test methods against One."""

import os
from pathlib import Path

import pandas as pd
from scipy.stats import ttest_rel


ROOT = Path(
    os.environ.get(
        "POPULATION_RESULT_ROOT",
        "results/population_one_significance_seed0_4_precision17",
    )
)
DATASETS = ["Twin-2K-500", "OpinionQA", "EEDI"]
SEEDS = list(range(5))
VARIANT_TO_METHOD = {
    "x_only": "w/o LLM",
    "x_one_llm": "One",
    "one_logprob": "One Logprob",
    "x_avg_llm": "Mean",
    "x_all_llm": "Vector",
}
METRICS = {
    "mae": "test_mae_model_original",
    "acc": "test_acc_model_mad",
    "ha": "test_acc_model_hard",
    "sa": "test_acc_model_soft",
}
BASE_METRICS = {
    "mae": "test_mae_base_original",
    "acc": "test_acc_base_mad",
    "ha": "test_acc_base_hard",
    "sa": "test_acc_base_soft",
}


def stars(p_value: float) -> str:
    if p_value < 0.01:
        return "***"
    if p_value < 0.05:
        return "**"
    if p_value < 0.10:
        return "*"
    return ""


seed_rows = []
base_rows = []
for dataset in DATASETS:
    for seed in SEEDS:
        path = ROOT / dataset / f"population_{dataset}_seed_{seed}.csv"
        frame = pd.read_csv(path)
        if len(frame) != len(VARIANT_TO_METHOD):
            raise ValueError(f"{path}: expected 5 rows, found {len(frame)}")
        if set(frame["variant"]) != set(VARIANT_TO_METHOD):
            raise ValueError(f"{path}: unexpected variants {set(frame['variant'])}")
        if frame["variant"].duplicated().any():
            raise ValueError(f"{path}: duplicate variants")

        for _, row in frame.iterrows():
            record = {
                "dataset": dataset,
                "seed": seed,
                "variant": row["variant"],
                "method": VARIANT_TO_METHOD[row["variant"]],
            }
            for metric, column in METRICS.items():
                record[metric] = float(row[column]) * 100.0
            seed_rows.append(record)

        base_source = frame.loc[frame["variant"] == "x_all_llm"].iloc[0]
        base_record = {"dataset": dataset, "seed": seed}
        for metric, column in BASE_METRICS.items():
            base_record[metric] = float(base_source[column]) * 100.0
        base_rows.append(base_record)

seed_frame = pd.DataFrame(seed_rows)
base_frame = pd.DataFrame(base_rows)
seed_frame.to_csv(ROOT / "population_seed_rows.csv", index=False)

summary_rows = []
for dataset in DATASETS:
    base = base_frame.loc[base_frame["dataset"] == dataset]
    for metric in METRICS:
        if base[metric].max() - base[metric].min() > 1e-10:
            raise ValueError(f"{dataset}/{metric}: Base LLM varies across seeds")
        row = {
            "dataset": dataset,
            "metric": metric,
            "base_llm": base[metric].iloc[0],
        }
        for variant, method in VARIANT_TO_METHOD.items():
            values = seed_frame.loc[
                (seed_frame["dataset"] == dataset)
                & (seed_frame["variant"] == variant),
                metric,
            ]
            row[f"{variant}_mean"] = values.mean()
            row[f"{variant}_std"] = values.std(ddof=1)
        summary_rows.append(row)

summary = pd.DataFrame(summary_rows)
summary.to_csv(ROOT / "population_5seed_summary.csv", index=False)

test_rows = []
for dataset in DATASETS:
    reference = seed_frame.loc[
        (seed_frame["dataset"] == dataset)
        & (seed_frame["variant"] == "x_one_llm")
    ].sort_values("seed")
    for variant in ["one_logprob", "x_avg_llm", "x_all_llm"]:
        candidate = seed_frame.loc[
            (seed_frame["dataset"] == dataset)
            & (seed_frame["variant"] == variant)
        ].sort_values("seed")
        if candidate["seed"].tolist() != reference["seed"].tolist():
            raise ValueError(f"{dataset}/{variant}: seed pairing mismatch")
        for metric in METRICS:
            statistic, p_value = ttest_rel(candidate[metric], reference[metric])
            test_rows.append(
                {
                    "dataset": dataset,
                    "comparison": f"{VARIANT_TO_METHOD[variant]} vs One",
                    "metric": metric,
                    "mean_difference": candidate[metric].mean()
                    - reference[metric].mean(),
                    "t_statistic": statistic,
                    "two_sided_p_value": p_value,
                    "stars": stars(p_value),
                }
            )

tests = pd.DataFrame(test_rows)
tests.to_csv(ROOT / "population_paired_vs_one.csv", index=False)
print(summary.to_string(index=False))
print()
print(tests.to_string(index=False))
