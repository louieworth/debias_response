#!/usr/bin/env python3
"""Summarize the corrected K=50 OpinionQA population rerun."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from scipy.stats import ttest_rel


ROOT = Path(
    "results/population_one_significance_k50fix_seed0_4_precision17/OpinionQA"
)
SEEDS = tuple(range(5))
METHODS = {
    "x_only": "w/o LLM",
    "x_one_llm": "One",
    "one_logprob": "One Logprob",
    "x_avg_llm": "Mean",
    "x_all_llm": "Vector",
}
QUERY_K = {
    "Base LLM": 50,
    "w/o LLM": 0,
    "One": 1,
    "One Logprob": 1,
    "Mean": 50,
    "Vector": 50,
}
METRICS = {
    "MAE": "test_mae_model_original",
    "Acc": "test_acc_model_mad",
    "HA": "test_acc_model_hard",
    "SA": "test_acc_model_soft",
}
BASE_METRICS = {
    "MAE": "test_mae_base_original",
    "Acc": "test_acc_base_mad",
    "HA": "test_acc_base_hard",
    "SA": "test_acc_base_soft",
}


def significance_stars(p_value: float) -> str:
    if p_value < 0.01:
        return "***"
    if p_value < 0.05:
        return "**"
    if p_value < 0.10:
        return "*"
    return ""


def main() -> int:
    frames = []
    for seed in SEEDS:
        path = ROOT / f"population_OpinionQA_seed_{seed}.csv"
        frame = pd.read_csv(path)
        if len(frame) != len(METHODS) or set(frame["variant"]) != set(METHODS):
            raise ValueError(f"{path}: incomplete or unexpected method rows")
        frame["seed"] = seed
        frames.append(frame)
    runs = pd.concat(frames, ignore_index=True)

    for variant in ("x_avg_llm", "x_all_llm"):
        lengths = runs.loc[runs["variant"].eq(variant), "llm_responses_length"]
        if not lengths.eq(50).all():
            raise ValueError(f"{variant}: expected recorded response budget K=50")

    seed_rows = []
    for _, row in runs.iterrows():
        record = {
            "seed": int(row["seed"]),
            "method": METHODS[row["variant"]],
            "k": QUERY_K[METHODS[row["variant"]]],
        }
        for metric, column in METRICS.items():
            record[metric] = float(row[column]) * 100.0
        seed_rows.append(record)

    vector_rows = runs.loc[runs["variant"].eq("x_all_llm")].sort_values("seed")
    for _, row in vector_rows.iterrows():
        record = {
            "seed": int(row["seed"]),
            "method": "Base LLM",
            "k": QUERY_K["Base LLM"],
        }
        for metric, column in BASE_METRICS.items():
            record[metric] = float(row[column]) * 100.0
        seed_rows.append(record)

    seed_frame = pd.DataFrame(seed_rows).sort_values(["method", "seed"])
    seed_frame.to_csv(ROOT / "seed_metrics.csv", index=False, float_format="%.17g")

    summary_rows = []
    for method, group in seed_frame.groupby("method", sort=False):
        record = {"method": method, "k": QUERY_K[method], "seeds": len(group)}
        for metric in METRICS:
            record[f"{metric}_mean"] = group[metric].mean()
            record[f"{metric}_sd"] = group[metric].std(ddof=1)
        summary_rows.append(record)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(ROOT / "summary.csv", index=False, float_format="%.17g")

    one = seed_frame.loc[seed_frame["method"].eq("One")].sort_values("seed")
    test_rows = []
    for method in ("One Logprob", "Mean", "Vector"):
        candidate = seed_frame.loc[seed_frame["method"].eq(method)].sort_values("seed")
        if candidate["seed"].tolist() != one["seed"].tolist():
            raise ValueError(f"{method}: seed pairing mismatch")
        for metric in METRICS:
            statistic, p_value = ttest_rel(candidate[metric], one[metric])
            test_rows.append(
                {
                    "comparison": f"{method} vs One",
                    "metric": metric,
                    "mean_difference": candidate[metric].mean() - one[metric].mean(),
                    "t_statistic": statistic,
                    "two_sided_p_value": p_value,
                    "stars": significance_stars(p_value),
                }
            )
    tests = pd.DataFrame(test_rows)
    tests.to_csv(ROOT / "pvalues_vs_one.csv", index=False, float_format="%.17g")
    print(summary.to_string(index=False))
    print()
    print(tests.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
