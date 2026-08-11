#!/usr/bin/env python3
"""Compare Base LLM, One, Vector, and SYN-DIGITS on held-out Twin questions."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from run.syn_digits.run_syn_digits_en_baselines import (  # noqa: E402
    DEFAULT_ALPHA,
    DEFAULT_L1_RATIO,
    DEFAULT_MIN_COLUMN_STD,
    elastic_net_fit_transfer,
    evaluate_predictions,
    load_individual_matrices,
)


SPLIT_ROOT = (
    PROJECT_ROOT
    / "dataset/Twin-2K-500/individual/syn_digits_question_holdout"
)
RAW_ROOT = PROJECT_ROOT / "results/syn_digits/question_holdout/raw"
OUTPUT_ROOT = PROJECT_ROOT / "results/syn_digits/question_holdout"
DATASET = "Twin-2K-500"
METHOD_ORDER = ["Base LLM", "One", "Vector", "SYN-DIGITS-EN"]
METRICS = ("MAE", "Acc", "HA", "SA")


def metric_row_from_result(row: pd.Series, *, method: str, baseline: bool) -> dict:
    prefix = "base" if baseline else "model"
    return {
        "method": method,
        "MAE": 100.0 * float(row[f"test_mae_{prefix}_original"]),
        "Acc": 100.0 * float(row[f"test_acc_{prefix}_mad"]),
        "HA": 100.0 * float(row[f"test_acc_{prefix}_hard"]),
        "SA": 100.0 * float(row[f"test_acc_{prefix}_soft"]),
    }


def predict_syn_digits() -> tuple[np.ndarray, pd.DataFrame, object]:
    matrices = load_individual_matrices(
        DATASET,
        train_path=SPLIT_ROOT / "individual_train.parquet",
        test_path=SPLIT_ROOT / "individual_test.parquet",
    )
    observed = np.isfinite(matrices.human_train)
    donor_columns = observed.all(axis=0)
    heldout_columns = ~observed.any(axis=0)
    target_indices = np.unique(matrices.test_question_indices)
    if donor_columns.sum() != 48 or heldout_columns.sum() != 12:
        raise ValueError(
            "question holdout must contain 48 complete donor columns and "
            "12 fully masked target columns"
        )
    if set(np.flatnonzero(heldout_columns)) != set(target_indices.tolist()):
        raise ValueError("test targets do not match the fully held-out questions")
    if not np.all(np.isfinite(matrices.synthetic)):
        raise ValueError("synthetic response matrix must be complete")

    predictions = np.full(len(matrices.test_truth), np.nan, dtype=float)
    diagnostics = []
    for target_index in target_indices:
        fit = elastic_net_fit_transfer(
            matrices.synthetic[:, donor_columns],
            matrices.synthetic[:, target_index],
            matrices.human_train[:, donor_columns],
            alpha=DEFAULT_ALPHA,
            l1_ratio=DEFAULT_L1_RATIO,
            min_column_std=DEFAULT_MIN_COLUMN_STD,
            human_normalization="separate",
        )
        target_mask = matrices.test_question_indices == target_index
        predictions[target_mask] = fit.predictions[
            matrices.test_twin_indices[target_mask]
        ]
        diagnostics.append(
            {
                "Variable_Name": matrices.question_ids[target_index],
                "respondents": int(target_mask.sum()),
                "donor_questions": int(donor_columns.sum()),
                "active_coefficients": fit.active_coefficients,
                "train_mse_normalized": fit.train_mse_normalized,
                "synthetic_target_mean": fit.synthetic_target_mean,
                "synthetic_target_std": fit.synthetic_target_std,
            }
        )
    if not np.all(np.isfinite(predictions)):
        raise RuntimeError("SYN-DIGITS produced missing predictions")
    return predictions, pd.DataFrame(diagnostics), matrices


def significance_stars(p_value: float) -> str:
    if p_value < 0.01:
        return "***"
    if p_value < 0.05:
        return "**"
    if p_value < 0.10:
        return "*"
    return ""


def summarize(seeds: tuple[int, ...]) -> tuple[pd.DataFrame, pd.DataFrame]:
    syn_prediction, diagnostics, matrices = predict_syn_digits()
    base_prediction = matrices.synthetic[
        matrices.test_twin_indices,
        matrices.test_question_indices,
    ]
    base_metrics = evaluate_predictions(
        matrices.test_truth,
        base_prediction,
        matrices.test_score_ranges,
    )
    syn_metrics = evaluate_predictions(
        matrices.test_truth,
        syn_prediction,
        matrices.test_score_ranges,
    )
    rows = []
    for seed in seeds:
        one_path = RAW_ROOT / f"seed{seed}_x_one_llm.csv"
        vector_path = RAW_ROOT / f"seed{seed}_x_all_llm.csv"
        if not one_path.exists() or not vector_path.exists():
            raise FileNotFoundError(f"missing trained result for seed {seed}")
        one = pd.read_csv(one_path).iloc[0]
        vector = pd.read_csv(vector_path).iloc[0]
        rows.extend(
            [
                {"seed": seed, "method": "Base LLM", **base_metrics},
                {
                    "seed": seed,
                    **metric_row_from_result(one, method="One", baseline=False),
                },
                {
                    "seed": seed,
                    **metric_row_from_result(vector, method="Vector", baseline=False),
                },
                {"seed": seed, "method": "SYN-DIGITS-EN", **syn_metrics},
            ]
        )
    seed_metrics = pd.DataFrame(rows)

    summary_rows = []
    for method in METHOD_ORDER:
        group = seed_metrics[seed_metrics["method"] == method]
        row = {"method": method, "seeds": len(group)}
        for metric in METRICS:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_std"] = group[metric].std(ddof=1)
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)

    one = seed_metrics[seed_metrics["method"] == "One"].sort_values("seed")
    test_rows = []
    for method in ("Base LLM", "Vector", "SYN-DIGITS-EN"):
        candidate = seed_metrics[seed_metrics["method"] == method].sort_values("seed")
        if candidate["seed"].tolist() != one["seed"].tolist():
            raise ValueError(f"seed mismatch for {method}")
        for metric in METRICS:
            statistic, p_value = ttest_rel(candidate[metric], one[metric])
            test_rows.append(
                {
                    "comparison": f"{method} vs One",
                    "metric": metric,
                    "candidate_mean": candidate[metric].mean(),
                    "one_mean": one[metric].mean(),
                    "paired_t": statistic,
                    "two_sided_p_value": p_value,
                    "stars": significance_stars(float(p_value)),
                }
            )
    tests = pd.DataFrame(test_rows)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    seed_metrics.to_csv(OUTPUT_ROOT / "seed_metrics.csv", index=False)
    summary.to_csv(OUTPUT_ROOT / "summary.csv", index=False)
    tests.to_csv(OUTPUT_ROOT / "paired_tests_vs_one.csv", index=False)
    diagnostics.to_csv(OUTPUT_ROOT / "syn_digits_diagnostics.csv", index=False)
    pd.DataFrame(
        {
            "Variable_Name": matrices.question_ids[matrices.test_question_indices],
            "twin_id": matrices.twin_ids[matrices.test_twin_indices],
            "human_response": matrices.test_truth,
            "prediction": syn_prediction,
            "score_min": matrices.test_score_ranges[:, 0],
            "score_max": matrices.test_score_ranges[:, 1],
        }
    ).to_csv(OUTPUT_ROOT / "syn_digits_predictions.csv", index=False)

    markdown = [
        "# Twin-2K-500 complete-question holdout comparison",
        "",
        "Train: 48 complete questions x 167 respondents = 8,016 records.  ",
        "Test: 12 complete questions x 167 respondents = 2,004 records.",
        "",
        summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Paired two-sided tests versus One",
        "",
        tests.to_markdown(index=False, floatfmt=".6g"),
        "",
    ]
    (OUTPUT_ROOT / "summary.md").write_text("\n".join(markdown), encoding="utf-8")
    return summary, tests


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    args = parser.parse_args()
    summary, tests = summarize(tuple(args.seeds))
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nPaired two-sided tests versus One:")
    print(tests.to_string(index=False, float_format=lambda value: f"{value:.6g}"))


if __name__ == "__main__":
    main()
