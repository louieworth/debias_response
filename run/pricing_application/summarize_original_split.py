#!/usr/bin/env python3
"""Managerial metrics on the paper's original Twin record-level split."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

from run_pricing_application import (
    PROJECT_ROOT,
    _metric_row,
    pricing_catalog,
    response_to_purchase_probability,
)


TRAIN_PATH = PROJECT_ROOT / "dataset/Twin-2K-500/individual/individual_train.parquet"
TEST_PATH = PROJECT_ROOT / "dataset/Twin-2K-500/individual/individual_test.parquet"
PREDICTION_ROOT = (
    PROJECT_ROOT
    / "results/pricing_application/original_split/unit_predictions/Twin-2K-500"
)
OUTPUT_ROOT = PROJECT_ROOT / "results/pricing_application/original_split"
VARIANTS = ("x_only", "x_one_llm", "x_avg_llm", "x_all_llm")
LABELS = {
    "base_llm": "Base LLM",
    "question_train_share_prior": "Question train-share prior",
    "x_only": "w/o LLM",
    "x_one_llm": "One",
    "x_avg_llm": "Mean",
    "x_all_llm": "Vector",
}


def prediction_path(seed: int, variant: str) -> Path:
    return PREDICTION_ROOT / f"seed{seed}_{variant}.csv"


def load_prediction_units(seed: int, variant: str) -> pd.DataFrame:
    path = prediction_path(seed, variant)
    if not path.exists():
        raise FileNotFoundError(path)
    prediction = pd.read_csv(path)
    pricing = prediction[
        prediction["question_id"].astype(str).str.startswith("QID9_")
    ].copy()
    if len(pricing) != 1362 or pricing["question_id"].nunique() != 40:
        raise ValueError(
            f"unexpected pricing predictions in {path}: "
            f"{len(pricing)} rows, {pricing['question_id'].nunique()} questions"
        )
    return pricing


def learned_method_units(seed: int, variant: str) -> pd.DataFrame:
    prediction = load_prediction_units(seed, variant)
    predicted_norm = prediction["y_pred"].to_numpy(dtype=float) - 1.0
    return pd.DataFrame(
        {
            "seed": seed,
            "variant": variant,
            "method": LABELS[variant],
            "question_id": prediction["question_id"].astype(str),
            "respondent_id": prediction["respondent_id"],
            "human_purchase": 2.0 - prediction["y_true"].to_numpy(dtype=float),
            "predicted_purchase_probability": response_to_purchase_probability(
                predicted_norm
            ),
        }
    )


def base_llm_units(seed: int) -> pd.DataFrame:
    prediction = load_prediction_units(seed, "x_all_llm")
    baseline_norm = prediction["y_baseline"].to_numpy(dtype=float) - 1.0
    return pd.DataFrame(
        {
            "seed": seed,
            "variant": "base_llm",
            "method": LABELS["base_llm"],
            "question_id": prediction["question_id"].astype(str),
            "respondent_id": prediction["respondent_id"],
            "human_purchase": 2.0 - prediction["y_true"].to_numpy(dtype=float),
            "predicted_purchase_probability": response_to_purchase_probability(
                baseline_norm
            ),
        }
    )


def question_train_share_prior_units(seed: int) -> pd.DataFrame:
    train = pd.read_parquet(
        TRAIN_PATH,
        columns=["Variable_Name", "Human_Response_norm"],
    )
    train = train[train["Variable_Name"].astype(str).str.startswith("QID9_")]
    train["human_purchase"] = response_to_purchase_probability(
        train["Human_Response_norm"].to_numpy()
    )
    priors = train.groupby("Variable_Name")["human_purchase"].mean()
    if len(priors) != 40:
        raise ValueError("expected training-side purchase shares for 40 questions")

    reference = load_prediction_units(seed, "x_all_llm")
    predicted = reference["question_id"].astype(str).map(priors)
    if predicted.isna().any():
        raise ValueError("a test pricing question has no training-side human share")
    return pd.DataFrame(
        {
            "seed": seed,
            "variant": "question_train_share_prior",
            "method": LABELS["question_train_share_prior"],
            "question_id": reference["question_id"].astype(str),
            "respondent_id": reference["respondent_id"],
            "human_purchase": 2.0 - reference["y_true"].to_numpy(dtype=float),
            "predicted_purchase_probability": predicted.to_numpy(dtype=float),
        }
    )


def paired_vector_tests(question_predictions: pd.DataFrame) -> pd.DataFrame:
    averaged = (
        question_predictions.groupby(
            ["variant", "method", "question_id"], as_index=False
        )
        .agg(
            human_purchase_share=("human_purchase_share", "first"),
            predicted_purchase_share=("predicted_purchase_share", "mean"),
            human_revenue=("human_revenue", "first"),
            predicted_revenue=("predicted_revenue", "mean"),
        )
    )
    averaged["share_abs_error"] = np.abs(
        averaged["predicted_purchase_share"]
        - averaged["human_purchase_share"]
    )
    averaged["revenue_abs_error"] = np.abs(
        averaged["predicted_revenue"] - averaged["human_revenue"]
    )
    vector = averaged[averaged["variant"] == "x_all_llm"].set_index("question_id")
    rows = []
    for comparator in LABELS:
        if comparator == "x_all_llm":
            continue
        other = averaged[averaged["variant"] == comparator].set_index("question_id")
        other = other.loc[vector.index]
        for metric in ("share_abs_error", "revenue_abs_error"):
            test = ttest_rel(
                vector[metric], other[metric], alternative="less"
            )
            rows.append(
                {
                    "comparison": f"Vector vs {LABELS[comparator]}",
                    "metric": metric,
                    "questions": len(vector),
                    "vector_mean_error": vector[metric].mean(),
                    "comparator_mean_error": other[metric].mean(),
                    "vector_advantage": other[metric].mean()
                    - vector[metric].mean(),
                    "paired_t": test.statistic,
                    "one_sided_p_value": test.pvalue,
                }
            )
    return pd.DataFrame(rows)


def summarize(seeds: tuple[int, ...]) -> pd.DataFrame:
    train = pd.read_parquet(TRAIN_PATH)
    test = pd.read_parquet(TEST_PATH)
    catalog = pricing_catalog(pd.concat([train, test], ignore_index=True))[
        ["question_id", "price"]
    ]

    units = []
    for seed in seeds:
        units.append(base_llm_units(seed))
        units.append(question_train_share_prior_units(seed))
        for variant in VARIANTS:
            units.append(learned_method_units(seed, variant))
    unit_predictions = pd.concat(units, ignore_index=True)
    expected = len(seeds) * len(LABELS) * 1362
    if len(unit_predictions) != expected:
        raise ValueError(f"unexpected unit count: {len(unit_predictions)} != {expected}")
    if unit_predictions.duplicated(
        ["seed", "variant", "question_id", "respondent_id"]
    ).any():
        raise ValueError("duplicate unit predictions")

    question_predictions = (
        unit_predictions.groupby(
            ["seed", "variant", "method", "question_id"], as_index=False
        )
        .agg(
            human_purchase_share=("human_purchase", "mean"),
            predicted_purchase_share=("predicted_purchase_probability", "mean"),
            respondents=("respondent_id", "nunique"),
        )
        .merge(catalog, on="question_id", validate="many_to_one")
    )
    question_predictions["human_revenue"] = (
        question_predictions["price"]
        * question_predictions["human_purchase_share"]
    )
    question_predictions["predicted_revenue"] = (
        question_predictions["price"]
        * question_predictions["predicted_purchase_share"]
    )

    rows = []
    for (seed, variant, method), questions in question_predictions.groupby(
        ["seed", "variant", "method"], sort=False
    ):
        method_units = unit_predictions[
            (unit_predictions["seed"] == seed)
            & (unit_predictions["variant"] == variant)
        ]
        rows.append(
            {
                "seed": seed,
                "variant": variant,
                "method": method,
                **_metric_row(questions, method_units),
            }
        )
    seed_metrics = pd.DataFrame(rows)
    metrics = [
        column
        for column in seed_metrics.columns
        if column not in {"seed", "variant", "method"}
    ]
    summary_rows = []
    for (variant, method), group in seed_metrics.groupby(
        ["variant", "method"], sort=False
    ):
        row = {"variant": variant, "method": method, "seeds": len(group)}
        for metric in metrics:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_std"] = group[metric].std(ddof=1)
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    order = {label: index for index, label in enumerate(LABELS.values())}
    summary["_order"] = summary["method"].map(order)
    summary = summary.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    tests = paired_vector_tests(question_predictions)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    unit_predictions.to_csv(
        OUTPUT_ROOT / "oof_unit_predictions.csv", index=False, float_format="%.17g"
    )
    question_predictions.to_csv(
        OUTPUT_ROOT / "oof_question_predictions.csv",
        index=False,
        float_format="%.17g",
    )
    seed_metrics.to_csv(
        OUTPUT_ROOT / "seed_metrics.csv", index=False, float_format="%.17g"
    )
    summary.to_csv(OUTPUT_ROOT / "summary.csv", index=False, float_format="%.17g")
    tests.to_csv(
        OUTPUT_ROOT / "paired_question_tests.csv",
        index=False,
        float_format="%.17g",
    )

    display_columns = [
        "method",
        "share_mae_pp_mean",
        "share_mae_pp_std",
        "share_spearman_r_mean",
        "share_spearman_r_std",
        "individual_brier_mean",
        "individual_brier_std",
        "revenue_mae_mean",
        "revenue_mae_std",
        "revenue_wape_pct_mean",
        "revenue_wape_pct_std",
        "revenue_spearman_r_mean",
        "revenue_spearman_r_std",
        "top5_revenue_regret_pct_mean",
        "top5_revenue_regret_pct_std",
    ]
    display = summary[display_columns]
    truth = question_predictions.drop_duplicates("question_id")[
        "human_purchase_share"
    ]
    markdown = [
        "# Twin-2K-500 pricing application (original record split)",
        "",
        (
            "All 40 pricing questions occur in both train and test. Each test "
            "offer has 25--41 held-out respondents, while its other 126--142 "
            "human responses are available during training."
        ),
        "",
        (
            f"Across test respondents, product-level human purchase shares have "
            f"mean {100.0 * truth.mean():.2f}%, standard deviation "
            f"{100.0 * truth.std(ddof=1):.2f} pp, and range "
            f"{100.0 * truth.min():.2f}%--{100.0 * truth.max():.2f}%."
        ),
        "",
        display.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Paired question-level tests",
        "",
        tests.to_markdown(index=False, floatfmt=".6g"),
        "",
    ]
    (OUTPUT_ROOT / "summary.md").write_text("\n".join(markdown), encoding="utf-8")
    print(display.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nPaired question-level tests (one-sided: Vector has lower error):")
    print(tests.to_string(index=False, float_format=lambda value: f"{value:.6g}"))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = parser.parse_args()
    summarize(tuple(args.seeds))


if __name__ == "__main__":
    main()
