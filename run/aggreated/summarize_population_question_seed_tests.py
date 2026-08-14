#!/usr/bin/env python3
"""Recompute population tests from paired question-by-seed predictions.

The primary test first averages each paired method-minus-One metric difference
over the five seeds and then performs a two-sided one-sample t test across held-
out questions.  The output also includes the old seed-only test, a deliberately
naive pooled question-by-seed test, and a two-way cluster-robust sensitivity
test that clusters by both question and seed.
"""

from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from debias.evaluate_variants import compute_accuracy_soft


PREDICTION_ROOT = (
    PROJECT_ROOT / "results" / "population_question_predictions_seed0_4_precision17"
)
PAPER_ROOT = (
    PROJECT_ROOT
    / "results"
    / "population_one_significance_k50fix_seed0_4_precision17"
)
HUMCAL_ROOT = PROJECT_ROOT / "results" / "humcal_mean_population_seed0_4"
SYN_DIGITS_ROOT = (
    PROJECT_ROOT / "results" / "syn_digits_en_indk1_popk50_seed0_4"
)

DATASETS = ("Twin-2K-500", "OpinionQA", "EEDI")
SEEDS = tuple(range(5))
VARIANT_TO_METHOD = {
    "x_only": "w/o LLM",
    "x_one_llm": "One",
    "one_logprob": "One Logprob",
    "x_avg_llm": "Mean",
    "x_all_llm": "Vector",
}
METHOD_ORDER = (
    "Base LLM",
    "w/o LLM",
    "One Logprob",
    "Mean",
    "Vector",
    "HuMCal-Mean",
    "SYN-DIGITS-EN-Mean",
)
METRICS = ("MAE", "Acc", "HA", "SA")
RESULT_COLUMNS = {
    "MAE": "test_mae_model_original",
    "Acc": "test_acc_model_mad",
    "HA": "test_acc_model_hard",
    "SA": "test_acc_model_soft",
}


def significance_stars(p_value: float) -> str:
    if not np.isfinite(p_value):
        return ""
    if p_value < 0.01:
        return "***"
    if p_value < 0.05:
        return "**"
    if p_value < 0.10:
        return "*"
    return ""


def metric_contributions(frame: pd.DataFrame) -> pd.DataFrame:
    truth = frame["y_true"].to_numpy(dtype=float)
    prediction = frame["y_pred"].to_numpy(dtype=float)
    score_min = frame["score_min"].to_numpy(dtype=float)
    score_max = frame["score_max"].to_numpy(dtype=float)
    widths = score_max - score_min
    absolute_error = np.abs(prediction - truth)

    acc = np.where(
        np.isclose(widths, 0.0),
        np.isclose(prediction, truth).astype(float),
        1.0 - absolute_error / widths,
    )
    rounded_truth = np.clip(np.rint(truth), score_min, score_max)
    rounded_prediction = np.clip(np.rint(prediction), score_min, score_max)
    hard = (rounded_truth == rounded_prediction).astype(float)
    soft = np.asarray(
        [
            compute_accuracy_soft(
                np.asarray([truth[index]], dtype=float),
                np.asarray([prediction[index]], dtype=float),
                np.asarray([[score_min[index], score_max[index]]], dtype=float),
            )
            for index in range(len(frame))
        ],
        dtype=float,
    )

    result = frame[
        ["dataset", "seed", "question_id", "method"]
    ].copy()
    result["MAE"] = absolute_error * 100.0
    result["Acc"] = acc * 100.0
    result["HA"] = hard * 100.0
    result["SA"] = soft * 100.0
    return result


def standardize_prediction_frame(
    frame: pd.DataFrame,
    *,
    dataset: str,
    method: str,
) -> pd.DataFrame:
    renamed = frame.rename(
        columns={
            "Variable_Name": "question_id",
            "human_mean": "y_true",
            "prediction": "y_pred",
        }
    ).copy()
    renamed["dataset"] = dataset
    renamed["method"] = method
    required = {
        "dataset",
        "seed",
        "question_id",
        "method",
        "y_true",
        "y_pred",
        "score_min",
        "score_max",
    }
    missing = required - set(renamed.columns)
    if missing:
        raise ValueError(f"{dataset}/{method}: missing prediction columns {sorted(missing)}")
    return renamed[list(required)].copy()


def load_predictions() -> pd.DataFrame:
    frames = []
    for dataset in DATASETS:
        for seed in SEEDS:
            for variant, method in VARIANT_TO_METHOD.items():
                path = (
                    PREDICTION_ROOT
                    / dataset
                    / f"{variant}_seed{seed}_predictions.csv"
                )
                frame = pd.read_csv(path)
                frame["dataset"] = dataset
                frame["method"] = method
                frames.append(
                    frame[
                        [
                            "dataset",
                            "seed",
                            "question_id",
                            "method",
                            "y_true",
                            "y_pred",
                            "score_min",
                            "score_max",
                        ]
                    ]
                )
                if variant == "x_all_llm":
                    baseline = frame.copy()
                    baseline["method"] = "Base LLM"
                    baseline["y_pred"] = baseline["y_baseline"]
                    frames.append(
                        baseline[
                            [
                                "dataset",
                                "seed",
                                "question_id",
                                "method",
                                "y_true",
                                "y_pred",
                                "score_min",
                                "score_max",
                            ]
                        ]
                    )

    humcal = pd.read_csv(HUMCAL_ROOT / "predictions.csv")
    for dataset in DATASETS:
        frames.append(
            standardize_prediction_frame(
                humcal.loc[humcal["dataset"].eq(dataset)],
                dataset=dataset,
                method="HuMCal-Mean",
            )
        )

    syn_digits = pd.read_csv(SYN_DIGITS_ROOT / "population_predictions.csv")
    for dataset in DATASETS:
        frames.append(
            standardize_prediction_frame(
                syn_digits.loc[syn_digits["dataset"].eq(dataset)],
                dataset=dataset,
                method="SYN-DIGITS-EN-Mean",
            )
        )

    combined = pd.concat(frames, ignore_index=True)
    combined["seed"] = combined["seed"].astype(int)
    combined["question_id"] = combined["question_id"].astype(str)
    key = ["dataset", "seed", "question_id", "method"]
    if combined.duplicated(key).any():
        duplicates = combined.loc[combined.duplicated(key, keep=False), key]
        raise ValueError(f"duplicate prediction keys:\n{duplicates.head()}")
    return combined


def load_current_tests() -> pd.DataFrame:
    records = []

    core = pd.read_csv(PAPER_ROOT / "population_paired_vs_one.csv")
    for row in core.itertuples(index=False):
        records.append(
            {
                "dataset": row.dataset,
                "method": row.comparison.removesuffix(" vs One"),
                "metric": {
                    "mae": "MAE",
                    "acc": "Acc",
                    "ha": "HA",
                    "sa": "SA",
                }[str(row.metric).lower()],
                "current_seed_p_value": float(row.two_sided_p_value),
                "current_seed_stars": str(row.stars) if pd.notna(row.stars) else "",
            }
        )

    humcal = pd.read_csv(HUMCAL_ROOT / "pvalues_vs_one.csv")
    for row in humcal.itertuples(index=False):
        records.append(
            {
                "dataset": row.dataset,
                "method": "HuMCal-Mean",
                "metric": str(row.metric),
                "current_seed_p_value": float(row.two_sided_p_value),
                "current_seed_stars": str(row.stars) if pd.notna(row.stars) else "",
            }
        )

    syn_digits = pd.read_csv(SYN_DIGITS_ROOT / "population_pvalues_vs_one.csv")
    for row in syn_digits.itertuples(index=False):
        records.append(
            {
                "dataset": row.dataset,
                "method": "SYN-DIGITS-EN-Mean",
                "metric": str(row.metric),
                "current_seed_p_value": float(row.two_sided_p_value),
                "current_seed_stars": str(row.stars) if pd.notna(row.stars) else "",
            }
        )
    return pd.DataFrame(records)


def cluster_variance(values: np.ndarray, groups: np.ndarray) -> float:
    residuals = values - np.mean(values)
    unique_groups, inverse = np.unique(groups, return_inverse=True)
    group_sums = np.bincount(inverse, weights=residuals)
    group_count = len(unique_groups)
    if group_count < 2:
        return math.nan
    correction = group_count / (group_count - 1.0)
    return float(correction * np.sum(group_sums**2) / len(values) ** 2)


def two_way_cluster_test(cell_differences: pd.DataFrame) -> tuple[float, float, int]:
    values = cell_differences["difference"].to_numpy(dtype=float)
    seeds = cell_differences["seed"].to_numpy()
    questions = cell_differences["question_id"].to_numpy()
    intersections = np.asarray(
        [f"{seed}\x1f{question}" for seed, question in zip(seeds, questions)]
    )
    variance = (
        cluster_variance(values, seeds)
        + cluster_variance(values, questions)
        - cluster_variance(values, intersections)
    )
    degrees_freedom = min(len(np.unique(seeds)) - 1, len(np.unique(questions)) - 1)
    if not np.isfinite(variance) or variance <= 0.0:
        return math.nan, math.nan, degrees_freedom
    statistic = float(np.mean(values) / math.sqrt(variance))
    p_value = float(2.0 * stats.t.sf(abs(statistic), df=degrees_freedom))
    return statistic, p_value, degrees_freedom


def compute_tests(contributions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset in DATASETS:
        dataset_frame = contributions.loc[contributions["dataset"].eq(dataset)]
        one = dataset_frame.loc[dataset_frame["method"].eq("One")]
        one = one.set_index(["seed", "question_id"])
        expected_seeds = sorted(one.index.get_level_values("seed").unique())
        expected_questions = sorted(one.index.get_level_values("question_id").unique())

        for method in METHOD_ORDER:
            candidate = dataset_frame.loc[dataset_frame["method"].eq(method)]
            candidate = candidate.set_index(["seed", "question_id"])
            if not candidate.index.equals(one.index):
                candidate = candidate.reindex(one.index)
            if candidate[list(METRICS)].isna().any().any():
                raise ValueError(f"{dataset}/{method}: prediction pairing mismatch")

            for metric in METRICS:
                difference = candidate[metric] - one[metric]
                cells = difference.rename("difference").reset_index()
                question_differences = cells.groupby("question_id")["difference"].mean()

                question_test = stats.ttest_1samp(question_differences, popmean=0.0)
                pooled_test = stats.ttest_1samp(difference.to_numpy(dtype=float), popmean=0.0)
                cluster_t, cluster_p, cluster_df = two_way_cluster_test(cells)

                rows.append(
                    {
                        "dataset": dataset,
                        "comparison": f"{method} vs One",
                        "method": method,
                        "metric": metric,
                        "seeds": len(expected_seeds),
                        "questions": len(expected_questions),
                        "seed_question_cells": len(difference),
                        "mean_difference_candidate_minus_one": float(difference.mean()),
                        "question_averaged_t": float(question_test.statistic),
                        "question_averaged_df": int(len(question_differences) - 1),
                        "question_averaged_p_value": float(question_test.pvalue),
                        "question_averaged_stars": significance_stars(
                            float(question_test.pvalue)
                        ),
                        "naive_pooled_t": float(pooled_test.statistic),
                        "naive_pooled_df": int(len(difference) - 1),
                        "naive_pooled_p_value": float(pooled_test.pvalue),
                        "naive_pooled_stars": significance_stars(float(pooled_test.pvalue)),
                        "two_way_cluster_t": cluster_t,
                        "two_way_cluster_df": cluster_df,
                        "two_way_cluster_p_value": cluster_p,
                        "two_way_cluster_stars": significance_stars(cluster_p),
                    }
                )
    return pd.DataFrame(rows)


def reproduction_audit(contributions: pd.DataFrame) -> pd.DataFrame:
    records = []
    for dataset in DATASETS:
        for seed in SEEDS:
            old_path = PAPER_ROOT / dataset / f"population_{dataset}_seed_{seed}.csv"
            old = pd.read_csv(old_path).set_index("variant")
            for variant, method in VARIANT_TO_METHOD.items():
                new_path = (
                    PREDICTION_ROOT / dataset / f"{variant}_seed{seed}_result.csv"
                )
                new = pd.read_csv(new_path).iloc[0]
                subset = contributions.loc[
                    contributions["dataset"].eq(dataset)
                    & contributions["seed"].eq(seed)
                    & contributions["method"].eq(method)
                ]
                for metric, result_column in RESULT_COLUMNS.items():
                    prediction_metric = float(subset[metric].mean())
                    rerun_metric = float(new[result_column]) * 100.0
                    paper_metric = float(old.loc[variant, result_column]) * 100.0
                    records.append(
                        {
                            "dataset": dataset,
                            "seed": seed,
                            "method": method,
                            "metric": metric,
                            "prediction_metric": prediction_metric,
                            "rerun_result_metric": rerun_metric,
                            "paper_result_metric": paper_metric,
                            "prediction_minus_rerun": prediction_metric - rerun_metric,
                            "rerun_minus_paper": rerun_metric - paper_metric,
                        }
                    )
    return pd.DataFrame(records)


def main() -> None:
    predictions = load_predictions()
    contributions = metric_contributions(predictions)
    tests = compute_tests(contributions)
    current = load_current_tests()
    tests = tests.merge(
        current,
        on=["dataset", "method", "metric"],
        how="left",
        validate="one_to_one",
    )
    tests["stars_match_current"] = np.where(
        tests["current_seed_p_value"].notna(),
        tests["question_averaged_stars"].eq(tests["current_seed_stars"]),
        pd.NA,
    )
    dataset_order = {dataset: index for index, dataset in enumerate(DATASETS)}
    method_order = {method: index for index, method in enumerate(METHOD_ORDER)}
    metric_order = {metric: index for index, metric in enumerate(METRICS)}
    tests = tests.sort_values(
        ["dataset", "method", "metric"],
        key=lambda column: column.map(
            dataset_order
            if column.name == "dataset"
            else method_order
            if column.name == "method"
            else metric_order
        ),
    )

    audit = reproduction_audit(contributions)
    PREDICTION_ROOT.mkdir(parents=True, exist_ok=True)
    tests.to_csv(
        PREDICTION_ROOT / "question_seed_pvalues_vs_one.csv",
        index=False,
        float_format="%.17g",
    )
    audit.to_csv(
        PREDICTION_ROOT / "reproduction_audit.csv",
        index=False,
        float_format="%.17g",
    )

    max_prediction_error = float(audit["prediction_minus_rerun"].abs().max())
    max_reproduction_error = float(audit["rerun_minus_paper"].abs().max())
    print(f"Maximum prediction/result metric difference: {max_prediction_error:.3g}")
    print(f"Maximum rerun/paper metric difference: {max_reproduction_error:.3g}")
    print()
    comparable = tests.loc[tests["current_seed_p_value"].notna()]
    changed = comparable.loc[~comparable["stars_match_current"].astype(bool)]
    print(
        f"Question-averaged stars match {len(comparable) - len(changed)}/"
        f"{len(comparable)} current starred comparisons."
    )
    if len(changed):
        print(
            changed[
                [
                    "dataset",
                    "comparison",
                    "metric",
                    "current_seed_p_value",
                    "current_seed_stars",
                    "question_averaged_p_value",
                    "question_averaged_stars",
                    "two_way_cluster_p_value",
                    "two_way_cluster_stars",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
