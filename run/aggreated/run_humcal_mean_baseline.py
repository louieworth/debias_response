#!/usr/bin/env python3
"""Run the mean-supervised HuMCal population baseline.

For each dataset, the baseline learns one global credibility vector over the
first K persona-aligned LLM responses:

    min_w mean_q (Y_q w - human_mean_q)^2
    subject to w >= 0 and sum(w) = 1.

The learned vector is frozen and evaluated on the existing held-out question
split.  The optimization problem and inputs are deterministic.  We still
materialize seeds 0--4 so the result has the same seed-indexed shape as the
population MLP experiments; seed is metadata rather than a source of random
variation for this convex baseline.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import ttest_rel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from debias.evaluate_variants import (  # noqa: E402
    compute_accuracy_hard,
    compute_accuracy_mad,
    compute_accuracy_soft,
)


DATASETS = {
    "Twin-2K-500": ("Twin-2K-500", "twin"),
    "OpinionQA": ("OpinionQA", "opinionqa"),
    "EEDI": ("EEDI", "eedi"),
}
DEFAULT_SEEDS = tuple(range(5))


@dataclass(frozen=True)
class PopulationSplit:
    variable_names: np.ndarray
    llm: np.ndarray
    human_norm: np.ndarray
    human_original: np.ndarray
    score_ranges: np.ndarray


def load_population_split(
    dataset: str,
    split: str,
    *,
    llm_field: str = "gpt-4o_norm",
    k: int = 50,
) -> PopulationSplit:
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset {dataset!r}")
    if split not in {"train", "test"}:
        raise ValueError(f"split must be train or test, got {split!r}")
    if k <= 0:
        raise ValueError("k must be positive")

    directory, prefix = DATASETS[dataset]
    path = PROJECT_ROOT / "dataset" / directory / "aggreated" / f"{prefix}_{split}.parquet"
    columns = [
        "Variable_Name",
        "Average_Human_Response",
        "Average_Human_Response_norm",
        "score_range",
        llm_field,
    ]
    frame = pd.read_parquet(path, columns=columns)

    response_rows = []
    for variable_name, response in zip(frame["Variable_Name"], frame[llm_field]):
        values = np.asarray(response, dtype=float)
        if values.ndim != 1 or len(values) < k:
            raise ValueError(
                f"{path}: {variable_name} has {len(values)} responses in {llm_field}; "
                f"expected at least K={k}"
            )
        response_rows.append(values[:k])

    llm = np.vstack(response_rows)
    human_norm = frame["Average_Human_Response_norm"].to_numpy(dtype=float)
    human_original = frame["Average_Human_Response"].to_numpy(dtype=float)
    score_ranges = np.vstack(
        [np.asarray(value, dtype=float)[[0, -1]] for value in frame["score_range"]]
    )

    arrays = (llm, human_norm, human_original, score_ranges)
    if any(not np.all(np.isfinite(array)) for array in arrays):
        raise ValueError(f"{path}: non-finite population values")
    if np.any(score_ranges[:, 1] <= score_ranges[:, 0]):
        raise ValueError(f"{path}: invalid score range")

    return PopulationSplit(
        variable_names=frame["Variable_Name"].astype(str).to_numpy(),
        llm=llm,
        human_norm=human_norm,
        human_original=human_original,
        score_ranges=score_ranges,
    )


def fit_humcal_mean(
    llm: np.ndarray,
    human_norm: np.ndarray,
    *,
    ftol: float = 1e-12,
    maxiter: int = 10_000,
) -> tuple[np.ndarray, dict[str, float | int | bool | str]]:
    """Fit a global simplex-constrained least-squares ensemble."""
    llm = np.asarray(llm, dtype=float)
    human_norm = np.asarray(human_norm, dtype=float)
    if llm.ndim != 2:
        raise ValueError("llm must be a two-dimensional question-by-persona matrix")
    if human_norm.shape != (len(llm),):
        raise ValueError("human_norm length must match the number of LLM rows")
    if len(llm) == 0 or llm.shape[1] == 0:
        raise ValueError("llm must be non-empty")
    if not np.all(np.isfinite(llm)) or not np.all(np.isfinite(human_norm)):
        raise ValueError("fit inputs must be finite")

    n_questions, k = llm.shape
    uniform = np.full(k, 1.0 / k, dtype=float)

    def objective(weights: np.ndarray) -> float:
        residual = llm @ weights - human_norm
        return float(np.mean(residual * residual))

    def gradient(weights: np.ndarray) -> np.ndarray:
        residual = llm @ weights - human_norm
        return 2.0 * (llm.T @ residual) / n_questions

    result = minimize(
        objective,
        uniform,
        jac=gradient,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * k,
        constraints=[
            {
                "type": "eq",
                "fun": lambda weights: float(np.sum(weights) - 1.0),
                "jac": lambda weights: np.ones_like(weights),
            }
        ],
        options={"ftol": ftol, "maxiter": maxiter, "disp": False},
    )
    if not result.success:
        raise RuntimeError(f"HuMCal optimization failed: {result.message}")

    # SLSQP already respects the bounds.  Remove only floating-point residue
    # before persisting the simplex vector.
    weights = np.clip(np.asarray(result.x, dtype=float), 0.0, None)
    weights /= weights.sum()
    fitted_objective = objective(weights)
    uniform_objective = objective(uniform)
    if fitted_objective > uniform_objective + 1e-10:
        raise RuntimeError("HuMCal objective is worse than its uniform initialization")
    if not np.isclose(weights.sum(), 1.0, atol=1e-10, rtol=0.0):
        raise RuntimeError("HuMCal weights do not sum to one")

    diagnostics: dict[str, float | int | bool | str] = {
        "success": bool(result.success),
        "message": str(result.message),
        "iterations": int(result.nit),
        "train_mse_norm": fitted_objective,
        "uniform_train_mse_norm": uniform_objective,
        "train_mse_reduction_pct": (
            (uniform_objective - fitted_objective) / uniform_objective * 100.0
            if uniform_objective > 0.0
            else 0.0
        ),
        "active_weights": int(np.count_nonzero(weights > 1e-8)),
        "min_weight": float(weights.min()),
        "max_weight": float(weights.max()),
    }
    return weights, diagnostics


def normalized_to_original(
    prediction_norm: np.ndarray, score_ranges: np.ndarray
) -> np.ndarray:
    prediction_norm = np.asarray(prediction_norm, dtype=float)
    score_ranges = np.asarray(score_ranges, dtype=float)
    return score_ranges[:, 0] + prediction_norm * (
        score_ranges[:, 1] - score_ranges[:, 0]
    )


def evaluate(
    split: PopulationSplit,
    prediction_norm: np.ndarray,
    uniform_norm: np.ndarray,
) -> dict[str, float]:
    prediction_norm = np.asarray(prediction_norm, dtype=float)
    uniform_norm = np.asarray(uniform_norm, dtype=float)
    prediction = normalized_to_original(prediction_norm, split.score_ranges)
    uniform = normalized_to_original(uniform_norm, split.score_ranges)
    truth = split.human_original

    return {
        "MSE_norm": float(mean_squared_error(split.human_norm, prediction_norm)),
        "MAE_norm": float(mean_absolute_error(split.human_norm, prediction_norm)),
        "R2_norm": float(r2_score(split.human_norm, prediction_norm)),
        "MSE_original": float(mean_squared_error(truth, prediction)),
        "MAE_original": float(mean_absolute_error(truth, prediction)),
        "MAE": float(mean_absolute_error(truth, prediction) * 100.0),
        "Acc": float(compute_accuracy_mad(truth, prediction, split.score_ranges) * 100.0),
        "HA": float(compute_accuracy_hard(truth, prediction, split.score_ranges) * 100.0),
        "SA": float(compute_accuracy_soft(truth, prediction, split.score_ranges) * 100.0),
        "Uniform_MAE": float(mean_absolute_error(truth, uniform) * 100.0),
        "Uniform_Acc": float(compute_accuracy_mad(truth, uniform, split.score_ranges) * 100.0),
        "Uniform_HA": float(compute_accuracy_hard(truth, uniform, split.score_ranges) * 100.0),
        "Uniform_SA": float(compute_accuracy_soft(truth, uniform, split.score_ranges) * 100.0),
    }


def build_comparison(
    summary: pd.DataFrame,
    *,
    main_summary_path: Path,
    pvalues: pd.DataFrame | None = None,
) -> str:
    lines = [
        "# HuMCal-Mean Population Baseline (Seeds 0--4)",
        "",
        "HuMCal-Mean learns one global nonnegative K=50 persona-weight vector "
        "whose entries sum to one, minimizing normalized train MSE. The vector "
        "is frozen on the held-out question split.",
        "",
        "The optimization is convex and deterministic for the fixed split. Seeds "
        "0--4 are materialized for table compatibility and produce identical "
        "predictions; the reported HuMCal standard deviation is therefore zero.",
        "",
        "| Dataset | Method | MAE ↓ | Acc ↑ (%) | HA ↑ (%) | SA ↑ (%) |",
        "|---|---|---:|---:|---:|---:|",
    ]

    current = None
    if main_summary_path.exists():
        current = pd.read_csv(main_summary_path)

    for dataset in DATASETS:
        row = summary.loc[summary["dataset"].eq(dataset)].iloc[0]
        humcal_cells = {}
        for metric in ("MAE", "Acc", "HA", "SA"):
            stars = ""
            if pvalues is not None:
                pvalue_row = pvalues.loc[
                    pvalues["dataset"].eq(dataset)
                    & pvalues["metric"].eq(metric)
                ]
                if len(pvalue_row) == 1:
                    stars = str(pvalue_row.iloc[0]["stars"])
            humcal_cells[metric] = (
                f"{row[f'{metric}_mean']:.2f} ± {row[f'{metric}_sd']:.2f}"
                f"{stars}"
            )
        lines.append(
            f"| {dataset} | HuMCal-Mean | "
            f"{humcal_cells['MAE']} | "
            f"{humcal_cells['Acc']} | "
            f"{humcal_cells['HA']} | "
            f"{humcal_cells['SA']} |"
        )
        if current is not None:
            for label, prefix in (("Mean", "x_avg_llm"), ("Vector", "x_all_llm")):
                values = {}
                for metric, metric_key in (("MAE", "mae"), ("Acc", "acc"), ("HA", "ha"), ("SA", "sa")):
                    metric_row = current.loc[
                        current["dataset"].eq(dataset)
                        & current["metric"].eq(metric_key)
                    ].iloc[0]
                    values[metric] = (
                        float(metric_row[f"{prefix}_mean"]),
                        float(metric_row[f"{prefix}_std"]),
                    )
                lines.append(
                    f"|  | {label} | "
                    f"{values['MAE'][0]:.2f} ± {values['MAE'][1]:.2f} | "
                    f"{values['Acc'][0]:.2f} ± {values['Acc'][1]:.2f} | "
                    f"{values['HA'][0]:.2f} ± {values['HA'][1]:.2f} | "
                    f"{values['SA'][0]:.2f} ± {values['SA'][1]:.2f} |"
                )

    lines.extend(
        [
            "",
            "Notes:",
            "",
            "- `MAE` follows the existing population table convention: original-scale MAE multiplied by 100.",
            "- `Acc` is MAD-based accuracy; `HA` and `SA` are the existing hard and soft accuracy metrics.",
            "- No human label from a test question is used to fit the global weights.",
            "- Mean and Vector rows are copied from the current five-seed population summary when available.",
            "- Stars are unadjusted paired two-sided tests against One over the matched seeds: "
            "`*` p<.10, `**` p<.05, `***` p<.01. They indicate a difference, not necessarily an improvement.",
            "- Because HuMCal-Mean is deterministic, these tests reflect variation in the five One fits only.",
            "",
        ]
    )
    return "\n".join(lines)


def significance_stars(p_value: float) -> str:
    if p_value < 0.01:
        return "***"
    if p_value < 0.05:
        return "**"
    if p_value < 0.10:
        return "*"
    return ""


def compute_pvalues_vs_one(
    humcal_seed_frame: pd.DataFrame,
    *,
    one_seed_path: Path,
) -> pd.DataFrame:
    """Paired two-sided seed tests following the population-table convention."""
    if not one_seed_path.exists():
        raise FileNotFoundError(f"Missing One seed results: {one_seed_path}")
    one_frame = pd.read_csv(one_seed_path)
    metric_columns = {
        "MAE": ("test_MAE", "mae", False),
        "Acc": ("test_Acc", "acc", True),
        "HA": ("test_HA", "ha", True),
        "SA": ("test_SA", "sa", True),
    }
    records = []
    for dataset in DATASETS:
        candidate = humcal_seed_frame.loc[
            humcal_seed_frame["dataset"].eq(dataset)
        ].sort_values("seed")
        reference = one_frame.loc[
            one_frame["dataset"].eq(dataset)
            & one_frame["variant"].eq("x_one_llm")
        ].sort_values("seed")
        if candidate["seed"].tolist() != reference["seed"].tolist():
            raise ValueError(f"{dataset}: HuMCal/One seed pairing mismatch")

        for metric, (candidate_column, reference_column, higher_is_better) in metric_columns.items():
            candidate_values = candidate[candidate_column].to_numpy(dtype=float)
            reference_values = reference[reference_column].to_numpy(dtype=float)
            statistic, p_value = ttest_rel(candidate_values, reference_values)
            difference = float(np.mean(candidate_values - reference_values))
            beneficial_difference = difference if higher_is_better else -difference
            direction = (
                "better"
                if beneficial_difference > 1e-12
                else "worse"
                if beneficial_difference < -1e-12
                else "tie"
            )
            records.append(
                {
                    "dataset": dataset,
                    "comparison": "HuMCal-Mean vs One",
                    "metric": metric,
                    "humcal_mean": float(np.mean(candidate_values)),
                    "one_mean": float(np.mean(reference_values)),
                    "mean_difference_humcal_minus_one": difference,
                    "direction": direction,
                    "t_statistic": float(statistic),
                    "two_sided_p_value": float(p_value),
                    "stars": significance_stars(float(p_value)),
                }
            )
    return pd.DataFrame(records)


def run(
    *,
    datasets: list[str],
    seeds: list[int],
    llm_field: str,
    k: int,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(seeds) == 0 or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be a non-empty list of unique integers")

    seed_rows = []
    prediction_rows = []
    weight_rows = []

    for dataset in datasets:
        train = load_population_split(dataset, "train", llm_field=llm_field, k=k)
        test = load_population_split(dataset, "test", llm_field=llm_field, k=k)

        reference_predictions = None
        reference_weights = None
        for seed in seeds:
            weights, diagnostics = fit_humcal_mean(train.llm, train.human_norm)
            train_prediction_norm = train.llm @ weights
            test_prediction_norm = test.llm @ weights
            train_uniform_norm = train.llm.mean(axis=1)
            test_uniform_norm = test.llm.mean(axis=1)
            train_metrics = evaluate(train, train_prediction_norm, train_uniform_norm)
            test_metrics = evaluate(test, test_prediction_norm, test_uniform_norm)

            if reference_predictions is None:
                reference_predictions = test_prediction_norm.copy()
                reference_weights = weights.copy()
            else:
                if not np.allclose(test_prediction_norm, reference_predictions, atol=1e-10, rtol=0.0):
                    raise RuntimeError(f"{dataset}: deterministic predictions vary across seeds")
                if not np.allclose(weights, reference_weights, atol=1e-10, rtol=0.0):
                    raise RuntimeError(f"{dataset}: deterministic weights vary across seeds")

            record = {
                "dataset": dataset,
                "seed": seed,
                "method": "HuMCal-Mean",
                "llm_field": llm_field,
                "k": k,
                "deterministic": True,
                "train_questions": len(train.llm),
                "test_questions": len(test.llm),
                **diagnostics,
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{f"test_{key}": value for key, value in test_metrics.items()},
            }
            seed_rows.append(record)

            prediction_original = normalized_to_original(
                test_prediction_norm, test.score_ranges
            )
            uniform_original = normalized_to_original(
                test_uniform_norm, test.score_ranges
            )
            for index, variable_name in enumerate(test.variable_names):
                prediction_rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "Variable_Name": variable_name,
                        "human_mean": test.human_original[index],
                        "prediction": prediction_original[index],
                        "uniform_prediction": uniform_original[index],
                        "score_min": test.score_ranges[index, 0],
                        "score_max": test.score_ranges[index, 1],
                    }
                )
            for persona_index, weight in enumerate(weights):
                weight_rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "persona_index": persona_index,
                        "weight": weight,
                    }
                )

    seed_frame = pd.DataFrame(seed_rows)
    prediction_frame = pd.DataFrame(prediction_rows)
    weight_frame = pd.DataFrame(weight_rows)

    summary_rows = []
    for dataset, group in seed_frame.groupby("dataset", sort=False):
        if sorted(group["seed"].tolist()) != sorted(seeds):
            raise RuntimeError(f"{dataset}: output seed mismatch")
        row = {
            "dataset": dataset,
            "method": "HuMCal-Mean",
            "seeds": len(group),
            "deterministic": bool(group["deterministic"].all()),
            "train_questions": int(group["train_questions"].iloc[0]),
            "test_questions": int(group["test_questions"].iloc[0]),
            "train_mse_norm": float(group["train_mse_norm"].mean()),
            "uniform_train_mse_norm": float(group["uniform_train_mse_norm"].mean()),
            "train_mse_reduction_pct": float(group["train_mse_reduction_pct"].mean()),
            "active_weights": int(group["active_weights"].iloc[0]),
        }
        for metric in ("MAE", "Acc", "HA", "SA"):
            values = group[f"test_{metric}"].to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_sd"] = (
                0.0
                if float(np.ptp(values)) <= 1e-12
                else float(np.std(values, ddof=1))
            )
        for metric in ("MAE", "Acc", "HA", "SA"):
            row[f"Uniform_{metric}"] = float(group[f"test_Uniform_{metric}"].iloc[0])
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    seed_frame.to_csv(output_dir / "per_seed.csv", index=False, float_format="%.17g")
    summary.to_csv(output_dir / "summary.csv", index=False, float_format="%.17g")
    prediction_frame.to_csv(
        output_dir / "predictions.csv", index=False, float_format="%.17g"
    )
    weight_frame.to_csv(output_dir / "weights.csv", index=False, float_format="%.17g")

    main_root = (
        PROJECT_ROOT
        / "results"
        / "population_one_significance_seed0_4_precision17"
    )
    pvalues = compute_pvalues_vs_one(
        seed_frame,
        one_seed_path=main_root / "population_seed_rows.csv",
    )
    pvalues.to_csv(
        output_dir / "pvalues_vs_one.csv", index=False, float_format="%.17g"
    )

    comparison = build_comparison(
        summary,
        main_summary_path=main_root / "population_5seed_summary.csv",
        pvalues=pvalues,
    )
    (output_dir / "comparison.md").write_text(comparison, encoding="utf-8")
    return seed_frame, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=list(DATASETS),
        default=list(DATASETS),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--llm-field", default="gpt-4o_norm")
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "humcal_mean_population_seed0_4",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _, summary = run(
        datasets=args.datasets,
        seeds=args.seeds,
        llm_field=args.llm_field,
        k=args.k,
        output_dir=args.output_dir,
    )
    print(summary.to_string(index=False))
    print(f"Wrote HuMCal-Mean results to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
