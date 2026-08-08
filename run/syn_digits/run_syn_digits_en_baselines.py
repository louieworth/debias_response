#!/usr/bin/env python3
"""Run mean-supervised SYN-DIGITS Elastic Net baselines.

This runner implements the fit--impute--normalize--transfer Elastic Net path
from Fan et al. (2026) for the repository's existing held-out records.

Individual ``SYN-DIGITS-EN`` uses exactly one canonical GPT-4o response for
each respondent--question pair (K=1).  Elastic Net is fit on the digital-twin
response matrix using the other questions as donors, then transferred to the
human donor matrix.  To adapt the paper's new-question method to this
repository's record-level split, all human test responses are masked before a
single dataset-level donor imputation; only training-side human entries can
therefore affect any transferred prediction.

Population ``SYN-DIGITS-EN-Mean`` uses 50 identity-aligned digital twins,
each queried once per question.  Thus K=50 in this repository's aggregate
query-budget notation, while the cell-level rollout count remains one.  The
Elastic Net is fit across the 50 digital-twin rows and transferred to the one
human-mean donor row.  This remains a mean-only aggregate adaptation rather
than the distributional persona-weighting method in SYN-DIGITS, which uses
full human marginal distributions.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error
from sklearn.utils.extmath import randomized_svd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from debias.evaluate_variants import (  # noqa: E402
    compute_accuracy_hard,
    compute_accuracy_mad,
    compute_accuracy_soft,
)


DATASETS = {
    "Twin-2K-500": {
        "directory": "Twin-2K-500",
        "population_prefix": "twin",
    },
    "OpinionQA": {
        "directory": "OpinionQA",
        "population_prefix": "opinionqa",
    },
    "EEDI": {
        "directory": "EEDI",
        "population_prefix": "eedi",
    },
}
DEFAULT_SEEDS = tuple(range(5))
GPT4O_SOURCE_INDEX = 4
DEFAULT_POPULATION_K = 50
DEFAULT_ALPHA = 0.01
DEFAULT_L1_RATIO = 0.3
DEFAULT_IMPUTATION_RANK = 5
DEFAULT_MIN_COLUMN_STD = 1.0


@dataclass(frozen=True)
class ImputationDiagnostics:
    iterations: int
    converged: bool
    missing_entries: int


@dataclass(frozen=True)
class TransferFit:
    predictions: np.ndarray
    coefficients: np.ndarray
    intercept: float
    train_mse_normalized: float
    synthetic_target_mean: float
    synthetic_target_std: float
    active_coefficients: int


@dataclass(frozen=True)
class IndividualMatrices:
    question_ids: np.ndarray
    twin_ids: np.ndarray
    human_train: np.ndarray
    synthetic: np.ndarray
    test_question_indices: np.ndarray
    test_twin_indices: np.ndarray
    test_truth: np.ndarray
    test_score_ranges: np.ndarray


def hard_impute_svd(
    matrix: np.ndarray,
    *,
    rank: int = DEFAULT_IMPUTATION_RANK,
    max_iter: int = 1_000,
    tol: float = 1e-4,
) -> tuple[np.ndarray, ImputationDiagnostics]:
    """Hard SVD imputation while preserving every observed entry.

    This follows the official SYN-DIGITS iteration.  A deterministic
    truncated randomized SVD is used because ``causaltensor``, the reference
    repository's SVD helper, is not a dependency of this project.
    """
    original = np.asarray(matrix, dtype=float)
    if original.ndim != 2 or min(original.shape) == 0:
        raise ValueError("matrix must be a non-empty two-dimensional array")
    if rank <= 0 or max_iter <= 0 or tol <= 0:
        raise ValueError("rank, max_iter, and tol must be positive")

    missing = np.isnan(original)
    missing_entries = int(missing.sum())
    if missing_entries == 0:
        return original.copy(), ImputationDiagnostics(0, True, 0)

    observed = ~missing
    counts = observed.sum(axis=0)
    sums = np.nansum(original, axis=0)
    column_means = np.divide(
        sums,
        counts,
        out=np.zeros_like(sums, dtype=float),
        where=counts > 0,
    )
    filled = np.where(missing, column_means[None, :], original)
    effective_rank = min(int(rank), min(filled.shape))
    converged = False

    for iteration in range(1, max_iter + 1):
        previous = filled
        u, singular_values, vt = randomized_svd(
            previous,
            n_components=effective_rank,
            n_iter=5,
            random_state=0,
        )
        approximation = (u * singular_values) @ vt
        filled = approximation
        filled[observed] = original[observed]
        denominator = np.linalg.norm(previous, ord="fro")
        change = np.linalg.norm(filled - previous, ord="fro")
        relative_change = change if denominator <= 1e-12 else change / denominator
        if relative_change < tol:
            converged = True
            break

    if not np.all(np.isfinite(filled)):
        raise RuntimeError("SVD imputation produced non-finite values")
    if not np.allclose(filled[observed], original[observed], atol=0.0, rtol=0.0):
        raise RuntimeError("SVD imputation changed observed values")
    return filled, ImputationDiagnostics(iteration, converged, missing_entries)


def _column_normalizer(
    matrix: np.ndarray,
    *,
    min_column_std: float,
) -> tuple[np.ndarray, np.ndarray]:
    means = np.mean(matrix, axis=0)
    standard_deviations = np.std(matrix, axis=0, ddof=0)
    safe = standard_deviations.copy()
    safe[safe < min_column_std] = 1.0
    return means, safe


def elastic_net_fit_transfer(
    synthetic_donors: np.ndarray,
    synthetic_target: np.ndarray,
    human_donors: np.ndarray,
    *,
    alpha: float = DEFAULT_ALPHA,
    l1_ratio: float = DEFAULT_L1_RATIO,
    min_column_std: float = DEFAULT_MIN_COLUMN_STD,
    human_normalization: str = "separate",
) -> TransferFit:
    """Fit Elastic Net on DT rows and transfer it to human donor rows."""
    synthetic_donors = np.asarray(synthetic_donors, dtype=float)
    synthetic_target = np.asarray(synthetic_target, dtype=float)
    human_donors = np.asarray(human_donors, dtype=float)
    if synthetic_donors.ndim != 2 or human_donors.ndim != 2:
        raise ValueError("donor matrices must be two-dimensional")
    if synthetic_donors.shape[1] != human_donors.shape[1]:
        raise ValueError("synthetic and human donor widths must match")
    if synthetic_target.shape != (synthetic_donors.shape[0],):
        raise ValueError("synthetic target length must match DT rows")
    if synthetic_donors.shape[1] == 0:
        raise ValueError("at least one donor question is required")
    if alpha < 0 or not 0 <= l1_ratio <= 1 or min_column_std <= 0:
        raise ValueError("invalid Elastic Net hyperparameters")
    if human_normalization not in {"separate", "synthetic"}:
        raise ValueError("human_normalization must be separate or synthetic")
    if not np.all(np.isfinite(synthetic_donors)):
        raise ValueError("synthetic donors must be finite after imputation")
    if not np.all(np.isfinite(human_donors)):
        raise ValueError("human donors must be finite after imputation")
    if not np.any(np.isfinite(synthetic_target)):
        raise ValueError("synthetic target must contain an observed response")

    synthetic_means, synthetic_stds = _column_normalizer(
        synthetic_donors,
        min_column_std=min_column_std,
    )
    synthetic_normalized = (
        synthetic_donors - synthetic_means[None, :]
    ) / synthetic_stds[None, :]

    if human_normalization == "separate":
        human_means, human_stds = _column_normalizer(
            human_donors,
            min_column_std=min_column_std,
        )
    else:
        human_means, human_stds = synthetic_means, synthetic_stds
    human_normalized = (
        human_donors - human_means[None, :]
    ) / human_stds[None, :]

    target_mean = float(np.nanmean(synthetic_target))
    target_std = float(np.nanstd(synthetic_target, ddof=0))
    if target_std < min_column_std:
        target_std = 1.0
    target_normalized = (synthetic_target - target_mean) / target_std
    target_normalized = np.where(np.isfinite(target_normalized), target_normalized, 0.0)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        model = ElasticNet(
            alpha=max(float(alpha), 1e-12),
            l1_ratio=float(l1_ratio),
            fit_intercept=True,
            max_iter=10_000,
            tol=1e-4,
            selection="cyclic",
        ).fit(synthetic_normalized, target_normalized)

    fitted_target = model.predict(synthetic_normalized)
    prediction_normalized = model.predict(human_normalized)
    predictions = prediction_normalized * target_std + target_mean
    coefficients = np.asarray(model.coef_, dtype=float)
    if not np.all(np.isfinite(predictions)):
        raise RuntimeError("Elastic Net transfer produced non-finite predictions")
    return TransferFit(
        predictions=predictions,
        coefficients=coefficients,
        intercept=float(model.intercept_),
        train_mse_normalized=float(
            np.mean((fitted_target - target_normalized) ** 2)
        ),
        synthetic_target_mean=target_mean,
        synthetic_target_std=target_std,
        active_coefficients=int(np.count_nonzero(np.abs(coefficients) > 1e-10)),
    )


def _dataset_path(dataset: str, level: str, split: str) -> Path:
    if dataset not in DATASETS:
        raise ValueError(f"unknown dataset {dataset!r}")
    info = DATASETS[dataset]
    if level == "individual":
        filename = f"individual_{split}.parquet"
        return PROJECT_ROOT / "dataset" / info["directory"] / level / filename
    if level == "aggreated":
        filename = f"{info['population_prefix']}_{split}.parquet"
        return PROJECT_ROOT / "dataset" / info["directory"] / level / filename
    raise ValueError(f"unknown level {level!r}")


def load_individual_matrices(dataset: str) -> IndividualMatrices:
    columns = [
        "Variable_Name",
        "score_range_min",
        "score_range_max",
        "LLM_Responses",
        "Human_Response",
        "twin_id",
    ]
    train = pd.read_parquet(
        _dataset_path(dataset, "individual", "train"),
        columns=columns,
    )
    test = pd.read_parquet(
        _dataset_path(dataset, "individual", "test"),
        columns=columns,
    )
    train = train.copy()
    test = test.copy()
    train["_split"] = "train"
    test["_split"] = "test"
    combined = pd.concat([train, test], ignore_index=True)
    question_ids = np.asarray(
        sorted(combined["Variable_Name"].astype(str).unique()),
        dtype=object,
    )
    twin_ids = np.asarray(
        sorted(combined["twin_id"].unique(), key=lambda value: str(value)),
        dtype=object,
    )
    question_index = {value: index for index, value in enumerate(question_ids)}
    twin_index = {value: index for index, value in enumerate(twin_ids)}
    shape = (len(twin_ids), len(question_ids))
    human_train_sum = np.zeros(shape, dtype=float)
    human_train_count = np.zeros(shape, dtype=int)
    synthetic = np.full(shape, np.nan, dtype=float)

    for (
        variable_name,
        respondent_id,
        responses_value,
        human_response,
        split_name,
    ) in zip(
        combined["Variable_Name"],
        combined["twin_id"],
        combined["LLM_Responses"],
        combined["Human_Response"],
        combined["_split"],
    ):
        question = str(variable_name)
        row_index = twin_index[respondent_id]
        column_index = question_index[question]
        responses = np.asarray(responses_value, dtype=float)
        if responses.ndim != 1 or len(responses) <= GPT4O_SOURCE_INDEX:
            raise ValueError(
                f"{dataset}/{respondent_id}/{question}: missing GPT-4o coordinate"
            )
        response = float(responses[GPT4O_SOURCE_INDEX])
        if not np.isfinite(response) or not np.isfinite(float(human_response)):
            raise ValueError(f"{dataset}/{respondent_id}/{question}: non-finite response")
        previous_synthetic = synthetic[row_index, column_index]
        if np.isfinite(previous_synthetic) and not np.isclose(
            previous_synthetic,
            response,
            atol=1e-12,
            rtol=0.0,
        ):
            raise ValueError(
                f"{dataset}/{respondent_id}/{question}: conflicting GPT-4o responses"
            )
        synthetic[row_index, column_index] = response
        if split_name == "train":
            human_train_sum[row_index, column_index] += float(human_response)
            human_train_count[row_index, column_index] += 1

    human_train = np.divide(
        human_train_sum,
        human_train_count,
        out=np.full(shape, np.nan, dtype=float),
        where=human_train_count > 0,
    )

    test_question_indices = np.asarray(
        [question_index[str(value)] for value in test["Variable_Name"]],
        dtype=int,
    )
    test_twin_indices = np.asarray(
        [twin_index[value] for value in test["twin_id"]],
        dtype=int,
    )
    test_score_ranges = test[
        ["score_range_min", "score_range_max"]
    ].to_numpy(dtype=float)
    if np.any(test_score_ranges[:, 1] <= test_score_ranges[:, 0]):
        raise ValueError(f"{dataset}: invalid individual score range")
    return IndividualMatrices(
        question_ids=question_ids,
        twin_ids=twin_ids,
        human_train=human_train,
        synthetic=synthetic,
        test_question_indices=test_question_indices,
        test_twin_indices=test_twin_indices,
        test_truth=test["Human_Response"].to_numpy(dtype=float),
        test_score_ranges=test_score_ranges,
    )


def evaluate_predictions(
    truth: np.ndarray,
    prediction: np.ndarray,
    score_ranges: np.ndarray,
) -> dict[str, float]:
    truth = np.asarray(truth, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    score_ranges = np.asarray(score_ranges, dtype=float)
    return {
        "MAE": float(mean_absolute_error(truth, prediction) * 100.0),
        "Acc": float(compute_accuracy_mad(truth, prediction, score_ranges) * 100.0),
        "HA": float(compute_accuracy_hard(truth, prediction, score_ranges) * 100.0),
        "SA": float(compute_accuracy_soft(truth, prediction, score_ranges) * 100.0),
    }


def predict_individual(
    dataset: str,
    *,
    alpha: float,
    l1_ratio: float,
    imputation_rank: int,
    imputation_max_iter: int,
    imputation_tol: float,
    min_column_std: float,
) -> tuple[np.ndarray, pd.DataFrame, IndividualMatrices]:
    matrices = load_individual_matrices(dataset)
    predictions = np.full(len(matrices.test_truth), np.nan, dtype=float)
    diagnostic_rows = []
    target_indices = np.unique(matrices.test_question_indices)

    # The current benchmark holds out respondent--question records rather than
    # whole question columns.  Impute the matrix once after masking every test
    # label, then use the other columns as donors for each target question.
    # This both respects the benchmark's training information and avoids
    # hundreds of redundant SVD runs on nearly identical donor matrices.
    human_imputed, human_diag = hard_impute_svd(
        matrices.human_train,
        rank=imputation_rank,
        max_iter=imputation_max_iter,
        tol=imputation_tol,
    )
    synthetic_imputed, synthetic_diag = hard_impute_svd(
        matrices.synthetic,
        rank=imputation_rank,
        max_iter=imputation_max_iter,
        tol=imputation_tol,
    )

    for target_index in target_indices:
        donors = np.ones(len(matrices.question_ids), dtype=bool)
        donors[target_index] = False
        fit = elastic_net_fit_transfer(
            synthetic_imputed[:, donors],
            matrices.synthetic[:, target_index],
            human_imputed[:, donors],
            alpha=alpha,
            l1_ratio=l1_ratio,
            min_column_std=min_column_std,
            human_normalization="separate",
        )
        test_mask = matrices.test_question_indices == target_index
        predictions[test_mask] = fit.predictions[
            matrices.test_twin_indices[test_mask]
        ]
        diagnostic_rows.append(
            {
                "dataset": dataset,
                "Variable_Name": matrices.question_ids[target_index],
                "respondents": len(matrices.twin_ids),
                "donor_questions": int(donors.sum()),
                "synthetic_target_observed": int(
                    np.isfinite(matrices.synthetic[:, target_index]).sum()
                ),
                "test_records": int(test_mask.sum()),
                "active_coefficients": fit.active_coefficients,
                "intercept": fit.intercept,
                "train_mse_normalized": fit.train_mse_normalized,
                "human_imputation_iterations": human_diag.iterations,
                "human_imputation_converged": human_diag.converged,
                "human_missing_donor_entries": human_diag.missing_entries,
                "synthetic_imputation_iterations": synthetic_diag.iterations,
                "synthetic_imputation_converged": synthetic_diag.converged,
                "synthetic_missing_donor_entries": synthetic_diag.missing_entries,
            }
        )

    if not np.all(np.isfinite(predictions)):
        raise RuntimeError(f"{dataset}: missing individual SYN-DIGITS predictions")
    return predictions, pd.DataFrame(diagnostic_rows), matrices


def load_population_split(
    dataset: str,
    split: str,
    *,
    population_k: int,
) -> pd.DataFrame:
    if population_k <= 0:
        raise ValueError("population_k must be positive")
    columns = [
        "Variable_Name",
        "Average_Human_Response",
        "score_range",
        "gpt-4o",
    ]
    frame = pd.read_parquet(
        _dataset_path(dataset, "aggreated", split),
        columns=columns,
    ).copy()
    population_responses = []
    score_min = []
    score_max = []
    for variable_name, responses_value, score_range_value in zip(
        frame["Variable_Name"],
        frame["gpt-4o"],
        frame["score_range"],
    ):
        responses = np.asarray(responses_value, dtype=float)
        score_range = np.asarray(score_range_value, dtype=float)
        if responses.ndim != 1 or len(responses) < population_k:
            raise ValueError(
                f"{dataset}/{variable_name}: requires {population_k} population "
                f"responses, found {len(responses) if responses.ndim == 1 else 0}"
            )
        if score_range.ndim != 1 or len(score_range) < 2:
            raise ValueError(f"{dataset}/{variable_name}: invalid score range")
        selected = responses[:population_k].copy()
        if not np.all(np.isfinite(selected)):
            raise ValueError(f"{dataset}/{variable_name}: non-finite population response")
        population_responses.append(selected)
        score_min.append(float(score_range[0]))
        score_max.append(float(score_range[-1]))
    frame["population_gpt4o_responses"] = population_responses
    frame["score_min"] = score_min
    frame["score_max"] = score_max
    numeric = frame[
        [
            "Average_Human_Response",
            "score_min",
            "score_max",
        ]
    ].to_numpy(dtype=float)
    if not np.all(np.isfinite(numeric)):
        raise ValueError(f"{dataset}/{split}: non-finite population values")
    return frame


def predict_population_mean(
    dataset: str,
    *,
    population_k: int,
    alpha: float,
    l1_ratio: float,
    min_column_std: float,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    train = load_population_split(
        dataset,
        "train",
        population_k=population_k,
    )
    test = load_population_split(
        dataset,
        "test",
        population_k=population_k,
    )
    # Rows are fixed digital-twin/persona identities and columns are donor
    # questions.  Each persona--question cell contributes exactly one answer.
    synthetic_donors = np.stack(
        train["population_gpt4o_responses"].to_list(),
        axis=1,
    )
    human_donors = train["Average_Human_Response"].to_numpy(dtype=float)[None, :]
    predictions = []
    diagnostic_rows = []
    for row in test.itertuples(index=False):
        synthetic_target = np.asarray(
            row.population_gpt4o_responses,
            dtype=float,
        )
        fit = elastic_net_fit_transfer(
            synthetic_donors,
            synthetic_target,
            human_donors,
            alpha=alpha,
            l1_ratio=l1_ratio,
            min_column_std=min_column_std,
            human_normalization="synthetic",
        )
        prediction = float(fit.predictions[0])
        predictions.append(prediction)
        diagnostic_rows.append(
            {
                "dataset": dataset,
                "Variable_Name": str(row.Variable_Name),
                "digital_twins": population_k,
                "cell_level_rollouts": 1,
                "donor_questions": len(train),
                "active_coefficients": fit.active_coefficients,
                "intercept": fit.intercept,
                "train_mse_normalized": fit.train_mse_normalized,
                "raw_dt_mean": float(np.mean(synthetic_target)),
                "prediction": prediction,
                "prediction_minus_raw_dt_mean": prediction
                - float(np.mean(synthetic_target)),
            }
        )
    prediction_array = np.asarray(predictions, dtype=float)
    if not np.all(np.isfinite(prediction_array)):
        raise RuntimeError(f"{dataset}: non-finite population predictions")
    return prediction_array, pd.DataFrame(diagnostic_rows), test


def significance_stars(p_value: float) -> str:
    if p_value < 0.01:
        return "***"
    if p_value < 0.05:
        return "**"
    if p_value < 0.10:
        return "*"
    return ""


def summarize_seed_rows(seed_rows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (level, dataset, method), group in seed_rows.groupby(
        ["level", "dataset", "method"],
        sort=False,
    ):
        row = {
            "level": level,
            "dataset": dataset,
            "method": method,
            "seeds": len(group),
            "deterministic": bool(group["deterministic"].all()),
            "k": int(group["k"].iloc[0]),
        }
        if group["k"].nunique() != 1:
            raise ValueError(f"{level}/{dataset}/{method}: inconsistent K")
        for metric in ("MAE", "Acc", "HA", "SA"):
            values = group[metric].to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_sd"] = (
                0.0
                if float(np.ptp(values)) <= 1e-12
                else float(np.std(values, ddof=1))
            )
        rows.append(row)
    return pd.DataFrame(rows)


def paired_tests_vs_one(
    candidate_rows: pd.DataFrame,
    *,
    reference_path: Path,
) -> pd.DataFrame:
    reference = pd.read_csv(reference_path)
    rows = []
    for (level, dataset), group in candidate_rows.groupby(
        ["level", "dataset"],
        sort=False,
    ):
        candidate = group.sort_values("seed")
        one = reference.loc[
            reference["dataset"].eq(dataset)
            & reference["variant"].eq("x_one_llm")
        ].sort_values("seed")
        if candidate["seed"].tolist() != one["seed"].astype(int).tolist():
            raise ValueError(f"{level}/{dataset}: SYN-DIGITS/One seed mismatch")
        for metric in ("MAE", "Acc", "HA", "SA"):
            candidate_values = candidate[metric].to_numpy(dtype=float)
            reference_values = one[metric.lower()].to_numpy(dtype=float)
            statistic, p_value = ttest_rel(candidate_values, reference_values)
            difference = float(np.mean(candidate_values - reference_values))
            beneficial = -difference if metric == "MAE" else difference
            direction = "better" if beneficial > 1e-12 else "worse" if beneficial < -1e-12 else "tie"
            rows.append(
                {
                    "level": level,
                    "dataset": dataset,
                    "comparison": f"{candidate['method'].iloc[0]} vs One",
                    "metric": metric,
                    "candidate_mean": float(np.mean(candidate_values)),
                    "one_mean": float(np.mean(reference_values)),
                    "mean_difference_candidate_minus_one": difference,
                    "direction": direction,
                    "t_statistic": float(statistic),
                    "two_sided_p_value": float(p_value),
                    "stars": significance_stars(float(p_value)),
                }
            )
    return pd.DataFrame(rows)


def run(
    *,
    datasets: list[str],
    seeds: list[int],
    alpha: float,
    l1_ratio: float,
    imputation_rank: int,
    imputation_max_iter: int,
    imputation_tol: float,
    min_column_std: float,
    population_k: int,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("seeds must be a non-empty list of unique integers")
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_rows = []
    individual_prediction_rows = []
    population_prediction_rows = []
    individual_diagnostics = []
    population_diagnostics = []

    for dataset in datasets:
        individual_prediction, diagnostics, matrices = predict_individual(
            dataset,
            alpha=alpha,
            l1_ratio=l1_ratio,
            imputation_rank=imputation_rank,
            imputation_max_iter=imputation_max_iter,
            imputation_tol=imputation_tol,
            min_column_std=min_column_std,
        )
        individual_metrics = evaluate_predictions(
            matrices.test_truth,
            individual_prediction,
            matrices.test_score_ranges,
        )
        individual_diagnostics.append(diagnostics)
        for seed in seeds:
            seed_rows.append(
                {
                    "level": "individual",
                    "dataset": dataset,
                    "seed": seed,
                    "method": "SYN-DIGITS-EN",
                    "k": 1,
                    "deterministic": True,
                    **individual_metrics,
                }
            )
            for index, prediction in enumerate(individual_prediction):
                individual_prediction_rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "Variable_Name": matrices.question_ids[
                            matrices.test_question_indices[index]
                        ],
                        "twin_id": matrices.twin_ids[
                            matrices.test_twin_indices[index]
                        ],
                        "human_response": matrices.test_truth[index],
                        "prediction": prediction,
                        "score_min": matrices.test_score_ranges[index, 0],
                        "score_max": matrices.test_score_ranges[index, 1],
                    }
                )

        population_prediction, diagnostics, population_test = predict_population_mean(
            dataset,
            population_k=population_k,
            alpha=alpha,
            l1_ratio=l1_ratio,
            min_column_std=min_column_std,
        )
        population_truth = population_test[
            "Average_Human_Response"
        ].to_numpy(dtype=float)
        population_ranges = population_test[
            ["score_min", "score_max"]
        ].to_numpy(dtype=float)
        population_metrics = evaluate_predictions(
            population_truth,
            population_prediction,
            population_ranges,
        )
        population_diagnostics.append(diagnostics)
        for seed in seeds:
            seed_rows.append(
                {
                    "level": "population",
                    "dataset": dataset,
                    "seed": seed,
                    "method": "SYN-DIGITS-EN-Mean",
                    "k": population_k,
                    "deterministic": True,
                    **population_metrics,
                }
            )
            for index, prediction in enumerate(population_prediction):
                population_prediction_rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "Variable_Name": str(
                            population_test["Variable_Name"].iloc[index]
                        ),
                        "human_mean": population_truth[index],
                        "prediction": prediction,
                        "score_min": population_ranges[index, 0],
                        "score_max": population_ranges[index, 1],
                    }
                )

    seed_frame = pd.DataFrame(seed_rows)
    summary = summarize_seed_rows(seed_frame)
    individual_seed_frame = seed_frame.loc[
        seed_frame["level"].eq("individual")
    ].copy()
    population_seed_frame = seed_frame.loc[
        seed_frame["level"].eq("population")
    ].copy()
    individual_tests = paired_tests_vs_one(
        individual_seed_frame,
        reference_path=(
            PROJECT_ROOT
            / "results"
            / "individual_test_tuned_a1e-6_d08_fulltrain_seed0_4"
            / "individual_seed_rows.csv"
        ),
    )
    population_tests = paired_tests_vs_one(
        population_seed_frame,
        reference_path=(
            PROJECT_ROOT
            / "results"
            / "population_one_significance_seed0_4_precision17"
            / "population_seed_rows.csv"
        ),
    )

    seed_frame.to_csv(output_dir / "per_seed.csv", index=False, float_format="%.17g")
    summary.to_csv(output_dir / "summary.csv", index=False, float_format="%.17g")
    pd.DataFrame(individual_prediction_rows).to_csv(
        output_dir / "individual_predictions.csv",
        index=False,
        float_format="%.17g",
    )
    pd.DataFrame(population_prediction_rows).to_csv(
        output_dir / "population_predictions.csv",
        index=False,
        float_format="%.17g",
    )
    pd.concat(individual_diagnostics, ignore_index=True).to_csv(
        output_dir / "individual_question_diagnostics.csv",
        index=False,
        float_format="%.17g",
    )
    pd.concat(population_diagnostics, ignore_index=True).to_csv(
        output_dir / "population_question_diagnostics.csv",
        index=False,
        float_format="%.17g",
    )
    individual_tests.to_csv(
        output_dir / "individual_pvalues_vs_one.csv",
        index=False,
        float_format="%.17g",
    )
    population_tests.to_csv(
        output_dir / "population_pvalues_vs_one.csv",
        index=False,
        float_format="%.17g",
    )
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
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--l1-ratio", type=float, default=DEFAULT_L1_RATIO)
    parser.add_argument(
        "--imputation-rank",
        type=int,
        default=DEFAULT_IMPUTATION_RANK,
    )
    parser.add_argument("--imputation-max-iter", type=int, default=1_000)
    parser.add_argument("--imputation-tol", type=float, default=1e-4)
    parser.add_argument(
        "--min-column-std",
        type=float,
        default=DEFAULT_MIN_COLUMN_STD,
    )
    parser.add_argument(
        "--population-k",
        type=int,
        default=DEFAULT_POPULATION_K,
        help=(
            "Number of identity-aligned digital twins used per population "
            "question; each twin contributes one response"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            PROJECT_ROOT
            / "results"
            / "syn_digits_en_indk1_popk50_seed0_4"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _, summary = run(
        datasets=args.datasets,
        seeds=args.seeds,
        alpha=args.alpha,
        l1_ratio=args.l1_ratio,
        imputation_rank=args.imputation_rank,
        imputation_max_iter=args.imputation_max_iter,
        imputation_tol=args.imputation_tol,
        min_column_std=args.min_column_std,
        population_k=args.population_k,
        output_dir=args.output_dir,
    )
    print(summary.to_string(index=False))
    print(f"Wrote SYN-DIGITS results to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
