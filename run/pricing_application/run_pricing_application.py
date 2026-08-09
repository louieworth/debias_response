#!/usr/bin/env python3
"""Whole-product holdout experiment for the Twin-2K-500 pricing items.

The 40 QID9 items are distinct product-price offers, each answered by every
respondent.  This script evaluates whether debiased synthetic respondents can
forecast human purchase incidence and gross revenue at those *observed* prices.
It deliberately does not estimate a demand curve, price elasticity, WTP, or an
optimal price because each product is observed at only one price.

The workflow has three stages:

1. ``prepare`` creates five deterministic whole-question folds.  Each test fold
   contains eight pricing questions and all 167 respondents.  Its training set
   contains every other question, so no human response to a held-out product is
   available during fitting.
2. ``run`` invokes the paper's individual-level MLP for One, Mean, and Vector
   over model seeds 0--4.  GPU workers execute sequential jobs on each assigned
   device to avoid multiple large MLPs contending for one GPU.
3. ``summarize`` pools out-of-fold predictions into purchase-share, observed-
   price gross-revenue, calibration, and offer-ranking metrics.

Running the script without a subcommand executes all three stages.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, ttest_rel


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_FILES = (
    PROJECT_ROOT / "dataset/Twin-2K-500/individual/individual_train.parquet",
    PROJECT_ROOT / "dataset/Twin-2K-500/individual/individual_test.parquet",
)
FOLD_ROOT = PROJECT_ROOT / "dataset/Twin-2K-500/pricing_application"
RESULT_ROOT = PROJECT_ROOT / "results/pricing_application"
LOG_ROOT = PROJECT_ROOT / "logs/pricing_application"

PRICING_PREFIX = "QID9_"
N_FOLDS = 5
FOLD_SEED = 20260808
DEFAULT_SEEDS = (0, 1, 2, 3, 4)
VARIANTS = ("x_only", "x_one_llm", "x_avg_llm", "x_all_llm")
METHOD_LABELS = {
    "base_llm": "Base LLM",
    "train_share_prior": "Train-share prior",
    "x_only": "w/o LLM",
    "x_one_llm": "One",
    "x_avg_llm": "Mean",
    "x_all_llm": "Vector",
}

PRICE_PATTERN = re.compile(r"priced at:\s*\$([0-9,]+(?:\.\d+)?)", re.IGNORECASE)


@dataclass(frozen=True)
class Job:
    fold: int
    seed: int
    variant: str

    @property
    def stem(self) -> str:
        return f"fold{self.fold}_seed{self.seed}_{self.variant}"

    @property
    def result_relative(self) -> Path:
        return Path("pricing_application/raw") / f"{self.stem}_metrics.csv"

    @property
    def prediction_relative(self) -> Path:
        return Path("pricing_application/predictions") / f"{self.stem}.csv"

    @property
    def result_path(self) -> Path:
        return PROJECT_ROOT / "results" / self.result_relative

    @property
    def prediction_path(self) -> Path:
        return PROJECT_ROOT / "results" / self.prediction_relative

    @property
    def log_path(self) -> Path:
        return LOG_ROOT / f"{self.stem}.log"


def _question_number(question_id: str) -> int:
    match = re.fullmatch(r"QID9_(\d+)", str(question_id))
    if not match:
        raise ValueError(f"invalid pricing question id: {question_id!r}")
    return int(match.group(1))


def extract_price(question: str) -> float:
    """Extract the single observed offer price from one pricing prompt."""
    matches = PRICE_PATTERN.findall(str(question))
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one observed price, found {len(matches)} in {question!r}"
        )
    price = float(matches[0].replace(",", ""))
    if not math.isfinite(price) or price < 0:
        raise ValueError(f"invalid observed price: {price}")
    return price


def response_to_purchase_probability(values: np.ndarray) -> np.ndarray:
    """Map Twin's normalized response (0=Yes, 1=No) to P(purchase)."""
    response = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(response)):
        raise ValueError("response scores must be finite")
    return 1.0 - np.clip(response, 0.0, 1.0)


def load_complete_data() -> pd.DataFrame:
    missing = [str(path) for path in SOURCE_FILES if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing source parquet files: {missing}")
    frame = pd.concat(
        [pd.read_parquet(path) for path in SOURCE_FILES], ignore_index=True
    )
    key = ["twin_id", "Variable_Name"]
    if frame.duplicated(key).any():
        duplicates = frame.loc[frame.duplicated(key, keep=False), key].head()
        raise ValueError(f"duplicate respondent-question rows:\n{duplicates}")
    return frame


def pricing_catalog(frame: pd.DataFrame) -> pd.DataFrame:
    pricing = frame[
        frame["Variable_Name"].astype(str).str.startswith(PRICING_PREFIX)
    ].copy()
    if pricing.empty:
        raise ValueError("no QID9 pricing questions found")

    respondent_count = frame["twin_id"].nunique()
    counts = pricing.groupby("Variable_Name")["twin_id"].nunique()
    if len(counts) != 40 or not (counts == respondent_count).all():
        raise ValueError(
            "pricing block must be a complete 40-question respondent matrix; "
            f"found {len(counts)} questions and counts {counts.describe().to_dict()}"
        )

    rows = []
    for question_id, group in pricing.groupby("Variable_Name", sort=False):
        prompts = group["Question"].drop_duplicates().tolist()
        if len(prompts) != 1:
            raise ValueError(f"question text varies within {question_id}")
        response_range = set(
            zip(group["score_range_min"], group["score_range_max"])
        )
        if response_range != {(1.0, 2.0)}:
            raise ValueError(f"{question_id} is not binary: {response_range}")
        rows.append(
            {
                "question_id": str(question_id),
                "question_number": _question_number(str(question_id)),
                "price": extract_price(prompts[0]),
                "human_purchase_share": float(
                    response_to_purchase_probability(
                        group["Human_Response_norm"].to_numpy()
                    ).mean()
                ),
                "respondents": int(group["twin_id"].nunique()),
                "question": prompts[0],
            }
        )
    catalog = pd.DataFrame(rows).sort_values("question_number").reset_index(drop=True)
    return catalog


def fold_paths(fold: int) -> tuple[Path, Path]:
    return (
        FOLD_ROOT / f"fold{fold}_train.parquet",
        FOLD_ROOT / f"fold{fold}_test.parquet",
    )


def prepare_folds() -> dict:
    frame = load_complete_data()
    catalog = pricing_catalog(frame)
    question_ids = catalog["question_id"].tolist()
    rng = np.random.default_rng(FOLD_SEED)
    shuffled = np.asarray(question_ids, dtype=object)[rng.permutation(len(question_ids))]
    folds = [sorted(chunk.tolist(), key=_question_number) for chunk in np.array_split(shuffled, N_FOLDS)]

    FOLD_ROOT.mkdir(parents=True, exist_ok=True)
    fold_records = []
    all_test_questions: set[str] = set()
    for fold, test_questions in enumerate(folds):
        test_set = set(test_questions)
        train = frame[~frame["Variable_Name"].isin(test_set)].copy()
        test = frame[frame["Variable_Name"].isin(test_set)].copy()
        train_path, test_path = fold_paths(fold)

        train_questions = set(train["Variable_Name"].astype(str))
        observed_test_questions = set(test["Variable_Name"].astype(str))
        if train_questions & observed_test_questions:
            raise AssertionError(f"question leakage in fold {fold}")
        if observed_test_questions != test_set:
            raise AssertionError(f"fold {fold} test question mismatch")
        if len(test) != 8 * frame["twin_id"].nunique():
            raise AssertionError(f"fold {fold} does not contain 8 complete questions")

        train.to_parquet(train_path, index=False)
        test.to_parquet(test_path, index=False)
        all_test_questions.update(test_set)
        fold_records.append(
            {
                "fold": fold,
                "test_questions": test_questions,
                "train_rows": len(train),
                "test_rows": len(test),
                "train_question_count": train["Variable_Name"].nunique(),
                "test_question_count": test["Variable_Name"].nunique(),
                "train_respondent_count": train["twin_id"].nunique(),
                "test_respondent_count": test["twin_id"].nunique(),
                "train_file": str(train_path.relative_to(PROJECT_ROOT)),
                "test_file": str(test_path.relative_to(PROJECT_ROOT)),
            }
        )

    if all_test_questions != set(question_ids):
        raise AssertionError("pricing questions are not covered exactly once")
    if sum(len(record["test_questions"]) for record in fold_records) != len(
        all_test_questions
    ):
        raise AssertionError("a pricing question occurs in more than one test fold")

    manifest = {
        "description": "Five-fold whole-product holdout for Twin-2K-500 QID9 offers",
        "fold_seed": FOLD_SEED,
        "folds": fold_records,
        "source_files": [str(path.relative_to(PROJECT_ROOT)) for path in SOURCE_FILES],
        "total_rows": len(frame),
        "respondents": frame["twin_id"].nunique(),
        "all_questions": frame["Variable_Name"].nunique(),
        "pricing_questions": len(catalog),
        "pricing_rows": len(catalog) * frame["twin_id"].nunique(),
        "estimand_scope": (
            "Purchase incidence and gross revenue at each observed product-price offer; "
            "not price elasticity, WTP, a demand curve, or an optimal price."
        ),
    }
    with (FOLD_ROOT / "fold_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    catalog.to_csv(FOLD_ROOT / "pricing_catalog.csv", index=False)
    print(
        f"Prepared {N_FOLDS} folds: {len(frame):,} total rows, "
        f"{len(catalog)} held-out pricing questions, "
        f"{frame['twin_id'].nunique()} respondents."
    )
    return manifest


def all_jobs(seeds: tuple[int, ...]) -> list[Job]:
    return [
        Job(fold=fold, seed=seed, variant=variant)
        for fold in range(N_FOLDS)
        for seed in seeds
        for variant in VARIANTS
    ]


def _command_for_job(job: Job, device: str) -> list[str]:
    train_path, test_path = fold_paths(job.fold)
    return [
        sys.executable,
        "-m",
        "debias.debias_variants",
        "--train_file",
        str(train_path),
        "--test_file",
        str(test_path),
        "--variant",
        job.variant,
        "--model_type",
        "mlp",
        "--device",
        device,
        "--random_state",
        str(job.seed),
        "--llm_vector_transform",
        "raw",
        "--result_file",
        str(job.result_relative),
        "--prediction_file",
        str(job.prediction_relative),
        "--hidden_layers",
        "6144,3072,1536,768,384",
        "--mlp_alpha",
        "1e-6",
        "--learning_rate_init",
        "0.0002",
        "--max_iter",
        "3500",
        "--batch_size",
        "512",
        "--mlp_dropout",
        "0.08",
        "--validation_fraction",
        "0",
        "--n_iter_no_change",
        "30",
        "--min_delta",
        "1e-6",
        "--result_precision",
        "17",
        "--no_split_results",
    ]


def _job_complete(job: Job) -> bool:
    if not job.result_path.exists() or not job.prediction_path.exists():
        return False
    try:
        metrics = pd.read_csv(job.result_path)
        predictions = pd.read_csv(job.prediction_path)
    except Exception:
        return False
    return (
        len(metrics) == 1
        and len(predictions) == 8 * 167
        and set(predictions["question_id"].dropna().astype(str))
        == set(json.loads((FOLD_ROOT / "fold_manifest.json").read_text())["folds"][job.fold]["test_questions"])
    )


def _run_worker(device: str, jobs: list[Job], skip_existing: bool) -> list[str]:
    failures = []
    for index, job in enumerate(jobs, start=1):
        if skip_existing and _job_complete(job):
            print(f"[{device}] skip complete {job.stem}", flush=True)
            continue
        job.result_path.parent.mkdir(parents=True, exist_ok=True)
        job.prediction_path.parent.mkdir(parents=True, exist_ok=True)
        job.log_path.parent.mkdir(parents=True, exist_ok=True)
        command = _command_for_job(job, device)
        print(
            f"[{device}] start {index}/{len(jobs)} {job.stem}",
            flush=True,
        )
        with job.log_path.open("w", encoding="utf-8") as log_handle:
            completed = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                check=False,
                text=True,
            )
        if completed.returncode != 0 or not _job_complete(job):
            failures.append(job.stem)
            print(
                f"[{device}] FAIL {job.stem}; see {job.log_path}",
                flush=True,
            )
        else:
            print(f"[{device}] done {job.stem}", flush=True)
    return failures


def run_jobs(
    seeds: tuple[int, ...], devices: tuple[str, ...], skip_existing: bool = True
) -> None:
    if not (FOLD_ROOT / "fold_manifest.json").exists():
        prepare_folds()
    jobs = all_jobs(seeds)
    if skip_existing:
        jobs = [job for job in jobs if not _job_complete(job)]
    if not jobs:
        print("All requested pricing jobs are already complete.")
        return
    queues = [jobs[index:: len(devices)] for index in range(len(devices))]
    print(
        f"Running {len(jobs)} jobs over {len(devices)} device workers: "
        f"{', '.join(devices)}"
    )
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=len(devices)) as executor:
        futures = {
            executor.submit(_run_worker, device, queue, skip_existing): device
            for device, queue in zip(devices, queues)
            if queue
        }
        for future in as_completed(futures):
            failures.extend(future.result())
    if failures:
        raise RuntimeError(f"{len(failures)} pricing jobs failed: {failures}")


def _safe_correlation(x: np.ndarray, y: np.ndarray, kind: str) -> float:
    if len(x) < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    if kind == "pearson":
        return float(pearsonr(x, y).statistic)
    if kind == "spearman":
        return float(spearmanr(x, y).statistic)
    raise ValueError(kind)


def _top_k_regret(question_frame: pd.DataFrame, k: int) -> float:
    oracle = question_frame.nlargest(k, "human_revenue")["human_revenue"].sum()
    selected = question_frame.nlargest(k, "predicted_revenue")["human_revenue"].sum()
    if oracle <= 0:
        return float("nan")
    return 100.0 * float(oracle - selected) / float(oracle)


def _metric_row(question_frame: pd.DataFrame, unit_frame: pd.DataFrame) -> dict:
    share_error = (
        question_frame["predicted_purchase_share"]
        - question_frame["human_purchase_share"]
    ).to_numpy()
    revenue_error = (
        question_frame["predicted_revenue"] - question_frame["human_revenue"]
    ).to_numpy()
    human_share = question_frame["human_purchase_share"].to_numpy()
    predicted_share = question_frame["predicted_purchase_share"].to_numpy()
    human_revenue = question_frame["human_revenue"].to_numpy()
    predicted_revenue = question_frame["predicted_revenue"].to_numpy()
    true_unit = unit_frame["human_purchase"].to_numpy()
    predicted_unit = unit_frame["predicted_purchase_probability"].to_numpy()
    revenue_denominator = float(np.abs(human_revenue).sum())
    return {
        "share_mae_pp": 100.0 * float(np.mean(np.abs(share_error))),
        "share_rmse_pp": 100.0 * float(np.sqrt(np.mean(share_error**2))),
        "share_bias_pp": 100.0 * float(np.mean(share_error)),
        "share_pearson_r": _safe_correlation(human_share, predicted_share, "pearson"),
        "share_spearman_r": _safe_correlation(human_share, predicted_share, "spearman"),
        "individual_brier": float(np.mean((predicted_unit - true_unit) ** 2)),
        "revenue_mae": float(np.mean(np.abs(revenue_error))),
        "revenue_rmse": float(np.sqrt(np.mean(revenue_error**2))),
        "revenue_wape_pct": (
            100.0 * float(np.abs(revenue_error).sum()) / revenue_denominator
            if revenue_denominator > 0
            else float("nan")
        ),
        "revenue_pearson_r": _safe_correlation(
            human_revenue, predicted_revenue, "pearson"
        ),
        "revenue_spearman_r": _safe_correlation(
            human_revenue, predicted_revenue, "spearman"
        ),
        "top5_revenue_regret_pct": _top_k_regret(question_frame, 5),
        "top10_revenue_regret_pct": _top_k_regret(question_frame, 10),
    }


def _load_method_units(job: Job, method: str) -> pd.DataFrame:
    predictions = pd.read_csv(job.prediction_path)
    if method == "base_llm":
        predicted_norm = predictions["y_baseline"].to_numpy() - 1.0
    else:
        predicted_norm = predictions["y_pred"].to_numpy() - 1.0
    unit = pd.DataFrame(
        {
            "fold": job.fold,
            "seed": job.seed,
            "method": METHOD_LABELS[method],
            "variant": method,
            "question_id": predictions["question_id"].astype(str),
            "respondent_id": predictions["respondent_id"],
            "human_purchase": 2.0 - predictions["y_true"].to_numpy(dtype=float),
            "predicted_purchase_probability": response_to_purchase_probability(
                predicted_norm
            ),
        }
    )
    return unit


def _load_train_share_prior_units(fold: int, seed: int) -> pd.DataFrame:
    """Forecast held-out offers with the other pricing items' human share.

    Every learned debiasing method receives the same training-side human
    labels, so this historical-incidence prior is a necessary managerial
    benchmark for new-offer forecasts.
    """
    train_path, _ = fold_paths(fold)
    train = pd.read_parquet(
        train_path, columns=["Variable_Name", "Human_Response_norm"]
    )
    train_pricing = train[
        train["Variable_Name"].astype(str).str.startswith(PRICING_PREFIX)
    ]
    if train_pricing["Variable_Name"].nunique() != 32:
        raise ValueError(f"fold {fold} does not contain 32 training pricing offers")
    prior = float(
        response_to_purchase_probability(
            train_pricing["Human_Response_norm"].to_numpy()
        ).mean()
    )
    reference = pd.read_csv(Job(fold, seed, "x_all_llm").prediction_path)
    return pd.DataFrame(
        {
            "fold": fold,
            "seed": seed,
            "method": METHOD_LABELS["train_share_prior"],
            "variant": "train_share_prior",
            "question_id": reference["question_id"].astype(str),
            "respondent_id": reference["respondent_id"],
            "human_purchase": 2.0 - reference["y_true"].to_numpy(dtype=float),
            "predicted_purchase_probability": np.full(len(reference), prior),
        }
    )


def _paired_question_tests(question_predictions: pd.DataFrame) -> pd.DataFrame:
    seed_averaged = (
        question_predictions.groupby(
            ["variant", "method", "question_id", "price"], as_index=False
        )
        .agg(
            human_purchase_share=("human_purchase_share", "first"),
            predicted_purchase_share=("predicted_purchase_share", "mean"),
            human_revenue=("human_revenue", "first"),
            predicted_revenue=("predicted_revenue", "mean"),
        )
    )
    seed_averaged["share_abs_error"] = np.abs(
        seed_averaged["predicted_purchase_share"]
        - seed_averaged["human_purchase_share"]
    )
    seed_averaged["revenue_abs_error"] = np.abs(
        seed_averaged["predicted_revenue"] - seed_averaged["human_revenue"]
    )
    vector = seed_averaged[seed_averaged["variant"] == "x_all_llm"].set_index(
        "question_id"
    )
    rows = []
    for comparator in (
        "base_llm",
        "train_share_prior",
        "x_only",
        "x_one_llm",
        "x_avg_llm",
    ):
        other = seed_averaged[seed_averaged["variant"] == comparator].set_index(
            "question_id"
        )
        if set(vector.index) != set(other.index):
            raise ValueError(f"question mismatch for Vector vs {comparator}")
        other = other.loc[vector.index]
        for metric in ("share_abs_error", "revenue_abs_error"):
            result = ttest_rel(
                vector[metric].to_numpy(),
                other[metric].to_numpy(),
                alternative="less",
            )
            rows.append(
                {
                    "comparison": f"Vector vs {METHOD_LABELS[comparator]}",
                    "metric": metric,
                    "questions": len(vector),
                    "vector_mean_error": vector[metric].mean(),
                    "comparator_mean_error": other[metric].mean(),
                    "vector_advantage": other[metric].mean()
                    - vector[metric].mean(),
                    "paired_t": result.statistic,
                    "one_sided_p_value": result.pvalue,
                }
            )
    return pd.DataFrame(rows)


def summarize(seeds: tuple[int, ...]) -> pd.DataFrame:
    catalog_path = FOLD_ROOT / "pricing_catalog.csv"
    if not catalog_path.exists():
        prepare_folds()
    catalog = pd.read_csv(catalog_path)[["question_id", "price"]]
    catalog["question_id"] = catalog["question_id"].astype(str)

    missing_jobs = [job.stem for job in all_jobs(seeds) if not _job_complete(job)]
    if missing_jobs:
        raise FileNotFoundError(
            f"cannot summarize; {len(missing_jobs)} jobs are incomplete: {missing_jobs[:8]}"
        )

    all_units = []
    for seed in seeds:
        for fold in range(N_FOLDS):
            vector_job = Job(fold, seed, "x_all_llm")
            all_units.append(_load_method_units(vector_job, "base_llm"))
            all_units.append(_load_train_share_prior_units(fold, seed))
            for variant in VARIANTS:
                all_units.append(_load_method_units(Job(fold, seed, variant), variant))
    unit_predictions = pd.concat(all_units, ignore_index=True)

    duplicated = unit_predictions.duplicated(
        ["seed", "variant", "question_id", "respondent_id"]
    )
    if duplicated.any():
        raise ValueError("duplicate out-of-fold unit predictions")
    expected_rows = len(seeds) * len(METHOD_LABELS) * 40 * 167
    if len(unit_predictions) != expected_rows:
        raise ValueError(
            f"unexpected OOF prediction count: {len(unit_predictions)} != {expected_rows}"
        )

    question_predictions = (
        unit_predictions.groupby(
            ["seed", "variant", "method", "fold", "question_id"], as_index=False
        )
        .agg(
            human_purchase_share=("human_purchase", "mean"),
            predicted_purchase_share=("predicted_purchase_probability", "mean"),
            respondents=("respondent_id", "nunique"),
        )
        .merge(catalog, on="question_id", how="left", validate="many_to_one")
    )
    if question_predictions["price"].isna().any():
        raise ValueError("missing price after prediction/catalog merge")
    question_predictions["human_revenue"] = (
        question_predictions["price"]
        * question_predictions["human_purchase_share"]
    )
    question_predictions["predicted_revenue"] = (
        question_predictions["price"]
        * question_predictions["predicted_purchase_share"]
    )

    metric_rows = []
    for (seed, variant, method), questions in question_predictions.groupby(
        ["seed", "variant", "method"], sort=False
    ):
        units = unit_predictions[
            (unit_predictions["seed"] == seed)
            & (unit_predictions["variant"] == variant)
        ]
        if len(questions) != 40 or len(units) != 40 * 167:
            raise ValueError(f"incomplete OOF predictions for {seed}/{method}")
        metric_rows.append(
            {
                "seed": seed,
                "variant": variant,
                "method": method,
                **_metric_row(questions, units),
            }
        )
    seed_metrics = pd.DataFrame(metric_rows)

    metric_columns = [
        column
        for column in seed_metrics.columns
        if column not in {"seed", "variant", "method"}
    ]
    summary_rows = []
    for (variant, method), group in seed_metrics.groupby(
        ["variant", "method"], sort=False
    ):
        row = {"variant": variant, "method": method, "seeds": len(group)}
        for metric in metric_columns:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_std"] = group[metric].std(ddof=1)
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    method_order = {label: index for index, label in enumerate(METHOD_LABELS.values())}
    summary["_order"] = summary["method"].map(method_order)
    summary = summary.sort_values("_order").drop(columns="_order").reset_index(drop=True)

    tests = _paired_question_tests(question_predictions)
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    unit_predictions.to_csv(
        RESULT_ROOT / "oof_unit_predictions.csv", index=False, float_format="%.17g"
    )
    question_predictions.to_csv(
        RESULT_ROOT / "oof_question_predictions.csv",
        index=False,
        float_format="%.17g",
    )
    seed_metrics.to_csv(
        RESULT_ROOT / "seed_metrics.csv", index=False, float_format="%.17g"
    )
    summary.to_csv(RESULT_ROOT / "summary.csv", index=False, float_format="%.17g")
    tests.to_csv(
        RESULT_ROOT / "paired_question_tests.csv", index=False, float_format="%.17g"
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
    display = summary[display_columns].copy()
    truth = question_predictions.drop_duplicates("question_id")
    human_share = truth["human_purchase_share"]
    markdown = [
        "# Twin-2K-500 pricing application (whole-product holdout)",
        "",
        "Five folds hold out all 167 human responses for eight pricing offers at a time. ",
        "The estimands are purchase incidence and gross revenue at observed prices; ",
        "the design does not identify WTP, elasticity, a demand curve, or an optimal price.",
        "",
        (
            f"Across the 40 offers, the human purchase share has mean "
            f"{100.0 * human_share.mean():.2f}%, standard deviation "
            f"{100.0 * human_share.std(ddof=1):.2f} pp, and range "
            f"{100.0 * human_share.min():.2f}%--{100.0 * human_share.max():.2f}%."
        ),
        "",
        display.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Paired question-level tests",
        "",
        tests.to_markdown(index=False, floatfmt=".6g"),
        "",
    ]
    (RESULT_ROOT / "summary.md").write_text("\n".join(markdown), encoding="utf-8")
    print(display.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nPaired question-level tests (one-sided: Vector has lower error):")
    print(tests.to_string(index=False, float_format=lambda value: f"{value:.6g}"))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        nargs="?",
        choices=("prepare", "run", "summarize", "all"),
        default="all",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--devices",
        nargs="+",
        default=[f"cuda:{index}" for index in range(8)],
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="rerun jobs even if both validated metric and prediction files exist",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = tuple(args.seeds)
    devices = tuple(args.devices)
    if not seeds:
        raise ValueError("at least one seed is required")
    if not devices:
        raise ValueError("at least one device is required")
    if args.stage in {"prepare", "all"}:
        prepare_folds()
    if args.stage in {"run", "all"}:
        run_jobs(seeds, devices, skip_existing=not args.rerun)
    if args.stage in {"summarize", "all"}:
        summarize(seeds)


if __name__ == "__main__":
    main()
