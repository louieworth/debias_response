#!/usr/bin/env python3
"""Summarize fixed-K source runs and test Mean/Vector against paired One."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


DATASETS = ("Twin-2K-500", "OpinionQA", "EEDI")
SOURCES = {
    "claude-3.5-haiku": "claude-3.5-haiku_norm",
    "deepseek-v3": "deepseek-v3_norm",
    "gpt-3.5-turbo": "gpt-3.5-turbo_norm",
    "gpt-4o-mini": "gpt-4o-mini_norm",
    "gpt-4o": "gpt-4o_norm",
    "gpt-5-mini": "gpt-5-mini_norm",
    "llama-3.3-70B": "llama-3.3-70B-instruct-turbo_norm",
    "mistral-7B-v0.3": "mistral-7B-instruct-v0.3_norm",
}
METHODS = {
    "One": "x_one_llm",
    "Mean": "x_avg_llm",
    "Vector": "x_all_llm",
}
METHOD_ORDER = ("Base", "w/o", "One", "Mean", "Vector")
METRICS = {
    "MAE": ("test_mae_base_original", "test_mae_model_original", "min"),
    "Acc": ("test_acc_base_mad", "test_acc_model_mad", "max"),
    "HA": ("test_acc_base_hard", "test_acc_model_hard", "max"),
    "SA": ("test_acc_base_soft", "test_acc_model_soft", "max"),
}
MAIN_METRICS = {"MAE": "mae", "Acc": "acc", "HA": "ha", "SA": "sa"}
MAIN_COLUMNS = {
    "Base": ("base_llm", None),
    "w/o": ("x_only_mean", "x_only_std"),
    "One": ("x_one_llm_mean", "x_one_llm_std"),
    "Mean": ("x_avg_llm_mean", "x_avg_llm_std"),
    "Vector": ("x_all_llm_mean", "x_all_llm_std"),
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


def load_dataset(root: Path, dataset: str) -> pd.DataFrame:
    frames = []
    for seed in range(5):
        path = root / dataset / f"population_source_{dataset}_seed_{seed}.csv"
        frame = pd.read_csv(path)
        if len(frame) != 25:
            raise ValueError(f"{path}: expected 25 rows, found {len(frame)}")
        frame["seed"] = seed
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    if not result["model_type"].eq("mlp").all():
        raise ValueError(f"{dataset}: found a non-MLP result")
    vector_rows = result.loc[result["variant"].eq("x_all_llm")]
    if not vector_rows["llm_responses_length"].eq(50).all():
        dimensions = vector_rows[
            ["llm_field", "llm_responses_length"]
        ].drop_duplicates()
        raise ValueError(f"{dataset}: non-K=50 Vector rows:\n{dimensions}")
    return result


def sorted_values(rows: pd.DataFrame, column: str) -> np.ndarray:
    values = rows.sort_values("seed")[column].to_numpy(dtype=float) * 100.0
    if len(values) != 5:
        raise ValueError(f"Expected five seed values for {column}, found {len(values)}")
    return values


def base_values(source_rows: pd.DataFrame, column: str) -> np.ndarray:
    values = sorted_values(
        source_rows.loc[source_rows["variant"].eq("x_avg_llm")], column
    )
    if not np.allclose(values, values[0], atol=1e-12, rtol=0.0):
        raise ValueError(f"Base is not fixed across seeds: {values}")
    return values


def learned_cell(mean: float, bold: bool, stars: str) -> str:
    displayed_mean = f"{mean:.1f}"
    value = rf"\textbf{{{displayed_mean}}}" if bold else displayed_mean
    if stars:
        return rf"\sourcesigcell{{{value}}}{{{stars}}}"
    return value


def base_cell(mean: float, bold: bool) -> str:
    displayed = f"{mean:.1f}"
    return rf"\textbf{{{displayed}}}" if bold else displayed


def verify_against_main(
    summary: pd.DataFrame,
    tests: pd.DataFrame,
    main_summary: pd.DataFrame,
    main_tests: pd.DataFrame,
) -> None:
    for dataset in DATASETS:
        for metric, main_metric in MAIN_METRICS.items():
            main_row = main_summary.loc[
                main_summary["dataset"].eq(dataset)
                & main_summary["metric"].eq(main_metric)
            ]
            if len(main_row) != 1:
                raise ValueError(f"Missing main summary for {dataset}/{metric}")
            main_row = main_row.iloc[0]
            for method, (mean_column, sd_column) in MAIN_COLUMNS.items():
                appendix_rows = summary.loc[
                    summary["dataset"].eq(dataset)
                    & summary["metric"].eq(metric)
                    & summary["method"].eq(method)
                ]
                # w/o is source-independent, so all eight rows must match main.
                if method == "w/o":
                    check_rows = appendix_rows
                else:
                    check_rows = appendix_rows.loc[
                        appendix_rows["source"].eq("gpt-4o")
                    ]
                expected_mean = float(main_row[mean_column])
                if not np.allclose(
                    check_rows["mean"], expected_mean, atol=1e-10, rtol=0.0
                ):
                    raise ValueError(
                        f"Main mean mismatch: {dataset}/{metric}/{method}"
                    )
                if sd_column is not None:
                    expected_sd = float(main_row[sd_column])
                    if not np.allclose(
                        check_rows["sd"], expected_sd, atol=1e-10, rtol=0.0
                    ):
                        raise ValueError(
                            f"Main s.d. mismatch: {dataset}/{metric}/{method}"
                        )

            for method in ("Mean", "Vector"):
                appendix_test = tests.loc[
                    tests["dataset"].eq(dataset)
                    & tests["source"].eq("gpt-4o")
                    & tests["metric"].eq(metric)
                    & tests["method"].eq(method)
                ]
                main_test = main_tests.loc[
                    main_tests["dataset"].eq(dataset)
                    & main_tests["metric"].eq(main_metric)
                    & main_tests["comparison"].eq(f"{method} vs One")
                ]
                if len(appendix_test) != 1 or len(main_test) != 1:
                    raise ValueError(
                        f"Missing p-value comparison: {dataset}/{metric}/{method}"
                    )
                appendix_p = float(appendix_test.iloc[0]["two_sided_p_value"])
                main_p = float(main_test.iloc[0]["two_sided_p_value"])
                if not np.isclose(appendix_p, main_p, atol=1e-10, rtol=0.0):
                    raise ValueError(
                        f"Main p-value mismatch: {dataset}/{metric}/{method}"
                    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(
            "results/population_source_significance_k50_seed0_4_precision17"
        ),
    )
    parser.add_argument(
        "--main-root",
        type=Path,
        default=Path("results/population_one_significance_seed0_4_precision17"),
    )
    args = parser.parse_args()

    summary_records = []
    test_records = []
    latex_by_dataset: dict[str, list[str]] = {}

    for dataset in DATASETS:
        frame = load_dataset(args.root, dataset)
        no_llm_rows = frame.loc[frame["variant"].eq("x_only")]
        if no_llm_rows.groupby("seed").size().to_dict() != {
            seed: 1 for seed in range(5)
        }:
            raise ValueError(f"{dataset}: expected one x_only row per seed")

        latex_rows = []
        for source, field in SOURCES.items():
            source_rows = frame.loc[frame["llm_field"].eq(field)]
            expected_counts = {
                variant: 5 for variant in ("x_one_llm", "x_avg_llm", "x_all_llm")
            }
            counts = source_rows.groupby("variant").size().to_dict()
            if counts != expected_counts:
                raise ValueError(
                    f"{dataset}/{source}: unexpected variant counts {counts}"
                )

            metric_values: dict[str, dict[str, np.ndarray]] = {}
            metric_stars: dict[str, dict[str, str]] = {}
            for metric, (base_column, model_column, _) in METRICS.items():
                values = {
                    "Base": base_values(source_rows, base_column),
                    "w/o": sorted_values(no_llm_rows, model_column),
                }
                for method, variant in METHODS.items():
                    values[method] = sorted_values(
                        source_rows.loc[source_rows["variant"].eq(variant)],
                        model_column,
                    )
                metric_values[metric] = values
                metric_stars[metric] = {}

                for method in ("Mean", "Vector"):
                    differences = values[method] - values["One"]
                    test = stats.ttest_rel(
                        values[method], values["One"], alternative="two-sided"
                    )
                    p_value = float(test.pvalue)
                    stars = significance_stars(p_value)
                    metric_stars[metric][method] = stars
                    test_records.append(
                        {
                            "dataset": dataset,
                            "source": source,
                            "method": method,
                            "comparison": f"{method} vs One",
                            "metric": metric,
                            "mean_difference": float(differences.mean()),
                            "t_statistic": float(test.statistic),
                            "two_sided_p_value": p_value,
                            "stars": stars,
                        }
                    )

            cells = []
            for metric, (_, _, direction) in METRICS.items():
                values = metric_values[metric]
                displayed = {
                    method: float(np.round(values[method].mean(), 1))
                    for method in METHOD_ORDER
                }
                best = (
                    min(displayed.values())
                    if direction == "min"
                    else max(displayed.values())
                )
                for method in METHOD_ORDER:
                    method_values = values[method]
                    mean = float(method_values.mean())
                    sd = 0.0 if method == "Base" else float(
                        method_values.std(ddof=1)
                    )
                    bold = displayed[method] == best
                    stars = metric_stars[metric].get(method, "")
                    summary_records.append(
                        {
                            "dataset": dataset,
                            "source": source,
                            "metric": metric,
                            "method": method,
                            "mean": mean,
                            "sd": sd,
                            "display_mean": displayed[method],
                            "display_sd": float(np.round(sd, 1)),
                            "bold": bold,
                            "stars_vs_one": stars,
                        }
                    )
                    cells.append(
                        base_cell(mean, bold)
                        if method == "Base"
                        else learned_cell(mean, bold, stars)
                    )
            latex_rows.append(
                rf"\texttt{{{source}}} & " + " & ".join(cells) + r" \\"
            )
        latex_by_dataset[dataset] = latex_rows

    summary = pd.DataFrame(summary_records)
    tests = pd.DataFrame(test_records)
    verify_against_main(
        summary,
        tests,
        pd.read_csv(args.main_root / "population_5seed_summary.csv"),
        pd.read_csv(args.main_root / "population_paired_vs_one.csv"),
    )

    args.root.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.root / "population_source_summary.csv", index=False)
    tests.to_csv(args.root / "population_source_pvalues_vs_one.csv", index=False)
    for dataset, rows in latex_by_dataset.items():
        output = args.root / f"{dataset}_latex_rows.tex"
        output.write_text("\n".join(rows) + "\n", encoding="utf-8")
        print(f"wrote {output}")
        print("\n".join(rows))
    print("Verified all gpt-4o rows, shared w/o rows, and gpt-4o p-values against main.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
