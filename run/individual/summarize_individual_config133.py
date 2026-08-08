"""Summarize the config-133 individual sweep for model seeds 0--4."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import ttest_1samp, ttest_rel


DATASETS = ["Twin-2K-500", "OpinionQA", "EEDI"]
VARIANTS = ["x_only", "x_one_llm", "one_logprob", "x_avg_llm", "x_all_llm"]
LABELS = {
    "x_only": "w/o LLM",
    "x_one_llm": "One",
    "one_logprob": "One Logprob",
    "x_avg_llm": "Mean",
    "x_all_llm": "Vector",
}
METRICS = ["mae", "acc", "ha", "sa"]
BASE = {
    "Twin-2K-500": {"mae": 62.30, "acc": 62.48, "ha": 51.55, "sa": 49.39},
    "OpinionQA": {"mae": 110.23, "acc": 72.44, "ha": 32.80, "sa": 31.09},
    "EEDI": {"mae": 37.88, "acc": 87.37, "ha": 67.53, "sa": 53.17},
}


def stars(p_value: float) -> str:
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-root",
        type=Path,
        default=Path(
            "results/individual_test_tuned_a1e-6_d08_fulltrain_seed0_4"
        ),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    return parser.parse_args()


def load_rows(root: Path, seeds: list[int]) -> pd.DataFrame:
    frames = []
    for dataset in DATASETS:
        path = root / dataset / f"individual_{dataset}_result.csv"
        frame = pd.read_csv(path)
        if len(frame) != len(seeds) * len(VARIANTS):
            raise ValueError(f"unexpected row count in {path}: {len(frame)}")
        if set(frame["seed"].astype(int)) != set(seeds):
            raise ValueError(f"unexpected seeds in {path}: {sorted(frame['seed'].unique())}")
        if set(frame["variant"]) != set(VARIANTS):
            raise ValueError(f"unexpected variants in {path}")
        if frame.duplicated(["seed", "variant"]).any():
            raise ValueError(f"duplicate seed/variant rows in {path}")
        frames.append(
            pd.DataFrame(
                {
                    "dataset": dataset,
                    "seed": frame["seed"].astype(int),
                    "variant": frame["variant"],
                    "method": frame["variant"].map(LABELS),
                    "mae": frame["test_mae_model_original"] * 100.0,
                    "acc": frame["test_acc_model_mad"] * 100.0,
                    "ha": frame["test_acc_model_hard"] * 100.0,
                    "sa": frame["test_acc_model_soft"] * 100.0,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def build_summary(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset in DATASETS:
        subset = data[data["dataset"] == dataset]
        for metric in METRICS:
            row = {"dataset": dataset, "metric": metric, "base_llm": BASE[dataset][metric]}
            for variant in VARIANTS:
                values = subset[subset["variant"] == variant][metric]
                row[f"{variant}_mean"] = values.mean()
                row[f"{variant}_std"] = values.std(ddof=1)
            rows.append(row)
    return pd.DataFrame(rows)


def build_base_tests(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset in DATASETS:
        for variant in VARIANTS:
            subset = data[(data["dataset"] == dataset) & (data["variant"] == variant)]
            for metric in METRICS:
                alternative = "less" if metric == "mae" else "greater"
                result = ttest_1samp(
                    subset[metric].to_numpy(),
                    BASE[dataset][metric],
                    alternative=alternative,
                )
                rows.append(
                    {
                        "dataset": dataset,
                        "variant": variant,
                        "method": LABELS[variant],
                        "metric": metric,
                        "mean": subset[metric].mean(),
                        "std": subset[metric].std(ddof=1),
                        "base": BASE[dataset][metric],
                        "alternative": alternative,
                        "t_statistic": result.statistic,
                        "p_value": result.pvalue,
                        "stars": stars(result.pvalue),
                    }
                )
    return pd.DataFrame(rows)


def build_vector_tests(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset in DATASETS:
        vector = data[
            (data["dataset"] == dataset) & (data["variant"] == "x_all_llm")
        ].set_index("seed")
        for variant in VARIANTS[:-1]:
            other = data[
                (data["dataset"] == dataset) & (data["variant"] == variant)
            ].set_index("seed")
            if list(vector.index) != list(other.index):
                raise ValueError(f"seed mismatch for {dataset}/{variant}")
            for metric in METRICS:
                alternative = "less" if metric == "mae" else "greater"
                directional = ttest_rel(
                    vector[metric], other[metric], alternative=alternative
                )
                two_sided = ttest_rel(vector[metric], other[metric])
                advantage = (
                    other[metric].mean() - vector[metric].mean()
                    if metric == "mae"
                    else vector[metric].mean() - other[metric].mean()
                )
                rows.append(
                    {
                        "dataset": dataset,
                        "comparison": f"Vector vs {LABELS[variant]}",
                        "metric": metric,
                        "vector_advantage": advantage,
                        "one_sided_p_value": directional.pvalue,
                        "two_sided_p_value": two_sided.pvalue,
                        "one_sided_stars": stars(directional.pvalue),
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    data = load_rows(args.result_root, args.seeds)
    summary = build_summary(data)
    base_tests = build_base_tests(data)
    vector_tests = build_vector_tests(data)

    data.to_csv(args.result_root / "individual_seed_rows.csv", index=False)
    summary.to_csv(args.result_root / "individual_5seed_summary.csv", index=False)
    base_tests.to_csv(args.result_root / "individual_pvalues_vs_base.csv", index=False)
    vector_tests.to_csv(args.result_root / "individual_paired_vs_vector.csv", index=False)

    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
