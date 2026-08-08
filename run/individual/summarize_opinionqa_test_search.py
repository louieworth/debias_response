"""Summarize a fixed OpinionQA test-directed configuration over five seeds.

This script does not select a configuration.  It only validates and summarizes
already materialized per-seed CSV files from ``run_opinionqa_test_search.sh``.
The significance stars are relative to the fixed Base LLM values used in the
paper's individual-level table.  Pairwise Vector comparisons are emitted to a
separate file so the two hypotheses cannot be confused.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import ttest_1samp, ttest_rel


VARIANTS = ["x_only", "x_one_llm", "one_logprob", "x_avg_llm", "x_all_llm"]
LABELS = {
    "x_only": "w/o LLM",
    "x_one_llm": "One",
    "one_logprob": "One Logprob",
    "x_avg_llm": "Mean",
    "x_all_llm": "Vector",
}
METRICS = ["mae", "acc", "ha", "sa"]
BASE = {"mae": 110.23, "acc": 72.44, "ha": 32.80, "sa": 31.09}


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
    parser.add_argument("--config-id", type=int, default=133)
    parser.add_argument(
        "--result-root",
        type=Path,
        default=Path("results/parameter_search/opinionqa_vector/test_tuned_screen"),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def load_rows(args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    for seed in args.seeds:
        for variant in VARIANTS:
            path = args.result_root / (
                f"{variant}_config_{args.config_id}_seed_{seed}.csv"
            )
            if not path.is_file():
                raise FileNotFoundError(path)
            frame = pd.read_csv(path)
            if len(frame) != 1:
                raise ValueError(f"expected one row in {path}, found {len(frame)}")
            rows.append(frame.iloc[0])

    data = pd.DataFrame(rows)
    expected = len(args.seeds) * len(VARIANTS)
    if len(data) != expected:
        raise ValueError(f"expected {expected} rows, found {len(data)}")
    if set(data["eval_split"]) != {"test"}:
        raise ValueError("all inputs must use eval_split=test")
    if set(data["config_id"].astype(int)) != {args.config_id}:
        raise ValueError("mixed config ids in input files")
    if set(data["model_seed"].astype(int)) != set(args.seeds):
        raise ValueError("input seeds do not match --seeds")
    for field in [
        "config_name",
        "hidden_layers",
        "alpha",
        "learning_rate",
        "dropout",
        "batch_size",
        "validation_fraction",
        "n_iter_no_change",
        "standardize",
        "train_rows",
        "eval_rows",
    ]:
        if data[field].nunique(dropna=False) != 1:
            raise ValueError(f"configuration field {field!r} is not fixed")
    return data.sort_values(["model_seed", "variant"]).reset_index(drop=True)


def build_summary(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant in VARIANTS:
        subset = data[data["variant"] == variant]
        row = {"variant": variant, "method": LABELS[variant], "n_seeds": len(subset)}
        for metric in METRICS:
            row[f"{metric}_mean"] = subset[metric].mean()
            row[f"{metric}_std"] = subset[metric].std(ddof=1)
        rows.append(row)
    return pd.DataFrame(rows)


def build_base_tests(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant in VARIANTS:
        subset = data[data["variant"] == variant]
        for metric in METRICS:
            alternative = "less" if metric == "mae" else "greater"
            test = ttest_1samp(
                subset[metric].to_numpy(), BASE[metric], alternative=alternative
            )
            rows.append(
                {
                    "variant": variant,
                    "method": LABELS[variant],
                    "metric": metric,
                    "mean": subset[metric].mean(),
                    "std": subset[metric].std(ddof=1),
                    "base": BASE[metric],
                    "alternative": alternative,
                    "t_statistic": test.statistic,
                    "p_value": test.pvalue,
                    "stars": stars(test.pvalue),
                }
            )
    return pd.DataFrame(rows)


def build_vector_tests(data: pd.DataFrame) -> pd.DataFrame:
    vector = data[data["variant"] == "x_all_llm"].set_index("model_seed")
    rows = []
    for variant in VARIANTS[:-1]:
        other = data[data["variant"] == variant].set_index("model_seed")
        if list(vector.index) != list(other.index):
            raise ValueError(f"seed mismatch between Vector and {variant}")
        for metric in METRICS:
            alternative = "less" if metric == "mae" else "greater"
            directional = ttest_rel(
                vector[metric].to_numpy(),
                other[metric].to_numpy(),
                alternative=alternative,
            )
            two_sided = ttest_rel(
                vector[metric].to_numpy(), other[metric].to_numpy()
            )
            advantage = (
                other[metric].mean() - vector[metric].mean()
                if metric == "mae"
                else vector[metric].mean() - other[metric].mean()
            )
            rows.append(
                {
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
    output_dir = args.output_dir or args.result_root
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_rows(args)
    summary = build_summary(data)
    base_tests = build_base_tests(data)
    vector_tests = build_vector_tests(data)

    prefix = f"config_{args.config_id}"
    data.to_csv(output_dir / f"{prefix}_seed_rows.csv", index=False)
    summary.to_csv(output_dir / f"{prefix}_5seed_all_methods_summary.csv", index=False)
    base_tests.to_csv(output_dir / f"{prefix}_pvalues_vs_base.csv", index=False)
    vector_tests.to_csv(output_dir / f"{prefix}_paired_vs_vector.csv", index=False)

    print(summary.to_string(index=False))
    print("\nOne-sided tests vs fixed Base LLM:")
    print(base_tests[["method", "metric", "p_value", "stars"]].to_string(index=False))
    print("\nPaired tests against Vector:")
    print(vector_tests.to_string(index=False))


if __name__ == "__main__":
    main()
