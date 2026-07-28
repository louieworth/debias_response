"""Side-by-side comparison: Qwen3-4B+8B as sample source vs closed-source.

Pulls the single-row Qwen3 CSVs under results/{group,individual}_qwen/
and the matching (x_all_llm, mlp, qwen3-4B_norm+qwen3-8B_norm -> closed)
rows from the existing closed-source multi-seed CSVs in results/{group,individual}/.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

PROJ_ROOT = Path("/home/jiangli/debias_response")
RES = PROJ_ROOT / "results"
DATASETS = ["EEDI", "OpinionQA", "Twin-2K-500"]
VARIANT = "x_all_llm"
MODEL = "mlp"
SEED = 1

# The test metrics we surface.
METRIC_COLS = [
    "test_acc_model_hard",
    "test_acc_model_soft",
    "test_acc_model_mad",
    "test_mae_model_original",
    "test_r2_norm",
    "test_pearson_r_model",
]
AGG_STEM = {"EEDI": "eedi", "OpinionQA": "opinionqa", "Twin-2K-500": "twin"}
AGG_SUBDIR = {"EEDI": "EEDI", "OpinionQA": "OpinionQA", "Twin-2K-500": "twin"}


def _qwen_csv(ds: str, level: str) -> Path:
    root = "group_qwen" if level == "aggreated" else "individual_qwen"
    return RES / root / ds / f"qwen_{VARIANT}_seed{SEED}.csv"


def _closed_csv(ds: str, level: str) -> Path:
    if level == "aggreated":
        if ds == "Twin-2K-500":
            return (
                RES
                / "group"
                / "twin"
                / "aggreated"
                / f"aggreated_twin_256_result_seed_{SEED}.csv"
            )
        return RES / "group" / AGG_SUBDIR[ds] / f"aggreated_{AGG_STEM[ds]}_multisource_result_seed_{SEED}.csv"
    return RES / "individual" / f"individual_{ds}_result.csv"


def _metrics(df: pd.DataFrame) -> dict:
    return {c: float(df[c].iloc[0]) for c in METRIC_COLS if c in df.columns}


def _pick_closed_row(df: pd.DataFrame, ds: str, level: str) -> pd.DataFrame | None:
    """Find the best-matching closed-source row.

    For aggregated: the multisource CSV holds rows for many llm_fields combinations
    at variant=x_all_llm, model_type=mlp. The best existing closed-source baseline
    to compare against is the all-8-LLM row (i.e. the full default DEFAULT_LLM_FIELDS).
    That row has the pipe-joined 8 closed LLMs in `llm_fields`.

    For individual: each row already corresponds to a (seed, variant) pair;
    filter on variant + random_state (if present) + LLM_Responses default.
    """
    if "seed" in df.columns:
        df = df[df["seed"] == SEED]
    if "variant" in df.columns:
        df = df[df["variant"] == VARIANT]
    if "model_type" in df.columns:
        df = df[df["model_type"] == MODEL]
    if level == "aggreated":
        # Pick the row that uses all 8 closed-source LLMs (x_all_llm with
        # the full 8-pipe field list in closed-source multisource CSVs).
        mask = df["llm_fields"].astype(str).str.count(r"\|") == 7
        pick = df[mask] if mask.any() else df
    else:
        pick = df
    if len(pick) == 0:
        return None
    return pick.head(1)


def main() -> None:
    summary = {}
    for ds in DATASETS:
        for level in ("aggreated", "individual"):
            qwen_p = _qwen_csv(ds, level)
            closed_p = _closed_csv(ds, level)

            qwen_m = _metrics(pd.read_csv(qwen_p)) if qwen_p.exists() else None
            closed_m = None
            if closed_p.exists():
                closed_df = pd.read_csv(closed_p)
                row = _pick_closed_row(closed_df, ds, level)
                closed_m = _metrics(row) if row is not None else None

            key = f"{ds}_{level}"
            summary[key] = {"qwen": qwen_m, "closed": closed_m}

            # Gap line for quick eyeballing (closed - qwen on test_acc_model_hard).
            if qwen_m and closed_m:
                q = qwen_m.get("test_acc_model_hard")
                c = closed_m.get("test_acc_model_hard")
                if q is not None and c is not None:
                    summary[key]["delta_acc_hard"] = round(c - q, 4)

    out = RES / "qwen_vs_closed.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(json.dumps(summary, indent=2, default=str))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
