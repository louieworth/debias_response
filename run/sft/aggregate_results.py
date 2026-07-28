"""Aggregate per-config eval outputs into results/sft.json.

For each of the 6 configs:
  - Read the per-config JSON written by verl.trainer.main_eval and extract
    the average hard-accuracy score.
  - Read the corresponding gen_results parquet and compute MAE of the
    extracted integer prediction vs. the raw float ground truth (only
    aggregated configs meaningfully differ from the int-gt case; both are
    reported for uniformity).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJ_ROOT = Path("/home/jiangli/debias_response")
RES_DIR = PROJ_ROOT / "results"
GEN_DIR = PROJ_ROOT / "gen_results"

sys.path.insert(0, str(PROJ_ROOT))
from run.sft.compute_score import extract_answer  # noqa: E402

CONFIGS = [
    "EEDI_aggreated",
    "EEDI_individual",
    "OpinionQA_aggreated",
    "OpinionQA_individual",
    "Twin-2K-500_aggreated",
    "Twin-2K-500_individual",
]


def find_scalar(obj):
    """Walk the main_eval JSON and return the first float metric found."""
    if isinstance(obj, (int, float)):
        return float(obj)
    if isinstance(obj, dict):
        for v in obj.values():
            r = find_scalar(v)
            if r is not None:
                return r
    return None


# (dataset, level) -> test parquet with per-sample score range columns.
_RANGE_SRC = {
    ("EEDI",        "aggreated"):  ("dataset/EEDI/aggreated/eedi_test.parquet", "score_range"),
    ("EEDI",        "individual"): ("dataset/EEDI/individual/individual_test.parquet", "split_range"),
    ("OpinionQA",   "aggreated"):  ("dataset/OpinionQA/aggreated/opinionqa_test.parquet", "score_range"),
    ("OpinionQA",   "individual"): ("dataset/OpinionQA/individual/individual_test.parquet", "split_range"),
    ("Twin-2K-500", "aggreated"):  ("dataset/Twin-2K-500/aggreated/twin_test.parquet", "score_range"),
    ("Twin-2K-500", "individual"): ("dataset/Twin-2K-500/individual/individual_test.parquet", "split_range"),
}


def _score_ranges_for(cfg: str):
    """Return (lo_list, hi_list) per test row for <dataset>_<level> config."""
    for (ds, lvl), (path, mode) in _RANGE_SRC.items():
        if cfg == f"{ds}_{lvl}":
            src = pd.read_parquet(PROJ_ROOT / path)
            if mode == "score_range":
                lo = [float(r[0]) for r in src["score_range"]]
                hi = [float(r[-1]) for r in src["score_range"]]
            else:
                lo = src["score_range_min"].astype(float).tolist()
                hi = src["score_range_max"].astype(float).tolist()
            return lo, hi
    return None, None


def metrics_from_gen(parquet_path: Path, cfg: str) -> dict:
    """Compute MAE (continuous), MAD (debias-style, per-sample range), and
    hard-accuracy (round match) from an SFT gen_results parquet."""
    out = {"mae": None, "mad": None, "hard_acc": None, "n_parsed": 0, "n_total": 0}
    if not parquet_path.exists():
        return out
    df = pd.read_parquet(parquet_path)
    out["n_total"] = len(df)
    if "responses" not in df.columns or "ground_truth_float" not in df.columns:
        return out
    lo, hi = _score_ranges_for(cfg)
    if lo is None or len(lo) != len(df):
        return out
    tot_abs, tot_mad, hard, n = 0.0, 0.0, 0, 0
    for i, row in enumerate(df.itertuples(index=False)):
        resp = getattr(row, "responses")
        if hasattr(resp, "tolist"):
            resp = resp.tolist()
        text = resp[0] if isinstance(resp, list) and resp else ""
        pred = extract_answer(text)
        if pred is None:
            continue
        gt = float(getattr(row, "ground_truth_float"))
        rng = hi[i] - lo[i]
        if rng <= 0:
            rng = 1.0
        err = abs(float(pred) - gt)
        tot_abs += err
        tot_mad += 1.0 - err / rng  # debias MAD: unclipped, can go <0
        if round(float(pred)) == round(gt):
            hard += 1
        n += 1
    if n == 0:
        return out
    out["mae"] = tot_abs / n
    out["mad"] = tot_mad / n
    out["hard_acc"] = hard / n
    out["n_parsed"] = n
    return out


def main() -> None:
    summary = {}
    for cfg in CONFIGS:
        acc_path = RES_DIR / f"{cfg}.json"
        gen_path = GEN_DIR / f"{cfg}.parquet"
        acc = None
        if acc_path.exists():
            with open(acc_path) as f:
                acc = find_scalar(json.load(f))
        m = metrics_from_gen(gen_path, cfg)
        summary[cfg] = {
            "acc": acc,  # hard-accuracy from main_eval (round(pred)==round(gt))
            "mae": m["mae"],  # continuous MAE: |pred_int - gt_float|
            "mad": m["mad"],  # debias-style: mean(1 - |err|/per_sample_range)
            "hard_acc": m["hard_acc"],
            "parse_rate": (m["n_parsed"] / m["n_total"]) if m["n_total"] else None,
        }

    # Primary metric the user asked for: average acc across configs.
    accs = [v["acc"] for v in summary.values() if v["acc"] is not None]
    mads = [v["mad"] for v in summary.values() if v["mad"] is not None]
    overall = {
        "avg_acc": sum(accs) / len(accs) if accs else None,
        "avg_mad": sum(mads) / len(mads) if mads else None,
        "per_config": summary,
    }
    out_path = RES_DIR / "sft.json"
    with open(out_path, "w") as f:
        json.dump(overall, f, indent=2)
    print(json.dumps(overall, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
