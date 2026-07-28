"""Parse open-source generation results and inject them back into the
aggregated / individual parquets so the existing debias pipeline can
consume them as just another LLM feature.

Aggregated: for each Variable_Name, collect responses across personas
ordered by persona_idx → save as list-valued column `<MODEL_TAG>` (and
normalized `<MODEL_TAG>_norm`) in the aggregated train/test parquets,
and dump a JSON file mirroring closed-source `<model>_converted.json`.

Individual: for each (Variable_Name, twin_id) join, add scalar column
`<MODEL_TAG>` (raw integer) and `<MODEL_TAG>_norm` (0-1) to the
individual train/test parquets.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJ_ROOT = Path("/home/jiangli/debias_response")
sys.path.insert(0, str(PROJ_ROOT))
from run.sft.compute_score import extract_answer  # noqa: E402


CFG_FILES = {
    "EEDI":        ("eedi",      "aggreated",  "individual"),
    "OpinionQA":   ("opinionqa", "aggreated",  "individual"),
    "Twin-2K-500": ("twin",      "aggreated", "individual"),
}
IND_STEM = "individual"


def _first_response_text(resp) -> str:
    if hasattr(resp, "tolist"):
        resp = resp.tolist()
    if isinstance(resp, list) and resp:
        return resp[0]
    return resp or ""


def _norm(val: float, lo: int, hi: int) -> float:
    span = max(hi - lo, 1)
    return float((val - lo) / span)


def inject_aggregated(dataset: str, model_tag: str, gen_parquet: Path) -> None:
    stem, agg_dir, _ = CFG_FILES[dataset]
    gen = pd.read_parquet(gen_parquet)
    lo = int(gen["score_range_min"].iloc[0])
    hi = int(gen["score_range_max"].iloc[0])

    # Extract predicted integer per row; fall back to midpoint if unparseable.
    mid = (lo + hi) // 2
    preds = []
    for _, r in gen.iterrows():
        text = _first_response_text(r["responses"])
        p = extract_answer(text)
        if p is None:
            p = mid
        p = int(round(float(p)))
        p = max(lo, min(hi, p))
        preds.append(p)
    gen = gen.assign(pred=preds)

    # Sort to guarantee persona_idx order per question.
    gen = gen.sort_values(["Variable_Name", "persona_idx"])
    grouped = gen.groupby("Variable_Name")["pred"].apply(list).to_dict()

    # JSON dump for parity with closed-source LLM_responses/.
    out_json = PROJ_ROOT / "dataset" / dataset / "LLM_responses" / f"{model_tag}_agg_converted.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump([{"Variable_Name": k, "Responses": v} for k, v in grouped.items()], f)
    print(f"[{dataset}:agg:{model_tag}] wrote JSON -> {out_json} ({len(grouped)} questions)")

    # Inject into aggregated train/test parquets.
    agg_path = PROJ_ROOT / "dataset" / dataset / agg_dir
    for split in ("train", "test"):
        pq = agg_path / f"{stem}_{split}.parquet"
        if not pq.exists():
            continue
        df = pd.read_parquet(pq)
        raws, norms = [], []
        for var in df["Variable_Name"]:
            vec = grouped.get(var, [])
            raws.append(np.array(vec, dtype=float))
            norms.append(np.array([_norm(v, lo, hi) for v in vec], dtype=float))
        df[model_tag] = raws
        df[f"{model_tag}_norm"] = norms
        df.to_parquet(pq, index=False)
        print(f"[{dataset}:agg:{model_tag}] injected into {pq} (rows={len(df)}, vec_len={len(raws[0]) if raws else 0})")


def inject_individual(dataset: str, model_tag: str, gen_parquet: Path) -> None:
    stem, _, ind_dir = CFG_FILES[dataset]
    gen = pd.read_parquet(gen_parquet)
    mid_default = lambda lo, hi: (lo + hi) // 2

    preds = []
    for _, r in gen.iterrows():
        text = _first_response_text(r["responses"])
        p = extract_answer(text)
        lo = int(r["score_range_min"])
        hi = int(r["score_range_max"])
        if p is None:
            p = mid_default(lo, hi)
        p = int(round(float(p)))
        p = max(lo, min(hi, p))
        preds.append(p)
    gen = gen.assign(pred=preds)

    # Key by (Variable_Name, twin_id).
    lookup = {(r["Variable_Name"], int(r["twin_id"])): int(r["pred"]) for _, r in gen.iterrows()}
    ind_path = PROJ_ROOT / "dataset" / dataset / ind_dir

    for split in ("train", "test"):
        pq = ind_path / f"{IND_STEM}_{split}.parquet"
        if not pq.exists():
            continue
        df = pd.read_parquet(pq)
        raws, norms = [], []
        missing = 0
        for _, r in df.iterrows():
            key = (r["Variable_Name"], int(r["twin_id"]))
            lo = int(r["score_range_min"])
            hi = int(r["score_range_max"])
            if key in lookup:
                raw = lookup[key]
            else:
                raw = mid_default(lo, hi)
                missing += 1
            raws.append(raw)
            norms.append(_norm(raw, lo, hi))
        df[model_tag] = raws
        df[f"{model_tag}_norm"] = norms
        df.to_parquet(pq, index=False)
        print(f"[{dataset}:ind:{model_tag}] {pq} rows={len(df)} missing_lookup={missing}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(CFG_FILES), required=True)
    ap.add_argument("--level", choices=["aggreated", "individual"], required=True)
    ap.add_argument("--model_tag", required=True, help="e.g. qwen3-4B")
    ap.add_argument("--gen_parquet", required=True)
    args = ap.parse_args()
    gen = Path(args.gen_parquet)
    if not gen.exists():
        raise SystemExit(f"gen parquet not found: {gen}")
    if args.level == "aggreated":
        inject_aggregated(args.dataset, args.model_tag, gen)
    else:
        inject_individual(args.dataset, args.model_tag, gen)


if __name__ == "__main__":
    main()
