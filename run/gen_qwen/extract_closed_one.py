"""Extract the single closed-source "strong" LLM response per (question, persona)
and add it as a scalar column to individual parquets so we can run
`--variant x_one_llm --llm_field gpt-4o_norm` cleanly.

Mapping (verified via Pearson-correlation matching against per-LLM files):
  EEDI       individual LLM_Responses[3] = gpt-4o                (r=0.994)
  OpinionQA  individual LLM_Responses[3] = gpt-4o                (r=0.970)
  Twin-2K-500 individual LLM_Responses[2] = GPT4.1 (no gpt-4o    (r=1.000)
              in Twin's closed pool; GPT4.1 is the flagship-class
              substitute chosen as baseline.)

We write two columns:
  <NAME>         raw-scale (score_range_min..score_range_max)
  <NAME>_norm    0-1 normalized (identical to LLM_Responses[i] already)
Column name is `gpt-4o` / `gpt-4o_norm` uniformly across all 3 datasets so
the debias CLI can be dataset-agnostic. For Twin this is actually GPT4.1;
documented here and in the comparison summary.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

PROJ_ROOT = Path("/home/jiangli/debias_response")

CONFIGS = [
    # (dataset, individual-parquet-stem, LLM_Responses slot index, actual identity)
    ("EEDI",        "individual", 3, "gpt-4o"),
    ("OpinionQA",   "individual", 3, "gpt-4o"),
    ("Twin-2K-500", "individual", 2, "GPT4.1"),
]
COL_NORM = "gpt-4o_norm"
COL_RAW = "gpt-4o"


def main() -> None:
    for ds, _, idx, identity in CONFIGS:
        for split in ("train", "test"):
            pq = PROJ_ROOT / "dataset" / ds / "individual" / f"individual_{split}.parquet"
            if not pq.exists():
                continue
            df = pd.read_parquet(pq)
            lo = df["score_range_min"].astype(float).values
            hi = df["score_range_max"].astype(float).values
            span = np.where(hi - lo > 0, hi - lo, 1.0)
            # EEDI/OpQA: `LLM_Responses` is already 0-1 normalized (float avg).
            # Twin    : `LLM_Responses` is raw 1-7; use `LLM_Responses_norm` for norm.
            if "LLM_Responses_norm" in df.columns:
                norms = np.fromiter(
                    (np.asarray(v, dtype=float)[idx] for v in df["LLM_Responses_norm"]),
                    dtype=float, count=len(df),
                )
                raws = np.fromiter(
                    (np.asarray(v, dtype=float)[idx] for v in df["LLM_Responses"]),
                    dtype=float, count=len(df),
                )
            else:
                norms = np.fromiter(
                    (np.asarray(v, dtype=float)[idx] for v in df["LLM_Responses"]),
                    dtype=float, count=len(df),
                )
                raws = norms * span + lo
            df[COL_NORM] = norms
            df[COL_RAW] = raws
            df.to_parquet(pq, index=False)
            print(f"[{ds}:{split}] {pq} rows={len(df)} idx={idx} ({identity}) "
                  f"norm_range=[{norms.min():.3f},{norms.max():.3f}]")


if __name__ == "__main__":
    main()
