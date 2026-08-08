#!/usr/bin/env python3
"""Create deterministic 200-wide bootstrap extensions for population ablations.

The original 50 gpt-4o draws are retained as the first 50 coordinates. The
remaining 150 coordinates are sampled with replacement from those same 50
values using a fixed data-construction seed. The derived inputs are therefore
nested at K=50, 100, and 200 and are shared across all optimizer seeds.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


BOOTSTRAP_SEED = 20260807
TARGET_K = 200
FIELD = "gpt-4o_norm"
OUTPUT_ROOT = Path(
    "results/population_k_ablation_k1_8_25_50_100_200_seed0_4_precision17/bootstrap_inputs"
)
SOURCES = {
    "Twin-2K-500": {
        "train": Path("dataset/Twin-2K-500/aggreated/twin_train.parquet"),
        "test": Path("dataset/Twin-2K-500/aggreated/twin_test.parquet"),
    },
    "OpinionQA": {
        "train": Path("dataset/OpinionQA/aggreated/opinionqa_train.parquet"),
        "test": Path("dataset/OpinionQA/aggreated/opinionqa_test.parquet"),
    },
    "EEDI": {
        "train": Path("dataset/EEDI/aggreated/eedi_train.parquet"),
        "test": Path("dataset/EEDI/aggreated/eedi_test.parquet"),
    },
}


def extend_frame(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    result = frame.copy()
    extended = []
    for value in result[FIELD]:
        original = np.asarray(value, dtype=float)
        if original.shape != (50,):
            raise ValueError(f"Expected exactly 50 {FIELD} draws, found {original.shape}")
        extra = rng.choice(original, size=TARGET_K - len(original), replace=True)
        extended.append(np.concatenate([original, extra]).tolist())
    result[FIELD] = extended
    lengths = result[FIELD].map(len)
    if not lengths.eq(TARGET_K).all():
        raise ValueError("Bootstrap extension did not produce width 200")
    return result


def main() -> int:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "bootstrap_seed": BOOTSTRAP_SEED,
        "field": FIELD,
        "original_k": 50,
        "target_k": TARGET_K,
        "construction": (
            "retain the original 50 coordinates, then append 150 draws sampled "
            "with replacement from those 50 coordinates"
        ),
        "files": [],
    }
    for dataset_index, (dataset, splits) in enumerate(SOURCES.items()):
        for split_index, (split, source_path) in enumerate(splits.items()):
            construction_seed = BOOTSTRAP_SEED + dataset_index * 10 + split_index
            frame = pd.read_parquet(source_path)
            output = extend_frame(frame, np.random.default_rng(construction_seed))
            output_path = OUTPUT_ROOT / f"{dataset}_{split}_gpt4o_bootstrap_k200.parquet"
            output.to_parquet(output_path, index=False)
            manifest["files"].append(
                {
                    "dataset": dataset,
                    "split": split,
                    "source": str(source_path),
                    "output": str(output_path),
                    "rows": len(output),
                    "construction_seed": construction_seed,
                }
            )
            print(f"[WRITE] {output_path} ({len(output)} rows)")
    manifest_path = OUTPUT_ROOT / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[WRITE] {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
