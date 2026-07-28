"""Build 6 SFT datasets from existing parquet files + personas.json.

For each (dataset, level) pair, read the train/test parquet, join with
personas.json (only for individual level — to recover `Demographic` text
from `twin_id`), render a chat-format prompt and write a new parquet with
the columns verl.trainer.sft_trainer expects:

  - messages:        list[dict] with at least one user turn (and an
                     assistant turn for train only).
  - enable_thinking: False  (disable Qwen3 thinking mode per-row).
  - ground_truth:    int target (rounded for aggregated, raw for individual).
  - data_source:     "<dataset>_<level>" tag for verl.main_eval routing.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

PROJ_ROOT = Path("/home/jiangli/debias_response")
DATASET_ROOT = PROJ_ROOT / "dataset"
OUT_ROOT = DATASET_ROOT / "sft"

SYSTEM_PROMPT = (
    "You are an expert response model. Output ONLY a JSON object of the form "
    '{"answer": <integer>} with no extra text.'
)

USER_AGG_TEMPLATE = (
    "Question: {question}\n"
    "Valid answer range: integers from {lo} to {hi}.\n"
    'Respond with JSON: {{"answer": <integer>}}'
)

USER_IND_TEMPLATE = (
    "Background:\n{demographic}\n\n"
    "Question: {question}\n"
    "Valid answer range: integers from {lo} to {hi}.\n"
    'Respond with JSON: {{"answer": <integer>}}'
)

# (dataset_dir, level, file_stem, source_dir). File stems:
#   aggregated train/test use <dataset>_train.parquet etc.
#   individual train/test use individual_{split}.parquet.
CONFIGS = [
    ("EEDI",        "aggreated",  "eedi",      "aggreated"),
    ("EEDI",        "individual", "individual", "individual"),
    ("OpinionQA",   "aggreated",  "opinionqa", "aggreated"),
    ("OpinionQA",   "individual", "individual", "individual"),
    ("Twin-2K-500", "aggreated",  "twin",      "aggreated"),
    ("Twin-2K-500", "individual", "individual", "individual"),
]


def load_personas(dataset_dir: Path) -> dict:
    """Return a dict of twin_id -> Demographic string."""
    with open(dataset_dir / "personas.json") as f:
        data = json.load(f)
    return {p["twin_id"]: p["Demographic"] for p in data}


def score_range_for_aggregated(row) -> tuple[int, int]:
    rng = row["score_range"]
    # numpy array or list of ints.
    lo = int(rng[0])
    hi = int(rng[-1])
    return lo, hi


def score_range_for_individual(row) -> tuple[int, int]:
    return int(row["score_range_min"]), int(row["score_range_max"])


def _pack(messages, gt_int, gt_float, cfg_name):
    return {
        "messages": messages,
        "enable_thinking": False,
        "ground_truth": gt_int,
        "ground_truth_float": gt_float,
        "reward_model": {"ground_truth": gt_int},
        "data_source": cfg_name,
    }


def build_row_aggregated(row, cfg_name: str, split: str):
    lo, hi = score_range_for_aggregated(row)
    user = USER_AGG_TEMPLATE.format(question=row["Question"], lo=lo, hi=hi)
    gt_float = float(row["Average_Human_Response"])
    gt_int = int(round(gt_float))
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
    if split == "train":
        messages.append({"role": "assistant", "content": f'{{"answer": {gt_int}}}'})
    return _pack(messages, gt_int, gt_float, cfg_name)


def build_row_individual(row, personas: dict, cfg_name: str, split: str):
    lo, hi = score_range_for_individual(row)
    demo = personas.get(row["twin_id"], "")
    user = USER_IND_TEMPLATE.format(
        demographic=demo, question=row["Question"], lo=lo, hi=hi
    )
    gt_float = float(row["Human_Response"])
    gt_int = int(round(gt_float))
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
    if split == "train":
        messages.append({"role": "assistant", "content": f'{{"answer": {gt_int}}}'})
    return _pack(messages, gt_int, gt_float, cfg_name)


def build_one(dataset: str, level: str, stem: str, source_dir: str) -> None:
    dataset_dir = DATASET_ROOT / dataset
    level_dir = dataset_dir / source_dir
    cfg_name = f"{dataset}_{level}"
    out_dir = OUT_ROOT / cfg_name
    out_dir.mkdir(parents=True, exist_ok=True)

    personas = load_personas(dataset_dir) if level == "individual" else None

    for split in ("train", "test"):
        src = level_dir / f"{stem}_{split}.parquet"
        df = pd.read_parquet(src)
        rows = []
        for _, row in df.iterrows():
            if level == "aggreated":
                rows.append(build_row_aggregated(row, cfg_name, split))
            else:
                rows.append(build_row_individual(row, personas, cfg_name, split))
        out_df = pd.DataFrame(rows)
        out_path = out_dir / f"{split}.parquet"
        out_df.to_parquet(out_path, index=False)
        print(f"[{cfg_name}:{split}] {len(out_df)} rows -> {out_path}")


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    for dataset, level, stem, source_dir in CONFIGS:
        build_one(dataset, level, stem, source_dir)


if __name__ == "__main__":
    main()
