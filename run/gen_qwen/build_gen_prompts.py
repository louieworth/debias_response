"""Build generation-input parquets for open-source LLM sampling.

For each (dataset, level) pair, produce a parquet that verl's
`main_generation_server` can consume. Each row contains:

  messages:         [{"role": "system", ...}, {"role": "user", ...}]
  enable_thinking:  False           # disable Qwen3 thinking mode
  Variable_Name:    <str>           # question id
  twin_id:          <int>           # persona id (or -1 for no-demographic rows)
  persona_idx:      <int>           # 0..K-1 index into the fixed persona pool
  score_range_min:  <int>
  score_range_max:  <int>
  data_source:      "<dataset>_<level>_qwenX"

Aggregated: for each question, prompt each of K fixed personas once.
           K = min(50, n_personas) — EEDI falls back to 13.
Individual: for each (question, twin_id) pair in train+test, prompt once.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd

PROJ_ROOT = Path("/home/jiangli/debias_response")
DATASET_ROOT = PROJ_ROOT / "dataset"
OUT_ROOT = DATASET_ROOT / "gen_prompts"

SYSTEM_PROMPT = (
    "You are an expert response model. Output ONLY a JSON object of the form "
    '{"answer": <integer>} with no extra text.'
)

USER_IND_TEMPLATE = (
    "Background:\n{demographic}\n\n"
    "Question: {question}\n"
    "Valid answer range: integers from {lo} to {hi}.\n"
    'Respond with JSON: {{"answer": <integer>}}'
)


def load_questions(dataset_dir: Path) -> list[dict]:
    with open(dataset_dir / "questions.json") as f:
        return json.load(f)


def load_personas(dataset_dir: Path) -> list[dict]:
    with open(dataset_dir / "personas.json") as f:
        return json.load(f)


def pick_fixed_personas(personas: list[dict], k: int, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    if len(personas) <= k:
        return list(personas)
    return rng.sample(personas, k)


def score_range_from_question(q: dict) -> tuple[int, int]:
    rng = q["score_range"]
    if isinstance(rng, list) and rng and isinstance(rng[0], str):
        return 1, len(rng)  # OpinionQA: list of option strings
    return int(rng[0]), int(rng[-1])


def build_aggregated(dataset_dir: Path, dataset_name: str, k: int, seed: int) -> pd.DataFrame:
    """Pull questions + score_range directly from the aggregated parquet.
    Twin-2K-500 uses a different Variable_Name convention in its aggregated
    parquet (human-readable scale names) vs questions.json (QID-style), so
    reading from the parquet is the only way to match inject-time lookups.
    """
    agg_stem, agg_dir, _, _ = CONFIGS[dataset_name]
    agg_train = dataset_dir / agg_dir / f"{agg_stem}_train.parquet"
    agg_test = dataset_dir / agg_dir / f"{agg_stem}_test.parquet"
    dfs = [pd.read_parquet(p, columns=["Variable_Name", "Question", "score_range"])
           for p in (agg_train, agg_test) if p.exists()]
    q_df = pd.concat(dfs, ignore_index=True).drop_duplicates("Variable_Name")

    personas = load_personas(dataset_dir)
    picked = pick_fixed_personas(personas, k, seed=seed)
    print(f"[{dataset_name}:agg] {len(q_df)} questions × {len(picked)} personas "
          f"= {len(q_df) * len(picked)} prompts")

    rows = []
    for _, q in q_df.iterrows():
        lo, hi = score_range_from_question({"score_range": q["score_range"]})
        for idx, p in enumerate(picked):
            user = USER_IND_TEMPLATE.format(
                demographic=p["Demographic"], question=q["Question"], lo=lo, hi=hi
            )
            rows.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                ],
                "enable_thinking": False,
                "Variable_Name": q["Variable_Name"],
                "twin_id": int(p["twin_id"]),
                "persona_idx": idx,
                "score_range_min": lo,
                "score_range_max": hi,
                "data_source": f"{dataset_name}_aggreated_qwen",
            })
    return pd.DataFrame(rows)


def build_individual(dataset_dir: Path, dataset_name: str, level_dir: Path, stem: str) -> pd.DataFrame:
    personas = {p["twin_id"]: p["Demographic"] for p in load_personas(dataset_dir)}
    qs_by_var = {q["Variable_Name"]: q for q in load_questions(dataset_dir)}
    rows = []
    for split in ("train", "test"):
        src = level_dir / f"{stem}_{split}.parquet"
        if not src.exists():
            continue
        df = pd.read_parquet(src, columns=[
            "Variable_Name", "twin_id", "score_range_min", "score_range_max"
        ])
        for _, r in df.iterrows():
            q = qs_by_var[r["Variable_Name"]]
            demo = personas[int(r["twin_id"])]
            lo, hi = int(r["score_range_min"]), int(r["score_range_max"])
            user = USER_IND_TEMPLATE.format(
                demographic=demo, question=q["Question"], lo=lo, hi=hi
            )
            rows.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                ],
                "enable_thinking": False,
                "Variable_Name": r["Variable_Name"],
                "twin_id": int(r["twin_id"]),
                "persona_idx": -1,
                "score_range_min": lo,
                "score_range_max": hi,
                "split": split,
                "data_source": f"{dataset_name}_individual_qwen",
            })
    print(f"[{dataset_name}:ind] {len(rows)} prompts (train+test)")
    return pd.DataFrame(rows)


CONFIGS = {
    "EEDI":        ("eedi",      "aggreated",  "individual", "individual"),
    "OpinionQA":   ("opinionqa", "aggreated",  "individual", "individual"),
    "Twin-2K-500": ("twin",      "aggreated", "individual", "individual"),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(CONFIGS), required=True)
    ap.add_argument("--level", choices=["aggreated", "individual"], required=True)
    ap.add_argument("--k_personas", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    agg_stem, agg_dir, ind_dir, ind_stem = CONFIGS[args.dataset]
    dataset_dir = DATASET_ROOT / args.dataset

    if args.level == "aggreated":
        df = build_aggregated(dataset_dir, args.dataset, args.k_personas, args.seed)
    else:
        level_path = dataset_dir / ind_dir
        df = build_individual(dataset_dir, args.dataset, level_path, ind_stem)

    out_dir = OUT_ROOT / f"{args.dataset}_{args.level}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "prompts.parquet"
    df.to_parquet(out_path, index=False)
    print(f"wrote {len(df)} rows -> {out_path}")


if __name__ == "__main__":
    main()
