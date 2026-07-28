#!/usr/bin/env python3
"""Build group-level scale ablation parquet files from filtered JSON data.

The output is intentionally separate from the existing ``aggreated`` datasets:

    dataset/<dataset>/aggreated_ablation/<prefix>_train.parquet
    dataset/<dataset>/aggreated_ablation/<prefix>_test.parquet
    dataset/<dataset>/aggreated_ablation/<prefix>_metadata.json

For each dataset, the builder chooses eight personas with complete question
coverage and creates three families of LLM input fields:

* ``persona_k_norm``: first k sampled personas for one anchor LLM.
* ``llm_k_norm``: first k LLMs for one sampled persona.
* ``all_64_norm``: all 8 sampled personas x all 8 LLMs, flattened
  persona-major then LLM-order.

These 8 + 8 + 1 fields support the requested 17 scale settings. Each field is
a fixed-length numeric list, so ``x_one_llm``, ``x_avg_llm``, and ``x_all_llm``
can reuse ``debias.debias_variants`` without special-case training code.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


DATASET_CONFIG = {
    "EEDI": {
        "root": "dataset/EEDI",
        "prefix": "eedi",
        "question_key": "Variable_Name",
        "anchor_llm": "gpt-4o",
    },
    "OpinionQA": {
        "root": "dataset/OpinionQA",
        "prefix": "opinionqa",
        "question_key": "Variable_Name",
        "anchor_llm": "gpt-4o",
    },
    "Twin": {
        "root": "dataset/Twin-2K-500",
        "prefix": "twin",
        "question_key": "question_id",
        "anchor_llm": "gpt-4o",
    },
}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def as_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(v):
        return None
    return v


def normalize_value(value: Any, score_range: Sequence[Any]) -> Optional[float]:
    v = as_float(value)
    if v is None or len(score_range) != 2:
        return None
    lo = as_float(score_range[0])
    hi = as_float(score_range[1])
    if lo is None or hi is None or np.isclose(lo, hi):
        return None
    if lo > hi:
        lo, hi = hi, lo
    return (v - lo) / (hi - lo)


def question_key_from_response(row: Dict[str, Any], preferred_key: str) -> Optional[str]:
    value = row.get(preferred_key)
    if isinstance(value, str) and value:
        return value
    value = row.get("Variable_Name")
    if isinstance(value, str) and value:
        return value
    value = row.get("question_id")
    if isinstance(value, str) and value:
        return value
    return None


def index_questions(questions: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    indexed = {}
    for q in questions:
        for key in ("Variable_Name", "question_id"):
            value = q.get(key)
            if isinstance(value, str) and value and value not in indexed:
                indexed[value] = q
    return indexed


def load_llm_order(dataset_root: Path, filtered_metadata: Dict[str, Any]) -> List[str]:
    order = filtered_metadata.get("llm_response_order")
    if order:
        return list(order)

    raw_metadata = dataset_root / "raw" / "responses_llm_metadata.json"
    if raw_metadata.exists():
        metadata = load_json(raw_metadata)
        canonical = metadata.get("canonical_order", [])
        order = [item["model"] for item in sorted(canonical, key=lambda x: x["index"])]
        if order:
            return order

    raise RuntimeError(
        f"Could not infer LLM response order for {dataset_root}. "
        "Expected filtered metadata llm_response_order or raw/responses_llm_metadata.json."
    )


def stable_persona_sort_key(value: Any) -> Tuple[int, Any]:
    if isinstance(value, (int, np.integer)):
        return (0, int(value))
    if isinstance(value, float) and value.is_integer():
        return (0, int(value))
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.lstrip("-").isdigit():
            return (0, int(stripped))
        return (1, stripped)
    return (1, str(value))


def choose_personas(
    personas: List[Dict[str, Any]],
    llm_feature_responses: List[Dict[str, Any]],
    questions_by_key: Dict[str, Dict[str, Any]],
    question_key: str,
    n: int,
    seed: int,
    preferred_ids: Optional[List[Any]] = None,
) -> List[Any]:
    ids = [p.get("twin_id") for p in personas if p.get("twin_id") is not None]
    ids = sorted(set(ids), key=stable_persona_sort_key)

    all_questions = set(questions_by_key.keys())
    persona_questions: Dict[Any, set] = {pid: set() for pid in ids}
    for row in llm_feature_responses:
        pid = row.get("twin_id")
        if pid not in persona_questions:
            continue
        llm_response = row.get("LLM_Response")
        if not isinstance(llm_response, list):
            continue
        qkey = question_key_from_response(row, question_key)
        if qkey in all_questions:
            persona_questions[pid].add(qkey)

    if preferred_ids is not None:
        preferred_ids = list(preferred_ids)
        missing_ids = [pid for pid in preferred_ids if pid not in persona_questions]
        incomplete_ids = [
            pid for pid in preferred_ids
            if pid in persona_questions and not all_questions.issubset(persona_questions[pid])
        ]
        if missing_ids or incomplete_ids or len(preferred_ids) < n:
            raise RuntimeError(
                f"Preferred persona ids are not a complete no-fill candidate pool. "
                f"missing_ids={missing_ids}, incomplete_ids={incomplete_ids}, "
                f"expected_at_least_n={n}, actual_n={len(preferred_ids)}"
            )
        if len(preferred_ids) == n:
            return preferred_ids
        rng = np.random.default_rng(seed)
        idx = sorted(rng.choice(len(preferred_ids), size=n, replace=False).tolist())
        return [preferred_ids[i] for i in idx]

    complete_ids = [
        pid for pid in ids
        if all_questions.issubset(persona_questions.get(pid, set()))
    ]
    if len(complete_ids) < n:
        top = sorted(
            ((len(persona_questions.get(pid, set())), pid) for pid in ids),
            reverse=True,
        )[:10]
        raise RuntimeError(
            f"Need at least {n} personas with complete coverage over "
            f"{len(all_questions)} questions, found {len(complete_ids)}. "
            f"Top partial coverages: {top}"
        )

    ids = complete_ids
    rng = np.random.default_rng(seed)
    idx = sorted(rng.choice(len(ids), size=n, replace=False).tolist())
    return [ids[i] for i in idx]


def build_dataframe(
    dataset_name: str,
    dataset_root: Path,
    prefix: str,
    seed: int,
    n_personas: int,
    test_size: float,
    anchor_llm_requested: str,
    persona_seed: Optional[int] = None,
    split_seed: Optional[int] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    persona_seed = seed if persona_seed is None else persona_seed
    split_seed = seed if split_seed is None else split_seed
    filtered_root = dataset_root / "filtered"
    questions = load_json(filtered_root / "questions.json")
    personas = load_json(filtered_root / "personas.json")
    responses = load_json(filtered_root / "responses.json")
    metadata = load_json(filtered_root / "metadata.json")
    sidecar_rows: List[Dict[str, Any]] = []
    sidecar_metadata: Dict[str, Any] = {}
    preferred_persona_ids: Optional[List[Any]] = None

    if dataset_name == "OpinionQA":
        sidecar_path = filtered_root / "llm_backfill_responses.json"
        sidecar_metadata_path = filtered_root / "llm_backfill" / "sidecar_metadata.json"
        if not sidecar_path.exists():
            raise RuntimeError(
                f"OpinionQA requires {sidecar_path} for no-fill group ablation LLM features."
            )
        sidecar_rows = load_json(sidecar_path)
        if sidecar_metadata_path.exists():
            sidecar_metadata = load_json(sidecar_metadata_path)
            preferred_persona_ids = sidecar_metadata.get("selected_persona_ids")

    question_key = metadata.get("question_key", {}).get("responses")
    if not question_key:
        question_key = DATASET_CONFIG[dataset_name]["question_key"]

    llm_order = load_llm_order(dataset_root, metadata)
    if len(llm_order) != 8:
        raise RuntimeError(f"Expected 8 LLMs for {dataset_name}, found {len(llm_order)}: {llm_order}")
    if sidecar_metadata:
        sidecar_order = sidecar_metadata.get("sidecar_llm_order") or sidecar_metadata.get("llm_order")
        if sidecar_order and list(sidecar_order) != list(llm_order):
            raise RuntimeError(
                f"OpinionQA sidecar LLM order does not match filtered/raw order. "
                f"sidecar={sidecar_order}, expected={llm_order}"
            )
    for row in sidecar_rows:
        if "Human_Response" in row:
            raise RuntimeError("OpinionQA LLM sidecar must not contain Human_Response.")

    llm_feature_responses = responses + sidecar_rows

    if anchor_llm_requested in llm_order:
        anchor_llm = anchor_llm_requested
    else:
        anchor_llm = llm_order[0]
        print(
            f"[WARN] {dataset_name}: requested anchor {anchor_llm_requested!r} not in "
            f"filtered LLM order. Falling back to {anchor_llm!r}."
        )
    anchor_idx = llm_order.index(anchor_llm)

    questions_by_key = index_questions(questions)
    sampled_personas = choose_personas(
        personas=personas,
        llm_feature_responses=llm_feature_responses,
        questions_by_key=questions_by_key,
        question_key=question_key,
        n=n_personas,
        seed=persona_seed,
        preferred_ids=preferred_persona_ids,
    )
    sampled_set = set(sampled_personas)

    grouped: Dict[str, Dict[str, Any]] = {}
    for row in responses:
        qkey = question_key_from_response(row, question_key)
        if qkey is None or qkey not in questions_by_key:
            continue
        human = as_float(row.get("Human_Response"))
        if human is None:
            continue
        llm_response = row.get("LLM_Response")
        if not isinstance(llm_response, list) or len(llm_response) != len(llm_order):
            continue

        q = questions_by_key[qkey]
        sr = q.get("score_range", [1, 5])
        human_norm = normalize_value(human, sr)
        if human_norm is None:
            continue

        if qkey not in grouped:
            grouped[qkey] = {
                "human_raw": [],
                "human_norm": [],
                "selected_llm_norm": {
                    pid: [None] * len(llm_order) for pid in sampled_personas
                },
            }
        grouped[qkey]["human_raw"].append(human)
        grouped[qkey]["human_norm"].append(human_norm)

    missing_target_questions = sorted(set(questions_by_key) - set(grouped))
    if missing_target_questions:
        raise RuntimeError(
            f"{dataset_name} has {len(missing_target_questions)} questions without real human targets. "
            f"Examples: {missing_target_questions[:10]}"
        )

    for row in llm_feature_responses:
        qkey = question_key_from_response(row, question_key)
        if qkey is None or qkey not in grouped:
            continue
        llm_response = row.get("LLM_Response")
        if not isinstance(llm_response, list) or len(llm_response) != len(llm_order):
            continue
        pid = row.get("twin_id")
        if pid in sampled_set:
            q = questions_by_key[qkey]
            sr = q.get("score_range", [1, 5])
            grouped[qkey]["selected_llm_norm"][pid] = [
                normalize_value(value, sr) for value in llm_response
            ]

    all_rows: List[Dict[str, Any]] = []
    for qkey in sorted(grouped.keys()):
        q = questions_by_key[qkey]
        sr = q.get("score_range", [1, 5])
        matrix = np.empty((n_personas, len(llm_order)), dtype=float)
        matrix[:] = np.nan
        for p_idx, pid in enumerate(sampled_personas):
            vals = grouped[qkey]["selected_llm_norm"][pid]
            for l_idx, value in enumerate(vals):
                if value is not None:
                    matrix[p_idx, l_idx] = float(value)

        if np.isnan(matrix).any():
            missing_positions = np.argwhere(np.isnan(matrix)).tolist()
            examples = [
                {
                    "persona": sampled_personas[p_idx],
                    "llm": llm_order[l_idx],
                }
                for p_idx, l_idx in missing_positions[:10]
            ]
            raise RuntimeError(
                f"{dataset_name} qkey={qkey} has missing selected LLM values. "
                f"No filling is allowed. Examples: {examples}"
            )

        row_out: Dict[str, Any] = {
            "Variable_Name": q.get("Variable_Name", qkey),
            "Question": q.get("Question"),
            "score_range": list(sr),
            "Average_Human_Response": float(np.mean(grouped[qkey]["human_raw"])),
            "Average_Human_Response_norm": float(np.mean(grouped[qkey]["human_norm"])),
            "Question_Embedding": q.get("Question_Embedding"),
            "persona_ids": list(sampled_personas),
            "llm_order": list(llm_order),
            "anchor_llm": anchor_llm,
            "Average_LLM_Response_norm": float(np.mean(matrix)),
        }

        anchor_values = matrix[:, anchor_idx].tolist()
        for k in range(1, n_personas + 1):
            values = anchor_values[:k]
            row_out[f"persona_{k}"] = values
            row_out[f"persona_{k}_norm"] = values

        anchor_persona_values = matrix[0, :].tolist()
        for l_idx in range(len(llm_order)):
            k = l_idx + 1
            values = anchor_persona_values[:k]
            row_out[f"llm_{k}"] = values
            row_out[f"llm_{k}_norm"] = values

        all_values = matrix.reshape(-1).tolist()
        row_out["all_64"] = all_values
        row_out["all_64_norm"] = all_values

        all_rows.append(row_out)

    df = pd.DataFrame(all_rows)
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=split_seed, shuffle=True)
    split_marker = pd.Series("train", index=train_df.index)
    split_marker = pd.concat([split_marker, pd.Series("test", index=test_df.index)])
    df = pd.concat([train_df, test_df], axis=0)
    df["split"] = split_marker.loc[df.index].values
    df = df.reset_index(drop=True)

    out_metadata = {
        "dataset": dataset_name,
        "source": str(filtered_root),
        "human_target_source": str(filtered_root / "responses.json"),
        "llm_feature_sources": [
            str(filtered_root / "responses.json"),
            *([str(filtered_root / "llm_backfill_responses.json")] if sidecar_rows else []),
        ],
        "sidecar_rows_used_for_llm_features_only": len(sidecar_rows),
        "sidecar_human_response_policy": (
            "ignored/not present; group labels are computed only from real filtered responses"
            if sidecar_rows else None
        ),
        "seed": seed,
        "persona_seed": persona_seed,
        "split_seed": split_seed,
        "test_size": test_size,
        "n_personas": n_personas,
        "sampled_personas": sampled_personas,
        "llm_scale_persona": sampled_personas[0],
        "llm_order": llm_order,
        "anchor_llm_requested": anchor_llm_requested,
        "anchor_llm_used": anchor_llm,
        "fields": {
            "persona_scale": [f"persona_{k}_norm" for k in range(1, n_personas + 1)],
            "llm_scale": [f"llm_{k}_norm" for k in range(1, len(llm_order) + 1)],
            "all_scale": ["all_64_norm"],
        },
        "all_scale_dim": n_personas * len(llm_order),
        "all_scale_shape": [n_personas, len(llm_order)],
        "all_scale_flatten_order": "persona_major_then_llm_order",
        "fill_policy": "none",
        "imputed_missing_selected_persona_llm_values": {},
        "rows_total": int(len(df)),
        "rows_train": int((df["split"] == "train").sum()),
        "rows_test": int((df["split"] == "test").sum()),
    }
    return df, out_metadata


def write_outputs(
    df: pd.DataFrame,
    metadata: Dict[str, Any],
    dataset_root: Path,
    prefix: str,
    output_suffix: Optional[str] = None,
) -> None:
    out_dir = dataset_root / "aggreated_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = prefix if not output_suffix else f"{prefix}_{output_suffix}"
    metadata = dict(metadata)
    metadata["output_suffix"] = output_suffix
    train_path = out_dir / f"{output_prefix}_train.parquet"
    test_path = out_dir / f"{output_prefix}_test.parquet"
    metadata_path = out_dir / f"{output_prefix}_metadata.json"

    train_df = df[df["split"] == "train"].drop(columns=["split"])
    test_df = df[df["split"] == "test"].drop(columns=["split"])
    train_df.to_parquet(train_path, index=False, compression="snappy")
    test_df.to_parquet(test_path, index=False, compression="snappy")
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"wrote {train_path} rows={len(train_df)} cols={len(train_df.columns)}")
    print(f"wrote {test_path} rows={len(test_df)} cols={len(test_df.columns)}")
    print(f"wrote {metadata_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["EEDI", "OpinionQA", "Twin"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_personas", type=int, default=8)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--anchor_llm", type=str, default="gpt-4o")
    parser.add_argument("--persona_seed", type=int, default=None)
    parser.add_argument("--split_seed", type=int, default=None)
    parser.add_argument("--output_suffix", type=str, default=None)
    args = parser.parse_args()

    for dataset_name in args.datasets:
        if dataset_name not in DATASET_CONFIG:
            raise ValueError(f"Unknown dataset {dataset_name}. Choose from {sorted(DATASET_CONFIG)}")
        cfg = DATASET_CONFIG[dataset_name]
        dataset_root = Path(cfg["root"])
        df, metadata = build_dataframe(
            dataset_name=dataset_name,
            dataset_root=dataset_root,
            prefix=cfg["prefix"],
            seed=args.seed,
            n_personas=args.n_personas,
            test_size=args.test_size,
            anchor_llm_requested=args.anchor_llm,
            persona_seed=args.persona_seed,
            split_seed=args.split_seed,
        )
        write_outputs(df, metadata, dataset_root, cfg["prefix"], args.output_suffix)


if __name__ == "__main__":
    main()
