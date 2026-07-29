#!/usr/bin/env python3
"""Materialize the canonical 279-question Twin population dataset.

The source of truth is filtered_new/strict_likert.  The aggregated interface
intentionally keeps the historical 26-column schema consumed by the group
experiment runners while recording the richer source dataset and incomplete
LLM coverage in metadata.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
TWIN_ROOT = HERE.parent
SOURCE_DIR = TWIN_ROOT / "filtered_new" / "strict_likert"

SOURCE_ALL = SOURCE_DIR / "twin_all.parquet"
SOURCE_TRAIN = SOURCE_DIR / "twin_train.parquet"
SOURCE_TEST = SOURCE_DIR / "twin_test.parquet"
SOURCE_METADATA = SOURCE_DIR / "metadata.json"

OUTPUT_TRAIN = HERE / "twin_train.parquet"
OUTPUT_TEST = HERE / "twin_test.parquet"
OUTPUT_METADATA = HERE / "twin_metadata.json"

RAW_LLM_FIELDS = [
    "claude-3.5-haiku",
    "deepseek-v3",
    "gpt-3.5-turbo",
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-5-mini",
    "llama-3.3-70B-instruct-turbo",
    "mistral-7B-instruct-v0.3",
]
NORM_LLM_FIELDS = [f"{field}_norm" for field in RAW_LLM_FIELDS]
ONE_LOGPROB_FIELDS = [
    "one_logprobs",
    "one_logprobs_expected_response",
    "one_logprobs_expected_response_norm",
    "one_logprobs_choice_count",
]
CANONICAL_COLUMNS = [
    "Variable_Name",
    "Question",
    "score_range",
    "Average_Human_Response",
    "Average_Human_Response_norm",
    "Question_Embedding",
    *RAW_LLM_FIELDS,
    *NORM_LLM_FIELDS,
    *ONE_LOGPROB_FIELDS,
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def atomic_json(value: dict[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def vector_length(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    try:
        return len(value)
    except TypeError:
        return None


def validate_source(
    all_items: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> dict[str, Any]:
    missing_columns = sorted(set(CANONICAL_COLUMNS) - set(all_items.columns))
    if missing_columns:
        raise ValueError(f"Source is missing canonical columns: {missing_columns}")
    if len(all_items) != 279 or len(train) != 223 or len(test) != 56:
        raise ValueError(
            "Expected strict split rows all/train/test=279/223/56, got "
            f"{len(all_items)}/{len(train)}/{len(test)}"
        )

    all_ids = all_items["Variable_Name"].astype(str).tolist()
    train_ids = train["Variable_Name"].astype(str).tolist()
    test_ids = test["Variable_Name"].astype(str).tolist()
    if len(set(all_ids)) != 279:
        raise ValueError("Variable_Name must be unique across 279 questions")
    if set(train_ids) & set(test_ids):
        raise ValueError("Train/test question IDs overlap")
    if set(train_ids) | set(test_ids) != set(all_ids):
        raise ValueError("Train/test IDs do not exactly partition twin_all")

    embedding_lengths = {
        vector_length(value) for value in all_items["Question_Embedding"]
    }
    if embedding_lengths != {256}:
        raise ValueError(
            f"Expected 256-dimensional embeddings, got {embedding_lengths}"
        )

    coverage: dict[str, int] = {}
    valid_vector_coverage: dict[str, int] = {}
    pending_ids: dict[str, list[str]] = {}
    invalid_lengths: dict[str, list[dict[str, Any]]] = {}
    for field in RAW_LLM_FIELDS:
        coverage[field] = int(all_items[field].notna().sum())
        pending_ids[field] = (
            all_items.loc[all_items[field].isna(), "Variable_Name"]
            .astype(str)
            .tolist()
        )
        bad = []
        valid = 0
        for _, row in all_items.loc[all_items[field].notna()].iterrows():
            raw_length = vector_length(row[field])
            norm_length = vector_length(row[f"{field}_norm"])
            if raw_length == 50 and norm_length == 50:
                valid += 1
            else:
                bad.append(
                    {
                        "Variable_Name": str(row["Variable_Name"]),
                        "raw_length": raw_length,
                        "normalized_length": norm_length,
                    }
                )
        valid_vector_coverage[field] = valid
        if bad:
            invalid_lengths[field] = bad

    one_logprob_lengths = {
        vector_length(value) for value in all_items["one_logprobs"]
    }
    if one_logprob_lengths != {11}:
        raise ValueError(
            f"Expected width-11 one_logprobs, got {one_logprob_lengths}"
        )
    probabilities = np.asarray(all_items["one_logprobs"].tolist(), dtype=float)
    if (
        not np.all(np.isfinite(probabilities))
        or np.any(probabilities < 0.0)
        or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-8)
    ):
        raise ValueError("Invalid one_logprobs probability vectors")

    complete_models = [
        field
        for field in RAW_LLM_FIELDS
        if coverage[field] == 279
        and valid_vector_coverage[field] == 279
    ]
    pending_models = [
        field for field in RAW_LLM_FIELDS if field not in complete_models
    ]
    return {
        "all_rows": len(all_items),
        "train_rows": len(train),
        "test_rows": len(test),
        "unique_questions": len(set(all_ids)),
        "train_test_overlap": 0,
        "embedding_dimension": 256,
        "llm_nonnull_coverage": coverage,
        "llm_valid_50_vector_coverage": valid_vector_coverage,
        "pending_question_ids_by_model": {
            field: ids for field, ids in pending_ids.items() if ids
        },
        "invalid_vector_lengths_by_model": invalid_lengths,
        "complete_persona_vector_models": complete_models,
        "pending_persona_vector_models": pending_models,
        "ready_for_eight_source_experiments": not pending_models,
        "one_logprobs_questions": len(all_items),
        "one_logprobs_width": 11,
    }


def main() -> int:
    source_all = pd.read_parquet(SOURCE_ALL)
    source_train = pd.read_parquet(SOURCE_TRAIN)
    source_test = pd.read_parquet(SOURCE_TEST)
    source_metadata = json.loads(SOURCE_METADATA.read_text(encoding="utf-8"))

    validation = validate_source(source_all, source_train, source_test)
    train = source_train.loc[:, CANONICAL_COLUMNS].copy()
    test = source_test.loc[:, CANONICAL_COLUMNS].copy()

    atomic_parquet(train, OUTPUT_TRAIN)
    atomic_parquet(test, OUTPUT_TEST)

    source_generation = (
        source_metadata.get("llm_features", {})
        .get("new_question_generation", {})
    )
    model_provenance = {
        field: source_generation[field]
        for field in RAW_LLM_FIELDS
        if field in source_generation
    }
    metadata = {
        "schema_version": 3,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Canonical Twin population-level strict Likert dataset "
            "with 279 questions"
        ),
        "canonical": True,
        "question_universe": {
            "definition": "strict Likert-type ordinal opinion questions",
            "rows": 279,
            "legacy_259_retained": 259,
            "new_questions_added": 20,
            "source_counts": source_metadata.get("counts", {}),
        },
        "split": source_metadata.get("split", {}),
        "llm_source_schema": {
            "compatibility_target": "dataset/OpinionQA/aggreated",
            "raw_field_order": RAW_LLM_FIELDS,
            "normalized_field_order": NORM_LLM_FIELDS,
            "normalized_fields_follow_raw_fields": True,
            "normalization": (
                "(response - canonical_min) / "
                "(canonical_max - canonical_min)"
            ),
            "model_provenance_for_new_20_questions": model_provenance,
        },
        "embedding": {
            "provider": "OpenAI",
            "model": "text-embedding-3-small",
            "dimensions": 256,
            "input_field": "Question",
        },
        "one_logprobs": {
            "coverage": 279,
            "vector_width": 11,
            "generation": (
                source_metadata.get("llm_features", {})
                .get("one_logprobs_generation", {})
            ),
        },
        "source": {
            "directory": str(SOURCE_DIR),
            "all": {
                "path": str(SOURCE_ALL),
                "sha256": sha256_file(SOURCE_ALL),
            },
            "train": {
                "path": str(SOURCE_TRAIN),
                "sha256": sha256_file(SOURCE_TRAIN),
            },
            "test": {
                "path": str(SOURCE_TEST),
                "sha256": sha256_file(SOURCE_TEST),
            },
            "metadata": {
                "path": str(SOURCE_METADATA),
                "sha256": sha256_file(SOURCE_METADATA),
                "updated_utc": source_metadata.get("updated_utc"),
            },
        },
        "output": {
            "columns": CANONICAL_COLUMNS,
            "train": {
                "path": str(OUTPUT_TRAIN),
                "rows": len(train),
                "sha256": sha256_file(OUTPUT_TRAIN),
            },
            "test": {
                "path": str(OUTPUT_TEST),
                "rows": len(test),
                "sha256": sha256_file(OUTPUT_TEST),
            },
        },
        "validation": validation,
        "usage_status": (
            "ready_for_complete_eight_source_experiments"
            if validation["ready_for_eight_source_experiments"]
            else "pending_local_llm_vectors_for_complete_eight_source_experiments"
        ),
    }
    atomic_json(metadata, OUTPUT_METADATA)

    print(
        json.dumps(
            {
                "status": "written",
                "train_rows": len(train),
                "test_rows": len(test),
                "total_rows": len(train) + len(test),
                "columns": len(CANONICAL_COLUMNS),
                "complete_models": validation[
                    "complete_persona_vector_models"
                ],
                "pending_models": validation[
                    "pending_persona_vector_models"
                ],
                "ready_for_eight_source_experiments": validation[
                    "ready_for_eight_source_experiments"
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
