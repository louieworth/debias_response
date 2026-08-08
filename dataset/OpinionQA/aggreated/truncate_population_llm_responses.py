#!/usr/bin/env python3
"""Materialize the OpinionQA population benchmark with exactly K=50 draws.

All raw and normalized per-model response arrays are truncated deterministically
to their first K coordinates. The canonical aggregate scalar fields are then
recomputed from the same retained GPT-4o coordinates used by the population
Base LLM, Mean, HuMCal-Mean, and Vector experiments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


MODEL_FIELDS = (
    "claude-3.5-haiku",
    "deepseek-v3",
    "gpt-3.5-turbo",
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-5-mini",
    "llama-3.3-70B-instruct-turbo",
    "mistral-7B-instruct-v0.3",
)
RESPONSE_FIELDS = MODEL_FIELDS + tuple(f"{field}_norm" for field in MODEL_FIELDS)
CANONICAL_RAW_FIELD = "gpt-4o"
CANONICAL_NORM_FIELD = "gpt-4o_norm"
DEFAULT_K = 50
BASE_DIR = Path(__file__).resolve().parent
DATA_FILES = (
    BASE_DIR / "opinionqa_flat.json",
    BASE_DIR / "opinionqa_train.parquet",
    BASE_DIR / "opinionqa_test.parquet",
)
METADATA_FILE = BASE_DIR / "opinionqa_population_k50_metadata.json"


def _as_list(value, *, field: str, row_index: int) -> list:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"row {row_index}: {field} is not a response sequence")
    responses = list(value)
    for item in responses:
        numeric = float(item)
        if not np.isfinite(numeric):
            raise ValueError(f"row {row_index}: {field} contains a non-finite value")
    return responses


def truncate_records(
    records: list[dict], *, k: int
) -> tuple[list[dict], dict[str, dict[str, int]]]:
    length_counts = {field: Counter() for field in RESPONSE_FIELDS}
    output = []
    for row_index, source in enumerate(records):
        row = dict(source)
        for field in RESPONSE_FIELDS:
            if field not in row:
                raise KeyError(f"row {row_index}: missing response field {field}")
            responses = _as_list(row[field], field=field, row_index=row_index)
            length_counts[field][len(responses)] += 1
            if len(responses) < k:
                raise ValueError(
                    f"row {row_index}: {field} has {len(responses)} responses; requires {k}"
                )
            row[field] = responses[:k]
        row["Average_LLM_Response"] = float(np.mean(row[CANONICAL_RAW_FIELD]))
        row["Average_LLM_Response_norm"] = float(
            np.mean(row[CANONICAL_NORM_FIELD])
        )
        output.append(row)
    serialized_counts = {
        field: {str(length): count for length, count in sorted(counts.items())}
        for field, counts in length_counts.items()
    }
    return output, serialized_counts


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, records: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_parquet(path: Path, records: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame.from_records(records).to_parquet(
        temporary,
        index=False,
        compression="snappy",
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    args = parser.parse_args()
    if args.k <= 0:
        raise ValueError("--k must be positive")

    metadata = {
        "k": args.k,
        "selection": "retain the first K coordinates for every model field",
        "canonical_source": "gpt-4o",
        "aggregate_scalars": (
            "Average_LLM_Response and Average_LLM_Response_norm are recomputed "
            "from the retained canonical-source coordinates"
        ),
        "files": [],
    }
    for path in DATA_FILES:
        if path.suffix == ".json":
            records = json.loads(path.read_text(encoding="utf-8"))
        else:
            records = pd.read_parquet(path).to_dict("records")
        truncated, source_length_counts = truncate_records(records, k=args.k)
        if path.suffix == ".json":
            _write_json(path, truncated)
        else:
            _write_parquet(path, truncated)
        metadata["files"].append(
            {
                "path": str(path.relative_to(BASE_DIR.parents[2])),
                "rows": len(truncated),
                "source_length_counts": source_length_counts,
                "sha256": _sha256(path),
            }
        )
        print(f"[WRITE] {path}: {len(truncated)} rows, K={args.k}")

    metadata_temporary = METADATA_FILE.with_suffix(".json.tmp")
    metadata_temporary.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    metadata_temporary.replace(METADATA_FILE)
    print(f"[WRITE] {METADATA_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
