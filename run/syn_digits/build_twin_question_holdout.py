#!/usr/bin/env python3
"""Build a complete-question holdout split for the Twin individual task."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "dataset/Twin-2K-500/individual"
DEFAULT_OUTPUT_ROOT = SOURCE_ROOT / "syn_digits_question_holdout"
EXPECTED_QUESTIONS = 60
EXPECTED_RESPONDENTS = 167


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_split(*, output_root: Path, random_state: int = 42) -> dict:
    source_train = SOURCE_ROOT / "individual_train.parquet"
    source_test = SOURCE_ROOT / "individual_test.parquet"
    source_hashes = {
        str(path.relative_to(PROJECT_ROOT)): file_sha256(path)
        for path in (source_train, source_test)
    }

    full = pd.concat(
        [pd.read_parquet(source_train), pd.read_parquet(source_test)],
        ignore_index=True,
    )
    required = {"Variable_Name", "twin_id"}
    missing = sorted(required - set(full.columns))
    if missing:
        raise ValueError(f"source data are missing required columns: {missing}")
    if full.duplicated(["Variable_Name", "twin_id"]).any():
        raise ValueError("source data contain duplicate respondent-question cells")

    question_ids = sorted(full["Variable_Name"].astype(str).unique().tolist())
    respondent_ids = sorted(full["twin_id"].unique().tolist(), key=str)
    if len(question_ids) != EXPECTED_QUESTIONS:
        raise ValueError(f"expected 60 questions, found {len(question_ids)}")
    if len(respondent_ids) != EXPECTED_RESPONDENTS:
        raise ValueError(f"expected 167 respondents, found {len(respondent_ids)}")
    if len(full) != EXPECTED_QUESTIONS * EXPECTED_RESPONDENTS:
        raise ValueError(f"expected 10,020 cells, found {len(full):,}")

    per_question = full.groupby("Variable_Name")["twin_id"].nunique()
    per_respondent = full.groupby("twin_id")["Variable_Name"].nunique()
    if not per_question.eq(EXPECTED_RESPONDENTS).all():
        raise ValueError("at least one question is missing respondent cells")
    if not per_respondent.eq(EXPECTED_QUESTIONS).all():
        raise ValueError("at least one respondent is missing question cells")

    train_questions, test_questions = train_test_split(
        question_ids,
        test_size=0.2,
        random_state=random_state,
        shuffle=True,
    )
    train_questions = sorted(train_questions)
    test_questions = sorted(test_questions)
    train_set = set(train_questions)
    test_set = set(test_questions)
    if train_set & test_set or train_set | test_set != set(question_ids):
        raise AssertionError("question partition is not disjoint and exhaustive")

    question_as_str = full["Variable_Name"].astype(str)
    train = full.loc[question_as_str.isin(train_set)].copy()
    test = full.loc[question_as_str.isin(test_set)].copy()
    train = train.sort_values(["Variable_Name", "twin_id"], key=lambda x: x.astype(str))
    test = test.sort_values(["Variable_Name", "twin_id"], key=lambda x: x.astype(str))
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

    expected_train = 48 * EXPECTED_RESPONDENTS
    expected_test = 12 * EXPECTED_RESPONDENTS
    if len(train) != expected_train or len(test) != expected_test:
        raise AssertionError(
            f"unexpected split sizes: train={len(train)}, test={len(test)}"
        )
    if set(train["Variable_Name"].astype(str)) & set(test["Variable_Name"].astype(str)):
        raise AssertionError("train and test questions overlap")
    if train["twin_id"].nunique() != EXPECTED_RESPONDENTS:
        raise AssertionError("train does not contain all respondents")
    if test["twin_id"].nunique() != EXPECTED_RESPONDENTS:
        raise AssertionError("test does not contain all respondents")

    output_root.mkdir(parents=True, exist_ok=True)
    train_path = output_root / "individual_train.parquet"
    test_path = output_root / "individual_test.parquet"
    manifest_path = output_root / "split_manifest.json"
    train.to_parquet(train_path, index=False, compression="snappy")
    test.to_parquet(test_path, index=False, compression="snappy")

    manifest = {
        "dataset": "Twin-2K-500",
        "task_level": "individual",
        "split_strategy": "complete-question holdout",
        "source_files": source_hashes,
        "source_matrix": {
            "questions": EXPECTED_QUESTIONS,
            "respondents": EXPECTED_RESPONDENTS,
            "records": len(full),
        },
        "selection": {
            "method": "train_test_split over sorted unique question IDs",
            "random_state": random_state,
            "test_fraction_of_questions": 0.2,
        },
        "train": {
            "file": str(train_path.relative_to(PROJECT_ROOT)),
            "questions": len(train_questions),
            "respondents": train["twin_id"].nunique(),
            "records": len(train),
            "question_ids": train_questions,
        },
        "test": {
            "file": str(test_path.relative_to(PROJECT_ROOT)),
            "questions": len(test_questions),
            "respondents": test["twin_id"].nunique(),
            "records": len(test),
            "question_ids": test_questions,
        },
        "question_overlap": 0,
        "respondent_overlap": EXPECTED_RESPONDENTS,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    if source_hashes != {
        str(path.relative_to(PROJECT_ROOT)): file_sha256(path)
        for path in (source_train, source_test)
    }:
        raise RuntimeError("a source split file changed while building the backup")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()
    manifest = build_split(
        output_root=args.output_root.resolve(),
        random_state=args.random_state,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
