#!/usr/bin/env python3
"""Run family-level persona-vs-LLM scale comparison.

Definitions:
  persona: fixed anchor LLM, average over all available pool personas.
           OpinionQA uses its 40 sidecar personas, Twin uses a fixed 40-persona
           sample from the complete pool, and EEDI uses all 8 complete personas.
  llm:     fixed persona, average over all 8 LLMs. For OpinionQA/Twin, sample
           8 personas for each persona-sample seed and average the per-persona
           results. For EEDI, use all 8 personas once.

The script is deliberately CPU-only and clears CUDA visibility for all child
processes.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[3]
HELPER_PATH = ROOT / "run/aggreated/ablation/build_group_scale_ablation.py"
LOG_ROOT = ROOT / "logs/runs/group_scale_ablation/family_scale_comparison"

spec = importlib.util.spec_from_file_location("group_scale_builder", HELPER_PATH)
builder = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(builder)

DATASET_INFO = {
    "EEDI": {
        "root": ROOT / "dataset/EEDI",
        "prefix": "eedi",
        "question_key": "Variable_Name",
        "result_dataset_dir": "EEDI",
    },
    "OpinionQA": {
        "root": ROOT / "dataset/OpinionQA",
        "prefix": "opinionqa",
        "question_key": "Variable_Name",
        "result_dataset_dir": "OpinionQA",
    },
    "Twin": {
        "root": ROOT / "dataset/Twin-2K-500",
        "prefix": "twin",
        "question_key": "question_id",
        "result_dataset_dir": "twin",
    },
}

METRIC_ROWS = {"EEDI": 9, "OpinionQA": 41, "Twin": 41}


def parse_int_list(value: str) -> List[int]:
    out: List[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return out


def result_path(dataset: str, seed: int, result_subdir: str) -> Path:
    info = DATASET_INFO[dataset]
    return (
        ROOT
        / "results/group"
        / info["result_dataset_dir"]
        / result_subdir
        / f"aggreated_{info['prefix']}_family_scale_comparison_seed_{seed}.csv"
    )


def is_complete_result(path: Path, expected_rows: int, expected_variant: str) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        df = pd.read_csv(path)
    except Exception:
        return False
    if len(df) != expected_rows:
        return False
    if sorted(df["variant"].unique().tolist()) != [expected_variant]:
        return False
    keys = ["variant", "llm_input_mode", "llm_input_name"]
    return not df.duplicated(keys).any()


def load_sidecar(dataset: str, filtered_root: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if dataset != "OpinionQA":
        return [], {}
    sidecar_path = filtered_root / "llm_backfill_responses.json"
    sidecar_metadata_path = filtered_root / "llm_backfill" / "sidecar_metadata.json"
    if not sidecar_path.exists():
        raise RuntimeError(f"OpinionQA requires {sidecar_path}")
    sidecar_rows = builder.load_json(sidecar_path)
    sidecar_metadata = builder.load_json(sidecar_metadata_path) if sidecar_metadata_path.exists() else {}
    for row in sidecar_rows:
        if "Human_Response" in row:
            raise RuntimeError("OpinionQA LLM sidecar must not contain Human_Response.")
    return sidecar_rows, sidecar_metadata


def complete_persona_ids(
    personas: List[Dict[str, Any]],
    llm_feature_responses: List[Dict[str, Any]],
    questions_by_key: Dict[str, Dict[str, Any]],
    question_key: str,
) -> List[Any]:
    ids = [p.get("twin_id") for p in personas if p.get("twin_id") is not None]
    ids = sorted(set(ids), key=builder.stable_persona_sort_key)
    all_questions = set(questions_by_key.keys())
    persona_questions: Dict[Any, set] = {pid: set() for pid in ids}
    for row in llm_feature_responses:
        pid = row.get("twin_id")
        if pid not in persona_questions:
            continue
        if not isinstance(row.get("LLM_Response"), list):
            continue
        qkey = builder.question_key_from_response(row, question_key)
        if qkey in all_questions:
            persona_questions[pid].add(qkey)
    return [pid for pid in ids if all_questions.issubset(persona_questions.get(pid, set()))]


def choose_pool(
    dataset: str,
    complete_ids: List[Any],
    sidecar_metadata: Dict[str, Any],
    twin_pool_size: int,
    twin_pool_seed: int,
) -> List[Any]:
    if dataset == "OpinionQA":
        preferred = sidecar_metadata.get("selected_persona_ids")
        if preferred is None:
            raise RuntimeError("OpinionQA sidecar metadata must provide selected_persona_ids.")
        complete_set = set(complete_ids)
        missing = [pid for pid in preferred if pid not in complete_set]
        if missing:
            raise RuntimeError(f"OpinionQA preferred personas are incomplete or missing: {missing[:10]}")
        return list(preferred)

    if dataset == "EEDI":
        return list(complete_ids)

    if dataset == "Twin":
        if len(complete_ids) < twin_pool_size:
            raise RuntimeError(f"Twin has only {len(complete_ids)} complete personas, need {twin_pool_size}.")
        rng = np.random.default_rng(twin_pool_seed)
        idx = sorted(rng.choice(len(complete_ids), size=twin_pool_size, replace=False).tolist())
        return [complete_ids[i] for i in idx]

    raise ValueError(f"Unknown dataset: {dataset}")


def build_family_dataframe(
    dataset: str,
    seed: int,
    test_size: float,
    anchor_llm_requested: str,
    twin_pool_size: int,
    twin_pool_seed: int,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    info = DATASET_INFO[dataset]
    dataset_root = info["root"]
    filtered_root = dataset_root / "filtered"
    questions = builder.load_json(filtered_root / "questions.json")
    personas = builder.load_json(filtered_root / "personas.json")
    responses = builder.load_json(filtered_root / "responses.json")
    metadata = builder.load_json(filtered_root / "metadata.json")
    sidecar_rows, sidecar_metadata = load_sidecar(dataset, filtered_root)

    question_key = metadata.get("question_key", {}).get("responses") or info["question_key"]
    llm_order = builder.load_llm_order(dataset_root, metadata)
    if len(llm_order) != 8:
        raise RuntimeError(f"Expected 8 LLMs for {dataset}, found {len(llm_order)}: {llm_order}")

    if sidecar_metadata:
        sidecar_order = sidecar_metadata.get("sidecar_llm_order") or sidecar_metadata.get("llm_order")
        if sidecar_order and list(sidecar_order) != list(llm_order):
            raise RuntimeError(f"OpinionQA sidecar order mismatch: {sidecar_order} vs {llm_order}")

    anchor_llm = anchor_llm_requested if anchor_llm_requested in llm_order else llm_order[0]
    if anchor_llm != anchor_llm_requested:
        print(f"[WARN] {dataset}: requested anchor {anchor_llm_requested!r}; using {anchor_llm!r}")
    anchor_idx = llm_order.index(anchor_llm)

    questions_by_key = builder.index_questions(questions)
    llm_feature_responses = responses + sidecar_rows
    complete_ids = complete_persona_ids(personas, llm_feature_responses, questions_by_key, question_key)
    pool_personas = choose_pool(dataset, complete_ids, sidecar_metadata, twin_pool_size, twin_pool_seed)
    pool_set = set(pool_personas)

    grouped: Dict[str, Dict[str, Any]] = {}
    for row in responses:
        qkey = builder.question_key_from_response(row, question_key)
        if qkey is None or qkey not in questions_by_key:
            continue
        human = builder.as_float(row.get("Human_Response"))
        if human is None:
            continue
        q = questions_by_key[qkey]
        sr = q.get("score_range", [1, 5])
        human_norm = builder.normalize_value(human, sr)
        if human_norm is None:
            continue
        if qkey not in grouped:
            grouped[qkey] = {
                "human_raw": [],
                "human_norm": [],
                "pool_llm_norm": {pid: [None] * len(llm_order) for pid in pool_personas},
            }
        grouped[qkey]["human_raw"].append(human)
        grouped[qkey]["human_norm"].append(human_norm)

    missing_target_questions = sorted(set(questions_by_key) - set(grouped))
    if missing_target_questions:
        raise RuntimeError(
            f"{dataset} has {len(missing_target_questions)} questions without real human targets. "
            f"Examples: {missing_target_questions[:10]}"
        )

    for row in llm_feature_responses:
        qkey = builder.question_key_from_response(row, question_key)
        if qkey is None or qkey not in grouped:
            continue
        llm_response = row.get("LLM_Response")
        if not isinstance(llm_response, list) or len(llm_response) != len(llm_order):
            continue
        pid = row.get("twin_id")
        if pid not in pool_set:
            continue
        q = questions_by_key[qkey]
        sr = q.get("score_range", [1, 5])
        grouped[qkey]["pool_llm_norm"][pid] = [
            builder.normalize_value(value, sr) for value in llm_response
        ]

    all_rows: List[Dict[str, Any]] = []
    for qkey in sorted(grouped.keys()):
        q = questions_by_key[qkey]
        sr = q.get("score_range", [1, 5])
        matrix = np.empty((len(pool_personas), len(llm_order)), dtype=float)
        matrix[:] = np.nan
        for p_idx, pid in enumerate(pool_personas):
            vals = grouped[qkey]["pool_llm_norm"][pid]
            for l_idx, value in enumerate(vals):
                if value is not None:
                    matrix[p_idx, l_idx] = float(value)

        if np.isnan(matrix).any():
            missing_positions = np.argwhere(np.isnan(matrix)).tolist()
            examples = [
                {"persona": pool_personas[p_idx], "llm": llm_order[l_idx]}
                for p_idx, l_idx in missing_positions[:10]
            ]
            raise RuntimeError(f"{dataset} qkey={qkey} has missing pool values: {examples}")

        row_out: Dict[str, Any] = {
            "Variable_Name": q.get("Variable_Name", qkey),
            "Question": q.get("Question"),
            "score_range": list(sr),
            "Average_Human_Response": float(np.mean(grouped[qkey]["human_raw"])),
            "Average_Human_Response_norm": float(np.mean(grouped[qkey]["human_norm"])),
            "Question_Embedding": q.get("Question_Embedding"),
            "pool_persona_ids": list(pool_personas),
            "llm_order": list(llm_order),
            "anchor_llm": anchor_llm,
            "Average_LLM_Response_norm": float(np.mean(matrix)),
            "persona_pool_norm": matrix[:, anchor_idx].tolist(),
        }
        for p_idx in range(len(pool_personas)):
            row_out[f"llm_pool_{p_idx}_norm"] = matrix[p_idx, :].tolist()
        all_rows.append(row_out)

    df = pd.DataFrame(all_rows)
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=seed, shuffle=True)
    split_marker = pd.Series("train", index=train_df.index)
    split_marker = pd.concat([split_marker, pd.Series("test", index=test_df.index)])
    df = pd.concat([train_df, test_df], axis=0)
    df["split"] = split_marker.loc[df.index].values
    df = df.reset_index(drop=True)

    metadata_out = {
        "dataset": dataset,
        "source": str(filtered_root),
        "human_target_source": str(filtered_root / "responses.json"),
        "llm_feature_sources": [
            str(filtered_root / "responses.json"),
            *([str(filtered_root / "llm_backfill_responses.json")] if sidecar_rows else []),
        ],
        "seed": seed,
        "test_size": test_size,
        "anchor_llm_requested": anchor_llm_requested,
        "anchor_llm_used": anchor_llm,
        "llm_order": llm_order,
        "complete_persona_count": len(complete_ids),
        "pool_persona_count": len(pool_personas),
        "pool_personas": pool_personas,
        "twin_pool_seed": twin_pool_seed if dataset == "Twin" else None,
        "fields": {
            "persona": "persona_pool_norm",
            "llm_by_pool_index": [f"llm_pool_{i}_norm" for i in range(len(pool_personas))],
        },
        "rows_total": int(len(df)),
        "rows_train": int((df["split"] == "train").sum()),
        "rows_test": int((df["split"] == "test").sum()),
    }
    return df, metadata_out


def build_parquet(dataset: str, seed: int, args: argparse.Namespace) -> Tuple[Path, Path, Path]:
    info = DATASET_INFO[dataset]
    out_dir = info["root"] / "aggreated_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = f"{info['prefix']}_family_scale_s{seed}"
    train_path = out_dir / f"{output_prefix}_train.parquet"
    test_path = out_dir / f"{output_prefix}_test.parquet"
    metadata_path = out_dir / f"{output_prefix}_metadata.json"
    if train_path.exists() and test_path.exists() and metadata_path.exists() and not args.force_build:
        return train_path, test_path, metadata_path

    df, metadata = build_family_dataframe(
        dataset=dataset,
        seed=seed,
        test_size=args.test_size,
        anchor_llm_requested=args.anchor_llm,
        twin_pool_size=args.twin_pool_size,
        twin_pool_seed=args.twin_pool_seed,
    )
    train_df = df[df["split"] == "train"].drop(columns=["split"])
    test_df = df[df["split"] == "test"].drop(columns=["split"])
    train_df.to_parquet(train_path, index=False, compression="snappy")
    test_df.to_parquet(test_path, index=False, compression="snappy")
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {train_path} rows={len(train_df)}")
    print(f"wrote {test_path} rows={len(test_df)}")
    print(f"wrote {metadata_path}")
    return train_path, test_path, metadata_path


def sample_pool_indices(dataset: str, pool_n: int, persona_sample_seed: int, n_sample: int) -> List[int]:
    if dataset == "EEDI" or pool_n <= n_sample:
        return list(range(pool_n))
    rng = np.random.default_rng(persona_sample_seed)
    return sorted(rng.choice(pool_n, size=n_sample, replace=False).tolist())


def metadata_pool(metadata_path: Path) -> List[Any]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return list(metadata["pool_personas"])


def safe_pid(value: Any) -> str:
    text = str(value)
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")


def base_command(
    py: str,
    train_file: Path,
    test_file: Path,
    out_rel: str,
    seed: int,
    args: argparse.Namespace,
) -> List[str]:
    cmd = [
        py,
        "-m",
        "debias.debias_variants",
        "--train_file",
        str(train_file),
        "--test_file",
        str(test_file),
        "--variant",
        args.variant,
        "--model_type",
        "mlp",
        "--random_state",
        str(seed),
        "--result_file",
        out_rel,
        "--no_split_results",
        "--device",
        "cpu",
        "--hidden_layers",
        args.mlp_hidden_layers,
        "--mlp_alpha",
        str(args.mlp_alpha),
        "--learning_rate_init",
        str(args.mlp_lr_init),
        "--max_iter",
        str(args.mlp_max_iter),
        "--batch_size",
        str(args.mlp_batch_size),
        "--mlp_dropout",
        str(args.mlp_dropout),
        "--validation_fraction",
        str(args.mlp_validation_fraction),
        "--n_iter_no_change",
        str(args.mlp_n_iter_no_change),
        "--min_delta",
        str(args.mlp_min_delta),
    ]
    if not args.mlp_standardize:
        cmd.append("--no_mlp_standardize")
    return cmd


def make_commands(
    py: str,
    dataset: str,
    train_file: Path,
    test_file: Path,
    metadata_path: Path,
    out_path: Path,
    seed: int,
    args: argparse.Namespace,
) -> List[List[str]]:
    out_rel = str(out_path.relative_to(ROOT / "results"))
    pool = metadata_pool(metadata_path)
    commands: List[List[str]] = []

    persona_cmd = base_command(py, train_file, test_file, out_rel, seed, args)
    persona_cmd.extend(
        [
            "--llm_field",
            "persona_pool_norm",
            "--llm_input_mode",
            "family_persona",
            "--llm_input_name",
            "persona_all_mean",
            "--llm_dim",
            str(len(pool)),
        ]
    )
    commands.append(persona_cmd)

    persona_sample_seeds = [0] if dataset == "EEDI" else parse_int_list(args.persona_sample_seeds)
    for persona_sample_seed in persona_sample_seeds:
        sampled_indices = sample_pool_indices(dataset, len(pool), persona_sample_seed, args.n_sampled_personas)
        for pos, pool_idx in enumerate(sampled_indices):
            pid_slug = safe_pid(pool[pool_idx])
            cmd = base_command(py, train_file, test_file, out_rel, seed, args)
            cmd.extend(
                [
                    "--llm_field",
                    f"llm_pool_{pool_idx}_norm",
                    "--llm_input_mode",
                    "family_llm",
                    "--llm_input_name",
                    f"llm_sample_{persona_sample_seed}_pos_{pos}_pool_{pool_idx}_pid_{pid_slug}",
                    "--llm_dim",
                    "8",
                ]
            )
            commands.append(cmd)
    return commands


def run_one_command(cmd: List[str], env: dict, log_path: Path) -> int:
    started = time.time()
    with log_path.open("a", encoding="utf-8") as log_f:
        log_f.write("\n" + "=" * 120 + "\n")
        log_f.write("CMD: " + " ".join(cmd) + "\n")
        log_f.flush()
        proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log_f, stderr=subprocess.STDOUT)
        log_f.write(f"RETURN_CODE: {proc.returncode} elapsed={time.time() - started:.1f}s\n")
    return proc.returncode


def run_dataset_seed(py: str, dataset: str, seed: int, args: argparse.Namespace, env: dict) -> None:
    out_path = result_path(dataset, seed, args.result_subdir)
    expected_rows = METRIC_ROWS[dataset]
    if is_complete_result(out_path, expected_rows, args.variant) and not args.force:
        print(f"[SKIP] {dataset} seed={seed} complete: {out_path}", flush=True)
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()
    lock_path = out_path.with_suffix(out_path.suffix + ".lock")
    if lock_path.exists():
        lock_path.unlink()

    train_file, test_file, metadata_path = build_parquet(dataset, seed, args)
    commands = make_commands(py, dataset, train_file, test_file, metadata_path, out_path, seed, args)
    if len(commands) != expected_rows:
        raise RuntimeError(f"{dataset} seed={seed} planned {len(commands)} commands, expected {expected_rows}")

    log_dir = LOG_ROOT / args.result_subdir / dataset.lower()
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"seed_{seed}.log"
    log_path.write_text(
        f"dataset={dataset}\nseed={seed}\ntrain_file={train_file}\ntest_file={test_file}\nout={out_path}\n",
        encoding="utf-8",
    )

    print(f"[RUN] {dataset} seed={seed} commands={len(commands)}", flush=True)
    failures = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as executor:
        future_to_cmd = {executor.submit(run_one_command, cmd, env, log_path): cmd for cmd in commands}
        for idx, future in enumerate(concurrent.futures.as_completed(future_to_cmd), start=1):
            rc = future.result()
            if rc != 0:
                failures += 1
            if idx % 10 == 0 or idx == len(commands):
                print(f"[PROGRESS] {dataset} seed={seed} {idx}/{len(commands)} failures={failures}", flush=True)
    if failures:
        raise RuntimeError(f"{dataset} seed={seed} had {failures} failed commands. See {log_path}")
    if not is_complete_result(out_path, expected_rows, args.variant):
        rows = len(pd.read_csv(out_path)) if out_path.exists() else 0
        raise RuntimeError(f"Incomplete result {out_path}: rows={rows}, expected={expected_rows}")
    print(f"[DONE] {dataset} seed={seed} -> {out_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["EEDI", "OpinionQA", "Twin"], choices=sorted(DATASET_INFO))
    parser.add_argument("--seeds", default="0-4")
    parser.add_argument("--persona_sample_seeds", default="0-4")
    parser.add_argument("--n_sampled_personas", type=int, default=8)
    parser.add_argument("--twin_pool_size", type=int, default=40)
    parser.add_argument("--twin_pool_seed", type=int, default=0)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--anchor_llm", default="gpt-4o")
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--py", default=sys.executable)
    parser.add_argument("--variant", default="x_avg_llm", choices=["x_avg_llm", "x_all_llm"])
    parser.add_argument("--result_subdir", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force_build", action="store_true")
    parser.add_argument("--mlp_hidden_layers", default="512,256,128")
    parser.add_argument("--mlp_alpha", type=float, default=0.01)
    parser.add_argument("--mlp_lr_init", type=float, default=0.0005)
    parser.add_argument("--mlp_max_iter", type=int, default=1500)
    parser.add_argument("--mlp_batch_size", type=int, default=64)
    parser.add_argument("--mlp_dropout", type=float, default=0.0)
    parser.add_argument("--mlp_validation_fraction", type=float, default=0.1)
    parser.add_argument("--mlp_n_iter_no_change", type=int, default=20)
    parser.add_argument("--mlp_min_delta", type=float, default=1e-6)
    parser.add_argument("--no_mlp_standardize", dest="mlp_standardize", action="store_false")
    parser.set_defaults(mlp_standardize=True)
    args = parser.parse_args()
    if args.result_subdir is None:
        args.result_subdir = (
            "family_scale_comparison"
            if args.variant == "x_avg_llm"
            else f"family_scale_comparison_{args.variant}"
        )

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["OMP_NUM_THREADS"] = env.get("OMP_NUM_THREADS", "1")
    env["MKL_NUM_THREADS"] = env.get("MKL_NUM_THREADS", "1")
    env["OPENBLAS_NUM_THREADS"] = env.get("OPENBLAS_NUM_THREADS", "1")
    env["NUMEXPR_NUM_THREADS"] = env.get("NUMEXPR_NUM_THREADS", "1")
    env["PYTHONUNBUFFERED"] = "1"

    seeds = parse_int_list(args.seeds)
    print("CPU-only family scale comparison")
    print(f"datasets={args.datasets}")
    print(f"seeds={seeds}")
    print(f"persona_sample_seeds={parse_int_list(args.persona_sample_seeds)}")
    print(f"variant={args.variant}")
    print(f"result_subdir={args.result_subdir}")
    print(f"jobs={args.jobs}")
    print("CUDA_VISIBLE_DEVICES is cleared for all subprocesses")

    for dataset in args.datasets:
        for seed in seeds:
            run_dataset_seed(args.py, dataset, seed, args, env)


if __name__ == "__main__":
    main()
