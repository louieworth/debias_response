#!/usr/bin/env python3
"""Run CPU-only group scale ablation robustness over persona samples.

For each dataset, persona sample seed, and model/split seed, this script:
1. Builds a distinct aggreated_ablation parquet pair with 8 sampled personas.
2. Runs the 17 scale settings x 3 variants = 51 MLP ablations.
3. Writes one result CSV per (dataset, persona_sample_seed, seed).

The script deliberately defaults to CPU-only execution and clears CUDA visibility
for subprocesses so it does not allocate H100 memory.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, List

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
BUILD_SCRIPT = ROOT / "run/aggreated/ablation/build_group_scale_ablation.py"
LOG_ROOT = ROOT / "logs/runs/group_scale_ablation/persona_robustness"

DATASET_INFO = {
    "OpinionQA": {
        "dataset_root": ROOT / "dataset/OpinionQA",
        "prefix": "opinionqa",
        "result_dataset_dir": "OpinionQA",
        "result_prefix": "opinionqa",
    },
    "Twin": {
        "dataset_root": ROOT / "dataset/Twin-2K-500",
        "prefix": "twin",
        "result_dataset_dir": "twin",
        "result_prefix": "twin",
    },
}
VARIANTS = ["x_one_llm", "x_avg_llm", "x_all_llm"]
SCALE_FAMILIES = ["persona", "llm", "all"]
EXPECTED_ROWS = 51


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


def result_path(dataset: str, persona_seed: int, seed: int) -> Path:
    info = DATASET_INFO[dataset]
    return (
        ROOT
        / "results/group"
        / info["result_dataset_dir"]
        / "scale_ablation_persona_robustness"
        / f"aggreated_{info['result_prefix']}_scale_ablation_persona_seed_{persona_seed}_seed_{seed}.csv"
    )


def is_complete_result(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        df = pd.read_csv(path)
    except Exception:
        return False
    if len(df) != EXPECTED_ROWS:
        return False
    keys = ["variant", "llm_input_mode", "llm_input_name"]
    return not df.duplicated(keys).any()


def build_parquet(py: str, dataset: str, persona_seed: int, seed: int, args: argparse.Namespace, env: dict) -> tuple[Path, Path, Path]:
    info = DATASET_INFO[dataset]
    suffix = f"robust_p{persona_seed}_s{seed}"
    out_dir = info["dataset_root"] / "aggreated_ablation"
    train_file = out_dir / f"{info['prefix']}_{suffix}_train.parquet"
    test_file = out_dir / f"{info['prefix']}_{suffix}_test.parquet"
    metadata_file = out_dir / f"{info['prefix']}_{suffix}_metadata.json"
    if train_file.exists() and test_file.exists() and metadata_file.exists() and not args.force_build:
        return train_file, test_file, metadata_file

    cmd = [
        py,
        str(BUILD_SCRIPT),
        "--datasets",
        dataset,
        "--seed",
        str(seed),
        "--persona_seed",
        str(persona_seed),
        "--split_seed",
        str(seed),
        "--n_personas",
        str(args.n_personas),
        "--test_size",
        str(args.test_size),
        "--anchor_llm",
        args.anchor_llm,
        "--output_suffix",
        suffix,
    ]
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)
    return train_file, test_file, metadata_file


def make_ablation_commands(py: str, train_file: Path, test_file: Path, out_rel: str, seed: int, args: argparse.Namespace) -> List[List[str]]:
    commands: List[List[str]] = []
    for family in SCALE_FAMILIES:
        k_values = [64] if family == "all" else list(range(1, args.n_personas + 1))
        for k in k_values:
            if family == "all":
                field = "all_64_norm"
                llm_dim = "64"
                input_name = "all_64"
            else:
                field = f"{family}_{k}_norm"
                llm_dim = str(k)
                input_name = f"{family}_{k}"
            for variant in VARIANTS:
                cmd = [
                    py,
                    "-m",
                    "debias.debias_variants",
                    "--train_file",
                    str(train_file),
                    "--test_file",
                    str(test_file),
                    "--variant",
                    variant,
                    "--model_type",
                    "mlp",
                    "--llm_field",
                    field,
                    "--llm_input_mode",
                    f"scale_{family}",
                    "--llm_input_name",
                    input_name,
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
                if variant in {"x_all_llm", "x_avg_llm"}:
                    cmd.extend(["--llm_dim", llm_dim])
                if not args.mlp_standardize:
                    cmd.append("--no_mlp_standardize")
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


def run_combo(py: str, dataset: str, persona_seed: int, seed: int, args: argparse.Namespace, env: dict) -> None:
    out_path = result_path(dataset, persona_seed, seed)
    if is_complete_result(out_path) and not args.force:
        print(f"[SKIP] {dataset} persona_seed={persona_seed} seed={seed} complete: {out_path}", flush=True)
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()
    lock_path = out_path.with_suffix(out_path.suffix + ".lock")
    if lock_path.exists():
        lock_path.unlink()

    train_file, test_file, _metadata_file = build_parquet(py, dataset, persona_seed, seed, args, env)
    out_rel = str(out_path.relative_to(ROOT / "results"))
    commands = make_ablation_commands(py, train_file, test_file, out_rel, seed, args)
    log_dir = LOG_ROOT / dataset.lower()
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"persona_seed_{persona_seed}_seed_{seed}.log"
    log_path.write_text(
        f"dataset={dataset}\npersona_seed={persona_seed}\nseed={seed}\ntrain_file={train_file}\ntest_file={test_file}\nout={out_path}\n",
        encoding="utf-8",
    )

    print(f"[RUN] {dataset} persona_seed={persona_seed} seed={seed} commands={len(commands)}", flush=True)
    failures = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as executor:
        future_to_cmd = {executor.submit(run_one_command, cmd, env, log_path): cmd for cmd in commands}
        for idx, future in enumerate(concurrent.futures.as_completed(future_to_cmd), start=1):
            rc = future.result()
            if rc != 0:
                failures += 1
            if idx % 10 == 0 or idx == len(commands):
                print(f"[PROGRESS] {dataset} p={persona_seed} s={seed} {idx}/{len(commands)} failures={failures}", flush=True)
    if failures:
        raise RuntimeError(f"{dataset} persona_seed={persona_seed} seed={seed} had {failures} failed commands. See {log_path}")
    if not is_complete_result(out_path):
        rows = len(pd.read_csv(out_path)) if out_path.exists() else 0
        raise RuntimeError(f"Incomplete result {out_path}: rows={rows}, expected={EXPECTED_ROWS}")
    print(f"[DONE] {dataset} persona_seed={persona_seed} seed={seed} -> {out_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["OpinionQA", "Twin"], choices=sorted(DATASET_INFO))
    parser.add_argument("--persona_seeds", default="0-9")
    parser.add_argument("--seeds", default="0-4")
    parser.add_argument("--n_personas", type=int, default=8)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--anchor_llm", default="gpt-4o")
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--py", default=sys.executable)
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

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["OMP_NUM_THREADS"] = env.get("OMP_NUM_THREADS", "1")
    env["MKL_NUM_THREADS"] = env.get("MKL_NUM_THREADS", "1")
    env["OPENBLAS_NUM_THREADS"] = env.get("OPENBLAS_NUM_THREADS", "1")
    env["NUMEXPR_NUM_THREADS"] = env.get("NUMEXPR_NUM_THREADS", "1")
    env["PYTHONUNBUFFERED"] = "1"

    persona_seeds = parse_int_list(args.persona_seeds)
    seeds = parse_int_list(args.seeds)
    print("CPU-only persona robustness run")
    print(f"datasets={args.datasets}")
    print(f"persona_seeds={persona_seeds}")
    print(f"seeds={seeds}")
    print(f"jobs={args.jobs}")
    print("CUDA_VISIBLE_DEVICES is cleared for all subprocesses")

    for dataset in args.datasets:
        for persona_seed in persona_seeds:
            for seed in seeds:
                run_combo(args.py, dataset, persona_seed, seed, args, env)


if __name__ == "__main__":
    main()
