#!/usr/bin/env python3
"""Run the output-head ablation in an isolated result namespace."""

from __future__ import annotations

import argparse
import csv
import queue
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = PROJECT_ROOT / "results" / "head_ablation" / "raw"
LOG_ROOT = PROJECT_ROOT / "logs" / "head_ablation"

DATASETS = {
    "Twin-2K-500": {
        "population": (
            "dataset/Twin-2K-500/aggreated/twin_train.parquet",
            "dataset/Twin-2K-500/aggreated/twin_test.parquet",
        ),
        "individual": (
            "dataset/Twin-2K-500/individual/individual_train.parquet",
            "dataset/Twin-2K-500/individual/individual_test.parquet",
        ),
    },
    "OpinionQA": {
        "population": (
            "dataset/OpinionQA/aggreated/opinionqa_train.parquet",
            "dataset/OpinionQA/aggreated/opinionqa_test.parquet",
        ),
        "individual": (
            "dataset/OpinionQA/individual/individual_train.parquet",
            "dataset/OpinionQA/individual/individual_test.parquet",
        ),
    },
    "EEDI": {
        "population": (
            "dataset/EEDI/aggreated/eedi_train.parquet",
            "dataset/EEDI/aggreated/eedi_test.parquet",
        ),
        "individual": (
            "dataset/EEDI/individual/individual_train.parquet",
            "dataset/EEDI/individual/individual_test.parquet",
        ),
    },
}

METHODS = {
    "one": {
        "variant": "x_one_llm",
        "head": "mse",
        "label": "One",
    },
    "gaussian": {
        "variant": "x_all_llm",
        "head": "gaussian",
        "label": "Vector, Gaussian Head (NLL)",
    },
    "beta": {
        "variant": "x_all_llm",
        "head": "beta",
        "label": "Vector, Beta Head (Beta NLL)",
    },
    "mse": {
        "variant": "x_all_llm",
        "head": "mse",
        "label": "Vector (Ours, MSE)",
    },
}

LEVEL_CONFIG = {
    "population": {
        "seeds": range(0, 5),
        "hidden_layers": "512,256,128",
        "mlp_alpha": "0.01",
        "learning_rate": "0.0005",
        "max_iter": "1500",
        "batch_size": "64",
        "dropout": "0.0",
        "patience": "20",
    },
    "individual": {
        "seeds": range(1, 6),
        "hidden_layers": "6144,3072,1536,768,384",
        "mlp_alpha": "0.0001",
        "learning_rate": "0.0002",
        "max_iter": "3500",
        "batch_size": "512",
        "dropout": "0.05",
        "patience": "60",
    },
}


@dataclass(frozen=True)
class Task:
    level: str
    dataset: str
    method: str
    seed: int

    @property
    def relative_result(self) -> Path:
        return (
            Path("head_ablation")
            / "raw"
            / self.level
            / self.dataset
            / self.method
            / f"seed_{self.seed}.csv"
        )

    @property
    def result_path(self) -> Path:
        return PROJECT_ROOT / "results" / self.relative_result

    @property
    def log_path(self) -> Path:
        return (
            LOG_ROOT
            / self.level
            / self.dataset
            / self.method
            / f"seed_{self.seed}.log"
        )

    @property
    def label(self) -> str:
        return f"{self.level}/{self.dataset}/{self.method}/seed={self.seed}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--levels",
        nargs="+",
        choices=sorted(LEVEL_CONFIG),
        default=list(LEVEL_CONFIG),
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=list(DATASETS),
        default=list(DATASETS),
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=list(METHODS),
        default=list(METHODS),
    )
    parser.add_argument(
        "--gpus",
        default="cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6,cuda:7",
        help="Comma-separated CUDA devices.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Concurrent jobs. Defaults to the number of configured GPUs.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def expected_metadata(task: Task) -> dict[str, str]:
    method = METHODS[task.method]
    return {
        "variant": method["variant"],
        "model_type": "mlp",
        "prediction_head": method["head"],
    }


def result_is_complete(task: Task) -> bool:
    if not task.result_path.exists():
        return False
    try:
        with task.result_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, csv.Error):
        return False
    if len(rows) != 1:
        return False
    expected = expected_metadata(task)
    return all(rows[0].get(key) == value for key, value in expected.items())


def build_command(task: Task, device: str) -> list[str]:
    method = METHODS[task.method]
    config = LEVEL_CONFIG[task.level]
    train_file, test_file = DATASETS[task.dataset][task.level]

    command = [
        sys.executable,
        "-m",
        "debias.debias_variants",
        "--train_file",
        train_file,
        "--test_file",
        test_file,
        "--variant",
        method["variant"],
        "--model_type",
        "mlp",
        "--mlp_head",
        method["head"],
        "--device",
        device,
        "--random_state",
        str(task.seed),
        "--result_file",
        str(task.relative_result),
        "--no_split_results",
        "--result_precision",
        "17",
        "--llm_vector_transform",
        "raw",
        "--llm_input_mode",
        "head_ablation",
        "--llm_input_name",
        method["label"],
        "--hidden_layers",
        config["hidden_layers"],
        "--mlp_alpha",
        config["mlp_alpha"],
        "--learning_rate_init",
        config["learning_rate"],
        "--max_iter",
        config["max_iter"],
        "--batch_size",
        config["batch_size"],
        "--mlp_dropout",
        config["dropout"],
        "--validation_fraction",
        "0.1",
        "--n_iter_no_change",
        config["patience"],
        "--min_delta",
        "0.000001",
    ]
    if task.level == "population":
        command.extend(["--llm_field", "gpt-4o_norm"])
        if method["variant"] == "x_all_llm":
            command.extend(["--llm_dim", "50"])
    return command


def run_task(task: Task, device_pool: queue.Queue[str], dry_run: bool) -> tuple:
    device = device_pool.get()
    try:
        command = build_command(task, device)
        if dry_run:
            return task, True, device, " ".join(command)

        task.result_path.parent.mkdir(parents=True, exist_ok=True)
        task.log_path.parent.mkdir(parents=True, exist_ok=True)
        with task.log_path.open("w", encoding="utf-8") as log_handle:
            completed = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        if completed.returncode != 0:
            tail = task.log_path.read_text(encoding="utf-8")[-4000:]
            return task, False, device, tail
        if not result_is_complete(task):
            return task, False, device, "Result file failed metadata validation"
        return task, True, device, str(task.result_path)
    finally:
        device_pool.put(device)


def main() -> int:
    args = parse_args()
    devices = [value.strip() for value in args.gpus.split(",") if value.strip()]
    if not devices:
        raise SystemExit("--gpus must contain at least one device")
    max_workers = args.max_workers or len(devices)
    if max_workers <= 0 or max_workers > len(devices):
        raise SystemExit("--max-workers must be between 1 and the number of GPUs")

    tasks = [
        Task(level, dataset, method, seed)
        for level in args.levels
        for dataset in args.datasets
        for method in args.methods
        for seed in LEVEL_CONFIG[level]["seeds"]
    ]
    pending = []
    for task in tasks:
        if args.overwrite and task.result_path.exists():
            task.result_path.unlink()
        if result_is_complete(task):
            print(f"[skip] {task.label}")
        else:
            pending.append(task)

    print(
        f"tasks={len(tasks)} pending={len(pending)} "
        f"workers={max_workers} devices={','.join(devices)}"
    )
    if not pending:
        return 0

    device_pool: queue.Queue[str] = queue.Queue()
    for device in devices[:max_workers]:
        device_pool.put(device)

    failures = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(run_task, task, device_pool, args.dry_run)
            for task in pending
        ]
        for index, future in enumerate(as_completed(futures), start=1):
            task, success, device, detail = future.result()
            status = "ok" if success else "fail"
            print(f"[{index}/{len(pending)}] [{status}] [{device}] {task.label}")
            if args.dry_run:
                print(detail)
            elif not success:
                print(detail)
                failures.append(task)

    if failures:
        print("Failed tasks:")
        for task in failures:
            print(f"  {task.label}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
