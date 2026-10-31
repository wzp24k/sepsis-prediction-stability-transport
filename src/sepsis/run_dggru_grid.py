# -*- coding: utf-8 -*-
"""Run the formal DG-GRU grid and reverse-order stress test in parallel."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT = Path(os.environ.get("SEPSIS_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
TRAINER = ROOT / "src" / "sepsis" / "train_dggru_formal.py"
LOGS = ROOT / "work" / "logs" / "dggru"


def formal_tasks() -> list[dict]:
    tasks = []
    for database in ("mimic4", "eicu"):
        for tag, head, temporal in (
            ("ordinal_notemporal", "ordinal", False),
            ("ordinal_temporal", "ordinal", True),
            ("nominal_notemporal", "nominal", False),
            ("nominal_temporal", "nominal", True),
        ):
            for seed in (1, 2, 3):
                tasks.append(
                    {
                        "kind": "formal",
                        "database": database,
                        "tag": tag,
                        "head": head,
                        "temporal": temporal,
                        "seed": seed,
                        "output": ROOT / "results" / "formal" / database,
                    }
                )
    return tasks


def reverse_tasks() -> list[dict]:
    return [
        {
            "kind": "reverse",
            "database": database,
            "tag": "reverse_notemporal",
            "head": "ordinal",
            "temporal": False,
            "seed": seed,
            "output": (
                ROOT
                / "results"
                / "sensitivity_v2"
                / "formal_reverse"
                / database
            ),
        }
        for database in ("mimic4", "eicu")
        for seed in (1, 2, 3)
    ]


def command(task: dict) -> list[str]:
    arguments = [
        sys.executable,
        str(TRAINER),
        "--database",
        task["database"],
        "--head",
        task["head"],
        "--tag",
        task["tag"],
        "--seed",
        str(task["seed"]),
        "--output_folder",
        str(task["output"]),
        "--smooth_weight",
        "0.04",
        "--monotone_weight",
        "0.06",
        "--epochs",
        "15",
        "--patience",
        "4",
        "--batch_size",
        "256",
    ]
    if task["temporal"]:
        arguments.append("--temporal")
    if task["kind"] == "reverse":
        arguments.append("--reverse_order")
    return arguments


def run_task(task: dict, index: int) -> dict:
    task["output"].mkdir(parents=True, exist_ok=True)
    stem = task["output"] / f"{task['tag']}_s{task['seed']}"
    for suffix in (".json", ".npz"):
        stem.with_suffix(suffix).unlink(missing_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    log_path = (
        LOGS
        / f"{task['kind']}_{task['database']}_{task['tag']}_s{task['seed']}.log"
    )
    environment = os.environ.copy()
    environment["SEPSIS_TORCH_THREADS"] = "1"
    environment["OMP_NUM_THREADS"] = "1"
    environment["MKL_NUM_THREADS"] = "1"
    environment["OPENBLAS_NUM_THREADS"] = "1"
    environment["NUMEXPR_NUM_THREADS"] = "1"
    started = time.time()
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        process = subprocess.run(
            command(task),
            cwd=ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    missing = [
        str(stem.with_suffix(suffix))
        for suffix in (".json", ".npz")
        if not stem.with_suffix(suffix).exists()
    ]
    return {
        "index": index,
        "task": task,
        "returncode": process.returncode,
        "missing": missing,
        "elapsed_seconds": round(time.time() - started, 1),
        "log": str(log_path.relative_to(ROOT)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--set", choices=("formal", "reverse", "all"), default="all"
    )
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    tasks = []
    if args.set in ("formal", "all"):
        tasks.extend(formal_tasks())
    if args.set in ("reverse", "all"):
        tasks.extend(reverse_tasks())
    started = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(run_task, task, index): task
            for index, task in enumerate(tasks, 1)
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            task = result["task"]
            print(
                result["index"],
                len(tasks),
                task["kind"],
                task["database"],
                task["tag"],
                f"s{task['seed']}",
                f"rc={result['returncode']}",
                f"{result['elapsed_seconds']}s",
                flush=True,
            )
    failed = [
        result
        for result in results
        if result["returncode"] != 0 or result["missing"]
    ]
    print(
        {
            "tasks": len(tasks),
            "failed": len(failed),
            "elapsed_seconds": round(time.time() - started, 1),
            "workers": args.workers,
        },
        flush=True,
    )
    if failed:
        for result in failed:
            print(result, flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
