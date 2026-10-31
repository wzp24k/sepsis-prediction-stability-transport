# -*- coding: utf-8 -*-
"""Align reversed-order DG-GRU runs to the original ordinal directions."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


ROOT = Path(os.environ.get("SEPSIS_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
STANDARD = ROOT / "results" / "formal"
REVERSED = ROOT / "results" / "sensitivity_v2" / "formal_reverse"
OUT = ROOT / "results" / "sensitivity_v2"
SEEDS = (1, 2, 3)


def metric(
    probability: np.ndarray, y: np.ndarray
) -> dict:
    prediction = probability.argmax(2)
    return {
        "auc_24": round(
            float(
                roc_auc_score(
                    y,
                    probability[:, 0, :],
                    multi_class="ovr",
                    average="macro",
                    labels=np.arange(3),
                )
            ),
            4,
        ),
        "auc_72": round(
            float(
                roc_auc_score(
                    y,
                    probability[:, 2, :],
                    multi_class="ovr",
                    average="macro",
                    labels=np.arange(3),
                )
            ),
            4,
        ),
        "flip_24_72": round(
            float((prediction[:, 0] != prediction[:, 2]).mean()), 4
        ),
        "tv_24_72": round(
            float(
                (
                    0.5
                    * np.abs(probability[:, 0, :] - probability[:, 2, :]).sum(1)
                ).mean()
            ),
            4,
        ),
        "return_to_24h_class": round(
            float(
                (
                    (prediction[:, 0] == prediction[:, 2])
                    & (prediction[:, 0] != prediction[:, 1])
                ).mean()
            ),
            4,
        ),
    }


def mean_sd(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)
    return {
        "mean": round(float(array.mean()), 4),
        "sd": round(float(array.std(ddof=1)), 4),
        "values": [round(float(value), 4) for value in array],
    }


def analyze_database(database: str) -> dict:
    standard_metrics = []
    reversed_metrics = []
    paired = {key: [] for key in ("auc_72", "flip_24_72", "tv_24_72")}
    for seed in SEEDS:
        standard = np.load(
            STANDARD / database / f"ordinal_notemporal_s{seed}.npz"
        )
        reversed_bundle = np.load(
            REVERSED / database / f"reverse_notemporal_s{seed}.npz"
        )
        if not np.array_equal(standard["ids"], reversed_bundle["ids"]):
            raise ValueError(f"{database} seed {seed}: row order differs")
        original_y = standard["y"].copy()
        expected_reversed_y = 2 - original_y
        if not np.array_equal(expected_reversed_y, reversed_bundle["y"]):
            raise ValueError(f"{database} seed {seed}: reversed outcome is invalid")
        # Reversed training used y' = 2 - y. Restore the original label axis
        # before comparing probabilities or decisions.
        restored_probability = reversed_bundle["probability"][:, :, ::-1]
        standard_record = metric(standard["probability"], original_y)
        reversed_record = metric(restored_probability, original_y)
        standard_metrics.append(standard_record)
        reversed_metrics.append(reversed_record)
        for key in paired:
            paired[key].append(
                reversed_record[key] - standard_record[key]
            )
    result = {
        "database": database,
        "standard_ordinal_notemporal": {
            key: mean_sd([record[key] for record in standard_metrics])
            for key in ("auc_72", "flip_24_72", "tv_24_72")
        },
        "reversed_order_restored": {
            key: mean_sd([record[key] for record in reversed_metrics])
            for key in ("auc_72", "flip_24_72", "tv_24_72")
        },
        "paired_reversed_minus_standard": {
            key: mean_sd(values) for key, values in paired.items()
        },
        "seeds": list(SEEDS),
        "run_scope": "three-seed label-direction pressure test",
        "interpretation": (
            "Reversing the label order and then restoring the probability axis "
            "tests dependence on the assumed home-facility-death direction. "
            "It is a coding-order stress test, not a second clinical outcome."
        ),
    }
    return result


def main() -> None:
    summary = {
        "mimic4": analyze_database("mimic4"),
        "eicu": analyze_database("eicu"),
    }
    with (OUT / "reverse_order_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
