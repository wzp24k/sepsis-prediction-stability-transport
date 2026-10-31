# -*- coding: utf-8 -*-
"""Describe complete 24/48/72-hour out-of-fold prediction trajectories.

The pattern categories are descriptive and mutually exclusive. Confidence
intervals resample patients for MIMIC-IV and MIMIC-III and hospitals for
eICU-CRD, conditional on the already fitted out-of-fold predictions.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(os.environ.get("SEPSIS_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
RESULTS = ROOT / "results"
OUT = RESULTS / "trajectory_patterns"

LANDMARKS = (24, 48, 72)
CLASS_LABELS = ("home", "facility", "death")
SOURCES = {
    "mimic4": {
        "label": "MIMIC-IV",
        "path": RESULTS / "shared_fold" / "mimic4_shared_fold.npz",
        "outcome_key": "y3",
        "cluster_unit": "patient",
    },
    "eicu": {
        "label": "eICU-CRD",
        "path": RESULTS / "shared_fold" / "eicu_shared_fold.npz",
        "outcome_key": "y3",
        "cluster_unit": "hospital",
    },
    "mimic3": {
        "label": "MIMIC-III",
        "path": RESULTS / "mimic3_shared" / "mimic3_shared.npz",
        "outcome_key": "y3",
        "cluster_unit": "patient",
    },
}

PATTERN_LABELS = {
    "stable_all": "Stable at 24, 48, and 72 h",
    "transient_48h_excursion": "Changed at 48 h and returned to the 24 h class by 72 h",
    "monotone_worsening": "Ordinal worsening without reversal",
    "monotone_improvement": "Ordinal improvement without reversal",
    "other_nonmonotone": "Other non-monotone trajectory",
}

CLINICAL_LABELS = {
    "any_adjacent_change": "Any change between adjacent landmarks",
    "any_24_to_72_change": "Change from 24 to 72 h",
    "major_upgrade_to_death_by_72": "Prediction below death at 24 h and death at 72 h",
    "transient_death_upgrade": "Non-death at 24 h, death at 48 h, below death at 72 h",
    "sustained_death_upgrade": "Non-death at 24 h and death at both 48 and 72 h",
    "late_death_upgrade": "Below death at 24 and 48 h, then death at 72 h",
}


def load_bundle(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as bundle:
        return {key: bundle[key].copy() for key in bundle.files}


def class_path(probability: dict[int, np.ndarray]) -> np.ndarray:
    return np.column_stack(
        [probability[milestone].argmax(axis=1) for milestone in LANDMARKS]
    ).astype(np.int8)


def indicators(path: np.ndarray) -> dict[str, np.ndarray]:
    first, second, third = path.T
    stable = (first == second) & (second == third)
    transient = (first == third) & (first != second)
    monotone_worsening = (
        ~stable
        & ~transient
        & (first <= second)
        & (second <= third)
    )
    monotone_improvement = (
        ~stable
        & ~transient
        & (first >= second)
        & (second >= third)
    )
    other = ~(
        stable
        | transient
        | monotone_worsening
        | monotone_improvement
    )
    patterns = {
        "stable_all": stable,
        "transient_48h_excursion": transient,
        "monotone_worsening": monotone_worsening,
        "monotone_improvement": monotone_improvement,
        "other_nonmonotone": other,
    }
    if not np.array_equal(
        np.sum(np.column_stack(list(patterns.values())), axis=1),
        np.ones(len(path), dtype=int),
    ):
        raise ValueError("Trajectory pattern categories are not mutually exclusive")

    clinical = {
        "any_adjacent_change": (first != second) | (second != third),
        "any_24_to_72_change": first != third,
        "major_upgrade_to_death_by_72": (first < 2) & (third == 2),
        "transient_death_upgrade": (
            (first < 2) & (second == 2) & (third < 2)
        ),
        "sustained_death_upgrade": (
            (first < 2) & (second == 2) & (third == 2)
        ),
        "late_death_upgrade": (
            (first < 2) & (second < 2) & (third == 2)
        ),
    }
    return {**patterns, **clinical}


def transition_matrix(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    matrix = np.zeros((3, 3), dtype=np.int64)
    np.add.at(matrix, (first, second), 1)
    return matrix


def cluster_bootstrap_intervals(
    values: dict[str, np.ndarray],
    clusters: np.ndarray,
    replicates: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    unique, inverse = np.unique(clusters.astype(str), return_inverse=True)
    keys = list(values)
    cluster_numerators = np.vstack(
        [
            np.bincount(
                inverse,
                weights=values[key].astype(np.float64),
                minlength=len(unique),
            )
            for key in keys
        ]
    )
    cluster_denominators = np.bincount(
        inverse, minlength=len(unique)
    ).astype(np.float64)
    rng = np.random.default_rng(seed)
    point = cluster_numerators.sum(axis=1) / cluster_denominators.sum()
    sampled = np.zeros((replicates, len(keys)), dtype=np.float64)
    batch = 100
    for start in range(0, replicates, batch):
        stop = min(replicates, start + batch)
        selected = rng.integers(
            0, len(unique), size=(stop - start, len(unique))
        )
        counts = np.vstack(
            [np.bincount(row, minlength=len(unique)) for row in selected]
        ).astype(np.float64)
        sampled[start:stop] = (
            counts @ cluster_numerators.T
            / (counts @ cluster_denominators)[:, None]
        )
    output = {}
    for index, key in enumerate(keys):
        lower, upper = np.quantile(sampled[:, index], [0.025, 0.975])
        output[key] = {
            "estimate": round(float(point[index]), 6),
            "ci95": [round(float(lower), 6), round(float(upper), 6)],
            "events": int(values[key].sum()),
        }
    return output


def analyze_database(
    database: str,
    spec: dict,
    replicates: int,
    seed: int,
) -> dict:
    bundle = load_bundle(spec["path"])
    probability = {
        milestone: np.asarray(bundle[f"p{milestone}"], dtype=np.float64)
        for milestone in LANDMARKS
    }
    path = class_path(probability)
    y = np.asarray(bundle[spec["outcome_key"]], dtype=np.int64)
    clusters = np.asarray(bundle["clusters"]).astype(str)
    if not (len(path) == len(y) == len(clusters)):
        raise ValueError(f"{database}: trajectory arrays do not align")
    rates = indicators(path)
    intervals = cluster_bootstrap_intervals(
        rates,
        clusters,
        replicates,
        seed,
    )
    exact: dict[str, int] = {}
    for row in path:
        key = "".join(str(int(value)) for value in row)
        exact[key] = exact.get(key, 0) + 1
    exact_sorted = dict(
        sorted(exact.items(), key=lambda item: (-item[1], item[0]))
    )
    return {
        "label": spec["label"],
        "source": str(spec["path"].relative_to(ROOT)),
        "n": int(len(path)),
        "clusters": int(len(np.unique(clusters))),
        "cluster_unit": spec["cluster_unit"],
        "outcome_counts": {
            CLASS_LABELS[index]: int((y == index).sum()) for index in range(3)
        },
        "patterns": {
            key: {
                "label": PATTERN_LABELS[key],
                **intervals[key],
            }
            for key in PATTERN_LABELS
        },
        "clinical_trajectories": {
            key: {
                "label": CLINICAL_LABELS[key],
                **intervals[key],
            }
            for key in CLINICAL_LABELS
        },
        "exact_trajectory_counts": exact_sorted,
        "transition_matrices": {
            "24_to_48": {
                "rows": list(CLASS_LABELS),
                "columns": list(CLASS_LABELS),
                "counts": transition_matrix(path[:, 0], path[:, 1]).tolist(),
            },
            "48_to_72": {
                "rows": list(CLASS_LABELS),
                "columns": list(CLASS_LABELS),
                "counts": transition_matrix(path[:, 1], path[:, 2]).tolist(),
            },
        },
    }


def percent(record: dict) -> str:
    return (
        f"{100.0 * record['estimate']:.2f}% "
        f"({100.0 * record['ci95'][0]:.2f} to "
        f"{100.0 * record['ci95'][1]:.2f})"
    )


def build_report(summary: dict) -> str:
    lines = [
        "# Complete Three-Landmark Prediction Trajectories",
        "",
        "## Scope",
        "",
        "Prediction trajectories use the exact complete-landmark cohorts and",
        "the same fitted out-of-fold predictions as the primary stability",
        "analysis. Categories are mutually exclusive. Intervals resample",
        "patients in MIMIC-IV and MIMIC-III and hospitals in eICU-CRD,",
        "conditional on the fitted predictions and original folds.",
        "",
        "## Trajectory patterns",
        "",
        "| Database | N | Stable all | Transient 48 h | Monotone worse | "
        "Monotone better | Other non-monotone |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for database in SOURCES:
        record = summary["databases"][database]
        patterns = record["patterns"]
        lines.append(
            f"| {record['label']} | {record['n']:,} | "
            f"{percent(patterns['stable_all'])} | "
            f"{percent(patterns['transient_48h_excursion'])} | "
            f"{percent(patterns['monotone_worsening'])} | "
            f"{percent(patterns['monotone_improvement'])} | "
            f"{percent(patterns['other_nonmonotone'])} |"
        )
    lines.extend(
        [
            "",
            "## Clinical trajectory indicators",
            "",
            "| Database | Any adjacent change | Any 24-72 h change | "
            "Transient death upgrade | Sustained death upgrade | "
            "Late death upgrade |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for database in SOURCES:
        record = summary["databases"][database]
        clinical = record["clinical_trajectories"]
        lines.append(
            f"| {record['label']} | "
            f"{percent(clinical['any_adjacent_change'])} | "
            f"{percent(clinical['any_24_to_72_change'])} | "
            f"{percent(clinical['transient_death_upgrade'])} | "
            f"{percent(clinical['sustained_death_upgrade'])} | "
            f"{percent(clinical['late_death_upgrade'])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "- A trajectory describes movement of fitted predictions, not",
            "  verified deterioration or recovery of the patient.",
            "- Home, facility, and death are ordinally coded for these summaries.",
            "  Facility combines heterogeneous post-acute destinations.",
            "- The cohort remains conditional on observation through 72 h.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261031)
    args = parser.parse_args()
    if args.replicates < 100:
        raise ValueError("At least 100 bootstrap replicates are required")

    databases = {
        database: analyze_database(
            database,
            spec,
            args.replicates,
            args.seed + index,
        )
        for index, (database, spec) in enumerate(SOURCES.items())
    }
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis": "complete_three_landmark_trajectory_patterns",
        "landmarks_hours": list(LANDMARKS),
        "class_order": list(CLASS_LABELS),
        "pattern_definitions": {
            "stable_all": "c24 = c48 = c72",
            "transient_48h_excursion": "c24 = c72 and c48 != c24",
            "monotone_worsening": (
                "c24 <= c48 <= c72 after excluding stable and transient patterns"
            ),
            "monotone_improvement": (
                "c24 >= c48 >= c72 after excluding stable and transient patterns"
            ),
            "other_nonmonotone": "None of the preceding categories",
        },
        "bootstrap": {
            "replicates": args.replicates,
            "seed": args.seed,
            "scope": (
                "Evaluation-sample uncertainty conditional on fitted "
                "out-of-fold predictions and original folds."
            ),
        },
        "databases": databases,
        "interpretation_boundary": [
            "Prediction trajectories do not estimate patient trajectories.",
            "The ordinal class order is an analysis assumption.",
            "The cohort is conditional on observation through 72 hours.",
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (OUT / "report.md").write_text(
        build_report(summary) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                database: {
                    "n": record["n"],
                    "stable_all": record["patterns"]["stable_all"],
                    "monotone_worsening": record["patterns"][
                        "monotone_worsening"
                    ],
                    "transient": record["patterns"][
                        "transient_48h_excursion"
                    ],
                }
                for database, record in databases.items()
            },
            indent=2,
        ),
        flush=True,
    )
    print(f"Wrote {OUT / 'summary.json'}", flush=True)
    print(f"Wrote {OUT / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
