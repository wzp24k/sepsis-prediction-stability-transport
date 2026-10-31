# -*- coding: utf-8 -*-
"""MIMIC-III complete-landmark stability replication with shared folds.

MIMIC-III is a same-institution, different-era database in this study. This
analysis is a replication, not an independent external validation.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier


ROOT = Path(os.environ.get("SEPSIS_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
DATA = ROOT / "data" / "processed" / "mimic3" / "design_all.parquet"
OUT = ROOT / "results" / "mimic3_shared"

LANDMARKS = (24, 48, 72)
CLASSES = np.asarray([0, 1, 2], dtype=np.int64)
CLASS_NAMES = {0: "home", 1: "facility", 2: "death"}
RANDOM_SEED = 20261029
BOOTSTRAP_REPLICATES = 2000
DEFAULT_THREADS = 16

XGB_PARAMETERS: dict[str, Any] = {
    "objective": "multi:softprob",
    "num_class": 3,
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.06,
    "subsample": 0.85,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
    "eval_metric": "mlogloss",
}


def feature_columns(frame: pd.DataFrame, milestone: int) -> list[str]:
    """Use the feature rule from build_mimic3_sensitivity.py."""
    laboratories = [
        "crea_min",
        "crea_max",
        "lac_min",
        "lac_max",
        "wbc_min",
        "wbc_max",
        "plt_min",
        "plt_max",
        "bili_min",
        "bili_max",
        "na_min",
        "na_max",
        "bun_min",
        "bun_max",
        "hgb_min",
        "hgb_max",
        "inr_max",
    ]
    static = ["age", "male"] + laboratories
    static += [
        column
        for column in frame.columns
        if column.startswith(("adm_", "unit_"))
    ]
    windows = [f"_w{index}_" for index in range(1, milestone // 24 + 1)]
    dynamic = [
        column
        for column in frame.columns
        if any(token in column for token in windows)
    ]
    columns = list(dict.fromkeys(static + dynamic))
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"missing features at {milestone} h: {missing}")
    return columns


def validate_shared_folds(
    folds: list[tuple[np.ndarray, np.ndarray]],
    groups: np.ndarray,
    n_rows: int,
) -> dict[str, Any]:
    tests = [test_index for _, test_index in folds]
    all_test = np.sort(np.concatenate(tests))
    held_groups = [set(groups[index].tolist()) for index in tests]
    overlaps = []
    for first in range(len(held_groups)):
        for second in range(first + 1, len(held_groups)):
            overlaps.append(len(held_groups[first] & held_groups[second]))
    if not np.array_equal(all_test, np.arange(n_rows)):
        raise ValueError("fold test partitions do not cover every row exactly once")
    if any(overlaps):
        raise ValueError("a subject_id appears in more than one held-out fold")
    return {
        "n_splits": len(folds),
        "test_sizes": [int(len(test_index)) for _, test_index in folds],
        "held_out_subject_counts": [len(values) for values in held_groups],
        "test_partitions_cover_every_row_once": True,
        "held_out_subject_overlap_counts": [int(value) for value in overlaps],
        "same_fold_partition_reused_at_all_landmarks": True,
    }


def bootstrap_mean_ci(
    values: np.ndarray,
    clusters: np.ndarray,
    seed: int,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> list[float]:
    """Patient-cluster bootstrap percentile interval for a row-level mean."""
    unique, inverse = np.unique(clusters, return_inverse=True)
    if len(unique) < 2:
        raise ValueError("bootstrap requires at least two patient clusters")
    sums = np.bincount(inverse, weights=values, minlength=len(unique))
    counts = np.bincount(inverse, minlength=len(unique)).astype(np.float64)
    rng = np.random.default_rng(seed)
    estimates: list[np.ndarray] = []
    generated = 0
    batch = 100
    while generated < replicates:
        size = min(batch, replicates - generated)
        selected = rng.integers(0, len(unique), size=(size, len(unique)))
        estimates.append(
            sums[selected].sum(axis=1) / counts[selected].sum(axis=1)
        )
        generated += size
    sample = np.concatenate(estimates)
    return [
        round(float(np.quantile(sample, 0.025)), 6),
        round(float(np.quantile(sample, 0.975)), 6),
    ]


def rounded(value: float) -> float:
    return round(float(value), 6)


def rate_record(
    indicator: np.ndarray, denominator: np.ndarray | None = None
) -> dict[str, Any]:
    indicator = np.asarray(indicator, dtype=bool)
    denominator = (
        np.ones(len(indicator), dtype=bool)
        if denominator is None
        else np.asarray(denominator, dtype=bool)
    )
    denominator_n = int(denominator.sum())
    numerator_n = int((indicator & denominator).sum())
    return {
        "n": denominator_n,
        "numerator": numerator_n,
        "estimate": (
            rounded(numerator_n / denominator_n) if denominator_n else None
        ),
    }


def macro_auc(y: np.ndarray, probability: np.ndarray) -> float:
    return rounded(
        roc_auc_score(
            y,
            probability,
            multi_class="ovr",
            average="macro",
            labels=CLASSES,
        )
    )


def build_and_train() -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    frame = pd.read_parquet(DATA)
    eligible = frame["y3"].notna() & (frame["los"] >= 72)
    cohort = frame.loc[eligible].copy().reset_index(drop=True)
    if cohort.empty:
        raise ValueError("complete three-landmark MIMIC-III cohort is empty")
    if not cohort["icustay_id"].is_unique:
        raise ValueError("icustay_id is not unique in the complete cohort")
    if not set(cohort["y3"].astype(int).unique()).issubset(set(CLASSES.tolist())):
        raise ValueError("unexpected y3 class")

    y = cohort["y3"].to_numpy(dtype=np.int64)
    ids = cohort["icustay_id"].to_numpy(dtype=np.int64)
    subjects = cohort["subject_id"].astype(str).to_numpy()
    folds = list(
        GroupKFold(n_splits=5).split(
            np.zeros((len(cohort), 1), dtype=np.float32), y, subjects
        )
    )
    fold_integrity = validate_shared_folds(folds, subjects, len(cohort))
    fold_id = np.full(len(cohort), -1, dtype=np.int8)
    for fold_index, (_, test_index) in enumerate(folds):
        fold_id[test_index] = fold_index
    if (fold_id < 0).any():
        raise ValueError("at least one row lacks a fold assignment")

    threads = int(
        os.environ.get("SEPSIS_XGB_THREADS", str(DEFAULT_THREADS))
    )
    parameters = {
        **XGB_PARAMETERS,
        "n_jobs": threads,
        "random_state": RANDOM_SEED,
        "verbosity": 0,
    }

    probability: dict[int, np.ndarray] = {}
    landmark_results: dict[str, Any] = {}
    fold_metrics: dict[str, list[dict[str, Any]]] = {}
    feature_names: dict[str, list[str]] = {}
    for milestone in LANDMARKS:
        columns = feature_columns(cohort, milestone)
        feature_names[str(milestone)] = columns
        x = cohort[columns].astype(np.float32)
        landmark_probability = np.zeros((len(cohort), 3), dtype=np.float64)
        current_fold_metrics: list[dict[str, Any]] = []
        for fold_index, (train_index, test_index) in enumerate(folds):
            model = XGBClassifier(**parameters)
            model.fit(x.iloc[train_index], y[train_index])
            fold_probability = model.predict_proba(x.iloc[test_index])
            landmark_probability[test_index] = fold_probability
            current_fold_metrics.append(
                {
                    "fold": fold_index,
                    "n_test": int(len(test_index)),
                    "macro_auc": macro_auc(y[test_index], fold_probability),
                }
            )
        probability[milestone] = landmark_probability
        landmark_results[str(milestone)] = {
            "n": int(len(cohort)),
            "macro_auc": macro_auc(y, landmark_probability),
            "fold_macro_auc": current_fold_metrics,
            "n_features": len(columns),
            "mean_max_probability": rounded(
                landmark_probability.max(axis=1).mean()
            ),
        }
        fold_metrics[str(milestone)] = current_fold_metrics
        print(
            "MIMIC-III shared-fold",
            milestone,
            landmark_results[str(milestone)],
            flush=True,
        )

    for milestone in LANDMARKS:
        if probability[milestone].shape != (len(cohort), 3):
            raise ValueError(f"invalid probability shape at {milestone} h")
        if not np.isfinite(probability[milestone]).all():
            raise ValueError(f"non-finite probability at {milestone} h")
        if not np.allclose(
            probability[milestone].sum(axis=1), 1.0, atol=1e-8
        ):
            raise ValueError(f"probability rows do not sum to one at {milestone} h")

    prediction = {
        milestone: probability[milestone].argmax(axis=1)
        for milestone in LANDMARKS
    }
    flip = prediction[24] != prediction[72]
    total_variation = 0.5 * np.abs(
        probability[24] - probability[72]
    ).sum(axis=1)
    flip_ci = bootstrap_mean_ci(
        flip.astype(np.float64), subjects, RANDOM_SEED + 1
    )
    tv_ci = bootstrap_mean_ci(total_variation, subjects, RANDOM_SEED + 2)

    transition_counts = np.zeros((3, 3), dtype=np.int64)
    np.add.at(
        transition_counts, (prediction[24], prediction[72]), 1
    )
    transition_row_proportions = np.divide(
        transition_counts,
        transition_counts.sum(axis=1, keepdims=True),
        out=np.zeros_like(transition_counts, dtype=np.float64),
        where=transition_counts.sum(axis=1, keepdims=True) > 0,
    )

    confidence_06 = probability[24].max(axis=1) >= 0.6
    return_24h = (
        (prediction[24] == prediction[72])
        & (prediction[24] != prediction[48])
    )
    major_upgrade_to_death = (
        (prediction[24] < 2) & (prediction[72] == 2)
    )
    home_to_death = (
        (prediction[24] == 0) & (prediction[72] == 2)
    )
    facility_to_death = (
        (prediction[24] == 1) & (prediction[72] == 2)
    )
    home_to_facility = (
        (prediction[24] == 0) & (prediction[72] == 1)
    )

    distance_24 = np.abs(prediction[24] - y)
    distance_72 = np.abs(prediction[72] - y)
    gain = distance_72 < distance_24
    loss = distance_72 > distance_24
    tie = distance_72 == distance_24

    row_index = np.arange(len(y))
    p_true_24 = probability[24][row_index, y]
    p_true_72 = probability[72][row_index, y]
    p_true_change = p_true_72 - p_true_24
    improved = p_true_change > 1e-12
    worsened = p_true_change < -1e-12
    unchanged = np.abs(p_true_change) <= 1e-12

    role = (
        "Same-institution, different-era replication. MIMIC-III is not an "
        "independent external validation and is not pooled into the primary "
        "MIMIC-IV/eICU estimate."
    )
    metadata = {
        "analysis": "mimic3_complete_landmark_shared_fold_stability_replication",
        "database": "MIMIC-III v1.4",
        "role": role,
        "class_mapping": {str(key): value for key, value in CLASS_NAMES.items()},
        "estimand": (
            "Out-of-fold probabilities and argmax discharge-destination "
            "predictions at 24, 48, and 72 hours in one complete-landmark "
            "cohort, with intervals for the proportions of patients whose "
            "24 h and 72 h predictions differ and for mean probability "
            "total variation."
        ),
        "complete_landmark_cohort": (
            "y3 is non-null and ICU length of stay is at least 72 hours; "
            "all three landmark predictions use the same rows."
        ),
        "fold_partition": (
            "One fixed patient-grouped five-fold partition by subject_id, "
            "shared by all three landmark models."
        ),
        "bootstrap": {
            "method": "patient-level nonparametric percentile bootstrap",
            "replicates": BOOTSTRAP_REPLICATES,
            "resampling_unit": "subject_id",
            "cluster_count": int(pd.unique(subjects).size),
            "conditioning": (
                "Conditional on the fitted out-of-fold probabilities and the "
                "fixed five-fold partition. Training-model and fold-selection "
                "uncertainty are not resampled and models are not refitted."
            ),
        },
    }

    summary = {
        "metadata": metadata,
        "cohort": {
            "source": str(DATA.relative_to(ROOT)),
            "n": int(len(cohort)),
            "patient_count": int(pd.unique(subjects).size),
            "outcome_counts": {
                str(value): int((y == value).sum()) for value in CLASSES
            },
            "duration_column": "los",
            "duration_threshold_hours": 72,
            "conditioning": (
                "Patients had a mapped outcome and remained observable through "
                "72 hours. Results do not generalize to patients discharged or "
                "dying before 72 hours."
            ),
        },
        "training": {
            "algorithm": "XGBoost",
            "parameters": parameters,
            "random_seed": RANDOM_SEED,
            "seed_count": 1,
            "fold_integrity": fold_integrity,
            "fold_id": fold_id.tolist(),
            "feature_columns": feature_names,
            "feature_rule": (
                "First-24-hour static/demographic/laboratory features plus "
                "non-overlapping 24-hour summary windows available through the "
                "landmark; rules copied from "
                "src/sepsis/build_mimic3_sensitivity.py."
            ),
        },
        "discrimination": {
            "target": "three-class discharge destination",
            "macro_auc": {
                str(milestone): landmark_results[str(milestone)]["macro_auc"]
                for milestone in LANDMARKS
            },
            "landmarks": landmark_results,
        },
        "stability": {
            "flip_24_72": {
                "estimate": rounded(flip.mean()),
                "ci95": flip_ci,
                "n": int(len(y)),
                "numerator": int(flip.sum()),
            },
            "total_variation_24_72": {
                "estimate": rounded(total_variation.mean()),
                "ci95": tv_ci,
                "n": int(len(y)),
            },
            "return_to_24h_class": rate_record(return_24h),
            "confidence_ge_0_6": {
                "coverage": rate_record(confidence_06),
                "subsequent_flip_24_72": rate_record(flip, confidence_06),
                "subsequent_major_upgrade_to_death": rate_record(
                    major_upgrade_to_death, confidence_06
                ),
            },
            "same_fold_partition_used_at_all_landmarks": True,
            "interpretation": (
                "Describes prediction instability through 72 hours in a "
                "complete-landmark cohort; it does not estimate the exact time "
                "at which predictions become stable."
            ),
        },
        "transition_24_72": {
            "counts": transition_counts.tolist(),
            "row_proportions": np.round(
                transition_row_proportions, 6
            ).tolist(),
            "row_labels": [CLASS_NAMES[value] for value in CLASSES],
            "column_labels": [CLASS_NAMES[value] for value in CLASSES],
        },
        "clinical_transitions": {
            "major_upgrade_to_death": rate_record(major_upgrade_to_death),
            "home_to_death": rate_record(home_to_death),
            "facility_to_death": rate_record(facility_to_death),
            "home_to_facility": rate_record(home_to_facility),
            "note": (
                "Major upgrade to death combines home-to-death and "
                "facility-to-death transitions."
            ),
        },
        "correctness": {
            "distance_definition": (
                "Absolute difference between the predicted and final "
                "three-class outcome: |argmax probability - y3|."
            ),
            "all": {
                "gain": rate_record(gain),
                "loss": rate_record(loss),
                "tie": rate_record(tie),
                "distance_24_mean": rounded(distance_24.mean()),
                "distance_72_mean": rounded(distance_72.mean()),
                "distance_change_72_minus_24_mean": rounded(
                    (distance_72 - distance_24).mean()
                ),
            },
            "among_flips": {
                "gain": rate_record(gain, flip),
                "loss": rate_record(loss, flip),
                "tie": rate_record(tie, flip),
            },
            "true_class_probability": {
                "definition": (
                    "p(y3) at each landmark; improvement means p72(y3) is "
                    "greater than p24(y3) by more than 1e-12."
                ),
                "p_true_24_mean": rounded(p_true_24.mean()),
                "p_true_72_mean": rounded(p_true_72.mean()),
                "change_72_minus_24_all_mean": rounded(
                    p_true_change.mean()
                ),
                "among_flips": {
                    "improved": rate_record(improved, flip),
                    "worsened": rate_record(worsened, flip),
                    "unchanged": rate_record(unchanged, flip),
                    "change_mean": rounded(p_true_change[flip].mean())
                    if flip.any()
                    else None,
                },
            },
        },
        "artifacts": {
            "summary": "results/mimic3_shared/summary.json",
            "probabilities": "results/mimic3_shared/mimic3_shared.npz",
            "report": "results/mimic3_shared/report.md",
        },
    }
    arrays = {
        "p24": probability[24].astype(np.float64),
        "p48": probability[48].astype(np.float64),
        "p72": probability[72].astype(np.float64),
        "ids": ids.astype(np.int64),
        "y": y.astype(np.int64),
        "y3": y.astype(np.int64),
        "groups": subjects.astype("U"),
        "clusters": subjects.astype("U"),
        "fold_id": fold_id.astype(np.int8),
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
    }
    return summary, arrays


def percent(value: float | None) -> str:
    return "NA" if value is None else f"{100.0 * value:.2f}%"


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] + ["---:"] * (len(headers) - 1)) + "|",
    ]
    lines += [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in rows
    ]
    return "\n".join(lines)


def write_report(summary: dict[str, Any]) -> str:
    cohort = summary["cohort"]
    stability = summary["stability"]
    transition = summary["transition_24_72"]
    clinical = summary["clinical_transitions"]
    correctness = summary["correctness"]
    aucs = summary["discrimination"]["macro_auc"]

    count_rows = []
    for source in range(3):
        count_rows.append(
            [
                CLASS_NAMES[source],
                *transition["counts"][source],
                *[percent(value) for value in transition["row_proportions"][source]],
            ]
        )

    all_correctness = correctness["all"]
    flips_correctness = correctness["among_flips"]
    true_probability = correctness["true_class_probability"]
    among_flip_probability = true_probability["among_flips"]

    lines = [
        "# MIMIC-III Complete-Landmark Shared-Fold Stability Replication",
        "",
        "Generated by `src/sepsis/analyze_mimic3_shared.py`.",
        "",
        "## Role and estimand",
        "",
        summary["metadata"]["role"],
        "",
        summary["metadata"]["estimand"],
        "",
        "The cohort requires a non-null three-class outcome and observed ICU "
        "length of stay of at least 72 hours. The same 24, 48, and 72 hour "
        "models use the same rows, identical XGBoost hyperparameters, and one "
        "fixed five-fold partition grouped by `subject_id`.",
        "",
        "## Cohort and discrimination",
        "",
        markdown_table(
            ["Cohort", "n", "Patients", "Home", "Facility", "Death"],
            [
                [
                    "MIMIC-III",
                    f"{cohort['n']:,}",
                    f"{cohort['patient_count']:,}",
                    f"{cohort['outcome_counts']['0']:,}",
                    f"{cohort['outcome_counts']['1']:,}",
                    f"{cohort['outcome_counts']['2']:,}",
                ]
            ],
        ),
        "",
        markdown_table(
            ["Landmark", "Macro-AUC", "Features"],
            [
                [
                    f"{milestone} h",
                    f"{aucs[str(milestone)]:.3f}",
                    summary["discrimination"]["landmarks"][str(milestone)][
                        "n_features"
                    ],
                ]
                for milestone in LANDMARKS
            ],
        ),
        "",
        "## Stability",
        "",
        markdown_table(
            ["Metric", "n / numerator", "Estimate", "95% CI"],
            [
                [
                    "24-to-72 h class flip",
                    f"{stability['flip_24_72']['n']:,} / "
                    f"{stability['flip_24_72']['numerator']:,}",
                    percent(stability["flip_24_72"]["estimate"]),
                    "[" + ", ".join(
                        percent(value)
                        for value in stability["flip_24_72"]["ci95"]
                    ) + "]",
                ],
                [
                    "24-to-72 h mean total variation",
                    f"{stability['total_variation_24_72']['n']:,}",
                    f"{stability['total_variation_24_72']['estimate']:.3f}",
                    "[" + ", ".join(
                        f"{value:.3f}"
                        for value in stability[
                            "total_variation_24_72"
                        ]["ci95"]
                    ) + "]",
                ],
                [
                    "Return to 24 h class after a 48 h change",
                    f"{stability['return_to_24h_class']['n']:,} / "
                    f"{stability['return_to_24h_class']['numerator']:,}",
                    percent(stability["return_to_24h_class"]["estimate"]),
                    "NA",
                ],
                [
                    "24 h confidence >= 0.6 coverage",
                    f"{stability['confidence_ge_0_6']['coverage']['n']:,} / "
                    f"{stability['confidence_ge_0_6']['coverage']['numerator']:,}",
                    percent(
                        stability["confidence_ge_0_6"]["coverage"]["estimate"]
                    ),
                    "NA",
                ],
                [
                    "Subsequent flip in confidence >= 0.6 stratum",
                    f"{stability['confidence_ge_0_6']['subsequent_flip_24_72']['n']:,} / "
                    f"{stability['confidence_ge_0_6']['subsequent_flip_24_72']['numerator']:,}",
                    percent(
                        stability["confidence_ge_0_6"][
                            "subsequent_flip_24_72"
                        ]["estimate"]
                    ),
                    "NA",
                ],
            ],
        ),
        "",
        "Bootstrap intervals use 2,000 patient-level percentile replicates. "
        "They are conditional on the fitted out-of-fold probabilities and the "
        "fixed fold partition; training and fold-selection uncertainty are not "
        "resampled.",
        "",
        "## Full 24-to-72 h transition",
        "",
        markdown_table(
            [
                "24 h class",
                "to home, n",
                "to facility, n",
                "to death, n",
                "to home, row %",
                "to facility, row %",
                "to death, row %",
            ],
            count_rows,
        ),
        "",
        "## Clinical transitions and correctness",
        "",
        markdown_table(
            ["Metric", "n / numerator", "Estimate"],
            [
                [
                    "Major upgrade to death",
                    f"{clinical['major_upgrade_to_death']['n']:,} / "
                    f"{clinical['major_upgrade_to_death']['numerator']:,}",
                    percent(clinical["major_upgrade_to_death"]["estimate"]),
                ],
                [
                    "Home to death",
                    f"{clinical['home_to_death']['n']:,} / "
                    f"{clinical['home_to_death']['numerator']:,}",
                    percent(clinical["home_to_death"]["estimate"]),
                ],
                [
                    "Facility to death",
                    f"{clinical['facility_to_death']['n']:,} / "
                    f"{clinical['facility_to_death']['numerator']:,}",
                    percent(clinical["facility_to_death"]["estimate"]),
                ],
                [
                    "Home to facility",
                    f"{clinical['home_to_facility']['n']:,} / "
                    f"{clinical['home_to_facility']['numerator']:,}",
                    percent(clinical["home_to_facility"]["estimate"]),
                ],
            ],
        ),
        "",
        markdown_table(
            ["Correctness relative to final class", "All, n / rate", "Among flips, n / rate"],
            [
                [
                    "Distance gain",
                    f"{all_correctness['gain']['numerator']:,} / "
                    f"{percent(all_correctness['gain']['estimate'])}",
                    f"{flips_correctness['gain']['numerator']:,} / "
                    f"{percent(flips_correctness['gain']['estimate'])}",
                ],
                [
                    "Distance loss",
                    f"{all_correctness['loss']['numerator']:,} / "
                    f"{percent(all_correctness['loss']['estimate'])}",
                    f"{flips_correctness['loss']['numerator']:,} / "
                    f"{percent(flips_correctness['loss']['estimate'])}",
                ],
                [
                    "Distance tie",
                    f"{all_correctness['tie']['numerator']:,} / "
                    f"{percent(all_correctness['tie']['estimate'])}",
                    f"{flips_correctness['tie']['numerator']:,} / "
                    f"{percent(flips_correctness['tie']['estimate'])}",
                ],
            ],
        ),
        "",
        "Among the flips, the true-class probability improved in "
        f"{among_flip_probability['improved']['numerator']:,} of "
        f"{among_flip_probability['improved']['n']:,} patients "
        f"({percent(among_flip_probability['improved']['estimate'])}), "
        "worsened in "
        f"{among_flip_probability['worsened']['numerator']:,} "
        f"({percent(among_flip_probability['worsened']['estimate'])}), and was "
        "unchanged in "
        f"{among_flip_probability['unchanged']['numerator']:,} "
        f"({percent(among_flip_probability['unchanged']['estimate'])}). "
        "The mean true-class probability change among flips was "
        f"{among_flip_probability['change_mean']:.3f}.",
        "",
        "## Interpretation limits",
        "",
        "- MIMIC-III is same-institution and different-era; this is replication, "
        "not independent external validation.",
        "- The complete-landmark cohort conditions on survival and observation "
        "through 72 hours.",
        "- Three landmarks describe persistence through 72 hours but do not "
        "identify the exact time at which a prediction becomes stable.",
        "- Correctness uses distance to the eventual three-class outcome; it "
        "does not establish clinical appropriateness or utility.",
        "",
        "## Artifacts",
        "",
        "- `results/mimic3_shared/summary.json`",
        "- `results/mimic3_shared/mimic3_shared.npz`",
        "- `results/mimic3_shared/report.md`",
        "- `src/sepsis/analyze_mimic3_shared.py`",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary, arrays = build_and_train()
    with (OUT / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    np.savez_compressed(OUT / "mimic3_shared.npz", **arrays)
    report = write_report(summary)
    (OUT / "report.md").write_text(report, encoding="utf-8")
    print(
        json.dumps(
            {
                "n": summary["cohort"]["n"],
                "macro_auc": summary["discrimination"]["macro_auc"],
                "flip_24_72": summary["stability"]["flip_24_72"],
                "total_variation_24_72": summary["stability"][
                    "total_variation_24_72"
                ],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
