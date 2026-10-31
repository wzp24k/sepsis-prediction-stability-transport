# -*- coding: utf-8 -*-
"""Train XGBoost landmarks on one fixed patient-grouped fold partition.

The three landmark models are restricted to the same complete landmark cohort
and use identical hyperparameters, random seed, rows, and cross-validation
folds. Only the feature window changes with the landmark.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier


ROOT = Path(os.environ.get("SEPSIS_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
DATA = ROOT / "data" / "processed"
FORMAL_SUMMARY = ROOT / "results" / "formal_summary"
OUT = ROOT / "results" / "shared_fold"
OUT.mkdir(parents=True, exist_ok=True)

LANDMARKS = (24, 48, 72)
CLASSES = np.asarray([0, 1, 2], dtype=np.int64)
RANDOM_SEED = 20261029
BOOTSTRAP_REPLICATES = 2000
DATABASES = ("mimic4", "eicu")

CONFIG: dict[str, dict[str, str]] = {
    "mimic4": {
        "duration": "los",
        "group": "subject_id",
        "id": "stay_id",
        "cluster": "subject_id",
    },
    "eicu": {
        "duration": "los_h",
        "group": "uniquepid",
        "id": "patientunitstayid",
        "cluster": "hospitalid",
    },
}

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
    """Return first-24-hour static features plus windows through the landmark."""
    laboratory = [
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
    static = ["age", "male"] + laboratory
    static += [
        column
        for column in frame.columns
        if column.startswith(("adm_", "unit_", "src_", "ut_"))
    ]
    windows = [f"_w{index}_" for index in range(1, milestone // 24 + 1)]
    dynamic = [
        column for column in frame.columns if any(token in column for token in windows)
    ]
    return [
        column
        for column in dict.fromkeys(static + dynamic)
        if column in frame.columns
    ]


def expected_calibration_error(
    y: np.ndarray, probability: np.ndarray, bins: int = 10
) -> float:
    confidence = probability.max(axis=1)
    prediction = probability.argmax(axis=1)
    correct = prediction == y
    edges = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        selected = (
            (confidence >= lower) & (confidence <= upper)
            if upper == 1.0
            else (confidence >= lower) & (confidence < upper)
        )
        if selected.any():
            value += (
                abs(correct[selected].mean() - confidence[selected].mean())
                * selected.mean()
            )
    return float(value)


def metric_block(y: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    prediction = probability.argmax(axis=1)
    observed = np.eye(3, dtype=np.float64)[y]
    return {
        "n": int(len(y)),
        "macro_auc": round(
            float(
                roc_auc_score(
                    y,
                    probability,
                    multi_class="ovr",
                    average="macro",
                    labels=CLASSES,
                )
            ),
            6,
        ),
        "class_auc": {
            str(class_index): round(
                float(roc_auc_score((y == class_index).astype(int), probability[:, class_index])),
                6,
            )
            for class_index in CLASSES
        },
        "accuracy": round(float(accuracy_score(y, prediction)), 6),
        "log_loss": round(
            float(log_loss(y, probability, labels=CLASSES)), 6
        ),
        "brier": round(float(np.mean(np.sum((probability - observed) ** 2, axis=1))), 6),
        "ece_10bin": round(expected_calibration_error(y, probability), 6),
        "mean_max_probability": round(float(probability.max(axis=1).mean()), 6),
    }


def metric_delta(
    first: dict[str, Any], second: dict[str, Any], fields: tuple[str, ...]
) -> dict[str, float]:
    return {
        field: round(float(first[field] - second[field]), 6)
        for field in fields
        if field in first and field in second
    }


def bootstrap_ratio_interval(
    values: np.ndarray,
    clusters: np.ndarray,
    seed: int,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> list[float] | None:
    """Cluster-bootstrap the mean of row-level values."""
    unique, inverse = np.unique(clusters, return_inverse=True)
    if len(unique) < 2:
        return None
    sums = np.bincount(inverse, weights=values, minlength=len(unique))
    counts = np.bincount(inverse, minlength=len(unique)).astype(np.float64)
    rng = np.random.default_rng(seed)
    estimates: list[np.ndarray] = []
    batch = 100
    generated = 0
    while generated < replicates:
        size = min(batch, replicates - generated)
        selected = rng.integers(0, len(unique), size=(size, len(unique)))
        estimates.append(sums[selected].sum(axis=1) / counts[selected].sum(axis=1))
        generated += size
    sample = np.concatenate(estimates)
    return [
        round(float(np.quantile(sample, 0.025)), 6),
        round(float(np.quantile(sample, 0.975)), 6),
    ]


def stability_block(
    y: np.ndarray,
    probability: dict[int, np.ndarray],
    clusters: np.ndarray,
    database: str,
) -> dict[str, Any]:
    prediction = {
        milestone: probability[milestone].argmax(axis=1) for milestone in LANDMARKS
    }
    transition = np.zeros((3, 3), dtype=np.int64)
    for first, second in zip(prediction[24], prediction[72]):
        transition[int(first), int(second)] += 1

    changed = prediction[24] != prediction[72]
    direction = []
    for first in CLASSES:
        for second in CLASSES:
            if first == second:
                continue
            count = int(transition[first, second])
            direction.append(
                {
                    "from": int(first),
                    "to": int(second),
                    "count": count,
                    "percent_all": round(float(count / len(y)), 6),
                    "percent_of_flips": (
                        round(float(count / changed.sum()), 6) if changed.any() else None
                    ),
                }
            )

    result: dict[str, Any] = {
        "n": int(len(y)),
        "transition_24_to_72_counts": transition.tolist(),
        "directions_24_to_72": direction,
        "return_to_24h_class": round(
            float(
                (
                    (prediction[24] == prediction[72])
                    & (prediction[24] != prediction[48])
                ).mean()
            ),
            6,
        ),
    }
    for first, second in ((24, 48), (48, 72), (24, 72)):
        flip = (prediction[first] != prediction[second]).astype(np.float64)
        variation = 0.5 * np.abs(
            probability[first] - probability[second]
        ).sum(axis=1)
        key = f"{first}_{second}"
        result[f"flip_{key}"] = round(float(flip.mean()), 6)
        result[f"flip_{key}_ci95"] = bootstrap_ratio_interval(
            flip,
            clusters,
            RANDOM_SEED + int(first * 100 + second) + (0 if database == "mimic4" else 1),
        )
        result[f"tv_{key}"] = round(float(variation.mean()), 6)
        result[f"tv_{key}_ci95"] = bootstrap_ratio_interval(
            variation,
            clusters,
            RANDOM_SEED + int(first * 100 + second) + (10 if database == "mimic4" else 11),
        )

    confidence = probability[24].max(axis=1)
    result["confidence_24h"] = {}
    for threshold in (0.5, 0.6, 0.7):
        selected = confidence >= threshold
        values = (prediction[24][selected] != prediction[72][selected]).astype(
            np.float64
        )
        result["confidence_24h"][str(threshold)] = {
            "coverage": round(float(selected.mean()), 6),
            "n": int(selected.sum()),
            "flip_24_72": round(float(values.mean()), 6) if selected.any() else None,
            "flip_24_72_ci95": (
                bootstrap_ratio_interval(
                    values,
                    clusters[selected],
                    RANDOM_SEED + int(threshold * 1000)
                    + (20 if database == "mimic4" else 21),
                )
                if selected.any()
                else None
            ),
        }

    result["prediction_distribution"] = {}
    for milestone in LANDMARKS:
        counts = np.bincount(prediction[milestone], minlength=3)
        result["prediction_distribution"][str(milestone)] = {
            str(index): int(count) for index, count in enumerate(counts)
        }
    return result


def validate_shared_folds(
    folds: list[tuple[np.ndarray, np.ndarray]],
    groups: np.ndarray,
    n_rows: int,
) -> dict[str, Any]:
    test_indices = [test for _, test in folds]
    all_test = np.sort(np.concatenate(test_indices))
    held_groups = [set(groups[test].tolist()) for test in test_indices]
    overlaps = []
    for first in range(len(held_groups)):
        for second in range(first + 1, len(held_groups)):
            overlaps.append(len(held_groups[first].intersection(held_groups[second])))
    if not np.array_equal(all_test, np.arange(n_rows)):
        raise ValueError("fold test partitions do not cover every row exactly once")
    if any(overlaps):
        raise ValueError("a patient group appears in more than one held-out fold")
    return {
        "n_splits": len(folds),
        "test_sizes": [int(len(test)) for _, test in folds],
        "held_out_group_counts": [len(values) for values in held_groups],
        "test_partitions_cover_every_row_once": True,
        "held_out_group_overlap_counts": [int(value) for value in overlaps],
        "same_fold_partition_reused_at_all_landmarks": True,
    }


def load_reference_dg(database: str) -> dict[str, np.ndarray]:
    path = FORMAL_SUMMARY / f"{database}_ordinal_temporal_ensemble.npz"
    if not path.exists():
        raise FileNotFoundError(f"missing DG-GRU reference: {path}")
    with np.load(path) as bundle:
        return {key: bundle[key].copy() for key in bundle.files}


def verify_formal_dg_fold_partition(
    database: str,
    ids: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    fold_id: np.ndarray,
) -> dict[str, Any]:
    """Rebuild the fold partition used to train the formal DG-GRU reference."""
    config = CONFIG[database]
    filename = "design.parquet" if database == "mimic4" else "design_v3.parquet"
    frame = pd.read_parquet(
        DATA / database / filename,
        columns=[config["id"], config["group"], "y3"],
    )
    reference_ids = frame[config["id"]].to_numpy(dtype=np.int64)
    if set(reference_ids.tolist()) != set(ids.tolist()):
        return {
            "verified": False,
            "reason": "formal design cohort does not have the exact id set",
        }
    reference_folds = list(
        GroupKFold(n_splits=5).split(
            np.zeros((len(frame), 1), dtype=np.float32),
            frame["y3"].to_numpy(dtype=np.int64),
            frame[config["group"]].astype(str).to_numpy(),
        )
    )
    reference_fold_id = np.full(len(frame), -1, dtype=np.int8)
    for index, (_, test_index) in enumerate(reference_folds):
        reference_fold_id[test_index] = index
    positions = pd.Index(reference_ids).get_indexer(ids)
    if (positions < 0).any():
        return {
            "verified": False,
            "exact_id_set_match": True,
            "reason": "some ids could not be aligned to the formal design cohort",
        }
    aligned_fold_id = reference_fold_id[positions]
    aligned_y = frame["y3"].to_numpy(dtype=np.int64)[positions]
    aligned_groups = frame[config["group"]].astype(str).to_numpy()[positions]
    verified = bool(np.array_equal(aligned_fold_id, fold_id))
    return {
        "verified": verified,
        "formal_design_file": str(DATA / database / filename),
        "exact_id_set_match": True,
        "outcome_match_after_id_alignment": bool(np.array_equal(aligned_y, y)),
        "group_match_after_id_alignment": bool(
            np.array_equal(aligned_groups, groups)
        ),
        "fold_ids_identical_after_id_alignment": verified,
    }


def align_reference(
    database: str,
    ids: np.ndarray,
    y: np.ndarray,
    reference: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    reference_ids = reference["ids"].astype(np.int64)
    positions = pd.Index(reference_ids).get_indexer(ids)
    if (positions < 0).any() or len(np.unique(positions)) != len(ids):
        raise ValueError(f"{database}: DG-GRU reference is not an exact patient match")
    aligned = {
        "ids": reference_ids[positions],
        "y": reference["y"].astype(np.int64)[positions],
        "probability": reference["probability"].astype(np.float64)[positions],
    }
    aligned["probability"] = np.clip(aligned["probability"], 0.0, 1.0)
    aligned["probability"] /= aligned["probability"].sum(axis=2, keepdims=True)
    if not np.array_equal(aligned["ids"], ids):
        raise ValueError(f"{database}: DG-GRU ids failed alignment")
    if not np.array_equal(aligned["y"], y):
        raise ValueError(f"{database}: DG-GRU outcomes failed alignment")
    return aligned, {
        "reference_file": str(
            FORMAL_SUMMARY / f"{database}_ordinal_temporal_ensemble.npz"
        ),
        "exact_id_set_match": True,
        "outcome_match_after_id_alignment": True,
        "reference_saved_in_fold_concatenation_order": (
            not np.array_equal(reference_ids, ids)
        ),
        "rows_reordered_by_id_for_pairing": int(
            np.sum(reference_ids[positions] != ids)
        ),
    }


def train_database(database: str, threads: int) -> dict[str, Any]:
    started = time.time()
    config = CONFIG[database]
    frame = pd.read_parquet(DATA / database / "design_all.parquet")
    eligible = frame["y3"].notna() & (frame[config["duration"]] >= 72)
    cohort = frame.loc[eligible].copy().reset_index(drop=True)
    if cohort.empty:
        raise ValueError(f"{database}: complete landmark cohort is empty")
    if not cohort[config["id"]].is_unique:
        raise ValueError(f"{database}: stay identifiers are not unique")

    y = cohort["y3"].to_numpy(dtype=np.int64)
    ids = cohort[config["id"]].to_numpy(dtype=np.int64)
    groups = cohort[config["group"]].astype(str).to_numpy()
    clusters = cohort[config["cluster"]].astype(str).to_numpy()

    # GroupKFold depends on row order, outcomes, and groups only. Reusing these
    # exact indices at every landmark fixes the partition without a new seed.
    folds = list(
        GroupKFold(n_splits=5).split(
            np.zeros((len(cohort), 1), dtype=np.float32), y, groups
        )
    )
    fold_integrity = validate_shared_folds(folds, groups, len(cohort))
    fold_id = np.full(len(cohort), -1, dtype=np.int8)
    for index, (_, test_index) in enumerate(folds):
        fold_id[test_index] = index
    if (fold_id < 0).any():
        raise ValueError(f"{database}: some rows do not have a fold assignment")
    formal_dg_fold_check = verify_formal_dg_fold_partition(
        database, ids, y, groups, fold_id
    )

    probability: dict[int, np.ndarray] = {}
    feature_names: dict[str, list[str]] = {}
    metrics: dict[str, Any] = {}
    fold_metrics: dict[str, list[dict[str, Any]]] = {}
    parameters = {
        **XGB_PARAMETERS,
        "n_jobs": int(threads),
        "random_state": RANDOM_SEED,
        "verbosity": 0,
    }

    for milestone in LANDMARKS:
        columns = feature_columns(cohort, milestone)
        if not columns:
            raise ValueError(f"{database}: no features available at {milestone} h")
        x = cohort[columns].astype(np.float32)
        landmark_probability = np.zeros((len(cohort), 3), dtype=np.float64)
        landmark_fold_metrics: list[dict[str, Any]] = []
        for fold_index, (train_index, test_index) in enumerate(folds):
            model = XGBClassifier(**parameters)
            model.fit(x.iloc[train_index], y[train_index])
            fold_probability = model.predict_proba(x.iloc[test_index]).astype(
                np.float64
            )
            fold_probability = np.clip(fold_probability, 0.0, 1.0)
            fold_probability /= fold_probability.sum(axis=1, keepdims=True)
            landmark_probability[test_index] = fold_probability
            landmark_fold_metrics.append(
                {
                    "fold": fold_index,
                    "n_test": int(len(test_index)),
                    **metric_block(y[test_index], fold_probability),
                }
            )
        if not np.allclose(landmark_probability.sum(axis=1), 1.0):
            raise ValueError(f"{database}: {milestone} h probabilities do not sum to one")
        probability[milestone] = landmark_probability
        feature_names[str(milestone)] = columns
        metrics[str(milestone)] = {
            **metric_block(y, landmark_probability),
            "feature_count": len(columns),
        }
        fold_metrics[str(milestone)] = landmark_fold_metrics
        print(
            database,
            milestone,
            {
                "features": len(columns),
                "macro_auc": metrics[str(milestone)]["macro_auc"],
                "elapsed_seconds": round(time.time() - started, 1),
            },
            flush=True,
        )

    stability = stability_block(y, probability, clusters, database)
    reference, alignment = align_reference(
        database, ids, y, load_reference_dg(database)
    )
    reference_metrics = {
        str(milestone): metric_block(y, reference["probability"][:, index, :])
        for index, milestone in enumerate(LANDMARKS)
    }
    reference_prediction = reference["probability"].argmax(axis=2)
    xgb_prediction = {
        milestone: probability[milestone].argmax(axis=1) for milestone in LANDMARKS
    }
    reference_flip = (
        reference_prediction[:, 0] != reference_prediction[:, 2]
    ).astype(np.float64)
    xgb_flip = (xgb_prediction[24] != xgb_prediction[72]).astype(np.float64)
    reference_tv = 0.5 * np.abs(
        reference["probability"][:, 0, :] - reference["probability"][:, 2, :]
    ).sum(axis=1)
    xgb_tv = 0.5 * np.abs(probability[24] - probability[72]).sum(axis=1)

    comparison = {
        "reference": "results/formal_summary/{database}_ordinal_temporal_ensemble.npz",
        "same_patient_queue_and_outcomes_after_id_alignment": True,
        "same_patient_fold_partition_as_formal_dg_reference": formal_dg_fold_check[
            "verified"
        ],
        "formal_dg_fold_partition_check": formal_dg_fold_check,
        "same_landmark_feature_information": False,
        "xgb_model": {
            "seeds": [RANDOM_SEED],
            "seed_ensemble_count": 1,
            "fold_scope": "one patient-grouped 5-fold partition shared by all landmarks",
        },
        "dg_gru_model": {
            "seeds": [1, 2, 3],
            "seed_ensemble_count": 3,
            "prediction": "mean of the three seed probabilities",
            "fold_scope": "patient-grouped 5-fold partition generated once on the same complete queue",
            "configuration": "ordinal temporal DG-GRU",
        },
        "comparability_warning": (
            "Predictions are paired on the same patients, outcomes, and shared "
            "fold partition, but XGBoost is a single-seed model and DG-GRU is a "
            "three-seed probability ensemble. Differences combine algorithm, "
            "feature representation, and seed-averaging effects."
        ),
        "alignment": alignment,
        "metrics": {
            "xgb": metrics,
            "dg_gru_ordinal_temporal_three_seed_ensemble": reference_metrics,
            "xgb_minus_dg": {
                str(milestone): metric_delta(
                    metrics[str(milestone)],
                    reference_metrics[str(milestone)],
                    ("macro_auc", "accuracy", "log_loss", "brier"),
                )
                for milestone in LANDMARKS
            },
        },
        "stability": {
            "xgb": {
                "flip_24_72": stability["flip_24_72"],
                "tv_24_72": stability["tv_24_72"],
            },
            "dg_gru_ordinal_temporal_three_seed_ensemble": {
                "flip_24_72": round(float(reference_flip.mean()), 6),
                "tv_24_72": round(float(reference_tv.mean()), 6),
            },
            "xgb_minus_dg": {
                "flip_24_72": round(float((xgb_flip - reference_flip).mean()), 6),
                "flip_24_72_percentage_points": round(
                    float((xgb_flip - reference_flip).mean() * 100.0), 4
                ),
                "flip_24_72_ci95": bootstrap_ratio_interval(
                    xgb_flip - reference_flip,
                    clusters,
                    RANDOM_SEED + (30 if database == "mimic4" else 31),
                ),
                "tv_24_72": round(float((xgb_tv - reference_tv).mean()), 6),
                "tv_24_72_ci95": bootstrap_ratio_interval(
                    xgb_tv - reference_tv,
                    clusters,
                    RANDOM_SEED + (40 if database == "mimic4" else 41),
                ),
            },
        },
    }

    output_path = OUT / f"{database}_shared_fold.npz"
    np.savez_compressed(
        output_path,
        p24=probability[24],
        p48=probability[48],
        p72=probability[72],
        y3=y,
        ids=ids,
        groups=groups.astype("U"),
        fold_id=fold_id,
        clusters=clusters.astype("U"),
    )

    outcome_counts = np.bincount(y, minlength=3)
    result = {
        "database": database,
        "elapsed_seconds": round(time.time() - started, 1),
        "cohort": {
            "definition": "y3 is non-null and observed duration is at least 72 hours",
            "duration_column": config["duration"],
            "id_column": config["id"],
            "cv_group_column": config["group"],
            "n": int(len(cohort)),
            "group_count": int(pd.unique(groups).size),
            "outcome_counts": {
                str(index): int(count) for index, count in enumerate(outcome_counts)
            },
            "conditioning": (
                "Stays observable through 72 hours; estimates do not generalize "
                "to stays discharged or dying before 72 hours."
            ),
        },
        "features": {
            "rule": (
                "First-24-hour static/demographic/laboratory features plus "
                "non-overlapping 24-hour summary windows available through the "
                "landmark; no later window is used at an earlier landmark."
            ),
            "columns_by_landmark": feature_names,
        },
        "training": {
            "algorithm": "XGBoost",
            "random_seed": RANDOM_SEED,
            "seed_count": 1,
            "parameters": parameters,
            "fold_integrity": fold_integrity,
            "fold_sizes": [int(len(test)) for _, test in folds],
            "fold_assignment": fold_id.tolist(),
        },
        "metrics": metrics,
        "fold_metrics": fold_metrics,
        "stability": stability,
        "dg_comparison": comparison,
        "uncertainty": {
            "method": "cluster bootstrap of the evaluation sample",
            "replicates": BOOTSTRAP_REPLICATES,
            "cluster_unit": config["cluster"],
            "cluster_count": int(pd.unique(clusters).size),
            "scope": (
                "Conditional on the fitted out-of-fold predictions and fixed "
                "fold partition; training and fold-selection uncertainty are not "
                "resampled."
            ),
        },
        "output_file": str(output_path),
    }
    print(
        database,
        "complete",
        {
            "n": len(cohort),
            "flip_24_72": stability["flip_24_72"],
            "tv_24_72": stability["tv_24_72"],
            "elapsed_seconds": result["elapsed_seconds"],
        },
        flush=True,
    )
    return result


def database_summary_from_file(database: str) -> dict[str, Any]:
    """Fallback loader for parallel workers that cannot return large records."""
    path = OUT / f"{database}_summary_partial.json"
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_database_partial(database: str, summary: dict[str, Any]) -> None:
    path = OUT / f"{database}_summary_partial.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)


def worker(database: str, threads: int) -> dict[str, Any]:
    summary = train_database(database, threads)
    write_database_partial(database, summary)
    # Keep the process return small and deterministic.
    return {
        "database": database,
        "n": summary["cohort"]["n"],
        "elapsed_seconds": summary["elapsed_seconds"],
    }


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] + ["---:"] * (len(headers) - 1)) + "|",
    ]
    lines += ["| " + " | ".join(str(value) for value in row) + " |" for row in rows]
    return "\n".join(lines)


def format_percent(value: float | None) -> str:
    return "NA" if value is None else f"{100.0 * value:.2f}%"


def write_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# XGBoost Shared-Fold Complete-Landmark Analysis",
        "",
        "Generated by `src/sepsis/train_xgb_shared_folds.py`.",
        "",
        "## Estimand",
        "",
        "The complete-landmark cohort requires a non-null three-class outcome "
        "(`y3`) and observed duration of at least 72 hours. The 24, 48, and 72 "
        "hour predictions use the same patients, the same outcome, identical "
        "XGBoost hyperparameters and random seed, and one fixed patient-grouped "
        "5-fold partition. Feature windows stop at each landmark.",
        "",
        "## Discrimination and stability",
        "",
    ]
    rows = []
    for database in DATABASES:
        result = summary["databases"][database]
        metrics = result["metrics"]
        stability = result["stability"]
        rows.append(
            [
                database,
                result["cohort"]["n"],
                result["cohort"]["group_count"],
                f"{metrics['24']['macro_auc']:.3f}",
                f"{metrics['48']['macro_auc']:.3f}",
                f"{metrics['72']['macro_auc']:.3f}",
                format_percent(stability["flip_24_72"]),
                f"{stability['tv_24_72']:.4f}",
            ]
        )
    lines.append(
        markdown_table(
            [
                "Database",
                "n",
                "Groups",
                "AUC 24 h",
                "AUC 48 h",
                "AUC 72 h",
                "Flip 24-72",
                "TV 24-72",
            ],
            rows,
        )
    )
    lines += ["", "## Directional changes", ""]
    direction_rows = []
    labels = {"0": "home", "1": "facility", "2": "death"}
    for database in DATABASES:
        stability = summary["databases"][database]["stability"]
        for direction in stability["directions_24_to_72"]:
            if direction["count"]:
                direction_rows.append(
                    [
                        database,
                        f"{labels[str(direction['from'])]} to {labels[str(direction['to'])]}",
                        direction["count"],
                        format_percent(direction["percent_all"]),
                        format_percent(direction["percent_of_flips"]),
                    ]
                )
    lines.append(
        markdown_table(
            ["Database", "Direction", "Count", "All stays", "Changed stays"],
            direction_rows,
        )
    )
    lines += ["", "## Paired comparison with the formal DG-GRU ensemble", ""]
    comparison_rows = []
    for database in DATABASES:
        result = summary["databases"][database]["dg_comparison"]
        xgb_metrics = result["metrics"]["xgb"]
        dg_metrics = result["metrics"]["dg_gru_ordinal_temporal_three_seed_ensemble"]
        deltas = result["metrics"]["xgb_minus_dg"]
        for milestone in (24, 48, 72):
            key = str(milestone)
            comparison_rows.append(
                [
                    database,
                    f"{milestone} h",
                    f"{xgb_metrics[key]['macro_auc']:.3f}",
                    f"{dg_metrics[key]['macro_auc']:.3f}",
                    f"{deltas[key]['macro_auc']:+.3f}",
                ]
            )
    lines.append(
        markdown_table(
            ["Database", "Landmark", "XGB AUC", "DG-GRU AUC", "XGB - DG"],
            comparison_rows,
        )
    )
    lines += [""]
    stability_rows = []
    for database in DATABASES:
        values = summary["databases"][database]["dg_comparison"]["stability"]
        stability_rows.append(
            [
                database,
                format_percent(values["xgb"]["flip_24_72"]),
                format_percent(
                    values["dg_gru_ordinal_temporal_three_seed_ensemble"][
                        "flip_24_72"
                    ]
                ),
                f"{values['xgb_minus_dg']['flip_24_72_percentage_points']:+.2f} pp",
                f"{values['xgb']['tv_24_72']:.4f}",
                f"{values['dg_gru_ordinal_temporal_three_seed_ensemble']['tv_24_72']:.4f}",
                f"{values['xgb_minus_dg']['tv_24_72']:+.4f}",
            ]
        )
    lines.append(
        markdown_table(
            [
                "Database",
                "XGB flip",
                "DG flip",
                "XGB - DG",
                "XGB TV",
                "DG TV",
                "XGB - DG TV",
            ],
            stability_rows,
        )
    )
    lines += [
        "",
        "## Interpretation boundary",
        "",
        "- Shared folds remove the component of 24-to-72 hour change caused by "
        "using a different patient fold partition at each landmark.",
        "- XGBoost still uses one random seed. The DG-GRU reference averages "
        "three seeds, so the paired model difference combines algorithm, input "
        "representation, and seed-averaging effects.",
        "- XGBoost uses static first-24-hour summaries and discrete 24-hour "
        "windows; DG-GRU uses the full hourly sequence. The paired patient "
        "sample is identical, but the models do not see identical inputs.",
        "- The cohort remains conditional on survival or observation through "
        "72 hours. Cluster-bootstrap intervals condition on the fitted folds and "
        "do not represent retraining uncertainty.",
        "",
    ]
    path = OUT / "summary.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)


def aggregate(databases: tuple[str, ...], threads: int) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for database in databases:
        partial = OUT / f"{database}_summary_partial.json"
        if partial.exists():
            summaries[database] = database_summary_from_file(database)
        else:
            summaries[database] = train_database(database, threads)
            write_database_partial(database, summaries[database])
    result = {
        "module": "xgb_complete_landmark_shared_folds",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "landmarks_hours": list(LANDMARKS),
        "complete_queue_rule": "y3 non-null and duration through landmarks >= 72 h",
        "xgb_configuration": {
            **XGB_PARAMETERS,
            "random_state": RANDOM_SEED,
            "seed_count": 1,
            "fold_scope": "one shared patient-grouped 5-fold split",
        },
        "databases": summaries,
        "interpretation": (
            "The corrected XGBoost stability estimate removes landmark-specific "
            "fold partitioning. It remains an out-of-fold prediction trajectory, "
            "not a clinical stability guarantee."
        ),
    }
    json_path = OUT / "summary.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    markdown_path = write_markdown(result)
    print(f"wrote {json_path}", flush=True)
    print(f"wrote {markdown_path}", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", choices=("mimic4", "eicu", "both"), default="both")
    parser.add_argument(
        "--threads",
        type=int,
        default=int(os.environ.get("SEPSIS_XGB_THREADS", "12")),
        help="Total XGBoost threads; split across databases when run in parallel.",
    )
    parser.add_argument(
        "--parallel-databases",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Train MIMIC-IV and eICU concurrently when --database both.",
    )
    args = parser.parse_args()
    databases = DATABASES if args.database == "both" else (args.database,)
    threads = max(1, int(args.threads))

    if len(databases) == 1:
        selected_threads = threads
    else:
        selected_threads = max(1, threads // len(databases))
    for database in databases:
        partial = OUT / f"{database}_summary_partial.json"
        if partial.exists():
            partial.unlink()

    if len(databases) == 2 and args.parallel_databases:
        with ProcessPoolExecutor(max_workers=2) as executor:
            futures = {
                executor.submit(worker, database, selected_threads): database
                for database in databases
            }
            for future in as_completed(futures):
                print("worker finished", future.result(), flush=True)
    else:
        for database in databases:
            if len(databases) == 1:
                summary = train_database(database, selected_threads)
                write_database_partial(database, summary)
            else:
                print(worker(database, selected_threads), flush=True)

    aggregate(databases, threads)


if __name__ == "__main__":
    main()
