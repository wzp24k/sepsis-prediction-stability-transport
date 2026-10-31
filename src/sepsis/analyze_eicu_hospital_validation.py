# -*- coding: utf-8 -*-
"""Hospital-held-out validation of eICU landmark stability predictions.

This bounded analysis reuses the complete three-landmark eICU cohort, the
shared-fold XGBoost landmark configuration, and the existing clinical flip
definitions. The only design change is that one hospital-grouped five-fold
partition, rather than a patient-grouped partition, is shared by all
landmarks. No existing project artifact is modified.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
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
DESIGN_FILE = DATA / "eicu" / "design_all.parquet"
SHARED_DIR = ROOT / "results" / "shared_fold"
SHARED_SUMMARY_FILE = SHARED_DIR / "summary.json"
SHARED_NPZ_FILE = SHARED_DIR / "eicu_shared_fold.npz"
OUT = ROOT / "results" / "eicu_hospital_validation"

LANDMARKS = (24, 48, 72)
CLASSES = np.asarray([0, 1, 2], dtype=np.int64)
CLASS_NAMES = {0: "home", 1: "facility", 2: "death"}
TRANSITIONS = (
    (0, 1),
    (0, 2),
    (1, 0),
    (1, 2),
    (2, 0),
    (2, 1),
)
RANDOM_SEED = 20261029
BOOTSTRAP_REPLICATES = 2000
N_SPLITS = 5
MIN_HOSPITAL_N = 50

# Identical model hyperparameters to src/sepsis/train_xgb_shared_folds.py.
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
    """Return the same landmark feature set used by the shared-fold analysis."""
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
    present = [int(value) for value in CLASSES if np.any(y == value)]
    class_auc: dict[str, float | None] = {}
    for class_index in CLASSES:
        binary = (y == class_index).astype(int)
        class_auc[str(class_index)] = (
            round(float(roc_auc_score(binary, probability[:, class_index])), 6)
            if len(np.unique(binary)) == 2
            else None
        )
    macro_auc = (
        round(
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
        )
        if len(present) == len(CLASSES)
        else None
    )
    return {
        "n": int(len(y)),
        "macro_auc": macro_auc,
        "class_auc": class_auc,
        "accuracy": round(float(accuracy_score(y, prediction)), 6),
        "log_loss": round(float(log_loss(y, probability, labels=CLASSES)), 6),
        "brier": round(float(np.mean(np.sum((probability - observed) ** 2, axis=1))), 6),
        "ece_10bin": round(expected_calibration_error(y, probability), 6),
        "mean_max_probability": round(float(probability.max(axis=1).mean()), 6),
    }


def bootstrap_mean(
    values: np.ndarray,
    clusters: np.ndarray,
    seed: int,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> list[float] | None:
    """Cluster-bootstrap the mean of row-level values."""
    values = np.asarray(values, dtype=np.float64)
    unique, inverse = np.unique(clusters.astype(str), return_inverse=True)
    if len(unique) < 2:
        return None
    sums = np.bincount(inverse, weights=values, minlength=len(unique))
    counts = np.bincount(inverse, minlength=len(unique)).astype(np.float64)
    rng = np.random.default_rng(seed)
    generated = 0
    samples: list[np.ndarray] = []
    while generated < replicates:
        size = min(100, replicates - generated)
        selected = rng.integers(0, len(unique), size=(size, len(unique)))
        denominator = counts[selected].sum(axis=1)
        numerator = sums[selected].sum(axis=1)
        samples.append(numerator / denominator)
        generated += size
    sample = np.concatenate(samples)
    return [
        round(float(np.quantile(sample, 0.025)), 6),
        round(float(np.quantile(sample, 0.975)), 6),
    ]


def rate_record(
    values: np.ndarray,
    clusters: np.ndarray,
    seed: int,
    denominator: np.ndarray | None = None,
) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    if denominator is None:
        selected = values
        denominator_n = int(len(values))
    else:
        selected_mask = np.asarray(denominator, dtype=np.float64) != 0
        selected = values[selected_mask]
        denominator_n = int(selected_mask.sum())
    if denominator_n == 0:
        return {"estimate": None, "ci95": None, "n": 0, "denominator_n": 0}
    return {
        "estimate": round(float(selected.mean()), 6),
        "ci95": bootstrap_mean(
            selected,
            np.asarray(clusters)[np.asarray(denominator, dtype=np.float64) != 0]
            if denominator is not None
            else clusters,
            seed,
        ),
        "n": int(selected.sum()),
        "denominator_n": denominator_n,
    }


def mean_record(
    values: np.ndarray,
    clusters: np.ndarray,
    seed: int,
    selected: np.ndarray | None = None,
) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    if selected is not None:
        values = values[np.asarray(selected, dtype=bool)]
        clusters = np.asarray(clusters)[np.asarray(selected, dtype=bool)]
    return {
        "estimate": round(float(values.mean()), 6) if len(values) else None,
        "ci95": bootstrap_mean(values, clusters, seed) if len(values) else None,
        "n": int(len(values)),
    }


def validate_shared_hospital_folds(
    folds: list[tuple[np.ndarray, np.ndarray]],
    hospitals: np.ndarray,
    patients: np.ndarray,
    y: np.ndarray,
    n_rows: int,
) -> dict[str, Any]:
    all_test = np.sort(np.concatenate([test for _, test in folds]))
    if not np.array_equal(all_test, np.arange(n_rows)):
        raise ValueError("hospital fold test sets do not cover each row exactly once")
    held = [set(hospitals[test].tolist()) for _, test in folds]
    overlaps = [
        len(held[first].intersection(held[second]))
        for first in range(len(held))
        for second in range(first + 1, len(held))
    ]
    if any(overlaps):
        raise ValueError("a hospital appears in more than one held-out fold")
    if sum(len(values) for values in held) != len(set(hospitals.tolist())):
        raise ValueError("hospital partition does not cover each hospital exactly once")
    held_patients = [set(patients[test].tolist()) for _, test in folds]
    patient_overlaps = [
        len(held_patients[first].intersection(held_patients[second]))
        for first in range(len(held_patients))
        for second in range(first + 1, len(held_patients))
    ]
    if any(patient_overlaps):
        raise ValueError("a patient appears in more than one held-out fold")
    fold_details = []
    for fold_index, (_, test) in enumerate(folds):
        counts = np.bincount(y[test], minlength=3)
        fold_details.append(
            {
                "fold": fold_index,
                "n_test": int(len(test)),
                "n_hospitals": int(len(np.unique(hospitals[test]))),
                "outcome_counts": {
                    str(index): int(count) for index, count in enumerate(counts)
                },
            }
        )
    return {
        "n_splits": len(folds),
        "test_partitions_cover_each_row_once": True,
        "hospital_membership_is_whole_fold": True,
        "hospital_overlap_counts": [int(value) for value in overlaps],
        "patient_membership_is_whole_fold": True,
        "patient_overlap_counts": [int(value) for value in patient_overlaps],
        "same_hospital_fold_reused_at_all_landmarks": True,
        "folds": fold_details,
    }


def normalized_transition(pred24: np.ndarray, pred72: np.ndarray) -> dict[str, Any]:
    counts = np.zeros((3, 3), dtype=np.int64)
    np.add.at(counts, (pred24, pred72), 1)
    row_sums = counts.sum(axis=1)
    row_normalized = np.divide(
        counts,
        row_sums[:, None],
        out=np.zeros_like(counts, dtype=np.float64),
        where=row_sums[:, None] != 0,
    )
    return {
        "counts_rows24_columns72": counts.tolist(),
        "row_normalized_rows24_columns72": np.round(row_normalized, 6).tolist(),
        "row_denominators_24h": row_sums.tolist(),
    }


def model_arrays(
    y: np.ndarray,
    p24: np.ndarray,
    p48: np.ndarray,
    p72: np.ndarray,
) -> dict[str, np.ndarray]:
    pred24 = p24.argmax(axis=1)
    pred48 = p48.argmax(axis=1)
    pred72 = p72.argmax(axis=1)
    distance24 = np.abs(pred24 - y)
    distance72 = np.abs(pred72 - y)
    flip = pred24 != pred72
    upgrade = pred72 > pred24
    downgrade = pred72 < pred24
    gain = distance72 < distance24
    loss = distance72 > distance24
    true_probability_change = (
        p72[np.arange(len(y)), y] - p24[np.arange(len(y)), y]
    )
    return {
        "pred24": pred24,
        "pred48": pred48,
        "pred72": pred72,
        "distance24": distance24,
        "distance72": distance72,
        "flip": flip,
        "upgrade": upgrade,
        "downgrade": downgrade,
        "gain": gain,
        "loss": loss,
        "tie": distance72 == distance24,
        "correct_upgrade": upgrade & gain,
        "incorrect_upgrade": upgrade & ~gain,
        "correct_downgrade": downgrade & gain,
        "incorrect_downgrade": downgrade & ~gain,
        "true_probability_change": true_probability_change,
    }


def stability_and_clinical_output(
    y: np.ndarray,
    probability: dict[int, np.ndarray],
    hospitals: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, np.ndarray]]:
    arrays = model_arrays(
        y,
        probability[24],
        probability[48],
        probability[72],
    )
    flip = arrays["flip"]
    one = np.ones(len(y), dtype=np.float64)

    stability: dict[str, Any] = {
        "n": int(len(y)),
        "return_rate": {},
        "flip": {},
        "total_variation": {},
        "confidence_24h": {},
        "prediction_distribution": {},
    }
    returned = (
        (arrays["pred24"] == arrays["pred72"])
        & (arrays["pred24"] != arrays["pred48"])
    )
    stability["return_rate"] = {
        "cohort_rate": rate_record(returned, hospitals, RANDOM_SEED + 1, one),
        "among_24_to_48_flips": rate_record(
            returned,
            hospitals,
            RANDOM_SEED + 2,
            arrays["pred24"] != arrays["pred48"],
        ),
    }
    for first, second in ((24, 48), (48, 72), (24, 72)):
        key = f"{first}_{second}"
        changed = arrays[f"pred{first}"] != arrays[f"pred{second}"]
        variation = 0.5 * np.abs(
            probability[first] - probability[second]
        ).sum(axis=1)
        stability["flip"][key] = rate_record(
            changed, hospitals, RANDOM_SEED + first * 100 + second
        )
        stability["total_variation"][key] = mean_record(
            variation, hospitals, RANDOM_SEED + 1000 + first * 100 + second
        )
    confidence = probability[24].max(axis=1)
    for threshold in (0.5, 0.6, 0.7):
        selected = confidence >= threshold
        subsequent_flip = arrays["pred24"] != arrays["pred72"]
        stability["confidence_24h"][str(threshold)] = {
            "coverage": round(float(selected.mean()), 6),
            "n": int(selected.sum()),
            "subsequent_flip_24_72": rate_record(
                subsequent_flip,
                hospitals,
                RANDOM_SEED + int(threshold * 1000) + 2000,
                selected,
            ),
        }
    for milestone in LANDMARKS:
        counts = np.bincount(arrays[f"pred{milestone}"], minlength=3)
        stability["prediction_distribution"][str(milestone)] = {
            str(index): int(count) for index, count in enumerate(counts)
        }

    transition = normalized_transition(arrays["pred24"], arrays["pred72"])
    transition["flip_24_72"] = rate_record(
        flip, hospitals, RANDOM_SEED + 3000
    )
    transition["directions"] = {}
    for source, destination in TRANSITIONS:
        indicator = (
            (arrays["pred24"] == source) & (arrays["pred72"] == destination)
        )
        transition["directions"][
            f"{CLASS_NAMES[source]}_to_{CLASS_NAMES[destination]}"
        ] = {
            "from": CLASS_NAMES[source],
            "to": CLASS_NAMES[destination],
            "count": int(indicator.sum()),
            "cohort_rate": rate_record(
                indicator, hospitals, RANDOM_SEED + 3000 + source * 10 + destination
            ),
            "share_of_flips": rate_record(
                indicator,
                hospitals,
                RANDOM_SEED + 4000 + source * 10 + destination,
                flip,
            ),
        }

    clinical = {
        "major_upgrade_to_death": (
            (arrays["pred24"] < 2) & (arrays["pred72"] == 2)
        ),
        "home_to_facility": (
            (arrays["pred24"] == 0) & (arrays["pred72"] == 1)
        ),
        "any_upgrade": arrays["upgrade"],
    }
    clinical_output = {
        name: rate_record(
            indicator, hospitals, RANDOM_SEED + 5000 + index
        )
        for index, (name, indicator) in enumerate(clinical.items())
    }

    correctness: dict[str, Any] = {"all": {}, "among_flips": {}}
    for scope, denominator in (("all", one), ("among_flips", flip.astype(float))):
        for index, name in enumerate(
            (
                "gain",
                "loss",
                "tie",
                "correct_upgrade",
                "incorrect_upgrade",
                "correct_downgrade",
                "incorrect_downgrade",
            )
        ):
            correctness[scope][name] = rate_record(
                arrays[name],
                hospitals,
                RANDOM_SEED + 6000 + (0 if scope == "all" else 100) + index,
                denominator,
            )
    correctness["among_flips"]["true_class_probability_improved"] = rate_record(
        arrays["true_probability_change"] > 1e-12,
        hospitals,
        RANDOM_SEED + 7000,
        flip.astype(float),
    )
    correctness["among_flips"]["true_class_probability_change_mean"] = mean_record(
        arrays["true_probability_change"],
        hospitals,
        RANDOM_SEED + 7001,
        flip,
    )
    correctness["all"]["true_class_probability_change_mean"] = mean_record(
        arrays["true_probability_change"], hospitals, RANDOM_SEED + 7002
    )

    death = y == 2
    death_output = {
        "n": int(death.sum()),
        "predicted_class_counts": {
            "24": np.bincount(arrays["pred24"][death], minlength=3).tolist(),
            "72": np.bincount(arrays["pred72"][death], minlength=3).tolist(),
        },
    }
    death_indicators = {
        "detected_at_24": arrays["pred24"] == 2,
        "detected_at_72": arrays["pred72"] == 2,
        "new_upgrade_24_to_72": (
            (arrays["pred24"] < 2) & (arrays["pred72"] == 2)
        ),
        "missed_at_72": arrays["pred72"] < 2,
    }
    for index, (name, indicator) in enumerate(death_indicators.items()):
        death_output[name] = rate_record(
            indicator, hospitals, RANDOM_SEED + 8000 + index, death
        )
    clinical_output["true_death_subgroup"] = death_output
    return stability, transition, clinical_output, correctness


def simple_auc_variance(auc: float, n_positive: int, n_negative: int) -> float:
    """Hanley-McNeil approximation to the variance of one binary AUC."""
    if n_positive < 2 or n_negative < 2 or not 0.0 < auc < 1.0:
        return float("nan")
    q1 = auc / (2.0 - auc)
    q2 = 2.0 * auc * auc / (1.0 + auc)
    numerator = (
        auc * (1.0 - auc)
        + (n_positive - 1) * (q1 - auc * auc)
        + (n_negative - 1) * (q2 - auc * auc)
    )
    variance = numerator / (n_positive * n_negative)
    return max(float(variance), 1e-12)


def approx_macro_auc_variance(
    y: np.ndarray, probability: np.ndarray, macro_auc: float
) -> dict[str, Any]:
    """Approximate macro-AUC SE from one-vs-rest class-specific variances."""
    class_values = []
    for class_index in CLASSES:
        binary = (y == class_index).astype(int)
        n_positive = int(binary.sum())
        n_negative = int(len(binary) - n_positive)
        auc = float(roc_auc_score(binary, probability[:, class_index]))
        variance = simple_auc_variance(auc, n_positive, n_negative)
        class_values.append(
            {
                "class": int(class_index),
                "n_positive": n_positive,
                "n_negative": n_negative,
                "auc": round(auc, 6),
                "variance": variance,
            }
        )
    variance = float(np.nansum([item["variance"] for item in class_values]) / 9.0)
    return {
        "variance": max(variance, 1e-12),
        "se": float(np.sqrt(max(variance, 1e-12))),
        "method": "mean of Hanley-McNeil class AUC variances divided by 9",
        "macro_auc": macro_auc,
        "class_components": class_values,
    }


def dersimonian_laird(effects: np.ndarray, variances: np.ndarray) -> dict[str, Any]:
    """DerSimonian-Laird random-effects summary of hospital AUCs."""
    effects = np.asarray(effects, dtype=np.float64)
    variances = np.asarray(variances, dtype=np.float64)
    valid = (
        np.isfinite(effects)
        & np.isfinite(variances)
        & (variances > 0)
    )
    effects = effects[valid]
    variances = variances[valid]
    k = len(effects)
    if k < 3:
        return {"k": k, "status": "insufficient_sites"}
    fixed_weights = 1.0 / variances
    fixed_effect = float(np.sum(fixed_weights * effects) / np.sum(fixed_weights))
    q = float(np.sum(fixed_weights * (effects - fixed_effect) ** 2))
    degrees = k - 1
    denominator = float(np.sum(fixed_weights) - np.sum(fixed_weights**2) / np.sum(fixed_weights))
    tau2 = max(0.0, (q - degrees) / denominator) if denominator > 0 else 0.0
    random_weights = 1.0 / (variances + tau2)
    pooled = float(np.sum(random_weights * effects) / np.sum(random_weights))
    pooled_se = float(np.sqrt(1.0 / np.sum(random_weights)))
    i2 = max(0.0, (q - degrees) / q) * 100.0 if q > 0 else 0.0
    return {
        "k": k,
        "status": "ok",
        "pooled_auc": round(pooled, 6),
        "pooled_ci95": [
            round(pooled - 1.96 * pooled_se, 6),
            round(pooled + 1.96 * pooled_se, 6),
        ],
        "pooled_se": round(pooled_se, 6),
        "tau2": round(tau2, 10),
        "tau": round(float(np.sqrt(tau2)), 6),
        "q": round(q, 6),
        "q_df": degrees,
        "i2_percent": round(i2, 3),
        "fixed_effect_auc": round(fixed_effect, 6),
        "within_hospital_variance": (
            "Hanley-McNeil approximation; macro-AUC variance approximated as "
            "the mean of the three one-vs-rest class variances divided by 9"
        ),
    }


def quantiles(values: np.ndarray, probabilities: tuple[float, ...]) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        f"p{int(100 * probability):02d}": round(
            float(np.quantile(values, probability)), 6
        )
        for probability in probabilities
    }


def hospital_level_analysis(
    y: np.ndarray,
    probability: dict[int, np.ndarray],
    hospitals: np.ndarray,
    arrays: dict[str, np.ndarray],
) -> tuple[dict[str, Any], pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for hospital in sorted(np.unique(hospitals)):
        mask = hospitals == hospital
        hospital_y = y[mask]
        hospital_flip = arrays["flip"][mask]
        distance24 = arrays["distance24"][mask]
        distance72 = arrays["distance72"][mask]
        row: dict[str, Any] = {
            "hospitalid": hospital,
            "n": int(mask.sum()),
            "observed_home": int((hospital_y == 0).sum()),
            "observed_facility": int((hospital_y == 1).sum()),
            "observed_death": int((hospital_y == 2).sum()),
            "sufficient_n_ge_50": bool(mask.sum() >= MIN_HOSPITAL_N),
            "all_outcomes_observed": bool(len(np.unique(hospital_y)) == 3),
            "all_outcome_counts_ge_5": bool(
                np.all(
                    [
                        int((hospital_y == class_index).sum()) >= 5
                        for class_index in CLASSES
                    ]
                )
            ),
            "flip_24_72": round(float(hospital_flip.mean()), 6),
            "tv_24_72": round(
                float(
                    (
                        0.5
                        * np.abs(
                            probability[24][mask] - probability[72][mask]
                        ).sum(axis=1)
                    ).mean()
                ),
                6,
            ),
            "gain_rate": round(float((distance72 < distance24).mean()), 6),
            "loss_rate": round(float((distance72 > distance24).mean()), 6),
            "major_upgrade_to_death": round(
                float(
                    (
                        (arrays["pred24"][mask] < 2)
                        & (arrays["pred72"][mask] == 2)
                    ).mean()
                ),
                6,
            ),
            "home_to_facility": round(
                float(
                    (
                        (arrays["pred24"][mask] == 0)
                        & (arrays["pred72"][mask] == 1)
                    ).mean()
                ),
                6,
            ),
        }
        for milestone in LANDMARKS:
            hospital_probability = probability[milestone][mask]
            hospital_prediction = hospital_probability.argmax(axis=1)
            metrics = metric_block(hospital_y, hospital_probability)
            row[f"macro_auc_{milestone}"] = metrics["macro_auc"]
            for class_index in CLASSES:
                row[f"auc_{milestone}_class_{class_index}"] = metrics["class_auc"][
                    str(class_index)
                ]
                row[f"pred_{milestone}_class_{class_index}"] = int(
                    (hospital_prediction == class_index).sum()
                )
            row[f"ece_10bin_{milestone}"] = metrics["ece_10bin"]
            row[f"brier_{milestone}"] = metrics["brier"]
            row[f"accuracy_{milestone}"] = metrics["accuracy"]
            row[f"mean_max_probability_{milestone}"] = metrics[
                "mean_max_probability"
            ]
        rows.append(row)

    frame = pd.DataFrame(rows)
    primary = frame[
        frame["sufficient_n_ge_50"] & frame["all_outcome_counts_ge_5"]
    ].copy()
    if primary.empty:
        raise ValueError("no hospitals meet the hospital-level analysis threshold")
    primary_rows = int(primary["n"].sum())

    distributions: dict[str, Any] = {}
    for milestone in LANDMARKS:
        for metric in (
            f"macro_auc_{milestone}",
            f"ece_10bin_{milestone}",
            f"brier_{milestone}",
            f"accuracy_{milestone}",
        ):
            distributions[metric] = {
                "mean": round(float(primary[metric].mean()), 6),
                "sd": round(float(primary[metric].std(ddof=1)), 6),
                **quantiles(
                    primary[metric].to_numpy(),
                    (0.10, 0.25, 0.50, 0.75, 0.90),
                ),
                "min": round(float(primary[metric].min()), 6),
                "max": round(float(primary[metric].max()), 6),
            }
    for metric in (
        "flip_24_72",
        "tv_24_72",
        "gain_rate",
        "loss_rate",
        "major_upgrade_to_death",
        "home_to_facility",
    ):
        distributions[metric] = {
            "mean": round(float(primary[metric].mean()), 6),
            "sd": round(float(primary[metric].std(ddof=1)), 6),
            **quantiles(
                primary[metric].to_numpy(),
                (0.10, 0.25, 0.50, 0.75, 0.90),
            ),
            "min": round(float(primary[metric].min()), 6),
            "max": round(float(primary[metric].max()), 6),
        }

    variance_details = []
    for _, row in primary.iterrows():
        mask = hospitals == row["hospitalid"]
        variance_details.append(
            approx_macro_auc_variance(
                y[mask],
                probability[72][mask],
                float(row["macro_auc_72"]),
            )
        )
    effects = np.asarray([item["macro_auc"] for item in variance_details])
    variances = np.asarray([item["variance"] for item in variance_details])
    random_effects = dersimonian_laird(effects, variances)

    worst_columns = [
        "hospitalid",
        "n",
        "observed_home",
        "observed_facility",
        "observed_death",
        "macro_auc_24",
        "macro_auc_48",
        "macro_auc_72",
        "ece_10bin_72",
        "flip_24_72",
        "tv_24_72",
        "major_upgrade_to_death",
        "home_to_facility",
    ]
    worst = (
        primary.sort_values(["macro_auc_72", "n"], ascending=[True, False])
        .head(10)[worst_columns]
        .to_dict(orient="records")
    )
    best = (
        primary.sort_values(["macro_auc_72", "n"], ascending=[False, False])
        .head(5)[worst_columns]
        .to_dict(orient="records")
    )

    output = {
        "definition": {
            "sufficient_sample_primary": (
                f"hospital n >= {MIN_HOSPITAL_N} and at least 5 observed stays "
                "in each of the three outcome classes"
            ),
            "n_hospital_sites": int(frame["hospitalid"].nunique()),
            "n_sites_n_ge_50": int(frame["sufficient_n_ge_50"].sum()),
            "n_sites_primary_auc_evaluable": int(len(primary)),
            "n_rows_primary": primary_rows,
            "fraction_of_cohort_primary": round(primary_rows / len(y), 6),
            "all_hospitals_written_to_csv": True,
        },
        "distributions_across_primary_sites": distributions,
        "worst_10_sites_by_72h_macro_auc": worst,
        "best_5_sites_by_72h_macro_auc": best,
        "der_simonian_laird_72h_macro_auc": random_effects,
        "hospital_variance_details_72h": variance_details,
    }
    return output, frame


def align_patient_grouped_reference(
    ids: np.ndarray,
    y: np.ndarray,
    hospitals: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if not SHARED_NPZ_FILE.exists():
        raise FileNotFoundError(f"missing patient-grouped predictions: {SHARED_NPZ_FILE}")
    with np.load(SHARED_NPZ_FILE) as bundle:
        reference = {key: bundle[key].copy() for key in bundle.files}
    positions = pd.Index(reference["ids"].astype(np.int64)).get_indexer(ids)
    if (positions < 0).any() or len(np.unique(positions)) != len(ids):
        raise ValueError("patient-grouped predictions are not an exact id match")
    aligned = {
        "ids": reference["ids"].astype(np.int64)[positions],
        "y3": reference["y3"].astype(np.int64)[positions],
        "p24": reference["p24"].astype(np.float64)[positions],
        "p48": reference["p48"].astype(np.float64)[positions],
        "p72": reference["p72"].astype(np.float64)[positions],
        "clusters": reference["clusters"].astype(str)[positions],
    }
    if not np.array_equal(aligned["ids"], ids):
        raise ValueError("patient-grouped ids failed row alignment")
    if not np.array_equal(aligned["y3"], y):
        raise ValueError("patient-grouped outcomes failed row alignment")
    if not np.array_equal(aligned["clusters"], hospitals.astype(str)):
        raise ValueError("patient-grouped hospital clusters failed alignment")
    return aligned, {
        "reference_file": str(SHARED_NPZ_FILE),
        "exact_id_set": True,
        "outcomes_aligned": True,
        "hospital_clusters_aligned": True,
        "rows_reordered_by_id": int(
            np.sum(reference["ids"].astype(np.int64)[positions] != ids)
        ),
    }


def fit_patient_grouped_reference(
    cohort: pd.DataFrame,
    y: np.ndarray,
    patients: np.ndarray,
    parameters: dict[str, Any],
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    folds = list(
        GroupKFold(n_splits=N_SPLITS).split(
            np.zeros((len(cohort), 1), dtype=np.float32),
            y,
            patients,
        )
    )
    held = [set(patients[test].tolist()) for _, test in folds]
    patient_overlaps = [
        len(held[first].intersection(held[second]))
        for first in range(len(held))
        for second in range(first + 1, len(held))
    ]
    if any(patient_overlaps):
        raise ValueError("patient-grouped reference has patient overlap across folds")

    probability: dict[int, np.ndarray] = {}
    for milestone in LANDMARKS:
        columns = feature_columns(cohort, milestone)
        x = cohort[columns].astype(np.float32)
        out_of_fold = np.zeros((len(cohort), 3), dtype=np.float64)
        for train_index, test_index in folds:
            model = XGBClassifier(**parameters)
            model.fit(x.iloc[train_index], y[train_index])
            fold_probability = model.predict_proba(
                x.iloc[test_index]
            ).astype(np.float64)
            fold_probability = np.clip(fold_probability, 0.0, 1.0)
            fold_probability /= fold_probability.sum(axis=1, keepdims=True)
            out_of_fold[test_index] = fold_probability
        probability[milestone] = out_of_fold
    return probability, {
        "reference_type": "same-row patient-grouped refit",
        "exact_id_set": True,
        "outcomes_aligned": True,
        "patient_membership_is_whole_fold": True,
        "patient_overlap_counts": [int(value) for value in patient_overlaps],
    }


def comparison_with_patient_grouping(
    y: np.ndarray,
    probability: dict[int, np.ndarray],
    hospitals: np.ndarray,
    arrays: dict[str, np.ndarray],
    reference: dict[int, np.ndarray],
    alignment: dict[str, Any],
) -> dict[str, Any]:
    reference_metrics = {
        str(milestone): metric_block(
            y, reference[milestone]
        )
        for milestone in LANDMARKS
    }
    hospital_metrics = {
        str(milestone): metric_block(y, probability[milestone])
        for milestone in LANDMARKS
    }
    reference_arrays = model_arrays(
        y,
        reference[24],
        reference[48],
        reference[72],
    )
    flip_difference = (
        arrays["flip"].astype(float) - reference_arrays["flip"].astype(float)
    )
    tv = 0.5 * np.abs(
        probability[24] - probability[72]
    ).sum(axis=1)
    reference_tv = 0.5 * np.abs(
        reference[24] - reference[72]
    ).sum(axis=1)
    major = (
        (arrays["pred24"] < 2) & (arrays["pred72"] == 2)
    ).astype(float)
    reference_major = (
        (reference_arrays["pred24"] < 2)
        & (reference_arrays["pred72"] == 2)
    ).astype(float)
    home_facility = (
        (arrays["pred24"] == 0) & (arrays["pred72"] == 1)
    ).astype(float)
    reference_home_facility = (
        (reference_arrays["pred24"] == 0)
        & (reference_arrays["pred72"] == 1)
    ).astype(float)
    return {
        "alignment": alignment,
        "patient_grouped_metrics_recomputed": reference_metrics,
        "hospital_grouped_metrics": hospital_metrics,
        "hospital_minus_patient_grouped": {
            "macro_auc": {
                str(milestone): round(
                    float(
                        hospital_metrics[str(milestone)]["macro_auc"]
                        - reference_metrics[str(milestone)]["macro_auc"]
                    ),
                    6,
                )
                for milestone in LANDMARKS
            },
            "ece_10bin": {
                str(milestone): round(
                    float(
                        hospital_metrics[str(milestone)]["ece_10bin"]
                        - reference_metrics[str(milestone)]["ece_10bin"]
                    ),
                    6,
                )
                for milestone in LANDMARKS
            },
            "brier": {
                str(milestone): round(
                    float(
                        hospital_metrics[str(milestone)]["brier"]
                        - reference_metrics[str(milestone)]["brier"]
                    ),
                    6,
                )
                for milestone in LANDMARKS
            },
            "flip_24_72": mean_record(
                flip_difference, hospitals, RANDOM_SEED + 9000
            ),
            "tv_24_72": mean_record(
                tv - reference_tv, hospitals, RANDOM_SEED + 9001
            ),
            "major_upgrade_to_death": mean_record(
                major - reference_major, hospitals, RANDOM_SEED + 9002
            ),
            "home_to_facility": mean_record(
                home_facility - reference_home_facility,
                hospitals,
                RANDOM_SEED + 9003,
            ),
        },
        "interpretation": (
            "Patient-grouped and hospital-held-out estimates use the same rows "
            "and outcomes but different data partitions. Hospital-held-out "
            "performance is the stricter internal-external estimate."
        ),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percent(value: float | None) -> str:
    return "NA" if value is None else f"{100.0 * value:.2f}%"


def percent_ci(record: dict[str, Any]) -> str:
    estimate = record.get("estimate")
    interval = record.get("ci95")
    if estimate is None:
        return "NA"
    if not interval:
        return percent(estimate)
    return f"{percent(estimate)} ({percent(interval[0])} to {percent(interval[1])})"


def number_ci(record: dict[str, Any]) -> str:
    estimate = record.get("estimate")
    interval = record.get("ci95")
    if estimate is None:
        return "NA"
    if not interval:
        return f"{estimate:.4f}"
    return f"{estimate:.4f} ({interval[0]:.4f} to {interval[1]:.4f})"


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] + ["---:"] * (len(headers) - 1)) + "|",
    ]
    lines.extend(
        "| " + " | ".join(str(value) for value in row) + " |" for row in rows
    )
    return "\n".join(lines)


def write_report(summary: dict[str, Any], path: Path) -> None:
    cohort = summary["cohort"]
    metrics = summary["metrics"]
    stability = summary["stability"]
    clinical = summary["clinical_changes"]
    correctness = summary["correctness"]
    comparison = summary["comparison_with_patient_grouped"]
    hospital = summary["hospital_level"]
    re_model = hospital["der_simonian_laird_72h_macro_auc"]

    overall_rows = []
    for milestone in LANDMARKS:
        record = metrics[str(milestone)]
        overall_rows.append(
            [
                f"{milestone} h",
                record["n"],
                f"{record['macro_auc']:.3f}",
                f"{record['accuracy']:.3f}",
                f"{record['brier']:.3f}",
                f"{record['ece_10bin']:.3f}",
                f"{record['mean_max_probability']:.3f}",
            ]
        )
    comparison_rows = []
    for milestone in LANDMARKS:
        patient = comparison["patient_grouped_metrics_recomputed"][str(milestone)]
        held = comparison["hospital_grouped_metrics"][str(milestone)]
        comparison_rows.append(
            [
                f"{milestone} h",
                f"{patient['macro_auc']:.3f}",
                f"{held['macro_auc']:.3f}",
                f"{held['macro_auc'] - patient['macro_auc']:+.3f}",
                f"{patient['ece_10bin']:.3f}",
                f"{held['ece_10bin']:.3f}",
                f"{held['ece_10bin'] - patient['ece_10bin']:+.3f}",
            ]
        )
    change_rows = [
        [
            "24-to-72 h flip",
            percent_ci(stability["flip"]["24_72"]),
            percent_ci(comparison["hospital_minus_patient_grouped"]["flip_24_72"]),
        ],
        [
            "24-to-72 h total variation",
            number_ci(stability["total_variation"]["24_72"]),
            number_ci(comparison["hospital_minus_patient_grouped"]["tv_24_72"]),
        ],
        [
            "Major upgrade to death",
            percent_ci(clinical["major_upgrade_to_death"]),
            percent_ci(
                comparison["hospital_minus_patient_grouped"][
                    "major_upgrade_to_death"
                ]
            ),
        ],
        [
            "Home to facility",
            percent_ci(clinical["home_to_facility"]),
            percent_ci(
                comparison["hospital_minus_patient_grouped"]["home_to_facility"]
            ),
        ],
    ]
    confidence_rows = []
    for threshold in ("0.5", "0.6", "0.7"):
        record = stability["confidence_24h"][threshold]
        confidence_rows.append(
            [
                threshold,
                percent(record["coverage"]),
                record["n"],
                percent_ci(record["subsequent_flip_24_72"]),
            ]
        )
    transition_matrix = stability["transition_matrix"][
        "counts_rows24_columns72"
    ]
    transition_rows = []
    labels = ["Home", "Facility", "Death"]
    for index, row in enumerate(transition_matrix):
        transition_rows.append([labels[index], *row])
    worst_rows = []
    for item in hospital["worst_10_sites_by_72h_macro_auc"][:5]:
        worst_rows.append(
            [
                item["hospitalid"],
                item["n"],
                f"{item['observed_home']}/{item['observed_facility']}/{item['observed_death']}",
                f"{item['macro_auc_24']:.3f}",
                f"{item['macro_auc_48']:.3f}",
                f"{item['macro_auc_72']:.3f}",
                f"{item['ece_10bin_72']:.3f}",
                percent(item["flip_24_72"]),
            ]
        )
    distributions = hospital["distributions_across_primary_sites"]
    distribution_rows = []
    for metric, label in (
        ("macro_auc_24", "Macro-AUC 24 h"),
        ("macro_auc_48", "Macro-AUC 48 h"),
        ("macro_auc_72", "Macro-AUC 72 h"),
        ("ece_10bin_72", "ECE 72 h"),
        ("flip_24_72", "Flip 24-72 h"),
        ("tv_24_72", "TV 24-72 h"),
    ):
        record = distributions[metric]
        distribution_rows.append(
            [
                label,
                f"{record['p10']:.3f}",
                f"{record['p25']:.3f}",
                f"{record['p50']:.3f}",
                f"{record['p75']:.3f}",
                f"{record['p90']:.3f}",
                f"{record['min']:.3f}",
                f"{record['max']:.3f}",
            ]
        )
    correctness_rows = []
    for scope, label in (
        ("all", "All stays"),
        ("among_flips", "Among flips"),
    ):
        for metric in (
            "gain",
            "loss",
            "tie",
            "correct_upgrade",
            "incorrect_upgrade",
            "correct_downgrade",
            "incorrect_downgrade",
        ):
            correctness_rows.append(
                [
                    label,
                    metric.replace("_", " "),
                    percent_ci(correctness[scope][metric]),
                ]
            )
    correctness_rows.append(
        [
            "Among flips",
            "true-class probability improved",
            percent_ci(
                correctness["among_flips"][
                    "true_class_probability_improved"
                ]
            ),
        ]
    )

    lines = [
        "# eICU-CRD Hospital-Held-Out Validation",
        "",
        "Generated by `src/sepsis/analyze_eicu_hospital_validation.py`.",
        "",
        "## Design and estimand",
        "",
        f"The cohort contained {cohort['n']:,} stays from "
        f"{cohort['n_hospitals']} hospitals, requiring non-null `y3` and "
        "`los_h >= 72`. A single hospital-grouped five-fold partition was "
        "shared by the 24, 48, and 72 h XGBoost landmark models. All rows for "
        "a hospital were held out together, and all three landmarks used the "
        "same fold assignment, rows, fixed seed, and hyperparameters.",
        "",
        "This is hospital-held-out internal-external validation. It is stronger "
        "than patient-grouped cross-validation for estimating transport to an "
        "unseen site. It is not prospective validation, a deployment study, or "
        "a clinical-utility evaluation.",
        "",
        "## Overall discrimination and calibration",
        "",
        markdown_table(
            [
                "Landmark",
                "n",
                "Macro-AUC",
                "Accuracy",
                "Brier",
                "ECE",
                "Mean max p",
            ],
            overall_rows,
        ),
        "",
        "## Comparison with patient-grouped validation",
        "",
        markdown_table(
            [
                "Landmark",
                "Patient AUC",
                "Hospital AUC",
                "Difference",
                "Patient ECE",
                "Hospital ECE",
                "ECE difference",
            ],
            comparison_rows,
        ),
        "",
        markdown_table(
            ["Metric", "Hospital-held-out", "Hospital minus patient-grouped"],
            change_rows,
        ),
        "",
        "## Stability and clinical transitions",
        "",
        f"The 24-to-72 h flip estimate was "
        f"{percent_ci(stability['flip']['24_72'])} by hospital-cluster "
        f"bootstrap. Total variation was "
        f"{number_ci(stability['total_variation']['24_72'])}. The cohort "
        f"return rate to the 24 h class after a 48 h deviation was "
        f"{percent_ci(stability['return_rate']['cohort_rate'])}; among 24-to-48 "
        f"h flippers it was "
        f"{percent_ci(stability['return_rate']['among_24_to_48_flips'])}.",
        "",
        markdown_table(
            ["From 24 h", "To home", "To facility", "To death"],
            transition_rows,
        ),
        "",
        markdown_table(
            ["Confidence at 24 h", "Coverage", "n", "Subsequent 24-to-72 h flip"],
            confidence_rows,
        ),
        "",
        markdown_table(
            ["Scope", "Correctness measure", "Estimate (95% CI)"],
            correctness_rows,
        ),
        "",
        "## Hospital-level heterogeneity",
        "",
        f"Primary hospital-level AUC summaries used "
        f"{hospital['definition']['n_sites_primary_auc_evaluable']} sites with "
        f"at least {MIN_HOSPITAL_N} stays and at least five stays in each "
        f"outcome class "
        f"({hospital['definition']['n_rows_primary']:,} stays; "
        f"{percent(hospital['definition']['fraction_of_cohort_primary'])} of the "
        "cohort).",
        "",
        markdown_table(
            ["Metric", "P10", "P25", "Median", "P75", "P90", "Min", "Max"],
            distribution_rows,
        ),
        "",
        "Worst five hospitals by 72 h macro-AUC:",
        "",
        markdown_table(
            [
                "Hospital",
                "n",
                "Observed home/facility/death",
                "AUC 24 h",
                "AUC 48 h",
                "AUC 72 h",
                "ECE 72 h",
                "Flip",
            ],
            worst_rows,
        ),
        "",
        (
            f"DerSimonian-Laird random-effects summary of 72 h hospital "
            f"macro-AUC: {re_model.get('pooled_auc', 'NA'):.3f} "
            f"(95% CI {re_model.get('pooled_ci95', ['NA', 'NA'])[0]:.3f} "
            f"to {re_model.get('pooled_ci95', ['NA', 'NA'])[1]:.3f}); "
            f"tau = {re_model.get('tau', float('nan')):.3f}, "
            f"I2 = {re_model.get('i2_percent', float('nan')):.1f}%."
            if re_model.get("status") == "ok"
            else "DerSimonian-Laird summary was not feasible."
        ),
        "",
        "Within-hospital AUC variance was approximated with the "
        "Hanley-McNeil formula. For macro-AUC, the three one-vs-rest variances "
        "were averaged and divided by nine. This random-effects summary should "
        "therefore be interpreted as an approximate descriptive synthesis, not "
        "as an exact DeLong covariance analysis.",
        "",
        "## Interpretation boundary",
        "",
        "- Hospital-held-out folds remove leakage from hospital identity and "
        "local practice patterns. Patients observed at more than one hospital "
        "were excluded so that no patient crossed a fold. The model is still "
        "retrained in eICU and evaluated retrospectively.",
        "- A lower hospital-held-out AUC would indicate site transport loss; a "
        "similar AUC would not prove prospective or cross-database "
        "transportability.",
        "- AUC and flip summaries do not establish clinical utility. Hospital "
        "case mix can change the meaning of the ordinal destination classes.",
        "- Hospital-level AUC is imprecise for smaller sites, which is why the "
        "primary heterogeneity analysis requires at least 50 stays and at "
        "least five stays in each outcome class.",
        "- Bootstrap intervals condition on the fitted out-of-fold predictions "
        "and fixed hospital partition; retraining and fold-selection "
        "uncertainty are not resampled.",
        "",
        "## Saved outputs",
        "",
        "- `results/eicu_hospital_validation/summary.json`",
        "- `results/eicu_hospital_validation/eicu_hospital_fold.npz`",
        "- `results/eicu_hospital_validation/hospital_metrics.csv`",
        "- `results/eicu_hospital_validation/report.md`",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    frame = pd.read_parquet(DESIGN_FILE)
    eligible = frame["y3"].notna() & (frame["los_h"] >= 72)
    patient_hospitals = (
        frame.loc[eligible, ["uniquepid", "hospitalid"]]
        .astype({"uniquepid": str, "hospitalid": str})
        .drop_duplicates()
        .groupby("uniquepid")["hospitalid"]
        .nunique()
    )
    cross_hospital_patients = set(patient_hospitals[patient_hospitals > 1].index)
    cross_hospital_rows = eligible & frame["uniquepid"].astype(str).isin(
        cross_hospital_patients
    )
    n_patients_excluded = int(
        frame.loc[cross_hospital_rows, "uniquepid"].astype(str).nunique()
    )
    n_rows_excluded = int(cross_hospital_rows.sum())
    eligible &= ~cross_hospital_rows
    cohort = frame.loc[eligible].copy().reset_index(drop=True)
    if cohort.empty:
        raise ValueError("complete three-landmark eICU cohort is empty")
    if not cohort["patientunitstayid"].is_unique:
        raise ValueError("patientunitstayid is not unique in the complete cohort")

    global ids
    ids = cohort["patientunitstayid"].to_numpy(dtype=np.int64)
    y = cohort["y3"].to_numpy(dtype=np.int64)
    hospitals = cohort["hospitalid"].astype(str).to_numpy()
    patients = cohort["uniquepid"].astype(str).to_numpy()
    folds = list(
        GroupKFold(n_splits=N_SPLITS).split(
            np.zeros((len(cohort), 1), dtype=np.float32),
            y,
            hospitals,
        )
    )
    fold_integrity = validate_shared_hospital_folds(
        folds, hospitals, patients, y, len(cohort)
    )
    fold_id = np.full(len(cohort), -1, dtype=np.int8)
    for index, (_, test_index) in enumerate(folds):
        fold_id[test_index] = index
    if (fold_id < 0).any():
        raise ValueError("not every row received a hospital fold")

    threads = max(1, int(os.environ.get("SEPSIS_XGB_THREADS", os.cpu_count() or 1)))
    parameters = {
        **XGB_PARAMETERS,
        "n_jobs": threads,
        "random_state": RANDOM_SEED,
        "verbosity": 0,
    }
    probability: dict[int, np.ndarray] = {}
    feature_names: dict[str, list[str]] = {}
    metrics: dict[str, Any] = {}
    fold_metrics: dict[str, list[dict[str, Any]]] = {}
    for milestone in LANDMARKS:
        columns = feature_columns(cohort, milestone)
        if not columns:
            raise ValueError(f"no features available at {milestone} h")
        x = cohort[columns].astype(np.float32)
        landmark_probability = np.zeros((len(cohort), 3), dtype=np.float64)
        landmark_folds = []
        for fold_index, (train_index, test_index) in enumerate(folds):
            model = XGBClassifier(**parameters)
            model.fit(x.iloc[train_index], y[train_index])
            fold_probability = model.predict_proba(
                x.iloc[test_index]
            ).astype(np.float64)
            fold_probability = np.clip(fold_probability, 0.0, 1.0)
            fold_probability /= fold_probability.sum(axis=1, keepdims=True)
            landmark_probability[test_index] = fold_probability
            landmark_folds.append(
                {
                    "fold": fold_index,
                    "test_hospitals": int(len(np.unique(hospitals[test_index]))),
                    **metric_block(y[test_index], fold_probability),
                }
            )
        if not np.allclose(landmark_probability.sum(axis=1), 1.0):
            raise ValueError(f"{milestone} h probabilities do not sum to one")
        probability[milestone] = landmark_probability
        feature_names[str(milestone)] = columns
        metrics[str(milestone)] = {
            **metric_block(y, landmark_probability),
            "feature_count": len(columns),
        }
        fold_metrics[str(milestone)] = landmark_folds
        print(
            {
                "landmark_hours": milestone,
                "features": len(columns),
                "macro_auc": metrics[str(milestone)]["macro_auc"],
                "elapsed_seconds": round(time.time() - started, 1),
            },
            flush=True,
        )

    stability, transition, clinical, correctness = stability_and_clinical_output(
        y, probability, hospitals
    )
    stability["transition_matrix"] = transition
    arrays = model_arrays(
        y,
        probability[24],
        probability[48],
        probability[72],
    )
    hospital_level, hospital_frame = hospital_level_analysis(
        y, probability, hospitals, arrays
    )
    patient_reference, patient_alignment = fit_patient_grouped_reference(
        cohort, y, patients, parameters
    )
    comparison = comparison_with_patient_grouping(
        y,
        probability,
        hospitals,
        arrays,
        patient_reference,
        patient_alignment,
    )

    output_npz = OUT / "eicu_hospital_fold.npz"
    np.savez_compressed(
        output_npz,
        p24=probability[24],
        p48=probability[48],
        p72=probability[72],
        y3=y,
        ids=ids,
        hospital_ids=hospitals.astype("U"),
        uniquepids=patients.astype("U"),
        fold_id=fold_id,
    )
    output_csv = OUT / "hospital_metrics.csv"
    hospital_frame.to_csv(output_csv, index=False)

    outcome_counts = np.bincount(y, minlength=3)
    patient_hospital_counts = (
        pd.DataFrame({"patient": patients, "hospital": hospitals})
        .drop_duplicates()
        .groupby("patient")["hospital"]
        .nunique()
    )
    cross_hospital_patient_count = int((patient_hospital_counts > 1).sum())
    summary = {
        "module": "eicu_hospital_held_out_xgb_validation",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.time() - started, 1),
        "design": {
            "database": "eICU-CRD",
            "validation_type": "hospital-grouped internal-external validation",
            "n_splits": N_SPLITS,
            "group_column": "hospitalid",
            "same_fold_partition_reused_at_all_landmarks": True,
            "fixed_xgboost_seed": RANDOM_SEED,
            "strictly_external_validation": False,
            "prospective_validation": False,
            "clinical_utility_evaluation": False,
            "statement": (
                "Hospital-held-out cross-validation is stronger internal-external "
                "validation than patient-grouped cross-validation, but it is not "
                "prospective or independent-site external validation."
            ),
            "partition_method_note": (
                "Deterministic sklearn GroupKFold has no random_state; the fixed "
                "seed applies to XGBoost stochastic subsampling and column sampling."
            ),
        },
        "cohort": {
            "definition": (
                "y3 is non-null, los_h >= 72, and the patient is observed "
                "at exactly one hospital in this validation cohort"
            ),
            "id_column": "patientunitstayid",
            "n": int(len(cohort)),
            "n_hospitals": int(len(np.unique(hospitals))),
            "n_unique_patients": int(len(np.unique(patients))),
            "patients_observed_in_more_than_one_hospital": (
                cross_hospital_patient_count
            ),
            "patients_excluded_for_multiple_hospitals": n_patients_excluded,
            "rows_excluded_for_multiple_hospitals": n_rows_excluded,
            "outcome_counts": {
                str(index): int(count) for index, count in enumerate(outcome_counts)
            },
            "observed_class_distribution": {
                str(index): round(float(count / len(y)), 6)
                for index, count in enumerate(outcome_counts)
            },
            "conditioning": (
                "Stays observable through 72 hours; estimates do not generalize "
                "to stays discharged or dying before 72 hours."
            ),
        },
        "features": {
            "rule": (
                "First-24-hour static/demographic/laboratory features plus "
                "non-overlapping 24-hour summary windows through the landmark."
            ),
            "columns_by_landmark": feature_names,
        },
        "training": {
            "algorithm": "XGBoost",
            "parameters": parameters,
            "fold_integrity": fold_integrity,
            "fold_assignment_counts": {
                str(index): int((fold_id == index).sum())
                for index in range(N_SPLITS)
            },
            "same_hyperparameters_as": "src/sepsis/train_xgb_shared_folds.py",
        },
        "metrics": metrics,
        "fold_metrics": fold_metrics,
        "stability": stability,
        "clinical_changes": clinical,
        "correctness": correctness,
        "comparison_with_patient_grouped": comparison,
        "hospital_level": hospital_level,
        "uncertainty": {
            "method": "hospital-cluster bootstrap of the evaluation sample",
            "replicates": BOOTSTRAP_REPLICATES,
            "cluster_unit": "hospitalid",
            "cluster_count": int(len(np.unique(hospitals))),
            "scope": (
                "Conditional on fitted out-of-fold predictions and the fixed "
                "hospital partition; retraining uncertainty is not resampled."
            ),
        },
        "reproducibility": {
            "design_file": str(DESIGN_FILE),
            "design_sha256": sha256_file(DESIGN_FILE),
            "patient_grouped_summary": str(SHARED_SUMMARY_FILE),
            "patient_grouped_predictions": str(SHARED_NPZ_FILE),
        },
        "outputs": {
            "summary_json": str(OUT / "summary.json"),
            "aligned_oof_npz": str(output_npz),
            "hospital_metrics_csv": str(output_csv),
            "report_md": str(OUT / "report.md"),
        },
        "interpretation": (
            "Hospital-held-out results provide a stricter retrospective estimate "
            "of transport to unseen eICU sites. They do not establish prospective "
            "validity, causal benefit, or clinical utility."
        ),
    }
    summary_path = OUT / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    write_report(summary, OUT / "report.md")
    print(
        {
            "n": int(len(cohort)),
            "hospitals": int(len(np.unique(hospitals))),
            "macro_auc": {
                str(milestone): metrics[str(milestone)]["macro_auc"]
                for milestone in LANDMARKS
            },
            "flip_24_72": stability["flip"]["24_72"],
            "patient_grouped_auc_72": comparison[
                "patient_grouped_metrics_recomputed"
            ]["72"]["macro_auc"],
            "hospital_der_simonian_laird": hospital_level[
                "der_simonian_laird_72h_macro_auc"
            ],
            "elapsed_seconds": round(time.time() - started, 1),
            "outputs": summary["outputs"],
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
