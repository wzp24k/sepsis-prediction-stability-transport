# -*- coding: utf-8 -*-
"""Six-direction fixed-weight transport and target recalibration learning curves.

The analysis extends the two-database transport audit to all directed pairs
among MIMIC-IV, eICU-CRD, and MIMIC-III. A source model is fit once at each
landmark and transported unchanged to each of the two target databases. Target
labels are used only in the explicitly separated local-recalibration analysis.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from xgboost import XGBClassifier


ROOT = Path(os.environ.get("SEPSIS_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sepsis.analyze_transport import (  # noqa: E402
    EXCLUDED_COLUMNS,
    LABS,
    PARAMS,
    STATIC,
    STATS,
    VITALS,
    expected_calibration_error,
    feature_columns_for_landmark,
    multiclass_brier,
    to_native,
)


DATA = ROOT / "data" / "processed"
OUT = ROOT / "results" / "transport_extended"
OUT.mkdir(parents=True, exist_ok=True)

LANDMARKS = (24, 48, 72)
CLASS_LABELS = {0: "home", 1: "facility", 2: "death"}
SEED = 20261029
DEFAULT_REPEATS = 50
CALIBRATION_NS = (0, 25, 50, 100, 200)
CALIBRATION_METHODS = ("multinomial_logistic", "temperature_scaling")
RECALIBRATION_METRICS = (
    "macro_auc",
    "accuracy",
    "log_loss",
    "brier_multiclass",
    "ece_top_label_10bin",
    "mean_max_probability",
)
N_JOBS = int(
    os.environ.get(
        "SEPSIS_XGB_THREADS",
        str(min(16, os.cpu_count() or 1)),
    )
)
MODEL_PARAMS = {**PARAMS, "n_jobs": N_JOBS}

MIMIC3_EXCLUSIONS = {
    "icustay_id",
    "discharge_location",
    "hospital_expire_flag",
}
EXACT_EXCLUSIONS = EXCLUDED_COLUMNS | MIMIC3_EXCLUSIONS

DATABASES = {
    "mimic4": {
        "label": "MIMIC-IV",
        "id": "stay_id",
        "group": "subject_id",
        "cluster": "subject_id",
        "duration": "los",
        "institution": "Beth Israel Deaconess Medical Center",
        "era": "modern",
    },
    "eicu": {
        "label": "eICU-CRD",
        "id": "patientunitstayid",
        "group": "uniquepid",
        "cluster": "hospitalid",
        "duration": "los_h",
        "institution": "multi-hospital consortium",
        "era": "2014-2015",
    },
    "mimic3": {
        "label": "MIMIC-III",
        "id": "icustay_id",
        "group": "subject_id",
        "cluster": "subject_id",
        "duration": "los",
        "institution": "Beth Israel Deaconess Medical Center",
        "era": "2001-2012",
    },
}

COMPARISON_TYPES = {
    ("mimic4", "eicu"): {
        "type": "cross_institution",
        "label": "cross-institution",
        "interpretation": (
            "Transport from the modern single-center MIMIC-IV database to the "
            "multi-hospital eICU-CRD database."
        ),
    },
    ("eicu", "mimic4"): {
        "type": "cross_institution",
        "label": "cross-institution",
        "interpretation": (
            "Transport from the multi-hospital eICU-CRD database to the modern "
            "single-center MIMIC-IV database."
        ),
    },
    ("mimic4", "mimic3"): {
        "type": "same_institution_cross_era",
        "label": "same-institution cross-era",
        "interpretation": (
            "Transport from modern MIMIC-IV to historical MIMIC-III at the same "
            "institution; this is not independent external validation."
        ),
    },
    ("mimic3", "mimic4"): {
        "type": "same_institution_cross_era",
        "label": "same-institution cross-era",
        "interpretation": (
            "Transport from historical MIMIC-III to modern MIMIC-IV at the same "
            "institution; this is not independent external validation."
        ),
    },
    ("mimic3", "eicu"): {
        "type": "cross_institution_and_era",
        "label": "cross-institution and cross-era",
        "interpretation": (
            "Transport from historical single-center MIMIC-III to the later "
            "multi-hospital eICU-CRD; institution and era both change."
        ),
    },
    ("eicu", "mimic3"): {
        "type": "cross_institution_and_era",
        "label": "cross-institution and cross-era",
        "interpretation": (
            "Transport from the later multi-hospital eICU-CRD to historical "
            "single-center MIMIC-III; institution and era both change."
        ),
    },
}

DIRECTIONS = tuple(
    f"{source}_to_{target}" for source, target in COMPARISON_TYPES
)


def cohort_mask(
    frame: pd.DataFrame, duration_column: str, milestone: int
) -> np.ndarray:
    """Outcome is defined and the stay reaches the requested landmark."""
    return (
        frame["y3"].notna().to_numpy()
        & (frame[duration_column].to_numpy(dtype=float) >= milestone)
    )


def fit_source_model(
    source: pd.DataFrame,
    source_mask: np.ndarray,
    features: list[str],
) -> tuple[XGBClassifier, dict]:
    x = (
        source.loc[source_mask, features]
        .astype(float)
        .replace([np.inf, -np.inf], np.nan)
    )
    y = source.loc[source_mask, "y3"].to_numpy(dtype=int)
    if len(np.unique(y)) != 3:
        raise ValueError(f"Source outcome lacks a class: {np.unique(y)}")
    model = XGBClassifier(**MODEL_PARAMS)
    model.fit(x, y)
    metadata = {
        "n": int(len(y)),
        "class_counts": {
            CLASS_LABELS[class_index]: int((y == class_index).sum())
            for class_index in range(3)
        },
        "feature_count": int(len(features)),
    }
    return model, metadata


def predict_fixed(
    model: XGBClassifier,
    target: pd.DataFrame,
    target_mask: np.ndarray,
    features: list[str],
) -> np.ndarray:
    x = (
        target.loc[target_mask, features]
        .astype(float)
        .replace([np.inf, -np.inf], np.nan)
    )
    probability = np.asarray(model.predict_proba(x), dtype=float)
    if probability.shape != (int(target_mask.sum()), 3):
        raise ValueError(f"Unexpected target probability shape: {probability.shape}")
    if not np.isfinite(probability).all():
        raise ValueError("Non-finite transported probabilities")
    return probability / probability.sum(axis=1, keepdims=True)


def basic_metrics(y: np.ndarray, probability: np.ndarray) -> dict:
    labels = np.arange(3)
    probability = np.asarray(probability, dtype=float)
    y = np.asarray(y, dtype=int)
    return {
        "n": int(len(y)),
        "class_counts": {
            CLASS_LABELS[class_index]: int((y == class_index).sum())
            for class_index in labels
        },
        "macro_auc": float(
            roc_auc_score(
                y,
                probability,
                multi_class="ovr",
                average="macro",
                labels=labels,
            )
        ),
        "accuracy": float(accuracy_score(y, probability.argmax(axis=1))),
        "log_loss": float(log_loss(y, probability, labels=labels)),
        "brier_multiclass": multiclass_brier(y, probability),
        "ece_top_label_10bin": expected_calibration_error(y, probability),
        "mean_max_probability": float(probability.max(axis=1).mean()),
    }


def rate(numerator: np.ndarray, denominator: np.ndarray | None = None) -> dict:
    numerator = np.asarray(numerator, dtype=bool)
    if denominator is None:
        denominator = np.ones(len(numerator), dtype=bool)
    else:
        denominator = np.asarray(denominator, dtype=bool)
    denominator_n = int(denominator.sum())
    event_n = int((numerator & denominator).sum())
    return {
        "n": event_n,
        "denominator_n": denominator_n,
        "rate": float(event_n / denominator_n) if denominator_n else None,
    }


def transition_analysis(
    y: np.ndarray,
    p24: np.ndarray,
    p72: np.ndarray,
) -> dict:
    y = np.asarray(y, dtype=int)
    pred24 = p24.argmax(axis=1)
    pred72 = p72.argmax(axis=1)
    flip = pred24 != pred72
    upgrade = pred72 > pred24
    downgrade = pred72 < pred24
    distance24 = np.abs(pred24 - y)
    distance72 = np.abs(pred72 - y)
    gain = distance72 < distance24
    loss = distance72 > distance24
    tie = distance72 == distance24
    row_index = np.arange(len(y))
    true_probability_change = (
        p72[row_index, y] - p24[row_index, y]
    )
    transition = np.zeros((3, 3), dtype=int)
    for first, second in zip(pred24, pred72):
        transition[int(first), int(second)] += 1

    return {
        "n": int(len(y)),
        "class_counts": {
            CLASS_LABELS[class_index]: int((y == class_index).sum())
            for class_index in range(3)
        },
        "flip": rate(flip),
        "total_variation_mean": float(
            (0.5 * np.abs(p24 - p72).sum(axis=1)).mean()
        ),
        "transition_matrix_rows_24_cols_72": transition.tolist(),
        "rows_24": [CLASS_LABELS[index] for index in range(3)],
        "cols_72": [CLASS_LABELS[index] for index in range(3)],
        "clinical_major_changes": {
            "major_upgrade_to_death": rate(
                (pred24 < 2) & (pred72 == 2)
            ),
            "home_to_facility": rate(
                (pred24 == 0) & (pred72 == 1)
            ),
            "any_upgrade": rate(upgrade),
            "any_downgrade": rate(downgrade),
        },
        "flip_correctness_relative_to_final_outcome": {
            "all": {
                "distance_gain": rate(gain),
                "distance_loss": rate(loss),
                "distance_tie": rate(tie),
                "correct_upgrade": rate(upgrade & gain),
                "incorrect_upgrade_including_tie": rate(upgrade & ~gain),
                "correct_downgrade": rate(downgrade & gain),
                "incorrect_downgrade_including_tie": rate(downgrade & ~gain),
                "tie_upgrade": rate(upgrade & tie),
                "tie_downgrade": rate(downgrade & tie),
                "true_class_probability_improved": rate(
                    true_probability_change > 1e-12
                ),
                "true_class_probability_worsened": rate(
                    true_probability_change < -1e-12
                ),
                "true_class_probability_change_mean": float(
                    true_probability_change.mean()
                ),
            },
            "among_flips": {
                "distance_gain": rate(gain, flip),
                "distance_loss": rate(loss, flip),
                "distance_tie": rate(tie, flip),
                "correct_upgrade": rate(upgrade & gain, flip),
                "incorrect_upgrade_including_tie": rate(upgrade & ~gain, flip),
                "correct_downgrade": rate(downgrade & gain, flip),
                "incorrect_downgrade_including_tie": rate(
                    downgrade & ~gain, flip
                ),
                "tie_upgrade": rate(upgrade & tie, flip),
                "tie_downgrade": rate(downgrade & tie, flip),
                "true_class_probability_improved": rate(
                    true_probability_change > 1e-12, flip
                ),
                "true_class_probability_change_mean": float(
                    true_probability_change[flip].mean()
                )
                if flip.any()
                else None,
            },
        },
    }


def stratified_allocation(counts: np.ndarray, total: int) -> np.ndarray:
    """Allocate a stratified calibration size with at least one per class."""
    counts = np.asarray(counts, dtype=int)
    if total < len(counts):
        raise ValueError("Calibration size must include every outcome class")
    if total > counts.sum():
        raise ValueError("Calibration size exceeds the target cohort")
    if (counts <= 0).any():
        raise ValueError("Every outcome class must be present for stratification")

    allocation = np.ones(len(counts), dtype=int)
    remaining = total - len(counts)
    weights = counts / counts.sum()
    quotas = remaining * weights
    additional = np.floor(quotas).astype(int)
    allocation += additional
    left = total - int(allocation.sum())
    if left:
        remainders = quotas - additional
        order = np.argsort(-remainders, kind="stable")
        for index in order[:left]:
            allocation[index] += 1
    if allocation.sum() != total or (allocation > counts).any():
        raise ValueError("Invalid stratified allocation")
    return allocation


def one_row_per_patient(
    probability: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep the first target stay for each patient before local calibration."""
    groups = np.asarray(groups, dtype=str)
    probability = np.asarray(probability, dtype=float)
    y = np.asarray(y, dtype=int)
    if not (len(probability) == len(y) == len(groups)):
        raise ValueError("Recalibration probabilities, outcomes, and groups misalign")
    keep = ~pd.Series(groups).duplicated(keep="first").to_numpy()
    return probability[keep], y[keep]


def split_seed(direction: str, milestone: int, repeat: int) -> int:
    material = f"{SEED}|{direction}|{milestone}|{repeat}".encode("utf-8")
    digest = hashlib.sha256(material).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def log_probability_features(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(
        np.asarray(probability, dtype=float),
        1e-8,
        1.0,
    )
    return np.log(clipped)


def fit_recalibrator(
    probability: np.ndarray,
    y: np.ndarray,
) -> LogisticRegression:
    x = log_probability_features(probability)
    model = LogisticRegression(
        C=1e6,
        solver="lbfgs",
        max_iter=5000,
        tol=1e-8,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(x, y)
    if not np.array_equal(np.asarray(model.classes_, dtype=int), np.arange(3)):
        raise ValueError("Recalibrator did not retain all three outcome classes")
    return model


def recalibrated_probability(
    model: LogisticRegression,
    probability: np.ndarray,
) -> np.ndarray:
    output = np.asarray(
        model.predict_proba(log_probability_features(probability)),
        dtype=float,
    )
    output = output / output.sum(axis=1, keepdims=True)
    if not np.isfinite(output).all():
        raise ValueError("Recalibrator returned non-finite probabilities")
    return output


def fit_temperature(
    probability: np.ndarray,
    y: np.ndarray,
) -> float:
    """Fit one positive temperature by minimising calibration-set log loss."""
    logits = log_probability_features(probability)
    y = np.asarray(y, dtype=int)
    labels = np.arange(3)

    def objective(theta: float) -> float:
        temperature = float(np.exp(theta))
        scaled = logits / temperature
        scaled -= scaled.max(axis=1, keepdims=True)
        probability_scaled = np.exp(scaled)
        probability_scaled /= probability_scaled.sum(axis=1, keepdims=True)
        return float(log_loss(y, probability_scaled, labels=labels))

    result = minimize_scalar(
        objective,
        bounds=(-3.0, 3.0),
        method="bounded",
        options={"xatol": 1e-7, "maxiter": 500},
    )
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError("Temperature scaling failed to converge")
    return float(np.exp(result.x))


def temperature_scaled_probability(
    probability: np.ndarray,
    temperature: float,
) -> np.ndarray:
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be a positive finite value")
    scaled = log_probability_features(probability) / temperature
    scaled -= scaled.max(axis=1, keepdims=True)
    output = np.exp(scaled)
    output /= output.sum(axis=1, keepdims=True)
    return output


def percentile_record(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "p2_5": float(np.percentile(array, 2.5)),
        "p97_5": float(np.percentile(array, 97.5)),
    }


def aggregate_calibration_rows(rows: list[dict]) -> dict:
    output = {
        "repetitions": int(len(rows)),
        "evaluation_n": int(rows[0]["evaluation_n"]),
    }
    for metric in RECALIBRATION_METRICS:
        output[metric] = percentile_record(
            [float(row[metric]) for row in rows]
        )
    return output


def run_recalibration_curves(
    direction: str,
    milestone: int,
    probability: np.ndarray,
    y: np.ndarray,
    repeats: int,
) -> tuple[dict[str, dict], dict[str, dict], dict[str, dict], list[dict]]:
    """Recalibrate on target labels and evaluate on different target patients."""
    y = np.asarray(y, dtype=int)
    counts = np.bincount(y, minlength=3)
    aggregate: dict[str, dict[str, dict]] = {
        method: {} for method in CALIBRATION_METHODS
    }
    replicate_rows: list[dict] = []

    baseline = basic_metrics(y, probability)
    for method in CALIBRATION_METHODS:
        aggregate[method]["0"] = {
            "requested_local_labels": 0,
            "method": method,
            "repetitions": 1,
            "evaluation_n": int(len(y)),
            "calibration_class_counts_mean": {
                CLASS_LABELS[index]: 0.0 for index in range(3)
            },
        }
        for metric in RECALIBRATION_METRICS:
            value = float(baseline[metric])
            aggregate[method]["0"][metric] = {
                "mean": value,
                "p2_5": value,
                "p97_5": value,
            }
    replicate_rows.append(
        {
            "direction": direction,
            "landmark_hours": milestone,
            "repeat": 0,
            "method": "untouched",
            "n_local_labels": 0,
            "evaluation_n": int(len(y)),
            "macro_auc": baseline["macro_auc"],
            "accuracy": baseline["accuracy"],
            "log_loss": baseline["log_loss"],
            "brier_multiclass": baseline["brier_multiclass"],
            "ece_top_label_10bin": baseline["ece_top_label_10bin"],
            "mean_max_probability": baseline["mean_max_probability"],
        }
    )

    request_to_rows: dict[str, dict[int, list[dict]]] = {
        method: defaultdict(list) for method in CALIBRATION_METHODS
    }
    request_to_class_counts: dict[int, list[np.ndarray]] = defaultdict(list)
    request_to_differences: dict[int, list[dict[str, float]]] = defaultdict(list)
    for repeat in range(repeats):
        rng = np.random.default_rng(split_seed(direction, milestone, repeat))
        ranked_by_class = [
            rng.permutation(np.flatnonzero(y == class_index))
            for class_index in range(3)
        ]
        for requested_n in CALIBRATION_NS[1:]:
            if requested_n > len(y):
                continue
            allocation = stratified_allocation(counts, requested_n)
            calibration_parts = [
                ranked_by_class[class_index][: allocation[class_index]]
                for class_index in range(3)
            ]
            calibration_index = np.concatenate(calibration_parts)
            calibration_index = np.sort(calibration_index)
            evaluation_mask = np.ones(len(y), dtype=bool)
            evaluation_mask[calibration_index] = False
            if len(calibration_index) != requested_n or not evaluation_mask.any():
                raise ValueError("Invalid calibration/evaluation split")

            logistic_recalibrator = fit_recalibrator(
                probability[calibration_index],
                y[calibration_index],
            )
            logistic_probability = recalibrated_probability(
                logistic_recalibrator,
                probability[evaluation_mask],
            )
            temperature = fit_temperature(
                probability[calibration_index],
                y[calibration_index],
            )
            temperature_probability = temperature_scaled_probability(
                probability[evaluation_mask],
                temperature,
            )
            method_probability = {
                "multinomial_logistic": logistic_probability,
                "temperature_scaling": temperature_probability,
            }
            method_metrics: dict[str, dict] = {}
            for method in CALIBRATION_METHODS:
                metrics = basic_metrics(
                    y[evaluation_mask],
                    method_probability[method],
                )
                method_metrics[method] = metrics
                row = {
                    "direction": direction,
                    "landmark_hours": milestone,
                    "repeat": repeat,
                    "method": method,
                    "n_local_labels": requested_n,
                    "evaluation_n": metrics["n"],
                    **{
                        metric: metrics[metric]
                        for metric in RECALIBRATION_METRICS
                    },
                }
                if method == "temperature_scaling":
                    row["temperature"] = temperature
                request_to_rows[method][requested_n].append(row)
            request_to_class_counts[requested_n].append(allocation)
            request_to_differences[requested_n].append(
                {
                    metric: (
                        method_metrics["temperature_scaling"][metric]
                        - method_metrics["multinomial_logistic"][metric]
                    )
                    for metric in RECALIBRATION_METRICS
                }
            )

    method_differences: dict[str, dict] = {}
    for requested_n in CALIBRATION_NS[1:]:
        for method in CALIBRATION_METHODS:
            rows = request_to_rows[method].get(requested_n, [])
            if not rows:
                aggregate[method][str(requested_n)] = {
                    "requested_local_labels": requested_n,
                    "method": method,
                    "repetitions": 0,
                    "available": False,
                    "reason": "requested calibration size unavailable",
                }
                continue
            record = aggregate_calibration_rows(rows)
            record = {
                "requested_local_labels": requested_n,
                "method": method,
                "available": True,
                **record,
                "calibration_class_counts_mean": {
                    CLASS_LABELS[index]: float(
                        np.mean(
                            [
                                allocation[index]
                                for allocation in request_to_class_counts[
                                    requested_n
                                ]
                            ]
                        )
                    )
                    for index in range(3)
                },
            }
            aggregate[method][str(requested_n)] = record
            replicate_rows.extend(rows)

        differences = request_to_differences.get(requested_n, [])
        method_differences[str(requested_n)] = {
            "comparison": "temperature_scaling_minus_multinomial_logistic",
            "repetitions": len(differences),
            **{
                metric: {
                    **percentile_record(
                        [float(row[metric]) for row in differences]
                    ),
                    "fraction_favoring_temperature_scaling": round(
                        float(
                            np.mean(
                                [
                                    float(row[metric]) > 0
                                    if metric in ("macro_auc", "accuracy")
                                    else float(row[metric]) < 0
                                    for row in differences
                                ]
                            )
                        ),
                        6,
                    ),
                }
                for metric in RECALIBRATION_METRICS
            },
        }
    return (
        aggregate["multinomial_logistic"],
        aggregate["temperature_scaling"],
        method_differences,
        replicate_rows,
    )


def format_percent(value: float | None) -> str:
    return "NA" if value is None else f"{100.0 * value:.1f}%"


def format_metric(value: float | None) -> str:
    return "NA" if value is None else f"{value:.3f}"


def format_interval(record: dict) -> str:
    return (
        f"{record['mean']:.3f} "
        f"({record['p2_5']:.3f}-{record['p97_5']:.3f})"
    )


def build_report(summary: dict) -> str:
    lines = [
        "# Six-Direction Fixed-Weight Transport and Target Recalibration",
        "",
        f"Generated: {summary['generated_at']}",
        "",
        "## Scope",
        "",
        (
            "All six directed pairs among MIMIC-IV, eICU-CRD, and MIMIC-III were "
            "evaluated. Each fixed model was fit on all source patients with a "
            "mapped three-class outcome who reached the landmark and was then "
            "applied unchanged to all target patients reaching the same landmark."
        ),
        "",
        (
            "MIMIC-IV and MIMIC-III are from the same institution in different "
            "eras. Their directions are same-institution cross-era transport and "
            "are not independent external validation."
        ),
        "",
        "| Direction | Comparison | Interpretation |",
        "|---|---|---|",
    ]
    for direction in DIRECTIONS:
        result = summary["directions"][direction]
        lines.append(
            f"| {direction} | {result['comparison_label']} | "
            f"{result['interpretation']} |"
        )

    lines.extend(
        [
            "",
            "## Cohorts and features",
            "",
            (
                f"Exact-name clinical features shared by all three databases: "
                f"{summary['feasibility']['exact_common_feature_count']}. "
                f"Landmark feature counts are 24 h "
                f"{summary['feasibility']['landmark_feature_counts']['24']}, "
                f"48 h {summary['feasibility']['landmark_feature_counts']['48']}, "
                f"and 72 h "
                f"{summary['feasibility']['landmark_feature_counts']['72']}."
            ),
            "",
            "| Database | Landmark | n | Home | Facility | Death |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for database in DATABASES:
        cohorts = summary["feasibility"]["cohorts"][database]
        for milestone in LANDMARKS:
            cohort = cohorts[str(milestone)]
            lines.append(
                f"| {DATABASES[database]['label']} | {milestone} h | "
                f"{cohort['n']} | {cohort['class_counts']['home']} | "
                f"{cohort['class_counts']['facility']} | "
                f"{cohort['class_counts']['death']} |"
            )

    lines.extend(
        [
            "",
            "## Fixed-weight transport metrics",
            "",
            (
                "Metrics use the target population reaching each landmark. "
                "No target fitting or recalibration enters these rows."
            ),
            "",
            "| Direction | Type | Landmark | n | Macro-AUC | Accuracy | "
            "Log loss | Brier | Top-label ECE |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for direction in DIRECTIONS:
        result = summary["directions"][direction]
        for milestone in LANDMARKS:
            metrics = result["transport_landmarks"][str(milestone)][
                "target_all_reaching_landmark"
            ]
            lines.append(
                f"| {direction} | {result['comparison_label']} | "
                f"{milestone} h | {metrics['n']} | "
                f"{metrics['macro_auc']:.3f} | {metrics['accuracy']:.3f} | "
                f"{metrics['log_loss']:.3f} | "
                f"{metrics['brier_multiclass']:.3f} | "
                f"{metrics['ece_top_label_10bin']:.3f} |"
            )

    lines.extend(
        [
            "",
            "## Fixed-weight 24-to-72 h prediction change",
            "",
            (
                "The target cohort is restricted to patients present at all three "
                "landmarks. The 24 h and 72 h models are independently fit on "
                "their source landmark cohorts and applied without target updates."
            ),
            "",
            "| Direction | n | Flip | Mean TV | Upgrade to death | "
            "home->facility | Distance gain among flips | Distance loss among flips | "
            "p_true improved among flips |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for direction in DIRECTIONS:
        transition = summary["directions"][direction][
            "fixed_24_to_72_target_complete"
        ]
        flips = transition["flip_correctness_relative_to_final_outcome"][
            "among_flips"
        ]
        clinical = transition["clinical_major_changes"]
        lines.append(
            f"| {direction} | {transition['n']} | "
            f"{format_percent(transition['flip']['rate'])} | "
            f"{transition['total_variation_mean']:.3f} | "
            f"{format_percent(clinical['major_upgrade_to_death']['rate'])} | "
            f"{format_percent(clinical['home_to_facility']['rate'])} | "
            f"{format_percent(flips['distance_gain']['rate'])} | "
            f"{format_percent(flips['distance_loss']['rate'])} | "
            f"{format_percent(flips['true_class_probability_improved']['rate'])} |"
        )

    lines.extend(
        [
            "",
            "## Minimal target recalibration",
            "",
            (
                "For each repeat, a deterministic outcome-stratified calibration "
                "sample was drawn from the target database. A multinomial "
                "logistic recalibrator and a one-parameter temperature scaler "
                "were fitted on the same transported log-probabilities and "
                "evaluated on the remaining target patients. One stay per "
                "patient was retained before calibration to keep calibration "
                "and evaluation patients disjoint. The n=0 row is the untouched "
                "transported model on that same population. Values are mean "
                "(2.5th-97.5th percentile across repeated splits)."
            ),
            "",
            "| Direction | Landmark | Local labels | Splits | Evaluation n | "
            "Macro-AUC | Top-label ECE | Log loss |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for direction in DIRECTIONS:
        recalibration = summary["recalibration"][direction]
        for milestone in LANDMARKS:
            curves = recalibration[str(milestone)]["curves"]
            for requested_n in CALIBRATION_NS:
                curve = curves[str(requested_n)]
                if not curve.get("available", True):
                    continue
                lines.append(
                    f"| {direction} | {milestone} h | {requested_n} | "
                    f"{curve['repetitions']} | {curve['evaluation_n']} | "
                    f"{format_interval(curve['macro_auc'])} | "
                    f"{format_interval(curve['ece_top_label_10bin'])} | "
                    f"{format_interval(curve['log_loss'])} |"
                )

    lines.extend(
        [
            "",
            "### Paired low-label calibration comparison at 72 h",
            "",
            (
                "Difference is temperature scaling minus multinomial logistic "
                "recalibration on the same split. Negative differences favour "
                "temperature scaling for calibration error, log loss, and Brier "
                "score; positive differences favour it for AUC."
            ),
            "",
            "| Direction | Local labels | Delta macro-AUC | Delta ECE | "
            "Delta log loss | Delta Brier | ECE favours temperature |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for direction in DIRECTIONS:
        record = summary["recalibration"][direction]["72"]
        comparisons = record[
            "temperature_scaling_minus_multinomial_logistic"
        ]
        for requested_n in CALIBRATION_NS[1:]:
            comparison = comparisons[str(requested_n)]
            if not comparison["repetitions"]:
                continue
            lines.append(
                f"| {direction} | {requested_n} | "
                f"{comparison['macro_auc']['mean']:+.3f} | "
                f"{comparison['ece_top_label_10bin']['mean']:+.3f} | "
                f"{comparison['log_loss']['mean']:+.3f} | "
                f"{comparison['brier_multiclass']['mean']:+.3f} | "
                f"{comparison['ece_top_label_10bin']['fraction_favoring_temperature_scaling']:.1%} |"
            )

    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            (
                "1. Fixed-weight transport measures retrospective transport of "
                "fitted predictions. It is not prospective clinical validation."
            ),
            (
                "2. MIMIC-IV to or from MIMIC-III changes era, not institution, "
                "and cannot be described as independent external validation."
            ),
            (
                "3. Directional performance differences combine case mix, coding "
                "differences, care era, missingness, and model fit rather than one "
                "isolated causal factor."
            ),
            (
                "4. Target recalibration uses observed local outcomes. Its curves "
                "estimate how much labeled local data may be needed for calibration; "
                "they do not establish clinical utility or a deployment policy."
            ),
            (
                "5. The temperature-scaled and multinomial calibrators were "
                "compared on identical splits. The simpler calibrator was not "
                "selected after inspecting direction-specific results."
            ),
            (
                "6. Facility combines heterogeneous post-acute destinations. A "
                "home-to-facility change is not equivalent in clinical severity to "
                "an upgrade to death."
            ),
            (
                "7. Repeated-split intervals describe sensitivity to the finite "
                "local calibration sample conditional on one fixed transported "
                "prediction per target patient. They are not model-refitting "
                "confidence intervals."
            ),
            "",
            "## Artifacts",
            "",
            "- `summary.json`: full machine-readable results and protocol.",
            "- `recalibration_learning_curves.csv`: aggregate metrics for both calibrators.",
            "- `recalibration_replicates.csv`: metrics for every deterministic split.",
            "- `analyze_transport_extended.py`: reproducible analysis script.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run six-direction fixed-weight transport and recalibration"
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=DEFAULT_REPEATS,
        help=f"Deterministic target calibration splits (default: {DEFAULT_REPEATS})",
    )
    args = parser.parse_args()
    if args.repeats < 20:
        raise ValueError("At least 20 repeated calibration splits are required")

    frames = {
        database: pd.read_parquet(DATA / database / "design_all.parquet")
        for database in DATABASES
    }
    common_columns = [
        column
        for column in frames["mimic4"].columns
        if all(column in frame.columns for frame in frames.values())
        and column not in EXACT_EXCLUSIONS
    ]
    feature_sets = {
        milestone: feature_columns_for_landmark(common_columns, milestone)
        for milestone in LANDMARKS
    }
    expected_counts = {
        24: len([column for column in STATIC if column in common_columns])
        + len(VITALS) * len(STATS),
        48: len([column for column in STATIC if column in common_columns])
        + 2 * len(VITALS) * len(STATS),
        72: len([column for column in STATIC if column in common_columns])
        + 3 * len(VITALS) * len(STATS),
    }
    if any(len(feature_sets[milestone]) != expected_counts[milestone] for milestone in LANDMARKS):
        raise RuntimeError("Feature alignment differs from the prespecified exact-name design")

    cohort_diagnostics: dict[str, dict] = {}
    for database, spec in DATABASES.items():
        frame = frames[database]
        cohort_diagnostics[database] = {}
        for milestone in LANDMARKS:
            mask = cohort_mask(frame, spec["duration"], milestone)
            y = frame.loc[mask, "y3"].to_numpy(dtype=int)
            cohort_diagnostics[database][str(milestone)] = {
                "n": int(mask.sum()),
                "class_counts": {
                    CLASS_LABELS[index]: int((y == index).sum())
                    for index in range(3)
                },
            }
        complete_mask = cohort_mask(frame, spec["duration"], 72)
        complete_y = frame.loc[complete_mask, "y3"].to_numpy(dtype=int)
        cohort_diagnostics[database]["complete_24_48_72"] = {
            "n": int(complete_mask.sum()),
            "class_counts": {
                CLASS_LABELS[index]: int((complete_y == index).sum())
                for index in range(3)
            },
        }

    directions_result: dict[str, dict] = {}
    recalibration_result: dict[str, dict] = {}
    calibration_replicate_rows: list[dict] = []

    # Train one source model per source database and landmark, then transport it
    # to both target databases. This is the same fixed model that separate
    # direction-specific fits would produce, without keeping nine models alive
    # at once.
    for source_name in DATABASES:
        source = frames[source_name]
        source_spec = DATABASES[source_name]
        models: dict[int, XGBClassifier] = {}
        training: dict[int, dict] = {}
        print(f"Fitting source models: {source_name}", flush=True)
        for milestone in LANDMARKS:
            source_mask = cohort_mask(
                source, source_spec["duration"], milestone
            )
            model, metadata = fit_source_model(
                source,
                source_mask,
                feature_sets[milestone],
            )
            models[milestone] = model
            training[milestone] = metadata
            print(
                f"  {source_name} {milestone} h "
                f"n={metadata['n']} features={metadata['feature_count']}",
                flush=True,
            )

        for target_name in DATABASES:
            if source_name == target_name:
                continue
            direction = f"{source_name}_to_{target_name}"
            comparison = COMPARISON_TYPES[(source_name, target_name)]
            target = frames[target_name]
            target_spec = DATABASES[target_name]
            probabilities: dict[int, np.ndarray] = {}
            outcomes: dict[int, np.ndarray] = {}
            target_groups: dict[int, np.ndarray] = {}
            target_masks: dict[int, np.ndarray] = {}
            transport_landmarks: dict[str, dict] = {}

            print(f"Running {direction}", flush=True)
            for milestone in LANDMARKS:
                target_mask = cohort_mask(
                    target, target_spec["duration"], milestone
                )
                probability = predict_fixed(
                    models[milestone],
                    target,
                    target_mask,
                    feature_sets[milestone],
                )
                y = target.loc[target_mask, "y3"].to_numpy(dtype=int)
                metrics = basic_metrics(y, probability)
                probabilities[milestone] = probability
                outcomes[milestone] = y
                target_groups[milestone] = target.loc[
                    target_mask, target_spec["group"]
                ].astype(str).to_numpy()
                target_masks[milestone] = target_mask
                transport_landmarks[str(milestone)] = {
                    "feature_count": int(len(feature_sets[milestone])),
                    "source_training": training[milestone],
                    "target_all_reaching_landmark": metrics,
                }
                print(
                    f"  {milestone} h n={metrics['n']} "
                    f"AUC={metrics['macro_auc']:.4f} "
                    f"ECE={metrics['ece_top_label_10bin']:.4f}",
                    flush=True,
                )

            complete_mask = cohort_mask(
                target, target_spec["duration"], 72
            )
            complete_ids = target.loc[
                complete_mask, target_spec["id"]
            ].to_numpy(dtype=np.int64)
            aligned_probability: dict[int, np.ndarray] = {}
            for milestone in (24, 72):
                ids = target.loc[
                    target_masks[milestone], target_spec["id"]
                ].to_numpy(dtype=np.int64)
                if len(np.unique(ids)) != len(ids):
                    raise ValueError(f"{target_name}: duplicate target identifiers")
                lookup = pd.Index(ids)
                index = lookup.get_indexer(complete_ids)
                if (index < 0).any():
                    raise ValueError(
                        "Complete target cohort is not contained in landmark cohort"
                    )
                aligned_probability[milestone] = probabilities[milestone][index]
            complete_y = target.loc[
                complete_mask, "y3"
            ].to_numpy(dtype=int)
            if not np.array_equal(complete_y, outcomes[72]):
                raise ValueError("Target complete-cohort outcomes do not align")

            transition = transition_analysis(
                complete_y,
                aligned_probability[24],
                aligned_probability[72],
            )
            directions_result[direction] = {
                "source_database": source_name,
                "target_database": target_name,
                "comparison_type": comparison["type"],
                "comparison_label": comparison["label"],
                "interpretation": comparison["interpretation"],
                "transport_landmarks": transport_landmarks,
                "fixed_24_to_72_target_complete": transition,
                "protocol": {
                    "source_fit": (
                        "One XGBoost model per source database and landmark on "
                        "all source patients with a mapped outcome who reached "
                        "that landmark."
                    ),
                    "model_reuse": (
                        "The same source-landmark model is transported to both "
                        "target databases; no target information enters fitting."
                    ),
                    "target_evaluation": (
                        "All target patients with a mapped outcome who reached "
                        "the same landmark."
                    ),
                    "transition_alignment": (
                        "Exact target IDs in the complete 24/48/72 h cohort. "
                        "Weights are fixed within each independent source-"
                        "landmark model; 24 h and 72 h weights are not shared, "
                        "so temporal change also includes model refitting."
                    ),
                },
            }

            direction_recalibration: dict[str, dict] = {}
            for milestone in LANDMARKS:
                calibration_probability, calibration_y = one_row_per_patient(
                    probabilities[milestone],
                    outcomes[milestone],
                    target_groups[milestone],
                )
                (
                    curves,
                    temperature_curves,
                    method_differences,
                    replicate_rows,
                ) = run_recalibration_curves(
                    direction,
                    milestone,
                    calibration_probability,
                    calibration_y,
                    args.repeats,
                )
                calibration_replicate_rows.extend(replicate_rows)
                direction_recalibration[str(milestone)] = {
                    "target_n": int(len(outcomes[milestone])),
                    "target_unique_patient_n": int(
                        len(np.unique(target_groups[milestone]))
                    ),
                    "calibration_population_n": int(len(calibration_y)),
                    "calibration_population_unique_patient_n": int(
                        len(np.unique(target_groups[milestone]))
                    ),
                    "curves": curves,
                    "curves_temperature_scaling": temperature_curves,
                    "temperature_scaling_minus_multinomial_logistic": (
                        method_differences
                    ),
                }
                print(
                    f"  recalibration {milestone} h complete: "
                    f"{args.repeats} splits per n",
                    flush=True,
                )
            recalibration_result[direction] = direction_recalibration

            del probabilities, outcomes, target_groups, aligned_probability
            gc.collect()

        del models
        gc.collect()

    if set(directions_result) != set(DIRECTIONS):
        raise RuntimeError("Not all six directed transport pairs were completed")

    summary = {
        "analysis": (
            "six-direction fixed-weight transport and minimal target "
            "recalibration learning curves"
        ),
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "random_seed": SEED,
        "xgboost_n_jobs": N_JOBS,
        "xgboost_hyperparameters": MODEL_PARAMS,
        "outcome": {
            "column": "y3",
            "labels": CLASS_LABELS,
        },
        "database_roles": {
            database: {
                key: value
                for key, value in spec.items()
                if key in ("label", "institution", "era", "duration")
            }
            for database, spec in DATABASES.items()
        },
        "direction_taxonomy": {
            direction: COMPARISON_TYPES[tuple(direction.split("_to_"))]
            for direction in DIRECTIONS
        },
        "feasibility": {
            "feasible": True,
            "exact_common_feature_count": int(len(common_columns)),
            "exact_common_features": common_columns,
            "landmark_feature_counts": {
                str(milestone): int(len(feature_sets[milestone]))
                for milestone in LANDMARKS
            },
            "landmark_features": {
                str(milestone): feature_sets[milestone]
                for milestone in LANDMARKS
            },
            "excluded_columns": sorted(EXACT_EXCLUSIONS),
            "cohorts": cohort_diagnostics,
        },
        "directions": directions_result,
        "recalibration_protocol": {
            "target_probability_source": (
                "Fixed source-model predictions on each target patient; target "
                "outcomes are never used to fit the transported XGBoost model."
            ),
            "calibration_features": (
                "Both methods use natural logarithms of the three transported "
                "class probabilities clipped to [1e-8, 1]."
            ),
            "calibrators": {
                "multinomial_logistic": (
                    "sklearn.linear_model.LogisticRegression with multinomial "
                    "three-class targets, C=1e6, lbfgs, max_iter=5000, tol=1e-8."
                ),
                "temperature_scaling": (
                    "One positive class-independent temperature fitted by "
                    "bounded scalar minimisation of calibration-set log loss."
                ),
            },
            "calibrator_comparison": (
                "Both calibrators are fitted on the same deterministic splits. "
                "Paired differences are descriptive and were not selected by "
                "direction-specific performance."
            ),
            "sampling": (
                "One target stay is retained per patient before repeated "
                "outcome-stratified sampling without replacement, with at least "
                "one patient from each class. Independent class permutations use "
                "a SHA-256 seed derived from the global seed, direction, landmark, "
                "and repeat."
            ),
            "evaluation": (
                "Recalibrated predictions for target patients not selected into "
                "the local calibration subset; calibration and evaluation patients "
                "are disjoint."
            ),
            "n_zero": (
                "Untouched transported predictions on the same one-stay-per-patient "
                "target population; no split is used because no local labels are "
                "consumed."
            ),
            "requested_local_label_counts": list(CALIBRATION_NS),
            "repeated_splits": int(args.repeats),
            "interval": (
                "2.5th and 97.5th percentiles over deterministic repeated target "
                "splits."
            ),
        },
        "recalibration": recalibration_result,
        "interpretation_boundary": [
            (
                "MIMIC-IV and MIMIC-III are same-institution, cross-era databases; "
                "their transport directions are not independent external validation."
            ),
            "Fixed-weight transport is retrospective and not prospective deployment.",
            (
                "Target recalibration estimates data requirements for calibration, "
                "not clinical utility, process improvement, or patient benefit."
            ),
            (
                "Facility is a heterogeneous administrative destination and does "
                "not form a verified continuous severity scale with home and death."
            ),
        ],
        "artifacts": {
            "script": str(Path(__file__).resolve()),
            "summary": str(OUT / "summary.json"),
            "report": str(OUT / "report.md"),
            "learning_curve_csv": str(
                OUT / "recalibration_learning_curves.csv"
            ),
            "replicate_csv": str(OUT / "recalibration_replicates.csv"),
        },
    }
    summary = to_native(summary)

    with (OUT / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    with (OUT / "report.md").open("w", encoding="utf-8") as handle:
        handle.write(build_report(summary))

    replicate_frame = pd.DataFrame(calibration_replicate_rows)
    replicate_frame.to_csv(
        OUT / "recalibration_replicates.csv",
        index=False,
        float_format="%.8f",
    )

    curve_rows = []
    for direction in DIRECTIONS:
        comparison = directions_result[direction]
        for milestone in LANDMARKS:
            recalibration_record = recalibration_result[direction][str(milestone)]
            for method, curve_key in (
                ("multinomial_logistic", "curves"),
                ("temperature_scaling", "curves_temperature_scaling"),
            ):
                curves = recalibration_record[curve_key]
                for requested_n in CALIBRATION_NS:
                    curve = curves[str(requested_n)]
                    if not curve.get("available", True):
                        continue
                    curve_rows.append(
                        {
                            "direction": direction,
                            "source_database": comparison["source_database"],
                            "target_database": comparison["target_database"],
                            "comparison_type": comparison["comparison_type"],
                            "landmark_hours": milestone,
                            "method": method,
                            "n_local_labels": requested_n,
                            "repetitions": curve["repetitions"],
                            "evaluation_n": curve["evaluation_n"],
                            **{
                                f"calibration_{CLASS_LABELS[index]}_mean": (
                                    curve["calibration_class_counts_mean"][
                                        CLASS_LABELS[index]
                                    ]
                                )
                                for index in range(3)
                            },
                            **{
                                f"{metric}_{statistic}": curve[metric][statistic]
                                for metric in RECALIBRATION_METRICS
                                for statistic in ("mean", "p2_5", "p97_5")
                            },
                        }
                    )
    pd.DataFrame(curve_rows).to_csv(
        OUT / "recalibration_learning_curves.csv",
        index=False,
        float_format="%.8f",
    )

    print(f"Wrote {OUT / 'summary.json'}", flush=True)
    print(f"Wrote {OUT / 'report.md'}", flush=True)
    print(f"Wrote {OUT / 'recalibration_learning_curves.csv'}", flush=True)
    print(f"Wrote {OUT / 'recalibration_replicates.csv'}", flush=True)


if __name__ == "__main__":
    main()
