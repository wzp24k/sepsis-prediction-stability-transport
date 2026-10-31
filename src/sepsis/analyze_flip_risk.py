# -*- coding: utf-8 -*-
"""Model and simulate 24-hour risk of subsequent landmark instability.

The analysis consumes existing out-of-fold (OOF) discharge-destination
predictions. At 24 hours, it predicts three future prediction behaviours:
any 24-to-72-hour class flip, a major upgrade to death, and home-to-facility.
It then performs fixed-model cross-database validation and a retrospective
simulation of enhanced monitoring through the 72-hour landmark.

This is a prediction-instability analysis, not a clinical utility study.
"""

from __future__ import annotations

import json
import math
import os
import warnings
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(os.environ.get("SEPSIS_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
DATA = ROOT / "data" / "processed"
OUT = ROOT / "results" / "flip_risk"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 20261030
BOOTSTRAP_REPLICATES = 1000
THRESHOLDS = (0.05, 0.10, 0.15, 0.20, 0.30)
OUTCOMES = OrderedDict(
    (
        ("any_flip", "Any 24-to-72-hour predicted class flip"),
        ("major_death_upgrade", "Major upgrade from a non-death to death prediction"),
        ("home_to_facility", "Predicted class changes from home to facility"),
    )
)

DATABASES = OrderedDict(
    (
        (
            "mimic4",
            {
                "label": "MIMIC-IV",
                "id": "stay_id",
                "group": "subject_id",
                "cluster": "subject_id",
            },
        ),
        (
            "eicu",
            {
                "label": "eICU-CRD",
                "id": "patientunitstayid",
                "group": "uniquepid",
                "cluster": "hospitalid",
            },
        ),
        (
            "mimic3",
            {
                "label": "MIMIC-III",
                "id": "icustay_id",
                "group": "subject_id",
                "cluster": "subject_id",
            },
        ),
    )
)

VITAL_CHANNELS = ("hr", "rr", "spo2", "sbp", "dbp", "mbp", "tempf", "gcs")
VITAL_STATS = ("mean", "min", "max", "last", "count")
LAB_FEATURES = (
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
)
CLINICAL_24H_FEATURES = (
    ("age", "male")
    + tuple(
        f"{channel}_w1_{statistic}"
        for channel in VITAL_CHANNELS
        for statistic in VITAL_STATS
    )
    + LAB_FEATURES
)
ADMISSION_FEATURES = (
    "admission_emergency",
    "admission_elective_or_surgical",
    "admission_transfer",
    "admission_other_or_unknown",
)
PREDICTION_FEATURES = (
    "p24_home",
    "p24_facility",
    "p24_death",
    "p24_max_probability",
    "p24_entropy",
    "p24_margin",
    "predicted_24h_home",
    "predicted_24h_facility",
    "predicted_24h_death",
)
FEATURE_COLUMNS = PREDICTION_FEATURES + CLINICAL_24H_FEATURES + ADMISSION_FEATURES


def prediction_path(database: str) -> Path:
    """Use a shared MIMIC-III run if one becomes available; otherwise use v2."""
    if database == "mimic4":
        return ROOT / "results" / "shared_fold" / "mimic4_shared_fold.npz"
    if database == "eicu":
        return ROOT / "results" / "shared_fold" / "eicu_shared_fold.npz"
    shared = ROOT / "results" / "mimic3_shared" / "mimic3_shared.npz"
    if shared.exists():
        return shared
    return ROOT / "results" / "sensitivity_v2" / "mimic3" / "mimic3_xgb.npz"


def any_indicator(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    available = [column for column in columns if column in frame.columns]
    if not available:
        return np.zeros(len(frame), dtype=np.float64)
    return frame[available].fillna(False).astype(bool).any(axis=1).to_numpy(
        dtype=np.float64
    )


def harmonized_admission(frame: pd.DataFrame, database: str) -> pd.DataFrame:
    """Map heterogeneous admission labels to four prespecified coarse groups."""
    if database == "mimic4":
        emergency = any_indicator(frame, ["adm_EW EMER.", "adm_DIRECT EMER."])
        elective = any_indicator(
            frame, ["adm_ELECTIVE", "adm_SURGICAL SAME DAY ADMISSION"]
        )
        transfer = np.zeros(len(frame), dtype=np.float64)
    elif database == "eicu":
        emergency = any_indicator(
            frame, ["src_Emergency Department", "src_Chest Pain Center"]
        )
        elective = any_indicator(
            frame, ["src_Operating Room", "src_PACU", "src_Recovery Room"]
        )
        transfer = any_indicator(frame, ["src_Other Hospital", "src_Other ICU"])
    elif database == "mimic3":
        emergency = any_indicator(frame, ["adm_EMERGENCY"])
        elective = any_indicator(frame, ["adm_ELECTIVE"])
        transfer = np.zeros(len(frame), dtype=np.float64)
    else:
        raise ValueError(f"Unknown database: {database}")

    used = np.clip(emergency + elective + transfer, 0.0, 1.0)
    return pd.DataFrame(
        {
            "admission_emergency": emergency,
            "admission_elective_or_surgical": elective,
            "admission_transfer": transfer,
            "admission_other_or_unknown": 1.0 - used,
        },
        index=frame.index,
    )


def prediction_shape(p24: np.ndarray) -> pd.DataFrame:
    probability = np.clip(np.asarray(p24, dtype=np.float64), 1e-9, 1.0)
    probability = probability / probability.sum(axis=1, keepdims=True)
    ordered = np.sort(probability, axis=1)
    predicted_class = probability.argmax(axis=1)
    entropy = -(probability * np.log(probability)).sum(axis=1)
    frame = pd.DataFrame(
        {
            "p24_home": probability[:, 0],
            "p24_facility": probability[:, 1],
            "p24_death": probability[:, 2],
            "p24_max_probability": probability.max(axis=1),
            "p24_entropy": entropy,
            "p24_margin": ordered[:, -1] - ordered[:, -2],
            "predicted_24h_home": (predicted_class == 0).astype(float),
            "predicted_24h_facility": (predicted_class == 1).astype(float),
            "predicted_24h_death": (predicted_class == 2).astype(float),
        }
    )
    return frame


def get_outcome_label(z: np.lib.npyio.NpzFile) -> np.ndarray:
    for key in ("y72", "y3"):
        if key in z.files:
            return z[key].astype(np.int64)
    raise KeyError("Prediction archive has neither y72 nor y3")


def load_database(database: str) -> dict[str, Any]:
    spec = DATABASES[database]
    path = prediction_path(database)
    if not path.exists():
        raise FileNotFoundError(path)
    z = np.load(path, allow_pickle=True)
    ids = z["ids"].astype(np.int64)
    if len(np.unique(ids)) != len(ids):
        raise ValueError(f"{database}: duplicate prediction IDs")

    design = pd.read_parquet(DATA / database / "design_all.parquet")
    id_column = spec["id"]
    if id_column not in design.columns:
        raise KeyError(f"{database}: missing ID column {id_column}")
    if design[id_column].duplicated().any():
        raise ValueError(f"{database}: duplicate design IDs")
    indexed = design.set_index(id_column)
    missing = np.setdiff1d(ids, indexed.index.to_numpy())
    if len(missing):
        raise ValueError(f"{database}: {len(missing)} prediction IDs absent from design")
    aligned = indexed.loc[ids].reset_index()

    if "groups" in z.files:
        groups = z["groups"].astype(str)
    else:
        groups = aligned[spec["group"]].astype(str).to_numpy()
    if len(groups) != len(ids):
        raise ValueError(f"{database}: patient groups do not align with predictions")
    if "clusters" in z.files:
        clusters = z["clusters"].astype(str)
    elif spec["cluster"] in aligned:
        clusters = aligned[spec["cluster"]].astype(str).to_numpy()
    else:
        clusters = groups.copy()
    if len(clusters) != len(ids):
        raise ValueError(f"{database}: cluster identifiers do not align")

    if "fold_id" in z.files:
        folds = z["fold_id"].astype(np.int64)
        fold_source = "existing patient-grouped prediction fold_id"
    else:
        folds = np.full(len(ids), -1, dtype=np.int64)
        splitter = GroupKFold(n_splits=5)
        placeholder = np.zeros(len(ids), dtype=np.int64)
        for fold, (_, test_index) in enumerate(
            splitter.split(aligned, placeholder, groups=groups)
        ):
            folds[test_index] = fold
        fold_source = "patient-grouped GroupKFold reconstructed for this analysis"

    if (folds < 0).any():
        raise ValueError(f"{database}: incomplete fold assignment")
    for fold in np.unique(folds):
        held_groups = set(groups[folds == fold])
        other_groups = set(groups[folds != fold])
        if held_groups.intersection(other_groups):
            raise ValueError(f"{database}: patient group crosses folds")

    p24 = np.asarray(z["p24"], dtype=np.float64)
    p72 = np.asarray(z["p72"], dtype=np.float64)
    if p24.shape != (len(ids), 3) or p72.shape != (len(ids), 3):
        raise ValueError(f"{database}: unexpected probability shape")
    predicted24 = p24.argmax(axis=1)
    predicted72 = p72.argmax(axis=1)

    feature_frame = pd.concat(
        [
            prediction_shape(p24).reset_index(drop=True),
            aligned.loc[:, CLINICAL_24H_FEATURES].reset_index(drop=True),
            harmonized_admission(aligned, database).reset_index(drop=True),
        ],
        axis=1,
    )
    missing_features = [column for column in FEATURE_COLUMNS if column not in feature_frame]
    if missing_features:
        raise ValueError(f"{database}: missing features: {missing_features}")
    feature_frame = feature_frame.loc[:, FEATURE_COLUMNS].astype(np.float64)

    events = pd.DataFrame(
        {
            "any_flip": (predicted24 != predicted72).astype(np.int64),
            "major_death_upgrade": (
                (predicted24 != 2) & (predicted72 == 2)
            ).astype(np.int64),
            "home_to_facility": (
                (predicted24 == 0) & (predicted72 == 1)
            ).astype(np.int64),
        }
    )

    return {
        "database": database,
        "label": spec["label"],
        "prediction_path": str(path.relative_to(ROOT)),
        "ids": ids,
        "native_label": get_outcome_label(z),
        "groups": groups,
        "clusters": clusters,
        "folds": folds,
        "fold_source": fold_source,
        "predicted24": predicted24,
        "predicted72": predicted72,
        "features": feature_frame,
        "events": events,
    }


def make_model() -> Pipeline:
    """Fixed, prespecified L2-regularized binary risk model."""
    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                    keep_empty_features=True,
                ),
            ),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    penalty="l2",
                    C=0.2,
                    solver="lbfgs",
                    max_iter=3000,
                    random_state=SEED,
                ),
            ),
        ]
    )


def fit_oof(
    features: pd.DataFrame, outcome: np.ndarray, folds: np.ndarray
) -> np.ndarray:
    probability = np.full(len(features), np.nan, dtype=np.float64)
    for fold in sorted(np.unique(folds)):
        train = folds != fold
        test = folds == fold
        if outcome[train].min() == outcome[train].max():
            raise ValueError(f"fold {fold}: training outcome has one class")
        model = make_model()
        model.fit(features.loc[train], outcome[train])
        probability[test] = model.predict_proba(features.loc[test])[:, 1]
    if np.isnan(probability).any():
        raise ValueError("OOF predictions contain missing values")
    return np.clip(probability, 1e-6, 1.0 - 1e-6)


def fit_transport(
    source_features: pd.DataFrame,
    source_outcome: np.ndarray,
    target_features: pd.DataFrame,
) -> np.ndarray:
    model = make_model()
    model.fit(source_features, source_outcome)
    return np.clip(model.predict_proba(target_features)[:, 1], 1e-6, 1.0 - 1e-6)


def expected_calibration_error(
    outcome: np.ndarray, probability: np.ndarray, bins: int = 10
) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        selected = (
            (probability >= lower) & (probability <= upper)
            if upper == 1.0
            else (probability >= lower) & (probability < upper)
        )
        if selected.any():
            value += selected.mean() * abs(
                outcome[selected].mean() - probability[selected].mean()
            )
    return float(value)


def calibration_parameters(
    outcome: np.ndarray, probability: np.ndarray
) -> tuple[float | None, float | None]:
    """Fit y ~ intercept + slope * logit(p), using a minimal ridge fallback."""
    logit = np.log(np.clip(probability, 1e-6, 1 - 1e-6) / (
        1 - np.clip(probability, 1e-6, 1 - 1e-6)
    ))
    design = sm.add_constant(logit, has_constant="add")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = sm.Logit(outcome, design).fit(disp=False, maxiter=500)
        return float(result.params[1]), float(result.params[0])
    except Exception:
        try:
            result = sm.GLM(outcome, design, family=sm.families.Binomial()).fit_regularized(
                alpha=1e-6, L1_wt=0.0, maxiter=500
            )
            return float(result.params[1]), float(result.params[0])
        except Exception:
            return None, None


def metric_block(outcome: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    outcome = np.asarray(outcome, dtype=np.int64)
    probability = np.asarray(probability, dtype=np.float64)
    slope, intercept = calibration_parameters(outcome, probability)
    auc = (
        float(roc_auc_score(outcome, probability))
        if len(np.unique(outcome)) == 2
        else None
    )
    return {
        "n": int(len(outcome)),
        "events": int(outcome.sum()),
        "event_rate": round(float(outcome.mean()), 6),
        "mean_predicted_risk": round(float(probability.mean()), 6),
        "auc": round(auc, 6) if auc is not None else None,
        "auprc": (
            round(float(average_precision_score(outcome, probability)), 6)
            if len(np.unique(outcome)) == 2
            else None
        ),
        "brier": round(float(np.mean((probability - outcome) ** 2)), 6),
        "calibration_slope": round(slope, 6) if slope is not None else None,
        "calibration_intercept": (
            round(intercept, 6) if intercept is not None else None
        ),
        "ece_10bin": round(
            expected_calibration_error(outcome, probability), 6
        ),
    }


def cluster_bootstrap_metric_intervals(
    outcome: np.ndarray,
    probability: np.ndarray,
    clusters: np.ndarray,
    seed: int,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, list[float]]:
    """Cluster-bootstrap discrimination and probability-score intervals."""
    unique, inverse = np.unique(np.asarray(clusters).astype(str), return_inverse=True)
    members = [np.flatnonzero(inverse == index) for index in range(len(unique))]
    rng = np.random.default_rng(seed)
    values = {
        "auc": [],
        "auprc": [],
        "brier": [],
        "ece_10bin": [],
    }
    generated = 0
    batch = 100
    while generated < replicates:
        size = min(batch, replicates - generated)
        selected = rng.integers(0, len(unique), size=(size, len(unique)))
        for row in selected:
            index = np.concatenate([members[cluster] for cluster in row])
            sample_y = outcome[index]
            sample_p = probability[index]
            if len(np.unique(sample_y)) == 2:
                values["auc"].append(float(roc_auc_score(sample_y, sample_p)))
                values["auprc"].append(
                    float(average_precision_score(sample_y, sample_p))
                )
            values["brier"].append(
                float(np.mean((sample_p - sample_y) ** 2))
            )
            values["ece_10bin"].append(
                expected_calibration_error(sample_y, sample_p)
            )
        generated += size
    return {
        key: [
            round(float(np.quantile(np.asarray(metric_values), 0.025)), 6),
            round(float(np.quantile(np.asarray(metric_values), 0.975)), 6),
        ]
        for key, metric_values in values.items()
        if metric_values
    }


def simulation_block(
    instability_risk: np.ndarray,
    major_death_upgrade: np.ndarray,
    any_flip: np.ndarray,
) -> dict[str, Any]:
    """Simulate enhanced monitoring through 72 hours after 24-hour selection."""
    instability_risk = np.asarray(instability_risk, dtype=np.float64)
    major_death_upgrade = np.asarray(major_death_upgrade, dtype=np.int64)
    any_flip = np.asarray(any_flip, dtype=np.int64)
    n = len(instability_risk)
    major_total = int(major_death_upgrade.sum())
    flip_total = int(any_flip.sum())

    def block(selected: np.ndarray) -> dict[str, Any]:
        reassessments = int(selected.sum())
        major_captured = int(major_death_upgrade[selected].sum())
        flip_captured = int(any_flip[selected].sum())
        false_triggers = reassessments - major_captured
        non_upgrades = n - major_total
        return {
            "n": n,
            "reassessments": reassessments,
            "trigger_coverage": round(float(selected.mean()), 6),
            "reassessments_reduced_vs_all": round(
                float(1.0 - selected.mean()), 6
            ),
            "any_flip_total": flip_total,
            "any_flip_captured": flip_captured,
            "any_flip_missed": int(flip_total - flip_captured),
            "any_flip_capture_fraction": (
                round(float(flip_captured / flip_total), 6)
                if flip_total
                else None
            ),
            "major_death_upgrades_total": major_total,
            "major_death_upgrades_captured": major_captured,
            "major_death_upgrades_missed": int(major_total - major_captured),
            "major_death_upgrade_capture_fraction": (
                round(float(major_captured / major_total), 6)
                if major_total
                else None
            ),
            "major_death_upgrades_per_100_reassessments": (
                round(float(100.0 * major_captured / reassessments), 6)
                if reassessments
                else None
            ),
            "major_death_upgrade_miss_fraction": (
                round(float(1.0 - major_captured / major_total), 6)
                if major_total
                else None
            ),
            "major_death_upgrade_capture_enrichment": (
                round(
                    float(
                        (major_captured / major_total)
                        / (reassessments / n)
                    ),
                    6,
                )
                if major_total and reassessments
                else None
            ),
            "false_trigger_fraction_among_triggered": (
                round(float(false_triggers / reassessments), 6)
                if reassessments
                else None
            ),
            "reassessment_without_major_death_upgrade_fraction": (
                round(float(false_triggers / reassessments), 6)
                if reassessments
                else None
            ),
            "false_positive_rate_among_non_upgrades": (
                round(float(false_triggers / non_upgrades), 6)
                if non_upgrades and reassessments
                else None
            ),
        }

    none_selected = np.zeros(n, dtype=bool)
    all_selected = np.ones(n, dtype=bool)
    rows = []
    for threshold in THRESHOLDS:
        row = {
            "threshold": threshold,
            **block(instability_risk >= threshold),
        }
        rows.append(row)
    return {
        "no_reassessment": block(none_selected),
        "reassess_all": block(all_selected),
        "risk_thresholds": rows,
    }


def simulation_bootstrap_intervals(
    instability_risk: np.ndarray,
    major_death_upgrade: np.ndarray,
    clusters: np.ndarray,
    seed: int,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[float, dict[str, list[float]]]:
    """Cluster-bootstrap coverage, capture, workload, and enrichment."""
    unique, inverse = np.unique(np.asarray(clusters).astype(str), return_inverse=True)
    members = [np.flatnonzero(inverse == index) for index in range(len(unique))]
    values: dict[float, dict[str, list[float]]] = {
        threshold: {
            "trigger_coverage": [],
            "reassessments_reduced_vs_all": [],
            "major_death_upgrade_capture_fraction": [],
            "major_death_upgrade_capture_enrichment": [],
            "reassessment_without_major_death_upgrade_fraction": [],
        }
        for threshold in THRESHOLDS
    }
    rng = np.random.default_rng(seed)
    for _ in range(replicates):
        selected_clusters = rng.integers(
            0, len(unique), size=len(unique)
        )
        index = np.concatenate(
            [members[cluster] for cluster in selected_clusters]
        )
        risk = instability_risk[index]
        event = major_death_upgrade[index]
        n = len(index)
        event_total = int(event.sum())
        for threshold in THRESHOLDS:
            selected = risk >= threshold
            reassessments = int(selected.sum())
            captured = int(event[selected].sum())
            values[threshold]["trigger_coverage"].append(
                reassessments / n
            )
            values[threshold]["reassessments_reduced_vs_all"].append(
                1.0 - reassessments / n
            )
            if event_total and reassessments:
                capture = captured / event_total
                coverage = reassessments / n
                values[threshold][
                    "major_death_upgrade_capture_fraction"
                ].append(capture)
                values[threshold][
                    "major_death_upgrade_capture_enrichment"
                ].append(capture / coverage)
                values[threshold][
                    "reassessment_without_major_death_upgrade_fraction"
                ].append((reassessments - captured) / reassessments)
    output = {}
    for threshold, metrics in values.items():
        output[threshold] = {
            key: [
                round(float(np.quantile(metric_values, 0.025)), 6),
                round(float(np.quantile(metric_values, 0.975)), 6),
            ]
            for key, metric_values in metrics.items()
            if metric_values
        }
    return output


def attach_simulation_intervals(
    block: dict[str, Any],
    intervals: dict[float, dict[str, list[float]]],
) -> dict[str, Any]:
    for row in block["risk_thresholds"]:
        row["ci95"] = intervals.get(row["threshold"], {})
    return block


def top_coefficients(
    features: pd.DataFrame, outcome: np.ndarray, limit: int = 8
) -> list[dict[str, Any]]:
    """Descriptive full-cohort fit used only for model transparency."""
    model = make_model()
    model.fit(features, outcome)
    names = model.named_steps["imputer"].get_feature_names_out(FEATURE_COLUMNS)
    coefficients = model.named_steps["model"].coef_[0]
    order = np.argsort(np.abs(coefficients))[::-1][:limit]
    return [
        {
            "feature": str(names[index]),
            "standardized_log_odds": round(float(coefficients[index]), 6),
            "odds_ratio": round(float(math.exp(coefficients[index])), 6),
        }
        for index in order
    ]


def percent(value: float | None) -> str:
    return "NA" if value is None else f"{100.0 * value:.1f}%"


def decimal(value: float | None) -> str:
    return "NA" if value is None else f"{value:.3f}"


def metric_table(rows: list[dict[str, Any]], group_label: str) -> str:
    header = (
        f"| {group_label} | Outcome | n | Events | AUC | Brier | "
        "Calibration slope | Intercept | ECE |"
    )
    separator = "|---|---|---:|---:|---:|---:|---:|---:|---:|"
    lines = [header, separator]
    for row in rows:
        lines.append(
            "| {group} | {outcome} | {n} | {events} | {auc} | {brier} | "
            "{slope} | {intercept} | {ece} |".format(
                group=row["group"],
                outcome=OUTCOMES[row["outcome"]],
                n=row["metrics"]["n"],
                events=row["metrics"]["events"],
                auc=decimal(row["metrics"]["auc"]),
                brier=decimal(row["metrics"]["brier"]),
                slope=decimal(row["metrics"]["calibration_slope"]),
                intercept=decimal(row["metrics"]["calibration_intercept"]),
                ece=decimal(row["metrics"]["ece_10bin"]),
            )
        )
    return "\n".join(lines)


def simulation_metric_table(
    rows: list[tuple[str, dict[str, Any]]],
) -> str:
    lines = [
        "| Analysis | Threshold | Triggered | Coverage | Any flip captured | "
        "Death upgrades captured | Missed | Captured/100 monitored | "
        "False-trigger fraction | Reduced vs reassess-all |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, row in rows:
        lines.append(
            "| {label} | {threshold} | {triggered} | {coverage} | {flip} | "
            "{death} | {missed} | {yield_} | {false} | {reduction} |".format(
                label=label,
                threshold=percent(row["threshold"]),
                triggered=row["reassessments"],
                coverage=percent(row["trigger_coverage"]),
                flip=(
                    f"{row['any_flip_captured']}/{row['any_flip_total']} "
                    f"({percent(row['any_flip_capture_fraction'])})"
                ),
                death=(
                    f"{row['major_death_upgrades_captured']}/"
                    f"{row['major_death_upgrades_total']} "
                    f"({percent(row['major_death_upgrade_capture_fraction'])})"
                ),
                missed=row["major_death_upgrades_missed"],
                yield_=decimal(
                    row["major_death_upgrades_per_100_reassessments"]
                ),
                false=percent(
                    row["false_trigger_fraction_among_triggered"]
                ),
                reduction=percent(row["reassessments_reduced_vs_all"]),
            )
        )
    return "\n".join(lines)


def build_report(
    cohort_rows: list[dict[str, Any]],
    oof_rows: list[dict[str, Any]],
    transport_rows: list[dict[str, Any]],
    simulations: dict[str, Any],
    coefficients: dict[str, Any],
    data_sources: dict[str, str],
) -> str:
    within_simulation_rows = []
    for database, result in simulations["instability_risk_trigger"][
        "within_database"
    ].items():
        for row in result["risk_thresholds"]:
            within_simulation_rows.append(
                (DATABASES[database]["label"], row)
            )
    transport_simulation_rows = []
    for direction, result in simulations["instability_risk_trigger"][
        "fixed_model_transport"
    ].items():
        source, target = direction.split("__")
        label = f"{DATABASES[source]['label']} -> {DATABASES[target]['label']}"
        for row in result["risk_thresholds"]:
            transport_simulation_rows.append((label, row))

    clinical_simulation_rows = []
    for database, result in simulations["death_upgrade_risk_trigger"][
        "within_database"
    ].items():
        for row in result["risk_thresholds"]:
            clinical_simulation_rows.append((DATABASES[database]["label"], row))

    oof_auc = {
        (row["database"], row["outcome"]): row["metrics"]["auc"]
        for row in oof_rows
    }
    transport_auc = {
        (row["direction"], row["outcome"]): row["metrics"]["auc"]
        for row in transport_rows
    }

    def auc_range(outcome: str, source: dict[tuple[str, str], float | None]) -> str:
        values = [
            value
            for (key, event), value in source.items()
            if event == outcome and value is not None
        ]
        if not values:
            return "NA"
        return f"{min(values):.3f}-{max(values):.3f}"

    baseline_lines = [
        "| Database | Rule | Reassessed | Coverage | Any flip captured | "
        "Death upgrades captured | Missed | Reduced vs reassess-all |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for database, result in simulations["instability_risk_trigger"][
        "within_database"
    ].items():
        for rule in ("no_reassessment", "reassess_all"):
            row = result[rule]
            baseline_lines.append(
                f"| {DATABASES[database]['label']} | {rule.replace('_', ' ')} | "
                f"{row['reassessments']} | {percent(row['trigger_coverage'])} | "
                f"{row['any_flip_captured']}/{row['any_flip_total']} | "
                f"{row['major_death_upgrades_captured']}/"
                f"{row['major_death_upgrades_total']} | "
                f"{row['major_death_upgrades_missed']} | "
                f"{percent(row['reassessments_reduced_vs_all'])} |"
            )

    cohort_lines = [
        "| Database | Analysis rows | Patient groups | Event: any flip | "
        "Event: death upgrade | Event: home-to-facility | Fold source |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in cohort_rows:
        cohort_lines.append(
            f"| {row['label']} | {row['n']} | {row['groups']} | "
            f"{row['event_counts']['any_flip']} "
            f"({percent(row['event_rates']['any_flip'])}) | "
            f"{row['event_counts']['major_death_upgrade']} "
            f"({percent(row['event_rates']['major_death_upgrade'])}) | "
            f"{row['event_counts']['home_to_facility']} "
            f"({percent(row['event_rates']['home_to_facility'])}) | "
            f"{row['fold_source']} |"
        )

    coefficient_lines = [
        "| Database | Outcome | Strongest standardized predictors |",
        "|---|---|---|",
    ]
    for database in DATABASES:
        for outcome in OUTCOMES:
            terms = coefficients[database][outcome][:4]
            formatted = "; ".join(
                f"{term['feature']} (OR {term['odds_ratio']:.2f})"
                for term in terms
            )
            coefficient_lines.append(
                f"| {DATABASES[database]['label']} | {OUTCOMES[outcome]} | "
                f"{formatted} |"
            )

    source_lines = "\n".join(
        f"- {DATABASES[database]['label']}: `{path}`"
        for database, path in data_sources.items()
    )

    return f"""# 24-Hour Risk of Subsequent Prediction Instability

Generated: {datetime.now(timezone.utc).isoformat()}

## Scope

This analysis asks whether information available at 24 hours can identify
patients whose discharge-destination prediction will later become unstable.
It uses existing patient-grouped out-of-fold predictions at 24 and 72 hours.
The enhanced-monitoring analysis is a retrospective decision simulation.
It does not establish clinical utility, workflow benefit, or improved patient
outcomes.

## Data sources

{source_lines}

The analysis cohort is the complete three-landmark subset in each prediction
archive. MIMIC-III is a same-institution different-era database, not an
independent external institution.

## Outcomes and prediction time

All predictors are available at 24 hours:

- Existing OOF probabilities at 24 hours: home, facility, death, maximum
  probability, entropy, and top-two margin.
- Predicted 24-hour class represented as one-hot indicators.
- Age, sex, first-24-hour vital-sign summaries, first-24-hour laboratory
  summaries, and four harmonized admission-source/type indicators.

The three binary outcomes are defined from the 24- and 72-hour OOF predicted
classes:

- Any flip: the argmax class is different at 72 hours.
- Major death upgrade: a non-death prediction at 24 hours becomes death at
  72 hours.
- Home-to-facility: home at 24 hours changes to facility at 72 hours.

No 48- or 72-hour covariate, probability, or prediction is used as a feature.

## Cohort and event counts

{chr(10).join(cohort_lines)}

## Model and validation

Each database/outcome combination uses an L2-regularized logistic regression
with median imputation, missing-value indicators, and standardization.
`C=0.2` was prespecified. When an existing patient-grouped fold identifier is
available, it is reused for all three outcomes; otherwise one patient-grouped
five-fold partition is reconstructed from the patient ID and reused across
outcomes. Fixed-model transport trains on all source rows and applies the
unchanged model to each target database.

Admission labels were coarsened because source definitions differ:
emergency, elective/surgical, interhospital/ICU transfer, and
other/unknown. This harmonization is approximate and does not create a common
clinical ontology.

### Within-database out-of-fold performance

{metric_table(oof_rows, "Database")}

### Cross-database fixed-model transport

{metric_table(transport_rows, "Direction")}

### Descriptive model coefficients

These coefficients come from full-database fits used only to describe the
models. They are not independent validation results.

{chr(10).join(coefficient_lines)}

## Enhanced-monitoring simulation

At 24 hours, patients at or above a predicted instability-risk threshold enter
enhanced monitoring through 72 hours. Capture is evaluated against the
24-to-72-hour prediction labels. "False-trigger fraction" is the proportion of
monitored patients who did not have a predicted-class upgrade to death.
"Reduction" compares the number monitored with the monitor-all strategy.

### Baseline strategies

{chr(10).join(baseline_lines)}

### Instability-risk trigger: within-database OOF

{simulation_metric_table(
    within_simulation_rows,
)}

### Instability-risk trigger: fixed-model transport

{simulation_metric_table(
    transport_simulation_rows,
)}

### Secondary clinical trigger: predicted death-upgrade risk, within database

This secondary rule directly thresholds the predicted risk of a major death
upgrade. It is reported to show the trade-off between targeting the clinical
event and targeting the broader instability event.

{simulation_metric_table(
    clinical_simulation_rows,
)}

## Summary

- Within-database OOF AUC ranges were {auc_range("any_flip", oof_auc)} for
  any flip, {auc_range("major_death_upgrade", oof_auc)} for major death
  upgrade, and {auc_range("home_to_facility", oof_auc)} for
  home-to-facility.
- Fixed-model transport AUC ranges across all six directions were
  {auc_range("any_flip", transport_auc)} for any flip,
  {auc_range("major_death_upgrade", transport_auc)} for major death upgrade,
  and {auc_range("home_to_facility", transport_auc)} for home-to-facility.
- Transport performance and calibration are direction-dependent. Pooling
  directions or selecting only favourable directions would obscure this.
- A threshold can reduce monitoring workload relative to monitor-all, but the
  missed upgrades to death and false-trigger fraction must be interpreted
  together. No threshold establishes clinical usefulness.

## Interpretation boundary

The target labels are themselves later OOF predictions, so this analysis
models instability of the prediction system rather than verified clinical
deterioration. The cohort must remain observable through 72 hours. The
simulation assumes that enhanced monitoring through 72 hours is feasible and
informative; it does not test whether monitoring changes management.
Calibration slope and intercept use logistic recalibration of the fixed
predictions, not refitting of the clinical risk model. Admission-source
harmonization is approximate. These results should be used to prioritize
prospective silent or pilot evaluation, not to support deployment or a claim
of improved care.
"""


def main() -> None:
    cohorts = OrderedDict(
        (database, load_database(database)) for database in DATABASES
    )

    prohibited_tokens = ("p48", "p72", "_w2_", "_w3_")
    if any(
        token in feature
        for feature in FEATURE_COLUMNS
        for token in prohibited_tokens
    ):
        raise AssertionError("A post-24-hour feature entered the risk model")

    cohort_rows: list[dict[str, Any]] = []
    for database, cohort in cohorts.items():
        cohort_rows.append(
            {
                "database": database,
                "label": cohort["label"],
                "n": int(len(cohort["ids"])),
                "groups": int(len(np.unique(cohort["groups"]))),
                "event_counts": {
                    outcome: int(cohort["events"][outcome].sum())
                    for outcome in OUTCOMES
                },
                "event_rates": {
                    outcome: round(
                        float(cohort["events"][outcome].mean()), 6
                    )
                    for outcome in OUTCOMES
                },
                "fold_source": cohort["fold_source"],
            }
        )

    oof: dict[str, dict[str, np.ndarray]] = {}
    oof_rows: list[dict[str, Any]] = []
    coefficients: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for database_index, (database, cohort) in enumerate(cohorts.items()):
        oof[database] = {}
        coefficients[database] = {}
        for outcome_index, outcome in enumerate(OUTCOMES):
            event = cohort["events"][outcome].to_numpy(dtype=np.int64)
            probability = fit_oof(
                cohort["features"], event, cohort["folds"]
            )
            oof[database][outcome] = probability
            metrics = metric_block(event, probability)
            metrics_ci95 = cluster_bootstrap_metric_intervals(
                event,
                probability,
                cohort["clusters"],
                seed=SEED + 1000 * database_index + 10 * outcome_index,
            )
            oof_rows.append(
                {
                    "database": database,
                    "group": cohort["label"],
                    "outcome": outcome,
                    "metrics": metrics,
                    "metrics_ci95": metrics_ci95,
                }
            )
            coefficients[database][outcome] = top_coefficients(
                cohort["features"], event
            )
            print(
                f"OOF {database:7s} {outcome:22s} "
                f"AUC={metrics['auc']} Brier={metrics['brier']}",
                flush=True,
            )

    transport: dict[str, dict[str, np.ndarray]] = {}
    transport_rows: list[dict[str, Any]] = []
    for source in DATABASES:
        for target in DATABASES:
            if source == target:
                continue
            direction = f"{source}__{target}"
            transport[direction] = {}
            for outcome in OUTCOMES:
                source_event = cohorts[source]["events"][outcome].to_numpy(
                    dtype=np.int64
                )
                target_event = cohorts[target]["events"][outcome].to_numpy(
                    dtype=np.int64
                )
                probability = fit_transport(
                    cohorts[source]["features"],
                    source_event,
                    cohorts[target]["features"],
                )
                transport[direction][outcome] = probability
                metrics = metric_block(target_event, probability)
                transport_rows.append(
                    {
                        "direction": direction,
                        "group": (
                            f"{cohorts[source]['label']} -> "
                            f"{cohorts[target]['label']}"
                        ),
                        "source": source,
                        "target": target,
                        "outcome": outcome,
                        "metrics": metrics,
                    }
                )
                print(
                    f"TRANSPORT {direction:15s} {outcome:22s} "
                    f"AUC={metrics['auc']} Brier={metrics['brier']}",
                    flush=True,
                )

    simulations: dict[str, Any] = {
        "instability_risk_trigger": {
            "within_database": {},
            "fixed_model_transport": {},
        },
        "death_upgrade_risk_trigger": {
            "within_database": {},
            "fixed_model_transport": {},
        },
    }
    for database_index, (database, cohort) in enumerate(cohorts.items()):
        major = cohort["events"]["major_death_upgrade"].to_numpy(dtype=np.int64)
        flip = cohort["events"]["any_flip"].to_numpy(dtype=np.int64)
        instability_block = simulation_block(
            oof[database]["any_flip"], major, flip
        )
        death_upgrade_block = simulation_block(
            oof[database]["major_death_upgrade"], major, flip
        )
        simulations["instability_risk_trigger"]["within_database"][
            database
        ] = attach_simulation_intervals(
            instability_block,
            simulation_bootstrap_intervals(
                oof[database]["any_flip"],
                major,
                cohort["clusters"],
                seed=SEED + 3000 + 10 * database_index,
            ),
        )
        simulations["death_upgrade_risk_trigger"]["within_database"][
            database
        ] = attach_simulation_intervals(
            death_upgrade_block,
            simulation_bootstrap_intervals(
                oof[database]["major_death_upgrade"],
                major,
                cohort["clusters"],
                seed=SEED + 3100 + 10 * database_index,
            ),
        )
    for direction_index, (direction, prediction) in enumerate(transport.items()):
        source, target = direction.split("__")
        del source
        major = cohorts[target]["events"]["major_death_upgrade"].to_numpy(
            dtype=np.int64
        )
        flip = cohorts[target]["events"]["any_flip"].to_numpy(dtype=np.int64)
        instability_block = simulation_block(
            prediction["any_flip"], major, flip
        )
        death_upgrade_block = simulation_block(
            prediction["major_death_upgrade"], major, flip
        )
        simulations["instability_risk_trigger"]["fixed_model_transport"][
            direction
        ] = attach_simulation_intervals(
            instability_block,
            simulation_bootstrap_intervals(
                prediction["any_flip"],
                major,
                cohorts[target]["clusters"],
                seed=SEED + 4000 + 10 * direction_index,
            ),
        )
        simulations["death_upgrade_risk_trigger"]["fixed_model_transport"][
            direction
        ] = attach_simulation_intervals(
            death_upgrade_block,
            simulation_bootstrap_intervals(
                prediction["major_death_upgrade"],
                major,
                cohorts[target]["clusters"],
                seed=SEED + 4100 + 10 * direction_index,
            ),
        )

    source_map = {
        database: cohorts[database]["prediction_path"] for database in DATABASES
    }
    validations = {
        "all_prediction_ids_merged_to_design": True,
        "only_24h_features_used": True,
        "patient_groups_do_not_cross_cv_folds": True,
        "same_cv_partition_used_for_all_three_outcomes": True,
        "six_fixed_model_transport_directions_completed": len(transport) == 6,
        "all_three_outcomes_modeled_in_each_database": all(
            set(oof[database]) == set(OUTCOMES) for database in DATABASES
        ),
        "mimic3_note": (
            "MIMIC-III is same-institution different-era; it is not an "
            "independent external institution."
        ),
        "simulation_note": (
            "Retrospective decision simulation; no clinical utility is claimed."
        ),
    }
    if not all(
        value
        for key, value in validations.items()
        if key not in ("mimic3_note", "simulation_note")
    ):
        raise AssertionError("Analysis validation failed")

    summary = {
        "schema_version": "1.0",
        "module": "flip_risk_and_enhanced_monitoring_simulation",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": SEED,
        "analysis_role": (
            "Retrospective prediction-instability risk modeling and "
            "cross-database fixed-model validation; not clinical utility."
        ),
        "data_sources": source_map,
        "labels": {
            "any_flip": (
                "argmax(p24) != argmax(p72); p72 is used only to define "
                "the outcome"
            ),
            "major_death_upgrade": (
                "argmax(p24) != death and argmax(p72) == death"
            ),
            "home_to_facility": (
                "argmax(p24) == home and argmax(p72) == facility"
            ),
            "class_order": ["home", "facility", "death"],
        },
        "features": {
            "columns": list(FEATURE_COLUMNS),
            "prediction_information": list(PREDICTION_FEATURES),
            "clinical_24h_information": list(CLINICAL_24H_FEATURES),
            "admission_harmonization": list(ADMISSION_FEATURES),
            "no_post_24h_features": True,
            "admission_harmonization_note": (
                "Approximate semantic mapping across databases; not a shared "
                "clinical ontology."
            ),
        },
        "model": {
            "estimator": "L2-regularized binary logistic regression",
            "C": 0.2,
            "imputation": (
                "training-fold median imputation with missing indicators"
            ),
            "standardization": "training-fold StandardScaler",
            "solver": "lbfgs",
            "max_iter": 3000,
            "class_weight": None,
            "cv_unit": "patient",
            "transport": (
                "Train on all source rows; apply unchanged pipeline to target"
            ),
        },
        "cohorts": cohort_rows,
        "within_database_oof": oof_rows,
        "cross_database_fixed_model_transport": transport_rows,
        "descriptive_full_cohort_coefficients": coefficients,
        "simulation_definition": {
            "timing": (
                "At 24 h, risk at or above threshold initiates enhanced "
                "monitoring through the 72 h landmark"
            ),
            "thresholds": list(THRESHOLDS),
            "primary_trigger": "predicted risk of any 24-to-72-hour class flip",
            "secondary_trigger": "predicted risk of major death upgrade",
            "no_reassessment": "No enhanced monitoring after the 24 h landmark",
            "reassess_all": (
                "Enhanced monitoring through 72 h for every patient"
            ),
            "evaluation_horizon": (
                "Prediction instability observed by the 72-hour landmark"
            ),
            "clinical_utility": False,
        },
        "simulations": simulations,
        "validation": {
            "passed": True,
            "checks": validations,
        },
    }

    with (OUT / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=True)

    prediction_payload: dict[str, np.ndarray] = {}
    for database, cohort in cohorts.items():
        prediction_payload[f"{database}__ids"] = cohort["ids"]
        prediction_payload[f"{database}__fold_id"] = cohort["folds"]
        for outcome in OUTCOMES:
            prediction_payload[f"{database}__{outcome}__oof"] = oof[database][
                outcome
            ]
            prediction_payload[f"{database}__{outcome}__event"] = cohort[
                "events"
            ][outcome].to_numpy(dtype=np.int64)
    for direction, prediction in transport.items():
        for outcome in OUTCOMES:
            prediction_payload[f"{direction}__{outcome}__fixed_model"] = (
                prediction[outcome]
            )
    np.savez_compressed(OUT / "predictions.npz", **prediction_payload)

    report = build_report(
        cohort_rows,
        oof_rows,
        transport_rows,
        simulations,
        coefficients,
        source_map,
    )
    (OUT / "report.md").write_text(report, encoding="utf-8")

    with (OUT / "summary.json").open("r", encoding="utf-8") as handle:
        json.load(handle)
    if "clinical utility" not in report.lower():
        raise AssertionError("Clinical-utility boundary missing from report")
    print(f"Wrote {OUT / 'summary.json'}", flush=True)
    print(f"Wrote {OUT / 'report.md'}", flush=True)
    print(f"Wrote {OUT / 'predictions.npz'}", flush=True)


if __name__ == "__main__":
    main()
