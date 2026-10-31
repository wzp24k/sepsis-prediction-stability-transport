# -*- coding: utf-8 -*-
"""Fixed-weight cross-database transport validation for the 3-class outcome.

The source database is used for model fitting only. All reported performance
metrics are computed on the target database with the fitted source model; no
cross-validated source predictions are included.
"""

from __future__ import annotations

import json
import os
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from xgboost import XGBClassifier


ROOT = Path(os.environ.get("SEPSIS_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
DATA = ROOT / "data" / "processed"
OUT = ROOT / "results" / "transport"
OUT.mkdir(parents=True, exist_ok=True)

LANDMARKS = (24, 48, 72)
CLASS_LABELS = {0: "home", 1: "facility", 2: "death"}
SEED = 20261029
N_JOBS = int(os.environ.get("SEPSIS_XGB_THREADS", str(os.cpu_count() or 1)))

PARAMS = {
    "objective": "multi:softprob",
    "num_class": 3,
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.06,
    "subsample": 0.85,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
    "n_jobs": N_JOBS,
    "random_state": SEED,
    "eval_metric": "mlogloss",
}

LABS = [
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
STATIC = ["age", "male"] + LABS
VITALS = ("hr", "rr", "spo2", "sbp", "dbp", "mbp", "tempf", "gcs")
STATS = ("mean", "min", "max", "last", "count")

EXCLUDED_COLUMNS = {
    # Outcomes and outcome-related labels
    "y",
    "y3",
    # Follow-up duration and time fields
    "los",
    "los_h",
    "intime",
    "outtime",
    "hospitaldischargeoffset",
    "unitdischargeoffset",
    "hospitaldischargeyear",
    # Patient, encounter, hospital, and row identifiers
    "stay_id",
    "subject_id",
    "hadm_id",
    "patientunitstayid",
    "uniquepid",
    "hospitalid",
    "unitvisitnumber",
    # Discharge information and raw demographic source fields
    "hospitaldischargestatus",
    "hospitaldischargelocation",
    "unitdischargelocation",
    "gender",
    # Administrative variables with non-identical categories across databases
    "admission_type",
    "first_careunit",
    "hospitaladmitsource",
    "unitadmitsource",
    "unittype",
    "admissionweight",
    "admissionheight",
}

DATABASES = {
    "mimic4": {
        "id": "stay_id",
        "group": "subject_id",
        "cluster": "subject_id",
        "duration": "los",
    },
    "eicu": {
        "id": "patientunitstayid",
        "group": "uniquepid",
        "cluster": "hospitalid",
        "duration": "los_h",
    },
}


def feature_columns_for_landmark(
    exact_common: list[str], milestone: int
) -> list[str]:
    """Return exact-name common clinical features available by a landmark."""
    features = [column for column in STATIC if column in exact_common]
    for window in range(1, milestone // 24 + 1):
        for channel in VITALS:
            for statistic in STATS:
                column = f"{channel}_w{window}_{statistic}"
                if column in exact_common:
                    features.append(column)
    return list(dict.fromkeys(features))


def to_native(value):
    if isinstance(value, dict):
        return {str(key): to_native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_native(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def expected_calibration_error(
    y: np.ndarray, probability: np.ndarray, bins: int = 10
) -> float:
    confidence = probability.max(axis=1)
    prediction = probability.argmax(axis=1)
    correct = prediction == y
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        selected = (
            (confidence >= lower) & (confidence <= upper)
            if upper == 1.0
            else (confidence >= lower) & (confidence < upper)
        )
        if selected.any():
            ece += (
                abs(float(correct[selected].mean()) - float(confidence[selected].mean()))
                * float(selected.mean())
            )
    return float(ece)


def multiclass_brier(y: np.ndarray, probability: np.ndarray) -> float:
    observed = np.eye(3, dtype=float)[y]
    return float(np.mean(np.sum((probability - observed) ** 2, axis=1)))


def class_calibration(y: np.ndarray, probability: np.ndarray) -> dict:
    calibration = {}
    for class_index in range(3):
        observed = (y == class_index).astype(int)
        if np.unique(observed).size < 2:
            calibration[str(class_index)] = {
                "label": CLASS_LABELS[class_index],
                "slope": None,
                "intercept": None,
                "prevalence": float(observed.mean()),
                "available": False,
            }
            continue
        predicted = np.clip(probability[:, class_index], 1e-6, 1.0 - 1e-6)
        logit = np.log(predicted / (1.0 - predicted)).reshape(-1, 1)
        model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(logit, observed)
        calibration[str(class_index)] = {
            "label": CLASS_LABELS[class_index],
            "slope": float(model.coef_[0, 0]),
            "intercept": float(model.intercept_[0]),
            "prevalence": float(observed.mean()),
            "available": True,
        }
    return calibration


def metric_block(y: np.ndarray, probability: np.ndarray) -> dict:
    y = np.asarray(y, dtype=int)
    probability = np.asarray(probability, dtype=float)
    labels = np.arange(3)
    return {
        "n": int(len(y)),
        "class_counts": {
            CLASS_LABELS[index]: int((y == index).sum()) for index in labels
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
        "class_calibration": class_calibration(y, probability),
    }


def missingness_diagnostics(frame: pd.DataFrame, features: list[str]) -> dict:
    missing = frame[features].isna().mean().mul(100.0)
    top = missing.sort_values(ascending=False).head(8)
    return {
        "median_feature_missing_percent": float(missing.median()),
        "p90_feature_missing_percent": float(missing.quantile(0.90)),
        "max_feature_missing_percent": float(missing.max()),
        "top_missing_features_percent": {
            column: float(value) for column, value in top.items()
        },
    }


def cohort_mask(
    frame: pd.DataFrame, duration_column: str, milestone: int, complete_72: bool
) -> np.ndarray:
    required_hours = 72 if complete_72 else milestone
    return (
        frame["y3"].notna().to_numpy()
        & (frame[duration_column].to_numpy(dtype=float) >= required_hours)
    )


def fit_model(
    source: pd.DataFrame,
    source_mask: np.ndarray,
    features: list[str],
) -> tuple[XGBClassifier, dict]:
    x = source.loc[source_mask, features].astype(float).replace([np.inf, -np.inf], np.nan)
    y = source.loc[source_mask, "y3"].to_numpy(dtype=int)
    if len(np.unique(y)) != 3:
        raise ValueError(f"Source training data do not contain all 3 outcome classes: {np.unique(y)}")
    model = XGBClassifier(**PARAMS)
    model.fit(x, y)
    metadata = {
        "n": int(len(y)),
        "class_counts": {
            CLASS_LABELS[index]: int((y == index).sum()) for index in range(3)
        },
        "feature_count": int(len(features)),
    }
    return model, metadata


def predict_model(
    model: XGBClassifier, target: pd.DataFrame, target_mask: np.ndarray, features: list[str]
) -> np.ndarray:
    x = target.loc[target_mask, features].astype(float).replace([np.inf, -np.inf], np.nan)
    probability = np.asarray(model.predict_proba(x), dtype=float)
    if probability.shape != (int(target_mask.sum()), 3):
        raise ValueError(f"Unexpected probability shape: {probability.shape}")
    if not np.isfinite(probability).all():
        raise ValueError("Model returned non-finite predicted probabilities")
    probability = probability / probability.sum(axis=1, keepdims=True)
    return probability


def split_direction(direction: str) -> tuple[str, str]:
    source, target = direction.split("_to_")
    return source, target


def transport_direction(
    direction: str,
    frames: dict[str, pd.DataFrame],
    feature_sets: dict[int, list[str]],
) -> tuple[dict, dict[str, np.ndarray]]:
    source_name, target_name = split_direction(direction)
    source = frames[source_name]
    target = frames[target_name]
    source_spec = DATABASES[source_name]
    target_spec = DATABASES[target_name]
    target_id = target_spec["id"]
    target_cluster = target_spec["cluster"]

    probabilities_for_npz: dict[str, np.ndarray] = {}
    landmarks = {}
    primary_models: dict[int, XGBClassifier] = {}
    primary_masks: dict[int, np.ndarray] = {}
    primary_predictions: dict[int, np.ndarray] = {}

    for milestone in LANDMARKS:
        features = feature_sets[milestone]
        source_mask = cohort_mask(
            source, source_spec["duration"], milestone, complete_72=False
        )
        target_mask = cohort_mask(
            target, target_spec["duration"], milestone, complete_72=False
        )
        model, training = fit_model(source, source_mask, features)
        probability = predict_model(model, target, target_mask, features)
        y = target.loc[target_mask, "y3"].to_numpy(dtype=int)
        primary_models[milestone] = model
        primary_masks[milestone] = target_mask
        primary_predictions[milestone] = probability

        target_complete72_mask = cohort_mask(
            target, target_spec["duration"], milestone, complete_72=True
        )
        complete_probability = predict_model(
            model, target, target_complete72_mask, features
        )
        complete_y = target.loc[target_complete72_mask, "y3"].to_numpy(dtype=int)

        landmarks[str(milestone)] = {
            "feature_count": len(features),
            "features": features,
            "training": training,
            "target_all_reaching_landmark": metric_block(y, probability),
            "target_complete_72h_using_primary_model": metric_block(
                complete_y, complete_probability
            ),
        }
        key = f"{direction}__primary__{milestone}h"
        probabilities_for_npz[f"{key}__probability"] = probability
        probabilities_for_npz[f"{key}__y3"] = y
        probabilities_for_npz[f"{key}__ids"] = target.loc[
            target_mask, target_id
        ].to_numpy(dtype=np.int64)
        probabilities_for_npz[f"{key}__clusters"] = (
            target.loc[target_mask, target_cluster].astype(str).to_numpy(dtype="U")
        )

        # Matched-cohort transport: fit on source complete-72h and evaluate on
        # target complete-72h. At 72 h this model is identical to the primary model.
        if milestone < 72:
            source_complete72_mask = cohort_mask(
                source, source_spec["duration"], milestone, complete_72=True
            )
            matched_model, matched_training = fit_model(
                source, source_complete72_mask, features
            )
            matched_probability = predict_model(
                matched_model, target, target_complete72_mask, features
            )
        else:
            matched_model = primary_models[milestone]
            matched_training = training
            matched_probability = complete_probability
        landmarks[str(milestone)]["target_complete_72h_matched_training"] = {
            "training": matched_training,
            "metrics": metric_block(complete_y, matched_probability),
        }
        matched_key = f"{direction}__matched72__{milestone}h"
        probabilities_for_npz[f"{matched_key}__probability"] = matched_probability
        probabilities_for_npz[f"{matched_key}__y3"] = complete_y
        probabilities_for_npz[f"{matched_key}__ids"] = target.loc[
            target_complete72_mask, target_id
        ].to_numpy(dtype=np.int64)
        probabilities_for_npz[f"{matched_key}__clusters"] = (
            target.loc[target_complete72_mask, target_cluster]
            .astype(str)
            .to_numpy(dtype="U")
        )

    # Fixed-weight 24->72 transition. Both landmark-specific models are fit
    # independently on the same source protocol, then applied without update.
    target_common_mask = cohort_mask(
        target, target_spec["duration"], 72, complete_72=True
    )
    common_ids = target.loc[target_common_mask, target_id].to_numpy(dtype=np.int64)
    ids_24 = target.loc[primary_masks[24], target_id].to_numpy(dtype=np.int64)
    ids_72 = target.loc[primary_masks[72], target_id].to_numpy(dtype=np.int64)
    index_24 = pd.Series(np.arange(len(ids_24)), index=ids_24)
    index_72 = pd.Series(np.arange(len(ids_72)), index=ids_72)
    if not np.isin(common_ids, ids_24).all() or not np.isin(common_ids, ids_72).all():
        raise ValueError("Complete-72h target cohort is not a subset of landmark cohorts")
    take_24 = index_24.reindex(common_ids).to_numpy(dtype=int)
    take_72 = index_72.reindex(common_ids).to_numpy(dtype=int)
    p24 = primary_predictions[24][take_24]
    p72 = primary_predictions[72][take_72]
    class_24 = p24.argmax(axis=1)
    class_72 = p72.argmax(axis=1)
    flip = class_24 != class_72
    variation = 0.5 * np.abs(p24 - p72).sum(axis=1)
    transition = np.zeros((3, 3), dtype=int)
    for first, second in zip(class_24, class_72):
        transition[int(first), int(second)] += 1
    clusters = (
        target.loc[target_common_mask, target_cluster]
        .astype(str)
        .to_numpy(dtype="U")
    )

    direction_output = {
        "source_database": source_name,
        "target_database": target_name,
        "protocol": {
            "primary": (
                "Fit one fixed-weight model on all source patients with outcome "
                "and source duration >= landmark; evaluate all target patients "
                "with outcome and target duration >= the same landmark."
            ),
            "matched_72h": (
                "Fit on source patients with outcome and source duration >=72 h; "
                "evaluate target patients with outcome and target duration >=72 h. "
                "At 72 h training is identical to primary."
            ),
            "no_target_fitting": True,
            "no_source_oof_metrics_reported": True,
            "fixed_24_to_72_rule": (
                "Train independent fixed landmark models at 24 and 72 h on the "
                "source primary cohort; compare their target predictions in the "
                "target complete-72h cohort."
            ),
        },
        "landmarks": landmarks,
        "fixed_rule_24_to_72_target_complete72": {
            "n": int(len(common_ids)),
            "class_counts": {
                CLASS_LABELS[index]: int((target.loc[target_common_mask, "y3"] == index).sum())
                for index in range(3)
            },
            "flip": float(flip.mean()),
            "tv": float(variation.mean()),
            "transition_matrix_rows_24_cols_72": transition.tolist(),
            "rows_24": [CLASS_LABELS[index] for index in range(3)],
            "cols_72": [CLASS_LABELS[index] for index in range(3)],
            "y3": target.loc[target_common_mask, "y3"].to_numpy(dtype=int),
            "p24": p24,
            "p72": p72,
            "clusters": clusters,
        },
    }
    probabilities_for_npz[f"{direction}__fixed24_to_72__probability_24h"] = p24
    probabilities_for_npz[f"{direction}__fixed24_to_72__probability_72h"] = p72
    probabilities_for_npz[f"{direction}__fixed24_to_72__y3"] = direction_output[
        "fixed_rule_24_to_72_target_complete72"
    ]["y3"]
    probabilities_for_npz[f"{direction}__fixed24_to_72__ids"] = common_ids
    probabilities_for_npz[f"{direction}__fixed24_to_72__clusters"] = clusters
    # Keep the JSON summary compact; IDs, outcomes, and probabilities live in NPZ.
    direction_output["fixed_rule_24_to_72_target_complete72"].pop("y3")
    direction_output["fixed_rule_24_to_72_target_complete72"].pop("p24")
    direction_output["fixed_rule_24_to_72_target_complete72"].pop("p72")
    direction_output["fixed_rule_24_to_72_target_complete72"].pop("clusters")
    return direction_output, probabilities_for_npz


def format_metric_row(direction: str, milestone: int, metrics: dict) -> str:
    slope = ", ".join(
        f"{metrics['class_calibration'][str(index)]['label']} "
        f"{metrics['class_calibration'][str(index)]['slope']:.3f}"
        for index in range(3)
    )
    intercept = ", ".join(
        f"{metrics['class_calibration'][str(index)]['label']} "
        f"{metrics['class_calibration'][str(index)]['intercept']:.3f}"
        for index in range(3)
    )
    return (
        f"| {direction} | {milestone} | {metrics['n']} | "
        f"{metrics['macro_auc']:.3f} | {metrics['accuracy']:.3f} | "
        f"{metrics['log_loss']:.3f} | {metrics['brier_multiclass']:.3f} | "
        f"{metrics['ece_top_label_10bin']:.3f} | {slope} | {intercept} |"
    )


def build_report(summary: dict) -> str:
    lines = [
        "# Fixed-Weight Cross-Database Transport Validation",
        "",
        f"Generated: {summary['generated_at']}",
        "",
        "## Scope and safeguards",
        "",
        (
            "The source database was used only for model fitting. Every performance "
            "estimate below is a target-database prediction from one fixed-weight "
            "model; source cross-validated/OOF predictions are not reported or mixed "
            "with transport metrics."
        ),
        "",
        (
            "All models use the 3-class outcome: 0 = home, 1 = facility/rehabilitation, "
            "and 2 = death. Fits at 24, 48, and 72 h use identical XGBoost "
            "hyperparameters, while the feature set expands with the observed "
            "24-hour windows."
        ),
        "",
        "## Feasibility and feature audit",
        "",
        f"- Exact-name common features after exclusions: {summary['feasibility']['exact_common_feature_count']}.",
        f"- Features by landmark: 24 h = {summary['feasibility']['landmark_feature_counts']['24']}, "
        f"48 h = {summary['feasibility']['landmark_feature_counts']['48']}, "
        f"72 h = {summary['feasibility']['landmark_feature_counts']['72']}.",
        (
            "- The shared feature set is restricted to age, male sex, 17 laboratory "
            "summaries, and aligned vital-sign window summaries. Outcome, duration, "
            "identifier, and discharge-information fields were excluded."
        ),
        (
            "- Administrative admission/unit categories were not forced across databases "
            "because their category dictionaries are not identical."
        ),
        f"- Feasible: {summary['feasibility']['feasible']}.",
        "",
        "| Database | Landmark | Eligible with outcome | Home | Facility | Death |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for database in ("mimic4", "eicu"):
        for milestone in LANDMARKS:
            cohort = summary["feasibility"]["cohorts"][database][str(milestone)]
            lines.append(
                f"| {database} | {milestone} h | {cohort['n']} | "
                f"{cohort['class_counts']['home']} | "
                f"{cohort['class_counts']['facility']} | "
                f"{cohort['class_counts']['death']} |"
            )

    lines.extend(
        [
            "",
            "## Primary transport: all target patients reaching each landmark",
            "",
            "| Direction | Landmark | n | Macro-AUC | Accuracy | Log loss | Brier | ECE | Class calibration slopes | Class intercepts |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
        ]
    )
    for direction in ("mimic4_to_eicu", "eicu_to_mimic4"):
        result = summary["transport"][direction]
        for milestone in LANDMARKS:
            metrics = result["landmarks"][str(milestone)][
                "target_all_reaching_landmark"
            ]
            lines.append(format_metric_row(direction, milestone, metrics))

    lines.extend(
        [
            "",
            "## Matched complete-72h transport",
            "",
            (
                "Here both source training and target evaluation are restricted to "
                "patients observed through 72 h. Models at 24 and 48 h still use only "
                "the data available through their landmark."
            ),
            "",
            "| Direction | Landmark | n | Macro-AUC | Accuracy | Log loss | Brier | ECE | Class calibration slopes | Class intercepts |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
        ]
    )
    for direction in ("mimic4_to_eicu", "eicu_to_mimic4"):
        result = summary["transport"][direction]
        for milestone in LANDMARKS:
            metrics = result["landmarks"][str(milestone)][
                "target_complete_72h_matched_training"
            ]["metrics"]
            lines.append(format_metric_row(direction, milestone, metrics))

    lines.extend(
        [
            "",
            "## Fixed 24-to-72 h prediction change",
            "",
            (
                "Separate 24 h and 72 h models were fit on their source landmark "
                "cohorts and applied unchanged to the target complete-72h cohort."
            ),
            "",
            "| Direction | n | Flip | Mean total variation |",
            "|---|---:|---:|---:|",
        ]
    )
    for direction in ("mimic4_to_eicu", "eicu_to_mimic4"):
        fixed = summary["transport"][direction][
            "fixed_rule_24_to_72_target_complete72"
        ]
        lines.append(
            f"| {direction} | {fixed['n']} | {fixed['flip']:.3f} | "
            f"{fixed['tv']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            (
                "1. This is fixed-weight cross-database transport, not prospective "
                "clinical validation. eICU hospitals are clustered and heterogeneous, "
                "whereas MIMIC-IV is a single academic center."
            ),
            (
                "2. The two outcome dictionaries are harmonized but not identical. "
                "Facility merges rehabilitation and several post-acute destinations "
                "with different meanings across databases."
            ),
            (
                "3. Models use all source patients eligible at a landmark. The matched "
                "complete-72h analysis addresses cohort-window alignment but remains "
                "conditional on surviving/staying in the ICU through 72 h."
            ),
            (
                "4. XGBoost handles missing features natively. Cross-database missingness "
                "is substantial, especially for GCS, bilirubin, INR, and third-day "
                "vital signs; missingness diagnostics are included in summary.json."
            ),
            (
                "5. Calibration slopes/intercepts are target-database evaluation "
                "summaries from class-wise logistic recalibration models and do not "
                "imply that the transported probabilities need no clinical recalibration."
            ),
            (
                "6. The 24-to-72 h flip is descriptive. A lower flip rate alone is not "
                "evidence of better prediction, and no claim of clinical utility is made."
            ),
            "",
            "## Artifacts",
            "",
            "- `results/transport/summary.json`: full machine-readable audit and metrics.",
            "- `results/transport/predictions.npz`: fixed-model probabilities, outcomes, IDs, and clusters.",
            "- `src/sepsis/analyze_transport.py`: reproducible analysis script.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    frames = {
        database: pd.read_parquet(DATA / database / "design_all.parquet")
        for database in DATABASES
    }
    common_columns = [
        column
        for column in frames["mimic4"].columns
        if column in frames["eicu"].columns and column not in EXCLUDED_COLUMNS
    ]
    feature_sets = {
        milestone: feature_columns_for_landmark(common_columns, milestone)
        for milestone in LANDMARKS
    }
    if not common_columns or len(feature_sets[72]) < 1:
        raise RuntimeError("No usable exact-name common clinical features")

    cohort_diagnostics = {}
    missingness = {}
    for database, spec in DATABASES.items():
        frame = frames[database]
        cohort_diagnostics[database] = {}
        missingness[database] = {}
        for milestone in LANDMARKS:
            mask = cohort_mask(
                frame, spec["duration"], milestone, complete_72=False
            )
            y = frame.loc[mask, "y3"].to_numpy(dtype=int)
            cohort_diagnostics[database][str(milestone)] = {
                "n": int(mask.sum()),
                "class_counts": {
                    CLASS_LABELS[index]: int((y == index).sum())
                    for index in range(3)
                },
            }
            missingness[database][str(milestone)] = missingness_diagnostics(
                frame.loc[mask], feature_sets[milestone]
            )
        complete72_mask = cohort_mask(
            frame, spec["duration"], 72, complete_72=True
        )
        complete_y = frame.loc[complete72_mask, "y3"].to_numpy(dtype=int)
        cohort_diagnostics[database]["complete_72h"] = {
            "n": int(complete72_mask.sum()),
            "class_counts": {
                CLASS_LABELS[index]: int((complete_y == index).sum())
                for index in range(3)
            },
        }

    all_probabilities: dict[str, np.ndarray] = {}
    transport = {}
    for direction in ("mimic4_to_eicu", "eicu_to_mimic4"):
        print(f"Running {direction}", flush=True)
        result, probability_arrays = transport_direction(
            direction, frames, feature_sets
        )
        transport[direction] = result
        all_probabilities.update(probability_arrays)
        for milestone in LANDMARKS:
            metrics = result["landmarks"][str(milestone)][
                "target_all_reaching_landmark"
            ]
            print(
                direction,
                milestone,
                {
                    "n": metrics["n"],
                    "macro_auc": round(metrics["macro_auc"], 4),
                    "accuracy": round(metrics["accuracy"], 4),
                    "log_loss": round(metrics["log_loss"], 4),
                    "brier": round(metrics["brier_multiclass"], 4),
                    "ece": round(metrics["ece_top_label_10bin"], 4),
                },
                flush=True,
            )

    summary = {
        "analysis": "fixed-weight cross-database transport validation",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "random_seed": SEED,
        "xgboost_n_jobs": N_JOBS,
        "hyperparameters": PARAMS,
        "outcome": {
            "column": "y3",
            "labels": CLASS_LABELS,
            "mapping_compatible": True,
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
            "excluded_columns": sorted(EXCLUDED_COLUMNS),
            "cohorts": cohort_diagnostics,
            "missingness_by_landmark": missingness,
            "reason_if_not_feasible": None,
        },
        "transport": transport,
        "artifacts": {
            "script": str(Path(__file__).resolve()),
            "summary": str(OUT / "summary.json"),
            "report": str(OUT / "report.md"),
            "predictions": str(OUT / "predictions.npz"),
        },
    }
    summary = to_native(summary)
    with (OUT / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    with (OUT / "report.md").open("w", encoding="utf-8") as handle:
        handle.write(build_report(summary))
    np.savez_compressed(OUT / "predictions.npz", **all_probabilities)
    print(f"Wrote {OUT / 'summary.json'}", flush=True)
    print(f"Wrote {OUT / 'report.md'}", flush=True)
    print(f"Wrote {OUT / 'predictions.npz'}", flush=True)


if __name__ == "__main__":
    main()
