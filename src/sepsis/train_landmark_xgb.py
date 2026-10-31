# -*- coding: utf-8 -*-
"""Train landmark-specific XGBoost models for both outcome definitions."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier


ROOT = Path(os.environ.get("SEPSIS_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
DATA = ROOT / "data" / "processed"
OUT = ROOT / "results" / "landmarks"
OUT.mkdir(parents=True, exist_ok=True)
LANDMARKS = (24, 48, 72)
SEED = 20261029


def feature_columns(frame: pd.DataFrame, milestone: int) -> list[str]:
    labs = [
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
    static = ["age", "male"] + labs
    static += [
        column
        for column in frame.columns
        if column.startswith(("adm_", "unit_", "src_", "ut_"))
    ]
    windows = [f"_w{index}_" for index in range(1, milestone // 24 + 1)]
    dynamic = [
        column
        for column in frame.columns
        if any(token in column for token in windows)
    ]
    columns = [column for column in static + dynamic if column in frame.columns]
    return list(dict.fromkeys(columns))


def expected_calibration_error(y: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    confidence = probability.max(axis=1)
    prediction = probability.argmax(axis=1)
    correct = prediction == y
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        if upper == 1.0:
            selected = (confidence >= lower) & (confidence <= upper)
        else:
            selected = (confidence >= lower) & (confidence < upper)
        if selected.any():
            ece += (
                abs(correct[selected].mean() - confidence[selected].mean())
                * selected.mean()
            )
    return float(ece)


def metric_block(
    y: np.ndarray, probability: np.ndarray, selected: np.ndarray | None = None
) -> dict:
    if selected is not None:
        y = y[selected]
        probability = probability[selected]
    classes = np.unique(y)
    return {
        "n": int(len(y)),
        "macro_auc": round(
            float(
                roc_auc_score(
                    y,
                    probability[:, classes],
                    multi_class="ovr",
                    average="macro",
                    labels=classes,
                )
            ),
            4,
        ),
        "accuracy": round(float(accuracy_score(y, probability[:, classes].argmax(1))), 4),
        "log_loss": round(float(log_loss(y, probability[:, classes], labels=classes)), 4),
        "brier": round(
            float(np.mean(np.sum((probability[:, classes] - np.eye(len(classes))[y]) ** 2, 1))),
            4,
        ),
        "ece_10bin": round(expected_calibration_error(y, probability[:, classes]), 4),
        "mean_max_probability": round(float(probability[:, classes].max(1).mean()), 4),
    }


def train_database(database: str) -> dict:
    folder = DATA / database
    frame = pd.read_parquet(folder / "design_all.parquet")
    duration = "los" if database == "mimic4" else "los_h"
    group = "subject_id" if database == "mimic4" else "uniquepid"
    id_column = "stay_id" if database == "mimic4" else "patientunitstayid"
    target_folder = OUT / database
    target_folder.mkdir(parents=True, exist_ok=True)
    summary = {
        "database": database,
        "training": {
            "random_seed": SEED,
            "seeds": 1,
            "fold_rule": "patient-grouped; regenerated separately at each landmark",
            "cv_folds": 5,
        },
        "definitions": {},
    }

    for class_count, target_column in ((3, "y3"), (4, "y")):
        definition = f"{class_count}class"
        summary["definitions"][definition] = {}
        for milestone in LANDMARKS:
            eligible = frame[target_column].notna() & (frame[duration] >= milestone)
            subset = frame.loc[eligible].copy()
            y = subset[target_column].to_numpy(dtype=int)
            groups = subset[group].astype(str).to_numpy()
            columns = feature_columns(subset, milestone)
            x = subset[columns].astype(float)
            folds = list(GroupKFold(n_splits=5).split(x, y, groups))
            probability = np.zeros((len(subset), class_count), dtype=np.float64)
            params = {
                "objective": "multi:softprob",
                "num_class": class_count,
                "n_estimators": 300,
                "max_depth": 4,
                "learning_rate": 0.06,
                "subsample": 0.85,
                "colsample_bytree": 0.8,
                "tree_method": "hist",
                "n_jobs": int(os.environ.get("SEPSIS_XGB_THREADS", "8")),
                "random_state": SEED,
                "eval_metric": "mlogloss",
            }
            for train_index, test_index in folds:
                model = XGBClassifier(**params)
                model.fit(x.iloc[train_index], y[train_index])
                probability[test_index] = model.predict_proba(x.iloc[test_index])
            metrics = metric_block(y, probability)
            metrics["landmark_hours"] = milestone
            metrics["feature_count"] = len(columns)
            summary["definitions"][definition][str(milestone)] = metrics
            np.savez_compressed(
                target_folder / f"{definition}_{milestone}h.npz",
                probability=probability,
                y=np.asarray(y, dtype=np.int64),
                ids=subset[id_column].to_numpy(dtype=np.int64),
                groups=groups.astype("U"),
            )
            print(database, definition, milestone, metrics, flush=True)

    with (target_folder / "xgb_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", choices=("mimic4", "eicu", "both"), default="both")
    args = parser.parse_args()
    databases = ("mimic4", "eicu") if args.database == "both" else (args.database,)
    summaries = [train_database(database) for database in databases]
    combined = {
        summary["database"]: summary
        for summary in [
            json.loads(
                (OUT / database / "xgb_summary.json").read_text(encoding="utf-8")
            )
            for database in ("mimic4", "eicu")
            if (OUT / database / "xgb_summary.json").exists()
        ]
    }
    with (OUT / "xgb_summary.json").open("w", encoding="utf-8") as handle:
        json.dump([combined[database] for database in ("mimic4", "eicu")], handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
