# -*- coding: utf-8 -*-
"""Run prespecified feasible sensitivity analyses after the primary analysis."""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier


ROOT = Path(os.environ.get("SEPSIS_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
DATA = ROOT / "data" / "processed"
LANDMARK = ROOT / "results" / "landmarks"
OUT = ROOT / "results" / "sensitivity"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 20261029


def static_columns(frame: pd.DataFrame) -> list[str]:
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
    columns = ["age", "male"] + labs
    columns += [
        column
        for column in frame.columns
        if column.startswith(("adm_", "unit_", "src_", "ut_"))
    ]
    return [column for column in dict.fromkeys(columns) if column in frame.columns]


def early_static_columns(frame: pd.DataFrame) -> list[str]:
    """Keep only variables that are available by the six-hour landmark."""
    columns = ["age", "male"]
    columns += [
        column
        for column in frame.columns
        if column.startswith(("adm_", "unit_", "src_", "ut_"))
    ]
    return [column for column in dict.fromkeys(columns) if column in frame.columns]


def six_hour_features(database: str) -> pd.DataFrame:
    folder = DATA / database
    hourly = pd.read_parquet(folder / "hourly_all.parquet")
    identifier = "stay_id" if database == "mimic4" else "patientunitstayid"
    channels = ["hr", "rr", "spo2", "sbp", "dbp", "mbp", "tempf", "gcs"]
    early = hourly.loc[hourly["hh"] < 6]
    grouped = early.groupby(identifier)[channels].agg(
        ["mean", "min", "max", "last", "count"]
    )
    grouped.columns = [f"{channel}_h6_{stat}" for channel, stat in grouped.columns]
    return grouped.reset_index()


def six_vs_48(database: str) -> dict:
    folder = DATA / database
    frame = pd.read_parquet(folder / "design_all.parquet")
    duration = "los" if database == "mimic4" else "los_h"
    group_column = "subject_id" if database == "mimic4" else "uniquepid"
    identifier = "stay_id" if database == "mimic4" else "patientunitstayid"
    feature_6h = six_hour_features(database)
    frame = frame.merge(feature_6h, on=identifier, how="left")
    eligible = frame["y3"].notna() & (frame[duration] >= 6)
    subset = frame.loc[eligible].copy()
    dynamic = [column for column in subset.columns if "_h6_" in column]
    # First-24-hour laboratory summaries are unavailable at six hours and are
    # deliberately excluded from this early-landmark sensitivity analysis.
    columns = early_static_columns(subset) + dynamic
    x = subset[columns].astype(float)
    y = subset["y3"].to_numpy(dtype=int)
    groups = subset[group_column].astype(str).to_numpy()
    probability = np.zeros((len(subset), 3))
    folds = list(GroupKFold(n_splits=5).split(x, y, groups))
    params = {
        "objective": "multi:softprob",
        "num_class": 3,
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.06,
        "subsample": 0.85,
        "colsample_bytree": 0.8,
        "tree_method": "hist",
        "n_jobs": 8,
        "random_state": SEED,
        "eval_metric": "mlogloss",
    }
    for train_index, test_index in folds:
        model = XGBClassifier(**params)
        model.fit(x.iloc[train_index], y[train_index])
        probability[test_index] = model.predict_proba(x.iloc[test_index])
    auc_6 = float(
        roc_auc_score(y, probability, multi_class="ovr", average="macro")
    )
    ids_6 = subset[identifier].to_numpy(dtype=np.int64)
    six_lookup = {int(identifier): index for index, identifier in enumerate(ids_6)}

    bundle_48 = np.load(LANDMARK / database / "3class_48h.npz")
    ids_48 = bundle_48["ids"].astype(np.int64)
    keep = np.isin(ids_48, ids_6)
    ids_common = ids_48[keep]
    probability_48 = bundle_48["probability"][keep]
    probability_6 = probability[
        np.asarray([six_lookup[int(identifier)] for identifier in ids_common])
    ]
    y_common = bundle_48["y"][keep]
    prediction_6 = probability_6.argmax(1)
    prediction_48 = probability_48.argmax(1)
    result = {
        "database": database,
        "six_hour": {
            "n": int(len(y)),
            "macro_auc": round(auc_6, 4),
        },
        "six_to_48h_complete_subset": {
            "n": int(len(y_common)),
            "flip": round(float((prediction_6 != prediction_48).mean()), 4),
            "tv": round(
                float(
                    (
                        0.5 * np.abs(probability_6 - probability_48).sum(1)
                    ).mean()
                ),
                4,
            ),
            "auc_6": round(
                float(
                    roc_auc_score(
                        y_common,
                        probability_6,
                        multi_class="ovr",
                        average="macro",
                    )
                ),
                4,
            ),
            "auc_48": round(
                float(
                    roc_auc_score(
                        y_common,
                        probability_48,
                        multi_class="ovr",
                        average="macro",
                    )
                ),
                4,
            ),
        },
    }
    np.savez_compressed(
        OUT / f"{database}_six_hour_oof.npz",
        probability=probability,
        y=y,
        ids=ids_6,
        groups=groups.astype("U"),
    )
    return result


def full_vs_complete(database: str) -> dict:
    result = {}
    complete = np.load(LANDMARK / database / "3class_72h.npz")["ids"].astype(np.int64)
    for milestone in (24, 48):
        bundle = np.load(LANDMARK / database / f"3class_{milestone}h.npz")
        ids = bundle["ids"].astype(np.int64)
        selected = np.isin(ids, complete)
        result[str(milestone)] = {
            "full_landmark": {
                "n": int(len(ids)),
                "macro_auc": round(
                    float(
                        roc_auc_score(
                            bundle["y"],
                            bundle["probability"],
                            multi_class="ovr",
                            average="macro",
                        )
                    ),
                    4,
                ),
            },
            "complete_window_subset": {
                "n": int(selected.sum()),
                "macro_auc": round(
                    float(
                        roc_auc_score(
                            bundle["y"][selected],
                            bundle["probability"][selected],
                            multi_class="ovr",
                            average="macro",
                        )
                    ),
                    4,
                ),
            },
        }
    return {"database": database, "landmarks": result}


def main() -> None:
    summary = {
        "six_vs_48": [six_vs_48(database) for database in ("mimic4", "eicu")],
        "full_vs_complete": [
            full_vs_complete(database) for database in ("mimic4", "eicu")
        ],
        "three_vs_four_class": "See results/landmarks/xgb_summary.json",
        "eicu_site_clustered_interval": "See results/stability_v3/primary_stability.json",
        "source_recovered_and_extended": (
            "See results/sensitivity_v2/sensitivity_v2_summary.json after running "
            "build_sensitivity_flags.py, run_sensitivity_v2.py, "
            "analyze_reverse_order.py, analyze_ordinal_clinical.py, and "
            "build_mimic3_sensitivity.py"
        ),
        "not_run": [
            "legacy time-reversal, time-permutation and generic-Laplace negative controls: not part of the final v3 estimand",
            "full lambda grid: formal runs used fixed weights 0.04/0.06; exploratory pilot settings were not retained as comparative evidence",
            "no-graph ablation: not part of the final v3 four-configuration ablation grid",
        ],
    }
    with (OUT / "sensitivity_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
