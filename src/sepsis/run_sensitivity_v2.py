# -*- coding: utf-8 -*-
"""Retrain XGBoost for ICU-death exclusion and MIMIC-IV ICD-era strata."""

from __future__ import annotations

import argparse
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
FLAGS = ROOT / "results" / "sensitivity_v2" / "flags"
OUT = ROOT / "results" / "sensitivity_v2" / "xgb"
OUT.mkdir(parents=True, exist_ok=True)
LANDMARKS = (24, 48, 72)
RNG = np.random.default_rng(20261030)


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
    tokens = [f"_w{index}_" for index in range(1, milestone // 24 + 1)]
    dynamic = [
        column for column in frame.columns if any(token in column for token in tokens)
    ]
    return list(dict.fromkeys(static + dynamic))


def cluster_bootstrap_mean(
    values: np.ndarray, clusters: np.ndarray, replicates: int = 2000
) -> list[float]:
    unique, inverse = np.unique(clusters, return_inverse=True)
    totals = np.bincount(inverse, weights=values, minlength=len(unique))
    counts = np.bincount(inverse, minlength=len(unique)).astype(float)
    estimates = np.empty(replicates)
    for index in range(replicates):
        selected = RNG.integers(0, len(unique), size=len(unique))
        estimates[index] = totals[selected].sum() / counts[selected].sum()
    return [
        round(float(np.quantile(estimates, 0.025)), 4),
        round(float(np.quantile(estimates, 0.975)), 4),
    ]


def metric_block(y: np.ndarray, probability: np.ndarray) -> dict:
    return {
        "n": int(len(y)),
        "macro_auc": round(
            float(
                roc_auc_score(
                    y,
                    probability,
                    multi_class="ovr",
                    average="macro",
                    labels=np.arange(3),
                )
            ),
            4,
        ),
    }


def train_scenario(
    database: str, scenario: str, selection: pd.Series | None
) -> dict:
    folder = DATA / database
    frame = pd.read_parquet(folder / "design_all.parquet")
    flag_path = FLAGS / f"{database}_sensitivity_flags.parquet"
    flag_frame = pd.read_parquet(flag_path)
    identifier = "stay_id" if database == "mimic4" else "patientunitstayid"
    group_column = "subject_id" if database == "mimic4" else "uniquepid"
    cluster_column = "subject_id" if database == "mimic4" else "hospitalid"
    duration = "los" if database == "mimic4" else "los_h"
    merge_columns = [identifier, "icu_death", "icd_era"]
    if cluster_column not in frame.columns:
        merge_columns.append(cluster_column)
    frame = frame.merge(
        flag_frame[list(dict.fromkeys(merge_columns))],
        on=identifier,
        how="left",
        validate="one_to_one",
    )
    if selection is not None:
        aligned = selection.reindex(frame[identifier]).fillna(False).to_numpy(bool)
    else:
        aligned = np.ones(len(frame), dtype=bool)

    probabilities: dict[int, np.ndarray] = {}
    ids_by_landmark: dict[int, np.ndarray] = {}
    y_by_landmark: dict[int, np.ndarray] = {}
    groups_by_landmark: dict[int, np.ndarray] = {}
    clusters_by_landmark: dict[int, np.ndarray] = {}
    landmarks = {}
    removed = {}
    params = {
        "objective": "multi:softprob",
        "num_class": 3,
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.06,
        "subsample": 0.85,
        "colsample_bytree": 0.8,
        "tree_method": "hist",
        "n_jobs": int(os.environ.get("SEPSIS_XGB_THREADS", "4")),
        "random_state": 20261029,
        "eval_metric": "mlogloss",
    }
    for milestone in LANDMARKS:
        base = frame["y3"].notna() & (frame[duration] >= milestone)
        eligible = base & aligned
        subset = frame.loc[eligible].copy()
        y = subset["y3"].to_numpy(dtype=int)
        groups = subset[group_column].astype(str).to_numpy()
        clusters = subset[cluster_column].astype(str).to_numpy()
        columns = feature_columns(subset, milestone)
        x = subset[columns].astype(float)
        folds = list(GroupKFold(n_splits=5).split(x, y, groups))
        probability = np.zeros((len(subset), 3), dtype=float)
        for train_index, test_index in folds:
            model = XGBClassifier(**params)
            model.fit(x.iloc[train_index], y[train_index])
            probability[test_index] = model.predict_proba(x.iloc[test_index])
        ids = subset[identifier].to_numpy(dtype=np.int64)
        probabilities[milestone] = probability
        ids_by_landmark[milestone] = ids
        y_by_landmark[milestone] = y
        groups_by_landmark[milestone] = groups
        clusters_by_landmark[milestone] = clusters
        landmarks[str(milestone)] = metric_block(y, probability)
        removed[str(milestone)] = int((base & ~aligned).sum())
        print(
            database,
            scenario,
            milestone,
            landmarks[str(milestone)],
            "removed",
            removed[str(milestone)],
            flush=True,
        )

    common = np.intersect1d(
        np.intersect1d(ids_by_landmark[24], ids_by_landmark[48]),
        ids_by_landmark[72],
    )
    aligned_probability = {}
    aligned_y = {}
    aligned_cluster = {}
    for milestone in LANDMARKS:
        lookup = {
            int(identifier_value): index
            for index, identifier_value in enumerate(ids_by_landmark[milestone])
        }
        index = np.asarray(
            [lookup[int(identifier_value)] for identifier_value in common], dtype=int
        )
        aligned_probability[milestone] = probabilities[milestone][index]
        aligned_y[milestone] = y_by_landmark[milestone][index]
        aligned_cluster[milestone] = clusters_by_landmark[milestone][index]
    y_common = aligned_y[24]
    prediction = {
        milestone: aligned_probability[milestone].argmax(1)
        for milestone in LANDMARKS
    }
    clusters = aligned_cluster[24]
    flip_24_72 = (
        prediction[24] != prediction[72]
    ).astype(float)
    tv_24_72 = 0.5 * np.abs(
        aligned_probability[24] - aligned_probability[72]
    ).sum(1)
    revert = (
        (prediction[24] == prediction[72]) & (prediction[24] != prediction[48])
    ).astype(float)
    confidence = aligned_probability[24].max(1)
    selected_confidence = confidence >= 0.6
    complete = {
        "n": int(len(common)),
        "class_counts": {
            str(value): int((y_common == value).sum()) for value in range(3)
        },
        "macro_auc": {
            str(milestone): metric_block(
                aligned_y[milestone], aligned_probability[milestone]
            )["macro_auc"]
            for milestone in LANDMARKS
        },
        "flip_24_72": round(float(flip_24_72.mean()), 4),
        "flip_24_72_ci95": cluster_bootstrap_mean(flip_24_72, clusters),
        "tv_24_72": round(float(tv_24_72.mean()), 4),
        "tv_24_72_ci95": cluster_bootstrap_mean(tv_24_72, clusters),
        "return_to_24h_class": round(float(revert.mean()), 4),
        "direction_more_dependent": round(
            float((prediction[72] > prediction[24]).mean()), 4
        ),
        "direction_less_dependent": round(
            float((prediction[72] < prediction[24]).mean()), 4
        ),
        "confidence_0_6": {
            "coverage": round(float(selected_confidence.mean()), 4),
            "n": int(selected_confidence.sum()),
            "flip": round(
                float(flip_24_72[selected_confidence].mean()), 4
            )
            if selected_confidence.any()
            else None,
        },
    }
    output = {
        "database": database,
        "scenario": scenario,
        "landmarks": landmarks,
        "removed_from_landmark_population": removed,
        "complete_three_landmark_subset": complete,
    }
    stem = OUT / f"{database}_{scenario}"
    np.savez_compressed(
        stem.with_suffix(".npz"),
        p24=aligned_probability[24],
        p48=aligned_probability[48],
        p72=aligned_probability[72],
        y24=aligned_y[24],
        y48=aligned_y[48],
        y72=aligned_y[72],
        ids=common,
        clusters=clusters.astype("U"),
    )
    with stem.with_suffix(".json").open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, ensure_ascii=False)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", choices=("mimic4", "eicu"), required=True)
    parser.add_argument(
        "--analysis",
        choices=("icu_death", "icd9", "icd10"),
        required=True,
    )
    args = parser.parse_args()
    flags = pd.read_parquet(
        FLAGS / f"{args.database}_sensitivity_flags.parquet"
    )
    identifier = "stay_id" if args.database == "mimic4" else "patientunitstayid"
    if args.analysis == "icu_death":
        selection = ~flags.set_index(identifier)["icu_death"].fillna(False)
        scenario = "exclude_icu_death"
    else:
        if args.database != "mimic4":
            raise ValueError("ICD-era strata are defined only for MIMIC-IV")
        era = "ICD-9" if args.analysis == "icd9" else "ICD-10"
        selection = flags.set_index(identifier)["icd_era"].eq(era)
        scenario = "icd9_era" if args.analysis == "icd9" else "icd10_era"
    print(json.dumps(train_scenario(args.database, scenario, selection), indent=2))


if __name__ == "__main__":
    main()
