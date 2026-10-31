# -*- coding: utf-8 -*-
"""Evaluation-sample cluster-bootstrap intervals for fitted model contrasts."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ROOT = Path(os.environ.get("SEPSIS_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
SUMMARY = ROOT / "results" / "formal_summary"
OUT = ROOT / "results" / "sensitivity_v2" / "model_contrast_bootstrap.json"
REPLICATES = int(os.environ.get("SEPSIS_BOOTSTRAP_REPLICATES", "2000"))
RNG = np.random.default_rng(20261030)


def ensemble(database: str, configuration: str) -> dict:
    bundle = np.load(SUMMARY / f"{database}_{configuration}_ensemble.npz")
    return {
        "p": bundle["probability"],
        "y": bundle["y"],
        "ids": bundle["ids"].astype(np.int64),
    }


def clusters_for(database: str, ids: np.ndarray) -> np.ndarray:
    if database == "mimic4":
        frame = pd.read_parquet(
            ROOT / "data" / "processed" / "mimic4" / "design_all.parquet",
            columns=["stay_id", "subject_id"],
        )
        lookup = dict(
            zip(frame["stay_id"].astype(int), frame["subject_id"].astype(str))
        )
        return np.asarray([lookup[int(value)] for value in ids])
    frame = pd.read_parquet(
        ROOT / "data" / "processed" / "eicu" / "design_all.parquet",
        columns=["patientunitstayid", "hospitalid"],
    )
    lookup = dict(
        zip(
            frame["patientunitstayid"].astype(int),
            frame["hospitalid"].astype(str),
        )
    )
    return np.asarray([lookup[int(value)] for value in ids])


def continuous_nri_idi(
    y: np.ndarray,
    reference: np.ndarray,
    comparison: np.ndarray,
    cutoff: int,
) -> tuple[float, float]:
    event = y >= cutoff
    reference_risk = reference[:, cutoff:].sum(1)
    comparison_risk = comparison[:, cutoff:].sum(1)
    change = comparison_risk - reference_risk
    nri = (
        (change[event] > 0).mean()
        - (change[event] < 0).mean()
        - (change[~event] > 0).mean()
        + (change[~event] < 0).mean()
    )
    idi = (
        comparison_risk[event].mean()
        - comparison_risk[~event].mean()
        - reference_risk[event].mean()
        + reference_risk[~event].mean()
    )
    return float(nri), float(idi)


def sampled_metrics(
    index: np.ndarray,
    temporal: dict,
    no_temporal: dict,
    nominal: dict,
    nominal_temporal: dict,
) -> dict:
    y = temporal["y"][index]
    temporal_p24 = temporal["p"][index, 0, :]
    temporal_p72 = temporal["p"][index, 2, :]
    no_temporal_p24 = no_temporal["p"][index, 0, :]
    no_temporal_p72 = no_temporal["p"][index, 2, :]
    nominal_p72 = nominal["p"][index, 2, :]
    nominal_temporal_p72 = nominal_temporal["p"][index, 2, :]
    temporal_auc = roc_auc_score(
        y,
        temporal_p72,
        multi_class="ovr",
        average="macro",
        labels=np.arange(3),
    )
    no_temporal_auc = roc_auc_score(
        y,
        no_temporal_p72,
        multi_class="ovr",
        average="macro",
        labels=np.arange(3),
    )
    output = {
        "temporal_auc72": float(temporal_auc - no_temporal_auc),
        "temporal_flip": float(
            (
                temporal_p24.argmax(1) != temporal_p72.argmax(1)
            ).mean()
            - (
                no_temporal_p24.argmax(1) != no_temporal_p72.argmax(1)
            ).mean()
        ),
        "temporal_tv": float(
            (
                0.5 * np.abs(temporal_p24 - temporal_p72).sum(1)
            ).mean()
            - (
                0.5 * np.abs(no_temporal_p24 - no_temporal_p72).sum(1)
            ).mean()
        ),
    }
    for cutoff, suffix in ((1, "non_home"), (2, "death")):
        nri, idi = continuous_nri_idi(
            y, nominal_p72, no_temporal_p72, cutoff
        )
        output[f"ordinal_nri_{suffix}"] = nri
        output[f"ordinal_idi_{suffix}"] = idi
        nri_temporal, idi_temporal = continuous_nri_idi(
            y, nominal_temporal_p72, temporal_p72, cutoff
        )
        output[f"ordinal_temporal_nri_{suffix}"] = nri_temporal
        output[f"ordinal_temporal_idi_{suffix}"] = idi_temporal
    return output


def interval(values: list[float], estimate: float) -> dict:
    array = np.asarray(values, dtype=float)
    return {
        "estimate": round(float(estimate), 4),
        "ci95": [
            round(float(np.quantile(array, 0.025)), 4),
            round(float(np.quantile(array, 0.975)), 4),
        ],
        "bootstrap_same_sign_fraction": round(
            float(max((array > 0).mean(), (array < 0).mean())), 4
        ),
    }


def analyze(database: str) -> dict:
    definitions = {
        "temporal": ensemble(database, "ordinal_temporal"),
        "no_temporal": ensemble(database, "ordinal_notemporal"),
        "nominal_no_temporal": ensemble(database, "nominal_notemporal"),
        "nominal_temporal": ensemble(database, "nominal_temporal"),
    }
    ids = definitions["temporal"]["ids"]
    for name, record in definitions.items():
        if not np.array_equal(ids, record["ids"]):
            raise ValueError(f"{database}: {name} row order differs")
        if not np.array_equal(definitions["temporal"]["y"], record["y"]):
            raise ValueError(f"{database}: {name} outcome order differs")
    point = sampled_metrics(
        np.arange(len(ids)),
        definitions["temporal"],
        definitions["no_temporal"],
        definitions["nominal_no_temporal"],
        definitions["nominal_temporal"],
    )
    clusters = clusters_for(database, ids)
    unique, inverse = np.unique(clusters, return_inverse=True)
    members = [np.where(inverse == index)[0] for index in range(len(unique))]
    values = {key: [] for key in point}
    for _ in range(REPLICATES):
        chosen = RNG.integers(0, len(unique), size=len(unique))
        index = np.concatenate([members[value] for value in chosen])
        record = sampled_metrics(
            index,
            definitions["temporal"],
            definitions["no_temporal"],
            definitions["nominal_no_temporal"],
            definitions["nominal_temporal"],
        )
        for key in values:
            values[key].append(record[key])
    return {
        "database": database,
        "clusters": int(len(unique)),
        "replicates": REPLICATES,
        "estimand": (
            "evaluation-sample uncertainty conditional on the fitted out-of-fold "
            "predictions and cross-validation folds; training uncertainty is not "
            "resampled"
        ),
        "contrasts": {
            "temporal_minus_none_ordinal": {
                "auc_72": interval(values["temporal_auc72"], point["temporal_auc72"]),
                "flip_24_72": interval(
                    values["temporal_flip"], point["temporal_flip"]
                ),
                "tv_24_72": interval(values["temporal_tv"], point["temporal_tv"]),
            },
            "ordinal_minus_nominal_no_temporal": {
                "continuous_nri_non_home": interval(
                    values["ordinal_nri_non_home"],
                    point["ordinal_nri_non_home"],
                ),
                "continuous_nri_death": interval(
                    values["ordinal_nri_death"],
                    point["ordinal_nri_death"],
                ),
                "idi_non_home": interval(
                    values["ordinal_idi_non_home"],
                    point["ordinal_idi_non_home"],
                ),
                "idi_death": interval(
                    values["ordinal_idi_death"],
                    point["ordinal_idi_death"],
                ),
            },
            "ordinal_minus_nominal_temporal": {
                "continuous_nri_non_home": interval(
                    values["ordinal_temporal_nri_non_home"],
                    point["ordinal_temporal_nri_non_home"],
                ),
                "continuous_nri_death": interval(
                    values["ordinal_temporal_nri_death"],
                    point["ordinal_temporal_nri_death"],
                ),
                "idi_non_home": interval(
                    values["ordinal_temporal_idi_non_home"],
                    point["ordinal_temporal_idi_non_home"],
                ),
                "idi_death": interval(
                    values["ordinal_temporal_idi_death"],
                    point["ordinal_temporal_idi_death"],
                ),
            },
        },
    }


def main() -> None:
    summary = {
        "mimic4": analyze("mimic4"),
        "eicu": analyze("eicu"),
        "cluster_unit": {
            "mimic4": "subject_id",
            "eicu": "hospitalid",
        },
        "interpretation": (
            "Intervals describe resampling of the evaluation sample by patient or "
            "hospital after model fitting. They are not intervals for the effect of "
            "retraining an algorithm across new datasets."
        ),
    }
    with OUT.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
