# -*- coding: utf-8 -*-
"""Primary three-class stability analysis with patient- and site-aware intervals."""

import json
import os
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


ROOT = Path(os.environ.get("SEPSIS_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
DATA = ROOT / "results" / "landmarks"
OUT = ROOT / "results" / "stability_v3"
OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(20261029)


def bootstrap_rate(
    indicator: np.ndarray,
    cluster_ids: np.ndarray,
    replicates: int = 2000,
) -> list[float]:
    clusters, inverse = np.unique(cluster_ids, return_inverse=True)
    total = np.bincount(inverse, weights=indicator.astype(float), minlength=len(clusters))
    count = np.bincount(inverse, minlength=len(clusters)).astype(float)
    estimates = np.empty(replicates)
    for i in range(replicates):
        selected = RNG.integers(0, len(clusters), size=len(clusters))
        estimates[i] = total[selected].sum() / count[selected].sum()
    return [
        float(np.quantile(estimates, 0.025)),
        float(np.quantile(estimates, 0.975)),
    ]


def bootstrap_mean(
    values: np.ndarray,
    cluster_ids: np.ndarray,
    replicates: int = 2000,
) -> list[float]:
    clusters, inverse = np.unique(cluster_ids, return_inverse=True)
    total = np.bincount(inverse, weights=values, minlength=len(clusters))
    count = np.bincount(inverse, minlength=len(clusters)).astype(float)
    estimates = np.empty(replicates)
    for i in range(replicates):
        selected = RNG.integers(0, len(clusters), size=len(clusters))
        estimates[i] = total[selected].sum() / count[selected].sum()
    return [
        float(np.quantile(estimates, 0.025)),
        float(np.quantile(estimates, 0.975)),
    ]


def load_definition(database: str, definition: str = "3class") -> dict:
    result = {}
    for milestone in (24, 48, 72):
        path = DATA / database / f"{definition}_{milestone}h.npz"
        # Existing artifacts may store a string group identifier as an object
        # array. These files are generated locally from credentialed data, never
        # downloaded, so loading them with pickle enabled is bounded to this path.
        bundle = np.load(path, allow_pickle=True)
        result[milestone] = {
            "p": bundle["probability"],
            "y": bundle["y"],
            "ids": bundle["ids"],
            "groups": bundle["groups"].astype(str),
        }
    return result


def aligned_complete_subset(database: str, definition: str = "3class") -> dict:
    loaded = load_definition(database, definition)
    common = np.intersect1d(
        np.intersect1d(loaded[24]["ids"], loaded[48]["ids"]),
        loaded[72]["ids"],
    )
    aligned = {}
    for milestone, bundle in loaded.items():
        lookup = {int(identifier): i for i, identifier in enumerate(bundle["ids"])}
        index = np.asarray([lookup[int(identifier)] for identifier in common], dtype=int)
        aligned[milestone] = {
            "p": bundle["p"][index],
            "y": bundle["y"][index],
            "ids": common,
            "groups": bundle["groups"][index],
        }
    return aligned


def stability(database: str) -> dict:
    data = aligned_complete_subset(database)
    prediction = {milestone: data[milestone]["p"].argmax(1) for milestone in data}
    probability = {milestone: data[milestone]["p"] for milestone in data}
    y = data[24]["y"]
    group = data[24]["groups"].astype(str)
    clusters = group
    if database == "eicu":
        source = np.load(DATA / database / "3class_72h.npz")
        source_ids = source["ids"].astype(np.int64)
        # Hospital identifiers are read from the derived model table so that the
        # interval reflects the multi-site design rather than patient bootstrap.
        import pandas as pd

        frame = pd.read_parquet(
            ROOT / "data" / "processed" / "eicu" / "design_all.parquet",
            columns=["patientunitstayid", "hospitalid"],
        )
        lookup = dict(zip(frame["patientunitstayid"].astype(int), frame["hospitalid"].astype(str)))
        clusters = np.asarray([lookup[int(i)] for i in data[24]["ids"]])
        del source_ids

    result = {
        "database": database,
        "definition": "3class",
        "n": int(len(y)),
        "class_counts": {str(i): int((y == i).sum()) for i in range(3)},
        "macro_auc": {},
        "flip": {},
        "total_variation": {},
        "return_to_24h_class": None,
        "direction_24_to_72": {},
        "confidence_24h": {},
        "transition_24_to_72": None,
    }
    for milestone in (24, 48, 72):
        result["macro_auc"][str(milestone)] = round(
            float(
                roc_auc_score(
                    y,
                    probability[milestone],
                    multi_class="ovr",
                    average="macro",
                    labels=np.arange(3),
                )
            ),
            4,
        )
    for first, second in ((24, 48), (48, 72), (24, 72)):
        indicator = (prediction[first] != prediction[second]).astype(float)
        tv = 0.5 * np.abs(probability[first] - probability[second]).sum(1)
        result["flip"][f"{first}_{second}"] = {
            "estimate": round(float(indicator.mean()), 4),
            "ci95": [round(value, 4) for value in bootstrap_rate(indicator, clusters)],
        }
        result["total_variation"][f"{first}_{second}"] = {
            "estimate": round(float(tv.mean()), 4),
            "ci95": [round(value, 4) for value in bootstrap_mean(tv, clusters)],
        }
    revert = ((prediction[24] == prediction[72]) & (prediction[24] != prediction[48])).astype(float)
    result["return_to_24h_class"] = {
        "estimate": round(float(revert.mean()), 4),
        "ci95": [round(value, 4) for value in bootstrap_rate(revert, clusters)],
    }
    result["direction_24_to_72"] = {
        "more_dependent": {
            "estimate": round(float((prediction[72] > prediction[24]).mean()), 4),
            "ci95": [
                round(value, 4)
                for value in bootstrap_rate((prediction[72] > prediction[24]).astype(float), clusters)
            ],
        },
        "less_dependent": {
            "estimate": round(float((prediction[72] < prediction[24]).mean()), 4),
            "ci95": [
                round(value, 4)
                for value in bootstrap_rate((prediction[72] < prediction[24]).astype(float), clusters)
            ],
        },
    }
    confidence = probability[24].max(1)
    for threshold in (0.5, 0.6, 0.7):
        selected = confidence >= threshold
        indicator = (prediction[24] != prediction[72]).astype(float)
        result["confidence_24h"][str(threshold)] = {
            "coverage": round(float(selected.mean()), 4),
            "n": int(selected.sum()),
            "flip": round(float(indicator[selected].mean()), 4) if selected.any() else None,
            "flip_ci95": [
                round(value, 4)
                for value in bootstrap_rate(indicator[selected], clusters[selected])
            ]
            if selected.sum() >= 10
            else None,
        }
    transition = np.zeros((3, 3), dtype=int)
    for first, second in zip(prediction[24], prediction[72]):
        transition[int(first), int(second)] += 1
    result["transition_24_to_72"] = transition.tolist()
    return result


def main() -> None:
    results = {database: stability(database) for database in ("mimic4", "eicu")}
    with (OUT / "primary_stability.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)
    print(json.dumps(results, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
