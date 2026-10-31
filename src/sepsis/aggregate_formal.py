# -*- coding: utf-8 -*-
"""Aggregate formal DG-GRU runs, calibration, decisions, and paired contrasts."""

import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score


ROOT = Path(os.environ.get("SEPSIS_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
FORMAL = ROOT / "results" / "formal"
OUT = ROOT / "results" / "formal_summary"
OUT.mkdir(parents=True, exist_ok=True)
CONFIGURATIONS = (
    "ordinal_notemporal",
    "ordinal_temporal",
    "nominal_notemporal",
    "nominal_temporal",
)
SEEDS = (1, 2, 3)
LANDMARKS = (24, 48, 72)
RNG = np.random.default_rng(20261029)


def mean_sd(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)
    return {
        "mean": round(float(array.mean()), 4),
        "sd": round(float(array.std(ddof=1)), 4),
        "values": [round(float(value), 4) for value in array],
    }


def top_label_calibration(y: np.ndarray, probability: np.ndarray) -> dict:
    prediction = probability.argmax(1)
    confidence = probability.max(1)
    correct = (prediction == y).astype(float)
    edges = np.linspace(0, 1, 11)
    bins = []
    ece = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        selected = (
            (confidence >= lower) & (confidence <= upper)
            if upper == 1
            else (confidence >= lower) & (confidence < upper)
        )
        if selected.any():
            observed = float(correct[selected].mean())
            predicted = float(confidence[selected].mean())
            weight = float(selected.mean())
            ece += abs(observed - predicted) * weight
            bins.append(
                {
                    "lower": round(float(lower), 2),
                    "upper": round(float(upper), 2),
                    "n": int(selected.sum()),
                    "predicted": round(predicted, 4),
                    "observed": round(observed, 4),
                }
            )
    return {"ece_10bin": round(float(ece), 4), "bins": bins}


def class_calibration(y: np.ndarray, probability: np.ndarray) -> dict:
    output = {}
    for class_index in range(probability.shape[1]):
        observed = (y == class_index).astype(int)
        predicted = np.clip(probability[:, class_index], 1e-6, 1 - 1e-6)
        logit = np.log(predicted / (1 - predicted)).reshape(-1, 1)
        model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        model.fit(logit, observed)
        output[str(class_index)] = {
            "slope": round(float(model.coef_[0, 0]), 4),
            "intercept": round(float(model.intercept_[0]), 4),
            "prevalence": round(float(observed.mean()), 4),
        }
    return output


def multiclass_brier(y: np.ndarray, probability: np.ndarray) -> float:
    observed = np.eye(probability.shape[1])[y]
    return float(np.mean(np.sum((probability - observed) ** 2, axis=1)))


def decision_curve(y: np.ndarray, probability: np.ndarray, class_index: int) -> list[dict]:
    observed = y == class_index
    prevalence = float(observed.mean())
    points = []
    for threshold in np.arange(0.05, 0.61, 0.05):
        selected = probability[:, class_index] >= threshold
        true_positive = int((selected & observed).sum())
        false_positive = int((selected & ~observed).sum())
        net_benefit = (
            true_positive / len(y)
            - false_positive / len(y) * threshold / (1 - threshold)
        )
        treat_all = prevalence - (1 - prevalence) * threshold / (1 - threshold)
        points.append(
            {
                "threshold": round(float(threshold), 2),
                "model": round(float(net_benefit), 4),
                "treat_all": round(float(treat_all), 4),
                "treat_none": 0.0,
                "selected_fraction": round(float(selected.mean()), 4),
            }
        )
    return points


def clustered_bootstrap(
    indicator: np.ndarray, cluster_ids: np.ndarray, replicates: int = 2000
) -> list[float]:
    clusters, inverse = np.unique(cluster_ids, return_inverse=True)
    total = np.bincount(
        inverse, weights=indicator.astype(float), minlength=len(clusters)
    )
    count = np.bincount(inverse, minlength=len(clusters)).astype(float)
    estimates = np.empty(replicates)
    for index in range(replicates):
        selected = RNG.integers(0, len(clusters), size=len(clusters))
        denominator = count[selected].sum()
        estimates[index] = total[selected].sum() / denominator if denominator else np.nan
    estimates = estimates[np.isfinite(estimates)]
    return [
        round(float(np.quantile(estimates, 0.025)), 4),
        round(float(np.quantile(estimates, 0.975)), 4),
    ]


def clustered_bootstrap_mean(
    values: np.ndarray, cluster_ids: np.ndarray, replicates: int = 2000
) -> list[float]:
    clusters, inverse = np.unique(cluster_ids, return_inverse=True)
    total = np.bincount(inverse, weights=values, minlength=len(clusters))
    count = np.bincount(inverse, minlength=len(clusters)).astype(float)
    estimates = np.empty(replicates)
    for index in range(replicates):
        selected = RNG.integers(0, len(clusters), size=len(clusters))
        denominator = count[selected].sum()
        estimates[index] = total[selected].sum() / denominator if denominator else np.nan
    estimates = estimates[np.isfinite(estimates)]
    return [
        round(float(np.quantile(estimates, 0.025)), 4),
        round(float(np.quantile(estimates, 0.975)), 4),
    ]


def stability(
    probability: np.ndarray, y: np.ndarray, cluster_ids: np.ndarray
) -> dict:
    prediction = probability.argmax(2)
    transition = np.zeros((3, 3), dtype=int)
    for first, second in zip(prediction[:, 0], prediction[:, 2]):
        transition[int(first), int(second)] += 1
    confidence = probability[:, 0, :].max(1)
    result = {
        "n": int(len(y)),
        "transition_24_to_72": transition.tolist(),
        "return_to_24h_class": round(
            float(
                (
                    (prediction[:, 0] == prediction[:, 2])
                    & (prediction[:, 0] != prediction[:, 1])
                ).mean()
            ),
            4,
        ),
        "confidence_24h": {},
    }
    for first, second in ((0, 1), (1, 2), (0, 2)):
        indicator = (prediction[:, first] != prediction[:, second]).astype(float)
        variation = 0.5 * np.abs(
            probability[:, first, :] - probability[:, second, :]
        ).sum(1)
        result[f"flip_{first}_{second}"] = round(float(indicator.mean()), 4)
        result[f"flip_{first}_{second}_ci95"] = clustered_bootstrap(
            indicator, cluster_ids
        )
        result[f"tv_{first}_{second}"] = round(float(variation.mean()), 4)
        result[f"tv_{first}_{second}_ci95"] = clustered_bootstrap_mean(
            variation, cluster_ids
        )
    for threshold in (0.5, 0.6, 0.7):
        selected = confidence >= threshold
        result["confidence_24h"][str(threshold)] = {
            "coverage": round(float(selected.mean()), 4),
            "n": int(selected.sum()),
            "flip": round(
                float(
                    (
                        prediction[selected, 0] != prediction[selected, 2]
                    ).mean()
                ),
                4,
            )
            if selected.any()
            else None,
            "flip_ci95": clustered_bootstrap(
                (prediction[selected, 0] != prediction[selected, 2]).astype(float),
                cluster_ids[selected],
            )
            if selected.any()
            else None,
        }
    return result


def cluster_ids_for(database: str, ids: np.ndarray) -> np.ndarray:
    if database == "mimic4":
        frame = pd.read_parquet(
            ROOT / "data" / "processed" / "mimic4" / "design_all.parquet",
            columns=["stay_id", "subject_id"],
        )
        lookup = dict(
            zip(frame["stay_id"].astype(int), frame["subject_id"].astype(str))
        )
    else:
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
    return np.asarray([lookup[int(identifier)] for identifier in ids])


def load_runs(database: str) -> tuple[dict, dict]:
    details = {}
    probabilities = {}
    for configuration in CONFIGURATIONS:
        details[configuration] = []
        probability_by_seed = []
        reference_ids = None
        reference_y = None
        for seed in SEEDS:
            stem = FORMAL / database / f"{configuration}_s{seed}"
            with (stem.with_suffix(".json")).open(encoding="utf-8") as handle:
                details[configuration].append(json.load(handle))
            bundle = np.load(stem.with_suffix(".npz"))
            probability = bundle["probability"]
            ids = bundle["ids"]
            y = bundle["y"]
            if reference_ids is None:
                reference_ids = ids
                reference_y = y
            elif not np.array_equal(reference_ids, ids):
                raise ValueError(f"{database}: seed row order differs")
            elif not np.array_equal(reference_y, y):
                raise ValueError(f"{database}: seed outcome order differs")
            probability_by_seed.append(probability)
        probabilities[configuration] = {
            "ids": reference_ids,
            "y": reference_y,
            "probability": np.mean(probability_by_seed, axis=0),
            "seed_probability": np.stack(probability_by_seed, axis=0),
        }
    return details, probabilities


def summarize_configuration(database: str, configuration: str, details: list[dict]) -> dict:
    metrics = [run["metrics"] for run in details]
    output = {"seed_metrics": metrics, "aggregate": {}}
    for landmark in LANDMARKS:
        key = str(landmark)
        output["aggregate"][f"auc_{key}"] = mean_sd(
            [metric["auc"][key] for metric in metrics]
        )
        output["aggregate"][f"accuracy_{key}"] = mean_sd(
            [metric["accuracy"][key] for metric in metrics]
        )
        output["aggregate"][f"log_loss_{key}"] = mean_sd(
            [metric["log_loss"][key] for metric in metrics]
        )
    for metric in (
        "flip_0_2",
        "tv_0_2",
        "flip_0_1",
        "flip_1_2",
        "return_to_24h_class",
    ):
        output["aggregate"][metric] = mean_sd([item[metric] for item in metrics])
    return output


def paired_contrast(details: dict, first: str, second: str) -> dict:
    output = {}
    for metric in ("auc_72", "auc_24", "flip_0_2", "tv_0_2"):
        if metric.startswith("auc_"):
            key = metric.split("_")[1]
            first_values = [run["metrics"]["auc"][key] for run in details[first]]
            second_values = [run["metrics"]["auc"][key] for run in details[second]]
        else:
            first_values = [run["metrics"][metric] for run in details[first]]
            second_values = [run["metrics"][metric] for run in details[second]]
        difference = np.asarray(first_values) - np.asarray(second_values)
        output[metric] = {
            "difference_mean": round(float(difference.mean()), 4),
            "difference_sd": round(float(difference.std(ddof=1)), 4),
            "difference_values": [round(float(value), 4) for value in difference],
        }
    return output


def database_summary(database: str) -> dict:
    details, probabilities = load_runs(database)
    result = {
        "database": database,
        "configurations": {
            configuration: summarize_configuration(
                database, configuration, details[configuration]
            )
            for configuration in CONFIGURATIONS
        },
        "paired_contrasts": {
            "temporal_vs_none_ordinal": paired_contrast(
                details, "ordinal_temporal", "ordinal_notemporal"
            ),
            "temporal_vs_none_nominal": paired_contrast(
                details, "nominal_temporal", "nominal_notemporal"
            ),
            "ordinal_vs_nominal_no_temporal": paired_contrast(
                details, "ordinal_notemporal", "nominal_notemporal"
            ),
            "ordinal_vs_nominal_temporal": paired_contrast(
                details, "ordinal_temporal", "nominal_temporal"
            ),
        },
        "ensemble": {},
    }
    for configuration in CONFIGURATIONS:
        bundle = probabilities[configuration]
        y = bundle["y"]
        probability = bundle["probability"]
        cluster_ids = cluster_ids_for(database, bundle["ids"])
        np.savez_compressed(
            OUT / f"{database}_{configuration}_ensemble.npz",
            probability=probability,
            y=y,
            ids=bundle["ids"],
        )
        result["ensemble"][configuration] = {
            "metrics": {
                "macro_auc": {
                    str(milestone): round(
                        float(
                            roc_auc_score(
                                y,
                                probability[:, index, :],
                                multi_class="ovr",
                                average="macro",
                                labels=np.arange(3),
                            )
                        ),
                        4,
                    )
                    for index, milestone in enumerate(LANDMARKS)
                },
                "log_loss": {
                    str(milestone): round(
                        float(
                            log_loss(
                                y,
                                probability[:, index, :],
                                labels=np.arange(3),
                            )
                        ),
                        4,
                    )
                    for index, milestone in enumerate(LANDMARKS)
                },
                "brier": {
                    str(milestone): round(
                        multiclass_brier(y, probability[:, index, :]), 4
                    )
                    for index, milestone in enumerate(LANDMARKS)
                },
                "accuracy": {
                    str(milestone): round(
                        float(
                            (
                                probability[:, index, :].argmax(1) == y
                            ).mean()
                        ),
                        4,
                    )
                    for index, milestone in enumerate(LANDMARKS)
                },
            },
            "stability": stability(probability, y, cluster_ids),
            "calibration": {},
            "decision_curve": {},
        }
        for index, milestone in enumerate(LANDMARKS):
            result["ensemble"][configuration]["calibration"][str(milestone)] = {
                "top_label": top_label_calibration(y, probability[:, index, :]),
                "one_vs_rest": class_calibration(y, probability[:, index, :]),
            }
            result["ensemble"][configuration]["decision_curve"][str(milestone)] = {
                "home": decision_curve(y, probability[:, index, :], 0),
                "death": decision_curve(y, probability[:, index, :], 2),
            }
    return result


def main() -> None:
    summaries = {database: database_summary(database) for database in ("mimic4", "eicu")}
    with (OUT / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summaries, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summaries, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
