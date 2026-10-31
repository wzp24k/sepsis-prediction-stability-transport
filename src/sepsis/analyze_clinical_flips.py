# -*- coding: utf-8 -*-
"""Clinical-direction and correctness audit of 24-to-72-hour prediction flips.

The analysis is conditional on the already fitted out-of-fold predictions.
MIMIC-IV evaluation stays are resampled by subject and eICU-CRD stays by
hospital. No model is retrained inside the bootstrap.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
SHARED_FOLD = RESULTS / "shared_fold"
FORMAL_SUMMARY = RESULTS / "formal_summary"
OUTPUT_DIR = RESULTS / "clinical_flips"

CLASS_NAMES = ("home", "facility", "death")
CLASS_CODES = ("H", "F", "D")
MODEL_LABELS = {
    "xgboost": "XGBoost shared-fold OOF",
    "ordinal_notemporal": "Ordinal DG-GRU, no temporal penalty",
    "ordinal_temporal": "Ordinal DG-GRU, temporal penalty",
}
DATABASE_LABELS = {
    "mimic4": "MIMIC-IV",
    "eicu": "eICU-CRD",
}
TRANSITIONS = (
    (0, 1),
    (0, 2),
    (1, 0),
    (1, 2),
    (2, 0),
    (2, 1),
)
TRANSITION_NAMES = {
    (0, 1): "home_to_facility",
    (0, 2): "home_to_death",
    (1, 0): "facility_to_home",
    (1, 2): "facility_to_death",
    (2, 0): "death_to_home",
    (2, 1): "death_to_facility",
}


def rounded(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    bundle = np.load(path, allow_pickle=False)
    return {key: bundle[key] for key in bundle.files}


def align_bundle(
    bundle: dict[str, np.ndarray],
    ids: np.ndarray,
) -> dict[str, np.ndarray]:
    source_ids = bundle["ids"].astype(np.int64)
    if len(np.unique(source_ids)) != len(source_ids):
        raise ValueError("Model artifact contains duplicate stay identifiers")
    lookup = {int(identifier): index for index, identifier in enumerate(source_ids)}
    missing = [int(identifier) for identifier in ids if int(identifier) not in lookup]
    if missing:
        raise ValueError(f"Aligned cohort is missing {len(missing)} identifiers")
    index = np.asarray([lookup[int(identifier)] for identifier in ids], dtype=int)
    output = {}
    for key, value in bundle.items():
        output[key] = value[index]
    return output


def load_database(database: str) -> dict:
    xgb = load_npz(SHARED_FOLD / f"{database}_shared_fold.npz")
    ordinal_notemporal = load_npz(
        FORMAL_SUMMARY / f"{database}_ordinal_notemporal_ensemble.npz"
    )
    ordinal_temporal = load_npz(
        FORMAL_SUMMARY / f"{database}_ordinal_temporal_ensemble.npz"
    )
    sources = {
        "xgboost_shared_fold": xgb,
        "ordinal_notemporal": ordinal_notemporal,
        "ordinal_temporal": ordinal_temporal,
    }
    common = None
    for source in sources.values():
        source_ids = set(source["ids"].astype(np.int64).tolist())
        common = source_ids if common is None else common.intersection(source_ids)
    if not common:
        raise ValueError(f"{database}: no common complete-landmark cohort")
    ids = np.asarray(sorted(common), dtype=np.int64)

    aligned = {
        name: align_bundle(source, ids) for name, source in sources.items()
    }
    y = aligned["xgboost_shared_fold"]["y3"].astype(int)
    if not np.isin(y, np.arange(3)).all():
        raise ValueError(f"{database}: outcome outside the three-class coding")
    for name, source in aligned.items():
        outcome = source["y3"] if "y3" in source else source["y"]
        if not np.array_equal(outcome.astype(int), y):
            raise ValueError(f"{database}: outcome mismatch in {name}")

    models = {
        "xgboost": {
            "p24": aligned["xgboost_shared_fold"]["p24"].astype(float),
            "p72": aligned["xgboost_shared_fold"]["p72"].astype(float),
        },
        "ordinal_notemporal": {
            "p24": aligned["ordinal_notemporal"]["probability"][:, 0, :].astype(float),
            "p72": aligned["ordinal_notemporal"]["probability"][:, 2, :].astype(float),
        },
        "ordinal_temporal": {
            "p24": aligned["ordinal_temporal"]["probability"][:, 0, :].astype(float),
            "p72": aligned["ordinal_temporal"]["probability"][:, 2, :].astype(float),
        },
    }
    for name, model in models.items():
        for key in ("p24", "p72"):
            probability = model[key]
            if probability.shape != (len(y), 3):
                raise ValueError(
                    f"{database}/{name}/{key}: expected {(len(y), 3)}, "
                    f"found {probability.shape}"
                )
            if not np.isfinite(probability).all():
                raise ValueError(f"{database}/{name}/{key}: non-finite probability")
            if not np.allclose(probability.sum(1), 1.0, atol=1e-5):
                raise ValueError(f"{database}/{name}/{key}: rows do not sum to one")

    return {
        "ids": ids,
        "y": y,
        "models": models,
        "clusters": aligned["xgboost_shared_fold"]["clusters"].astype(str),
        "fold_id": aligned["xgboost_shared_fold"]["fold_id"].astype(int),
        "source_sizes": {
            name: int(len(source["ids"])) for name, source in sources.items()
        },
        "provenance": {
            "xgboost": (
                "results/shared_fold/{database}_shared_fold.npz: one fixed "
                "patient-grouped 5-fold partition shared by all landmarks"
            ),
            "dg_gru": (
                "results/formal_summary/{database}_ordinal_*_ensemble.npz: "
                "three-seed probability ensembles"
            ),
        },
    }


class ClusterBootstrap:
    """Common cluster resamples for all ratios in one database."""

    def __init__(self, clusters: np.ndarray, replicates: int, seed: int) -> None:
        unique, inverse = np.unique(clusters, return_inverse=True)
        self.unique = unique
        self.inverse = inverse
        self.replicates = replicates
        self.samples = np.random.default_rng(seed).integers(
            0, len(unique), size=(replicates, len(unique))
        )

    def cluster_total(self, values: np.ndarray) -> np.ndarray:
        return np.bincount(
            self.inverse,
            weights=np.asarray(values, dtype=float),
            minlength=len(self.unique),
        )


class MetricBatch:
    """Collect numerator/denominator pairs, then bootstrap all of them at once."""

    def __init__(self, bootstrap: ClusterBootstrap) -> None:
        self.bootstrap = bootstrap
        self.specs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.output: dict[str, dict] = {}

    def add_rate(
        self,
        key: str,
        numerator: np.ndarray,
        denominator: np.ndarray | None = None,
    ) -> None:
        numerator = np.asarray(numerator, dtype=float)
        if denominator is None:
            denominator = np.ones(len(numerator), dtype=float)
        denominator = np.asarray(denominator, dtype=float)
        if numerator.shape != denominator.shape:
            raise ValueError(f"{key}: numerator and denominator differ in shape")
        # All ratios in this module are defined within a denominator domain
        # (for example, only true deaths or only patients who changed class).
        # Zero out contributions from rows outside that domain.
        numerator = np.where(denominator != 0, numerator, 0.0)
        self.specs[key] = (numerator, denominator)

    def add_mean(
        self,
        key: str,
        values: np.ndarray,
        denominator: np.ndarray | None = None,
    ) -> None:
        values = np.asarray(values, dtype=float)
        if denominator is None:
            denominator = np.ones(len(values), dtype=float)
        self.add_rate(key, values, denominator)

    def compute(self) -> dict[str, dict]:
        if not self.specs:
            return {}
        keys = list(self.specs)
        numerator_cluster = np.vstack(
            [
                self.bootstrap.cluster_total(self.specs[key][0])
                for key in keys
            ]
        )
        denominator_cluster = np.vstack(
            [
                self.bootstrap.cluster_total(self.specs[key][1])
                for key in keys
            ]
        )
        point_numerator = numerator_cluster.sum(1)
        point_denominator = denominator_cluster.sum(1)
        point = np.divide(
            point_numerator,
            point_denominator,
            out=np.full_like(point_numerator, np.nan),
            where=point_denominator != 0,
        )

        n_clusters = len(self.bootstrap.unique)
        cluster_count = np.zeros(
            (self.bootstrap.replicates, n_clusters), dtype=np.float32
        )
        for replicate, selected in enumerate(self.bootstrap.samples):
            cluster_count[replicate] = np.bincount(
                selected, minlength=n_clusters
            ).astype(np.float32)
        boot_numerator = cluster_count @ numerator_cluster.T.astype(np.float32)
        boot_denominator = cluster_count @ denominator_cluster.T.astype(np.float32)
        boot = np.full_like(boot_numerator, np.nan, dtype=float)
        np.divide(
            boot_numerator,
            boot_denominator,
            out=boot,
            where=boot_denominator != 0,
        )

        output = {}
        for index, key in enumerate(keys):
            lower, upper = np.nanquantile(boot[:, index], [0.025, 0.975])
            output[key] = {
                "estimate": rounded(point[index]),
                "ci95": [rounded(lower), rounded(upper)],
            }
        self.output = output
        return output

    def __getitem__(self, key: str) -> dict:
        if not self.output:
            raise RuntimeError("MetricBatch.compute() must be called first")
        return self.output[key]


def rate_record(
    batch: MetricBatch,
    key: str,
    numerator: np.ndarray,
    denominator: np.ndarray | None = None,
) -> dict:
    numerator_array = np.asarray(numerator)
    if denominator is None:
        denominator_n = len(numerator)
        selected_numerator = numerator_array
    else:
        denominator_array = np.asarray(denominator)
        denominator_n = int(denominator_array.sum())
        selected_numerator = numerator_array[denominator_array != 0]
    record = dict(batch[key])
    if np.all((selected_numerator == 0) | (selected_numerator == 1)):
        record["n"] = int(selected_numerator.sum())
    record["denominator_n"] = denominator_n
    return record


def mean_record(batch: MetricBatch, key: str, n: int) -> dict:
    record = dict(batch[key])
    record["n"] = int(n)
    return record


def model_arrays(y: np.ndarray, model: dict) -> dict:
    pred24 = model["p24"].argmax(1)
    pred72 = model["p72"].argmax(1)
    distance24 = np.abs(pred24 - y)
    distance72 = np.abs(pred72 - y)
    flip = pred24 != pred72
    upgrade = pred72 > pred24
    downgrade = pred72 < pred24
    gain = distance72 < distance24
    return {
        "pred24": pred24,
        "pred72": pred72,
        "distance24": distance24,
        "distance72": distance72,
        "flip": flip,
        "upgrade": upgrade,
        "downgrade": downgrade,
        "gain": gain,
        "loss": distance72 > distance24,
        "tie": distance72 == distance24,
        "correct_upgrade": upgrade & gain,
        "incorrect_upgrade": upgrade & ~gain,
        "correct_downgrade": downgrade & gain,
        "incorrect_downgrade": downgrade & ~gain,
        "tie_upgrade": upgrade & (distance72 == distance24),
        "tie_downgrade": downgrade & (distance72 == distance24),
        "pt_true_difference": (
            model["p72"][np.arange(len(y)), y]
            - model["p24"][np.arange(len(y)), y]
        ),
    }


def add_model_metrics(
    batch: MetricBatch,
    database: str,
    model_name: str,
    y: np.ndarray,
    model: dict,
    arrays: dict,
) -> None:
    prefix = f"{database}.{model_name}"
    one = np.ones(len(y), dtype=float)
    flip = arrays["flip"]
    batch.add_rate(f"{prefix}.flip_rate", flip, one)
    for source, destination in TRANSITIONS:
        transition = (
            (arrays["pred24"] == source) & (arrays["pred72"] == destination)
        )
        name = TRANSITION_NAMES[(source, destination)]
        batch.add_rate(f"{prefix}.transition.{name}.rate", transition, one)
        batch.add_rate(
            f"{prefix}.transition.{name}.share_of_flips", transition, flip
        )

    clinical = {
        "to_death": (arrays["pred24"] < 2) & (arrays["pred72"] == 2),
        "home_to_facility": (
            (arrays["pred24"] == 0) & (arrays["pred72"] == 1)
        ),
        "any_upgrade": arrays["upgrade"],
    }
    for name, indicator in clinical.items():
        batch.add_rate(f"{prefix}.clinical.{name}", indicator, one)

    for name in (
        "gain",
        "loss",
        "tie",
        "correct_upgrade",
        "incorrect_upgrade",
        "correct_downgrade",
        "incorrect_downgrade",
        "tie_upgrade",
        "tie_downgrade",
    ):
        batch.add_rate(f"{prefix}.correctness.all.{name}", arrays[name], one)
        batch.add_rate(f"{prefix}.correctness.flips.{name}", arrays[name], flip)

    batch.add_mean(
        f"{prefix}.correctness.all.distance24_mean", arrays["distance24"]
    )
    batch.add_mean(
        f"{prefix}.correctness.all.distance72_mean", arrays["distance72"]
    )
    batch.add_mean(
        f"{prefix}.correctness.all.distance_change",
        arrays["distance72"] - arrays["distance24"],
    )
    batch.add_mean(
        f"{prefix}.probability.all.pt_true24_mean",
        model["p24"][np.arange(len(y)), y],
    )
    batch.add_mean(
        f"{prefix}.probability.all.pt_true72_mean",
        model["p72"][np.arange(len(y)), y],
    )
    probability_change = arrays["pt_true_difference"]
    batch.add_mean(
        f"{prefix}.probability.all.pt_true_change", probability_change
    )
    batch.add_rate(
        f"{prefix}.probability.all.improved",
        probability_change > 1e-12,
        one,
    )
    batch.add_rate(
        f"{prefix}.probability.all.worsened",
        probability_change < -1e-12,
        one,
    )
    batch.add_rate(
        f"{prefix}.probability.all.unchanged",
        np.abs(probability_change) <= 1e-12,
        one,
    )
    batch.add_mean(
        f"{prefix}.probability.flips.pt_true_change",
        probability_change,
        flip,
    )
    batch.add_rate(
        f"{prefix}.probability.flips.improved",
        probability_change > 1e-12,
        flip,
    )

    death = y == 2
    death_metrics = {
        "already_death_at24": arrays["pred24"] == 2,
        "detected_death_at72": arrays["pred72"] == 2,
        "missed_death_at72": arrays["pred72"] < 2,
        "new_upgrade_to_death": (arrays["pred24"] < 2) & (arrays["pred72"] == 2),
        "any_upgrade": arrays["upgrade"],
        "retained_death": (arrays["pred24"] == 2) & (arrays["pred72"] == 2),
        "lost_detected_death": (arrays["pred24"] == 2) & (arrays["pred72"] < 2),
    }
    for name, indicator in death_metrics.items():
        batch.add_rate(f"{prefix}.death_subgroup.{name}", indicator, death)


def add_temporal_contrasts(
    batch: MetricBatch,
    database: str,
    y: np.ndarray,
    notemporal_arrays: dict,
    temporal_arrays: dict,
) -> None:
    prefix = f"{database}.temporal_vs_notemporal"
    one = np.ones(len(y), dtype=float)
    notemporal_flip = notemporal_arrays["flip"]
    temporal_flip = temporal_arrays["flip"]
    batch.add_rate(f"{prefix}.flip_rate_notemporal", notemporal_flip, one)
    batch.add_rate(f"{prefix}.flip_rate_temporal", temporal_flip, one)
    batch.add_rate(
        f"{prefix}.flip_rate_change_paired",
        temporal_flip.astype(float) - notemporal_flip.astype(float),
        one,
    )
    for name in (
        "correct_upgrade",
        "incorrect_upgrade",
        "correct_downgrade",
        "incorrect_downgrade",
        "tie_upgrade",
        "tie_downgrade",
    ):
        batch.add_rate(
            f"{prefix}.category.{name}.notemporal",
            notemporal_arrays[name],
            one,
        )
        batch.add_rate(
            f"{prefix}.category.{name}.temporal",
            temporal_arrays[name],
            one,
        )
        batch.add_rate(
            f"{prefix}.category.{name}.change",
            temporal_arrays[name].astype(float)
            - notemporal_arrays[name].astype(float),
            one,
        )

    death = y == 2
    for name in (
        "already_death_at24",
        "detected_death_at72",
        "missed_death_at72",
        "new_upgrade_to_death",
        "any_upgrade",
        "retained_death",
        "lost_detected_death",
    ):
        if name == "already_death_at24":
            notemporal_indicator = notemporal_arrays["pred24"] == 2
            temporal_indicator = temporal_arrays["pred24"] == 2
        elif name == "detected_death_at72":
            notemporal_indicator = notemporal_arrays["pred72"] == 2
            temporal_indicator = temporal_arrays["pred72"] == 2
        elif name == "missed_death_at72":
            notemporal_indicator = notemporal_arrays["pred72"] < 2
            temporal_indicator = temporal_arrays["pred72"] < 2
        elif name == "new_upgrade_to_death":
            notemporal_indicator = (
                (notemporal_arrays["pred24"] < 2)
                & (notemporal_arrays["pred72"] == 2)
            )
            temporal_indicator = (
                (temporal_arrays["pred24"] < 2)
                & (temporal_arrays["pred72"] == 2)
            )
        elif name == "any_upgrade":
            notemporal_indicator = notemporal_arrays["upgrade"]
            temporal_indicator = temporal_arrays["upgrade"]
        elif name == "retained_death":
            notemporal_indicator = (
                (notemporal_arrays["pred24"] == 2)
                & (notemporal_arrays["pred72"] == 2)
            )
            temporal_indicator = (
                (temporal_arrays["pred24"] == 2)
                & (temporal_arrays["pred72"] == 2)
            )
        else:
            notemporal_indicator = (
                (notemporal_arrays["pred24"] == 2)
                & (notemporal_arrays["pred72"] < 2)
            )
            temporal_indicator = (
                (temporal_arrays["pred24"] == 2)
                & (temporal_arrays["pred72"] < 2)
            )
        batch.add_rate(
            f"{prefix}.death_subgroup.{name}.notemporal",
            notemporal_indicator,
            death,
        )
        batch.add_rate(
            f"{prefix}.death_subgroup.{name}.temporal",
            temporal_indicator,
            death,
        )
        batch.add_rate(
            f"{prefix}.death_subgroup.{name}.change",
            temporal_indicator.astype(float) - notemporal_indicator.astype(float),
            death,
        )


def transition_matrix(arrays: dict) -> np.ndarray:
    matrix = np.zeros((3, 3), dtype=int)
    np.add.at(matrix, (arrays["pred24"], arrays["pred72"]), 1)
    return matrix


def model_output(
    batch: MetricBatch,
    database: str,
    model_name: str,
    y: np.ndarray,
    model: dict,
    arrays: dict,
) -> dict:
    prefix = f"{database}.{model_name}"
    one = np.ones(len(y), dtype=float)
    flip = arrays["flip"]
    matrix = transition_matrix(arrays)
    transitions = {}
    for source, destination in TRANSITIONS:
        name = TRANSITION_NAMES[(source, destination)]
        indicator = (
            (arrays["pred24"] == source) & (arrays["pred72"] == destination)
        )
        transitions[name] = {
            "from": CLASS_NAMES[source],
            "to": CLASS_NAMES[destination],
            "count": int(indicator.sum()),
            "cohort_rate": rate_record(
                batch, f"{prefix}.transition.{name}.rate", indicator, one
            ),
            "share_of_flips": rate_record(
                batch,
                f"{prefix}.transition.{name}.share_of_flips",
                indicator,
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
    clinical_output = {}
    for output_name, indicator in clinical.items():
        input_name = {
            "major_upgrade_to_death": "to_death",
            "home_to_facility": "home_to_facility",
            "any_upgrade": "any_upgrade",
        }[output_name]
        clinical_output[output_name] = rate_record(
            batch, f"{prefix}.clinical.{input_name}", indicator, one
        )

    correctness = {}
    for scope in ("all", "flips"):
        denominator = one if scope == "all" else flip
        scope_output = {}
        for name in (
            "gain",
            "loss",
            "tie",
            "correct_upgrade",
            "incorrect_upgrade",
            "correct_downgrade",
            "incorrect_downgrade",
            "tie_upgrade",
            "tie_downgrade",
        ):
            scope_output[name] = rate_record(
                batch,
                f"{prefix}.correctness.{scope}.{name}",
                arrays[name],
                denominator,
            )
        if scope == "all":
            scope_output["distance24_mean"] = mean_record(
                batch,
                f"{prefix}.correctness.all.distance24_mean",
                len(y),
            )
            scope_output["distance72_mean"] = mean_record(
                batch,
                f"{prefix}.correctness.all.distance72_mean",
                len(y),
            )
            scope_output["distance_change_72_minus_24_mean"] = mean_record(
                batch,
                f"{prefix}.correctness.all.distance_change",
                len(y),
            )
        scope_output["pt_true_change_mean"] = (
            mean_record(
                batch,
                f"{prefix}.probability.all.pt_true_change",
                len(y),
            )
            if scope == "all"
            else mean_record(
                batch,
                f"{prefix}.probability.flips.pt_true_change",
                int(flip.sum()),
            )
        )
        correctness[scope] = scope_output

    probability = {
        "pt_true24_mean": mean_record(
            batch, f"{prefix}.probability.all.pt_true24_mean", len(y)
        ),
        "pt_true72_mean": mean_record(
            batch, f"{prefix}.probability.all.pt_true72_mean", len(y)
        ),
        "pt_true_change_mean": correctness["all"]["pt_true_change_mean"],
        "improved": rate_record(
            batch,
            f"{prefix}.probability.all.improved",
            arrays["pt_true_difference"] > 1e-12,
            one,
        ),
        "worsened": rate_record(
            batch,
            f"{prefix}.probability.all.worsened",
            arrays["pt_true_difference"] < -1e-12,
            one,
        ),
        "unchanged": rate_record(
            batch,
            f"{prefix}.probability.all.unchanged",
            np.abs(arrays["pt_true_difference"]) <= 1e-12,
            one,
        ),
        "among_flips": {
            "pt_true_change_mean": correctness["flips"]["pt_true_change_mean"],
            "improved": rate_record(
                batch,
                f"{prefix}.probability.flips.improved",
                arrays["pt_true_difference"] > 1e-12,
                flip,
            ),
        },
    }
    death = y == 2
    death_counts = {
        "pred24": np.bincount(
            arrays["pred24"][death], minlength=3
        ).tolist(),
        "pred72": np.bincount(
            arrays["pred72"][death], minlength=3
        ).tolist(),
    }
    death_metrics = {
        "already_death_at24": arrays["pred24"] == 2,
        "detected_death_at72": arrays["pred72"] == 2,
        "missed_death_at72": arrays["pred72"] < 2,
        "new_upgrade_to_death": (
            (arrays["pred24"] < 2) & (arrays["pred72"] == 2)
        ),
        "any_upgrade": arrays["upgrade"],
        "retained_death": (
            (arrays["pred24"] == 2) & (arrays["pred72"] == 2)
        ),
        "lost_detected_death": (
            (arrays["pred24"] == 2) & (arrays["pred72"] < 2)
        ),
    }
    death_output = {
        "n": int(death.sum()),
        "predicted_class_counts": death_counts,
        **{
            name: rate_record(
                batch,
                f"{prefix}.death_subgroup.{name}",
                indicator,
                death,
            )
            for name, indicator in death_metrics.items()
        },
    }

    return {
        "label": MODEL_LABELS[model_name],
        "flip_24_72": rate_record(batch, f"{prefix}.flip_rate", flip, one),
        "transition_matrix_rows_24h_columns_72h": matrix.tolist(),
        "transition_composition": transitions,
        "clinical_major_upgrades": clinical_output,
        "correctness_relative_to_final_outcome": correctness,
        "true_class_probability": probability,
        "true_death_subgroup": death_output,
    }


def temporal_contrast_output(
    batch: MetricBatch,
    database: str,
    y: np.ndarray,
    notemporal_arrays: dict,
    temporal_arrays: dict,
) -> dict:
    prefix = f"{database}.temporal_vs_notemporal"
    one = np.ones(len(y), dtype=float)
    notemporal_flip = notemporal_arrays["flip"]
    temporal_flip = temporal_arrays["flip"]
    category_output = {}
    for name in (
        "correct_upgrade",
        "incorrect_upgrade",
        "correct_downgrade",
        "incorrect_downgrade",
        "tie_upgrade",
        "tie_downgrade",
    ):
        category_output[name] = {
            "notemporal": rate_record(
                batch,
                f"{prefix}.category.{name}.notemporal",
                notemporal_arrays[name],
                one,
            ),
            "temporal": rate_record(
                batch,
                f"{prefix}.category.{name}.temporal",
                temporal_arrays[name],
                one,
            ),
            "change_temporal_minus_notemporal": rate_record(
                batch,
                f"{prefix}.category.{name}.change",
                temporal_arrays[name].astype(float)
                - notemporal_arrays[name].astype(float),
                one,
            ),
        }

    death = y == 2
    death_output = {}
    for name in (
        "already_death_at24",
        "detected_death_at72",
        "missed_death_at72",
        "new_upgrade_to_death",
        "any_upgrade",
        "retained_death",
        "lost_detected_death",
    ):
        if name == "already_death_at24":
            notemporal_indicator = notemporal_arrays["pred24"] == 2
            temporal_indicator = temporal_arrays["pred24"] == 2
        elif name == "detected_death_at72":
            notemporal_indicator = notemporal_arrays["pred72"] == 2
            temporal_indicator = temporal_arrays["pred72"] == 2
        elif name == "missed_death_at72":
            notemporal_indicator = notemporal_arrays["pred72"] < 2
            temporal_indicator = temporal_arrays["pred72"] < 2
        elif name == "new_upgrade_to_death":
            notemporal_indicator = (
                (notemporal_arrays["pred24"] < 2)
                & (notemporal_arrays["pred72"] == 2)
            )
            temporal_indicator = (
                (temporal_arrays["pred24"] < 2)
                & (temporal_arrays["pred72"] == 2)
            )
        elif name == "any_upgrade":
            notemporal_indicator = notemporal_arrays["upgrade"]
            temporal_indicator = temporal_arrays["upgrade"]
        elif name == "retained_death":
            notemporal_indicator = (
                (notemporal_arrays["pred24"] == 2)
                & (notemporal_arrays["pred72"] == 2)
            )
            temporal_indicator = (
                (temporal_arrays["pred24"] == 2)
                & (temporal_arrays["pred72"] == 2)
            )
        else:
            notemporal_indicator = (
                (notemporal_arrays["pred24"] == 2)
                & (notemporal_arrays["pred72"] < 2)
            )
            temporal_indicator = (
                (temporal_arrays["pred24"] == 2)
                & (temporal_arrays["pred72"] < 2)
            )
        death_output[name] = {
            "notemporal": rate_record(
                batch,
                f"{prefix}.death_subgroup.{name}.notemporal",
                notemporal_indicator,
                death,
            ),
            "temporal": rate_record(
                batch,
                f"{prefix}.death_subgroup.{name}.temporal",
                temporal_indicator,
                death,
            ),
            "change_temporal_minus_notemporal": rate_record(
                batch,
                f"{prefix}.death_subgroup.{name}.change",
                temporal_indicator.astype(float)
                - notemporal_indicator.astype(float),
                death,
            ),
        }

    flip_rate_change = rate_record(
        batch,
        f"{prefix}.flip_rate_change_paired",
        temporal_flip.astype(float) - notemporal_flip.astype(float),
        one,
    )
    category_sum_change = sum(
        category_output[name]["change_temporal_minus_notemporal"]["estimate"]
        for name in (
            "correct_upgrade",
            "incorrect_upgrade",
            "correct_downgrade",
            "incorrect_downgrade",
        )
    )
    if abs(category_sum_change - flip_rate_change["estimate"]) > 1e-5:
        raise AssertionError("Flip category decomposition does not sum to flip change")

    correct_upgrade = category_output["correct_upgrade"][
        "change_temporal_minus_notemporal"
    ]
    incorrect_upgrade = category_output["incorrect_upgrade"][
        "change_temporal_minus_notemporal"
    ]
    correct_downgrade = category_output["correct_downgrade"][
        "change_temporal_minus_notemporal"
    ]
    incorrect_downgrade = category_output["incorrect_downgrade"][
        "change_temporal_minus_notemporal"
    ]
    upgrade_reduction = -(
        correct_upgrade["estimate"] + incorrect_upgrade["estimate"]
    )
    correct_upgrade_share = (
        -correct_upgrade["estimate"] / upgrade_reduction
        if upgrade_reduction > 0
        else None
    )
    incorrect_upgrade_share = (
        -incorrect_upgrade["estimate"] / upgrade_reduction
        if upgrade_reduction > 0
        else None
    )

    new_death_upgrade = death_output["new_upgrade_to_death"][
        "change_temporal_minus_notemporal"
    ]
    missed_death = death_output["missed_death_at72"][
        "change_temporal_minus_notemporal"
    ]
    lost_death = death_output["lost_detected_death"][
        "change_temporal_minus_notemporal"
    ]
    suppression = suppression_assessment(
        new_death_upgrade, missed_death, lost_death
    )

    return {
        "direction": "temporal penalty minus no temporal penalty",
        "flip_rate_notemporal": rate_record(
            batch, f"{prefix}.flip_rate_notemporal", notemporal_flip, one
        ),
        "flip_rate_temporal": rate_record(
            batch, f"{prefix}.flip_rate_temporal", temporal_flip, one
        ),
        "flip_rate_change": flip_rate_change,
        "flip_category_decomposition": {
            **category_output,
            "sum_of_four_primary_category_changes": rounded(category_sum_change),
            "upgrades_reduced": rounded(upgrade_reduction),
            "correct_upgrade_share_of_upgrade_reduction": (
                rounded(correct_upgrade_share)
                if correct_upgrade_share is not None
                else None
            ),
            "incorrect_upgrade_share_of_upgrade_reduction": (
                rounded(incorrect_upgrade_share)
                if incorrect_upgrade_share is not None
                else None
            ),
        },
        "true_death_subgroup": {
            "n": int(death.sum()),
            **death_output,
        },
        "suppression_assessment": suppression,
    }


def suppression_assessment(
    new_death_upgrade: dict,
    missed_death: dict,
    lost_death: dict,
) -> dict:
    upgrade_estimate = new_death_upgrade["estimate"]
    missed_estimate = missed_death["estimate"]
    lost_estimate = lost_death["estimate"]
    adverse_direction = upgrade_estimate < 0 and missed_estimate > 0
    upgrade_ci_negative = new_death_upgrade["ci95"][1] < 0
    missed_ci_positive = missed_death["ci95"][0] > 0
    lost_ci_positive = lost_death["ci95"][0] > 0
    if adverse_direction and upgrade_ci_negative and missed_ci_positive:
        evidence = "supported"
        interpretation = (
            "Temporal regularization significantly reduced correct death "
            "upgrades and significantly increased missed death at 72 h."
        )
    elif adverse_direction:
        evidence = "directional_but_uncertain"
        interpretation = (
            "Point estimates are consistent with suppression of death "
            "recognition, but the paired interval does not establish both "
            "components."
        )
    else:
        evidence = "not_observed"
        interpretation = (
            "The aligned contrast does not show the combined pattern of fewer "
            "correct death upgrades and more missed deaths."
        )
    if lost_estimate > 0 and lost_ci_positive:
        evidence += "_with_significant_loss_of_detected_death"
    return {
        "evidence": evidence,
        "criteria": (
            "Directional suppression requires fewer new upgrades to death and "
            "more missed deaths at 72 h; simultaneous 95% CI exclusion of zero "
            "strengthens the assessment."
        ),
        "interpretation": interpretation,
        "new_upgrade_to_death_change": new_death_upgrade,
        "missed_death_at72_change": missed_death,
        "lost_detected_death_change": lost_death,
    }


def analyze_database(
    database: str,
    replicates: int,
    seed: int,
) -> dict:
    loaded = load_database(database)
    ids = loaded["ids"]
    y = loaded["y"]
    clusters = loaded["clusters"]
    bootstrap_unit = "subject_id" if database == "mimic4" else "hospitalid"
    bootstrap = ClusterBootstrap(clusters, replicates=replicates, seed=seed)
    batch = MetricBatch(bootstrap)
    arrays = {}
    for model_name, model in loaded["models"].items():
        arrays[model_name] = model_arrays(y, model)
        add_model_metrics(
            batch,
            database,
            model_name,
            y,
            model,
            arrays[model_name],
        )
    add_temporal_contrasts(
        batch,
        database,
        y,
        arrays["ordinal_notemporal"],
        arrays["ordinal_temporal"],
    )
    batch.compute()

    model_outputs = {
        model_name: model_output(
            batch,
            database,
            model_name,
            y,
            loaded["models"][model_name],
            arrays[model_name],
        )
        for model_name in loaded["models"]
    }
    contrast = temporal_contrast_output(
        batch,
        database,
        y,
        arrays["ordinal_notemporal"],
        arrays["ordinal_temporal"],
    )
    # Rate intervals for the flip baselines are made available with the same
    # cluster bootstrap as the paired change.
    unique_clusters, cluster_sizes = np.unique(clusters, return_counts=True)
    return {
        "label": DATABASE_LABELS[database],
        "n": int(len(y)),
        "bootstrap": {
            "unit": bootstrap_unit,
            "clusters": int(len(unique_clusters)),
            "replicates": replicates,
            "seed": seed,
            "cluster_size_median": float(np.median(cluster_sizes)),
            "cluster_size_max": int(cluster_sizes.max()),
        },
        "cohort": {
            "complete_landmark_ids": int(len(ids)),
            "shared_fold_count": int(len(np.unique(loaded["fold_id"]))),
            "outcome_counts": {
                CLASS_NAMES[index]: int((y == index).sum()) for index in range(3)
            },
            "source_sizes_before_alignment": loaded["source_sizes"],
            "prediction_provenance": {
                key: value.format(database=database)
                for key, value in loaded["provenance"].items()
            },
        },
        "models": model_outputs,
        "temporal_vs_notemporal": contrast,
    }


def pct(value: float, digits: int = 2) -> str:
    return f"{100.0 * value:.{digits}f}%"


def pct_ci(record: dict, digits: int = 2) -> str:
    return (
        f"{pct(record['estimate'], digits)} "
        f"({pct(record['ci95'][0], digits)} to "
        f"{pct(record['ci95'][1], digits)})"
    )


def pp_ci(record: dict, digits: int = 2) -> str:
    return (
        f"{100.0 * record['estimate']:+.{digits}f} pp "
        f"({100.0 * record['ci95'][0]:+.{digits}f} to "
        f"{100.0 * record['ci95'][1]:+.{digits}f})"
    )


def signed(record: dict, digits: int = 4) -> str:
    return (
        f"{record['estimate']:+.{digits}f} "
        f"({record['ci95'][0]:+.{digits}f} to "
        f"{record['ci95'][1]:+.{digits}f})"
    )


def build_report(summary: dict) -> str:
    lines = [
        "# Clinical Flip Audit: Direction, Correctness, and Temporal Penalization",
        "",
        "## Scope and Definitions",
        "",
        f"- Generated (UTC): {summary['metadata']['generated_at_utc']}",
        f"- Cluster bootstrap: {summary['metadata']['replicates']:,} replicates "
        "per database, conditional on the fitted out-of-fold predictions.",
        "- Class order: home (0) < facility (1) < death (2).",
        "- Complete cohort: the same stays with 24, 48, and 72 h XGBoost "
        "predictions on one fixed patient-grouped fold partition and matched "
        "DG-GRU ensemble predictions.",
        "- Primary clinical major upgrade: a 24 h prediction below death that "
        "changes to death at 72 h. Home-to-facility and any class upgrade are "
        "reported separately.",
        "- Correctness distance: `abs(predicted_class - final_class)`. Gain "
        "means a shorter distance, loss a longer distance, and tie no change.",
        "- Temporal contrast: temporal penalty minus no temporal penalty in "
        "the same patients. Intervals resample the evaluation sample; they do "
        "not represent retraining or fold-partition uncertainty.",
        "",
        "## Cohort",
        "",
        "| Database | N | home | facility | death | Bootstrap unit | Clusters |",
        "|---|---:|---:|---:|---:|---|---:|",
    ]
    for database in ("mimic4", "eicu"):
        record = summary["databases"][database]
        outcome = record["cohort"]["outcome_counts"]
        lines.append(
            f"| {record['label']} | {record['n']:,} | "
            f"{outcome['home']:,} | {outcome['facility']:,} | "
            f"{outcome['death']:,} | {record['bootstrap']['unit']} | "
            f"{record['bootstrap']['clusters']:,} |"
        )

    lines.extend(
        [
            "",
            "## XGBoost 24-to-72-Hour Transition Composition",
            "",
            "Rate uses the complete cohort as the denominator. Share uses all "
            "patients whose predicted class changed between 24 and 72 h.",
            "",
            "| Database | Transition | n | Cohort rate (95% CI) | Share of flips (95% CI) |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for database in ("mimic4", "eicu"):
        record = summary["databases"][database]
        model = record["models"]["xgboost"]
        for name, transition in model["transition_composition"].items():
            lines.append(
                f"| {record['label']} | {transition['from']}->"
                f"{transition['to']} | {transition['count']:,} | "
                f"{pct_ci(transition['cohort_rate'])} | "
                f"{pct_ci(transition['share_of_flips'])} |"
            )
        matrix = model["transition_matrix_rows_24h_columns_72h"]
        lines.extend(
            [
                f"| {record['label']} | 24->72 any flip | "
                f"{model['flip_24_72']['n']:,} | "
                f"{pct_ci(model['flip_24_72'])} | 100.00% |",
                "",
                f"{record['label']} transition matrix (rows = 24 h, "
                "columns = 72 h; H/F/D):",
                "",
                "```text",
                f"H {matrix[0][0]:>6} {matrix[0][1]:>6} {matrix[0][2]:>6}",
                f"F {matrix[1][0]:>6} {matrix[1][1]:>6} {matrix[1][2]:>6}",
                f"D {matrix[2][0]:>6} {matrix[2][1]:>6} {matrix[2][2]:>6}",
                "```",
                "",
            ]
        )

    lines.extend(
        [
            "## Clinical Major Upgrades",
            "",
            "| Database | Model | Any upgrade to death | home->facility | Any upgrade |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for database in ("mimic4", "eicu"):
        record = summary["databases"][database]
        for model_name, model in record["models"].items():
            clinical = model["clinical_major_upgrades"]
            lines.append(
                f"| {record['label']} | {model['label']} | "
                f"{pct_ci(clinical['major_upgrade_to_death'])} | "
                f"{pct_ci(clinical['home_to_facility'])} | "
                f"{pct_ci(clinical['any_upgrade'])} |"
            )

    lines.extend(
        [
            "",
            "## Flip Correctness and True-Class Probability",
            "",
            "Distance and probability changes are 72 h minus 24 h. A positive "
            "probability change means that the final true-class probability "
            "increased.",
            "",
            "| Database | Model | Distance gain | Distance loss | Distance tie | "
            "Mean distance change | Mean p_true change | p_true improved |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for database in ("mimic4", "eicu"):
        record = summary["databases"][database]
        for model in record["models"].values():
            correctness = model["correctness_relative_to_final_outcome"]["all"]
            probability = model["true_class_probability"]
            lines.append(
                f"| {record['label']} | {model['label']} | "
                f"{pct_ci(correctness['gain'])} | "
                f"{pct_ci(correctness['loss'])} | "
                f"{pct_ci(correctness['tie'])} | "
                f"{signed(correctness['distance_change_72_minus_24_mean'])} | "
                f"{signed(probability['pt_true_change_mean'])} | "
                f"{pct_ci(probability['improved'])} |"
            )

    lines.extend(
        [
            "",
            "## Paired Temporal-Penalty Decomposition",
            "",
            "Change is the temporal-model rate minus the no-temporal-model "
            "rate. The four primary, mutually exclusive flip categories "
            "(correct/incorrect by upgrade/downgrade) sum to the total flip "
            "rate change. Directional ties have no correctness gain and are "
            "included with incorrect changes; separate tie rows are provided.",
            "",
        ]
    )
    for database in ("mimic4", "eicu"):
        record = summary["databases"][database]
        contrast = record["temporal_vs_notemporal"]
        lines.extend(
            [
                f"### {record['label']}",
                "",
                f"- No-temporal flip rate: "
                f"{pct_ci(contrast['flip_rate_notemporal'])}",
                f"- Temporal flip rate: "
                f"{pct_ci(contrast['flip_rate_temporal'])}",
                f"- Paired change: {pp_ci(contrast['flip_rate_change'])}",
                "",
                "| Category | No temporal | Temporal | Change (temporal - none) |",
                "|---|---:|---:|---:|",
            ]
        )
        for name, category in contrast["flip_category_decomposition"].items():
            if not isinstance(category, dict) or "notemporal" not in category:
                continue
            lines.append(
                f"| {name} | {pct_ci(category['notemporal'])} | "
                f"{pct_ci(category['temporal'])} | "
                f"{pp_ci(category['change_temporal_minus_notemporal'])} |"
            )
        lines.extend(
            [
                "",
                (
                    "Net reduction in upgrades: "
                    f"{100 * contrast['flip_category_decomposition']['upgrades_reduced']:+.2f} pp; "
                    "correct upgrades account for "
                    f"{contrast['flip_category_decomposition']['correct_upgrade_share_of_upgrade_reduction']:.1%} "
                    "and incorrect upgrades for "
                    f"{contrast['flip_category_decomposition']['incorrect_upgrade_share_of_upgrade_reduction']:.1%}."
                    if contrast["flip_category_decomposition"][
                        "correct_upgrade_share_of_upgrade_reduction"
                    ]
                    is not None
                    else "The temporal model did not produce a net reduction "
                    "in upgrade flips."
                ),
                "",
                "Patients with a final death outcome:",
                "",
                "| Metric | No temporal | Temporal | Change |",
                "|---|---:|---:|---:|",
            ]
        )
        death = contrast["true_death_subgroup"]
        for name in (
            "new_upgrade_to_death",
            "detected_death_at72",
            "missed_death_at72",
            "lost_detected_death",
            "any_upgrade",
        ):
            metric_record = death[name]
            lines.append(
                f"| {name} | {pct_ci(metric_record['notemporal'])} | "
                f"{pct_ci(metric_record['temporal'])} | "
                f"{pp_ci(metric_record['change_temporal_minus_notemporal'])} |"
            )
        assessment = contrast["suppression_assessment"]
        lines.extend(
            [
                "",
                f"**Suppression assessment: {assessment['evidence']}.** "
                f"{assessment['interpretation']}",
                "",
            ]
        )

    lines.extend(
        [
            "## Interpretation and Limitations",
            "",
            "- This analysis describes how fitted predictions move between "
            "landmarks; it does not estimate algorithm stability after "
            "retraining. Bootstrap intervals condition on the original folds.",
            "- MIMIC-IV is resampled by patient, allowing multiple stays from "
            "the same patient to enter a replicate together. eICU-CRD is "
            "resampled by hospital; intervals reflect variation across 184 "
            "hospitals.",
            "- XGBoost is a single-seed shared-fold result; DG-GRU is a "
            "three-seed probability ensemble. Cross-model differences still "
            "include algorithm, feature-representation, and seed-averaging "
            "effects.",
            "- Facility combines rehabilitation, SNF, LTACH, nursing home, "
            "and some surviving hospice discharges. The ordering "
            "home < facility < death is an analysis assumption.",
            "- Gain, loss, and tie compare only the distance to the final "
            "three-class outcome; they do not establish clinical pathway "
            "appropriateness.",
            "- The analysis is exploratory and has no multiplicity adjustment. "
            "It should not be interpreted as clinical utility.",
            "",
            "## Output Files",
            "",
            "- `results/clinical_flips/summary.json`",
            "- `results/clinical_flips/report.md`",
            "- `src/sepsis/analyze_clinical_flips.py`",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit clinical direction and correctness of 24-to-72 h flips"
    )
    parser.add_argument(
        "--replicates",
        type=int,
        default=2000,
        help="Cluster-bootstrap replicates per database (default: 2000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20261030,
        help="Random seed for the bootstrap",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.replicates < 1:
        raise ValueError("--replicates must be positive")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    databases = {
        database: analyze_database(database, args.replicates, args.seed)
        for database in ("mimic4", "eicu")
    }
    summary = {
        "metadata": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "analysis": "clinical_flip_direction_correctness_and_temporal_penalty",
            "replicates": args.replicates,
            "seed": args.seed,
            "xgboost_prediction_source": (
                "results/shared_fold/{database}_shared_fold.npz "
                "(p24 and p72; one fixed patient-grouped 5-fold partition)"
            ),
            "dg_gru_prediction_source": (
                "results/formal_summary/{database}_ordinal_notemporal_ensemble.npz "
                "and results/formal_summary/{database}_ordinal_temporal_ensemble.npz"
            ),
            "alignment": (
                "All three model series are aligned on the exact common ids and "
                "checked for identical y/y3 outcomes before analysis."
            ),
            "class_order": list(CLASS_NAMES),
            "complete_landmark_hours": [24, 48, 72],
            "models": MODEL_LABELS,
            "estimand": (
                "Evaluation-sample uncertainty conditional on already fitted "
                "out-of-fold predictions and original folds; models are not "
                "retrained within bootstrap replicates."
            ),
            "bootstrap_unit": {
                "mimic4": "subject_id (patient)",
                "eicu": "hospitalid (hospital cluster)",
            },
            "correctness_distance": (
                "Absolute distance between the predicted argmax class and the "
                "final three-class outcome."
            ),
            "clinical_major_upgrade": (
                "Primary: any 24 h non-death prediction changing to death at "
                "72 h. Secondary: home-to-facility and any class upgrade."
            ),
            "multiplicity": "No multiplicity adjustment; exploratory analysis.",
        },
        "databases": databases,
        "interpretation_boundary": [
            "The analysis describes stability of fitted predictions, not the "
            "effect of retraining an algorithm.",
            "Facility is a heterogeneous destination group.",
            "Home, facility, and death are ordinally coded for analysis; this "
            "ordering is a modelling assumption.",
            "A lower flip rate is not sufficient evidence of better prediction.",
        ],
    }
    summary_path = OUTPUT_DIR / "summary.json"
    report_path = OUTPUT_DIR / "report.md"
    summary_path.write_bytes(
        (json.dumps(summary, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    )
    report_path.write_bytes((build_report(summary) + "\n").encode("utf-8"))

    print(
        json.dumps(
            {
                database: {
                    "n": record["n"],
                    "clusters": record["bootstrap"]["clusters"],
                    "xgboost_flip": record["models"]["xgboost"]["flip_24_72"],
                    "temporal_flip_change": record["temporal_vs_notemporal"][
                        "flip_rate_change"
                    ],
                    "suppression_evidence": record["temporal_vs_notemporal"][
                        "suppression_assessment"
                    ]["evidence"],
                }
                for database, record in databases.items()
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )
    print(f"Wrote {summary_path}", flush=True)
    print(f"Wrote {report_path}", flush=True)


if __name__ == "__main__":
    main()
