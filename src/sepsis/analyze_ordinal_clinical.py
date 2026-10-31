# -*- coding: utf-8 -*-
"""Continuous NRI and IDI for ordinal versus nominal readouts at 72 h."""

from __future__ import annotations

import json
import os
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import numpy as np


ROOT = Path(os.environ.get("SEPSIS_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
FORMAL = ROOT / "results" / "formal"
SUMMARY = ROOT / "results" / "formal_summary"
OUT = ROOT / "results" / "sensitivity_v2"
TABLES = ROOT / "results" / "tables"
SEEDS = (1, 2, 3)


def fixed(value: float) -> str:
    rounded_four = Decimal(str(value)).quantize(
        Decimal("0.0001"), rounding=ROUND_HALF_UP
    )
    digits = (
        4
        if Decimal("0") < abs(rounded_four) < Decimal("0.01")
        else 2
    )
    quantum = Decimal(1).scaleb(-digits)
    rounded = Decimal(str(value)).quantize(quantum, rounding=ROUND_HALF_UP)
    return f"{rounded:+.{digits}f}"


def continuous_nri_idi(
    y: np.ndarray,
    reference: np.ndarray,
    comparison: np.ndarray,
) -> dict:
    output = {}
    for cutoff, label in ((1, "non-home"), (2, "death")):
        event = y >= cutoff
        reference_risk = reference[:, cutoff:].sum(1)
        comparison_risk = comparison[:, cutoff:].sum(1)
        change = comparison_risk - reference_risk
        event_up = float((change[event] > 0).mean())
        event_down = float((change[event] < 0).mean())
        non_event_up = float((change[~event] > 0).mean())
        non_event_down = float((change[~event] < 0).mean())
        nri = (event_up - event_down) - (non_event_up - non_event_down)
        reference_gap = float(
            reference_risk[event].mean() - reference_risk[~event].mean()
        )
        comparison_gap = float(
            comparison_risk[event].mean() - comparison_risk[~event].mean()
        )
        idi = comparison_gap - reference_gap
        output[label] = {
            "event_n": int(event.sum()),
            "non_event_n": int((~event).sum()),
            "continuous_nri": round(nri, 4),
            "idi": round(idi, 4),
            "reference_discrimination_slope": round(reference_gap, 4),
            "comparison_discrimination_slope": round(comparison_gap, 4),
        }
    return output


def mean_sd(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)
    return {
        "mean": round(float(array.mean()), 4),
        "sd": round(float(array.std(ddof=1)), 4),
        "values": [round(float(value), 4) for value in values],
    }


def analyze_contrast(
    database: str, reference_name: str, comparison_name: str
) -> dict:
    seed_records = []
    for seed in SEEDS:
        reference = np.load(
            FORMAL / database / f"{reference_name}_s{seed}.npz"
        )
        comparison = np.load(
            FORMAL / database / f"{comparison_name}_s{seed}.npz"
        )
        if not np.array_equal(reference["ids"], comparison["ids"]):
            raise ValueError(f"{database} seed {seed}: row order differs")
        record = continuous_nri_idi(
            reference["y"],
            reference["probability"][:, 2, :],
            comparison["probability"][:, 2, :],
        )
        seed_records.append(record)
    reference_ensemble = np.load(
        SUMMARY / f"{database}_{reference_name}_ensemble.npz"
    )
    comparison_ensemble = np.load(
        SUMMARY / f"{database}_{comparison_name}_ensemble.npz"
    )
    if not np.array_equal(reference_ensemble["ids"], comparison_ensemble["ids"]):
        raise ValueError(f"{database}: ensemble row order differs")
    ensemble_record = continuous_nri_idi(
        reference_ensemble["y"],
        reference_ensemble["probability"][:, 2, :],
        comparison_ensemble["probability"][:, 2, :],
    )
    output = {"ensemble": ensemble_record}
    for outcome in ("non-home", "death"):
        output[outcome] = {
            metric: mean_sd(
                [record[outcome][metric] for record in seed_records]
            )
            for metric in ("continuous_nri", "idi")
        }
    return output


def build_table(summary: dict, bootstrap: dict) -> str:
    lines = [
        "| Database | Contrast | Outcome | Continuous NRI (95% CI) | IDI (95% CI) |",
        "|---|---|---|---:|---:|",
    ]
    for database, label in (("mimic4", "MIMIC-IV"), ("eicu", "eICU-CRD")):
        for contrast, bootstrap_key, description in (
            (
                "ordinal_vs_nominal_no_temporal",
                "ordinal_minus_nominal_no_temporal",
                "Ordinal vs nominal, no temporal penalty",
            ),
            (
                "ordinal_vs_nominal_temporal",
                "ordinal_minus_nominal_temporal",
                "Ordinal vs nominal, temporal penalty",
            ),
        ):
            record = summary[database][contrast]
            for outcome in ("non-home", "death"):
                suffix = "non_home" if outcome == "non-home" else "death"
                if bootstrap_key in bootstrap[database]["contrasts"]:
                    nri = bootstrap[database]["contrasts"][bootstrap_key][
                        f"continuous_nri_{suffix}"
                    ]
                    idi = bootstrap[database]["contrasts"][bootstrap_key][
                        f"idi_{suffix}"
                    ]
                else:
                    nri = record[outcome]["continuous_nri"]
                    idi = record[outcome]["idi"]
                    if "mean" in nri:
                        nri = {
                            "estimate": nri["mean"],
                            "ci95": [nri["mean"] - nri["sd"], nri["mean"] + nri["sd"]],
                        }
                        idi = {
                            "estimate": idi["mean"],
                            "ci95": [idi["mean"] - idi["sd"], idi["mean"] + idi["sd"]],
                        }
                lines.append(
                    f"| {label} | {description} | {outcome} | "
                    f"{fixed(nri['estimate'])} "
                    f"({fixed(nri['ci95'][0])} to {fixed(nri['ci95'][1])}) | "
                    f"{fixed(idi['estimate'])} "
                    f"({fixed(idi['ci95'][0])} to {fixed(idi['ci95'][1])}) |"
                )
    return "\n".join(lines) + "\n"


def main() -> None:
    summary = {}
    for database in ("mimic4", "eicu"):
        summary[database] = {
            "ordinal_vs_nominal_no_temporal": analyze_contrast(
                database, "nominal_notemporal", "ordinal_notemporal"
            ),
            "ordinal_vs_nominal_temporal": analyze_contrast(
                database, "nominal_temporal", "ordinal_temporal"
            ),
        }
    summary["definitions"] = {
        "analysis_type": "exploratory",
        "multiplicity_adjustment": "none",
        "direction": "comparison minus reference",
        "continuous_nri": (
            "(up in events - down in events) - "
            "(up in non-events - down in non-events)"
        ),
        "idi": (
            "change in the difference in mean cumulative risk between events "
            "and non-events"
        ),
        "cutoffs": "y >= 1 (non-home) and y >= 2 (death) at 72 h",
    }
    with (OUT / "ordinal_clinical_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    bootstrap_path = OUT / "model_contrast_bootstrap.json"
    if not bootstrap_path.exists():
        raise FileNotFoundError(
            f"{bootstrap_path}; run bootstrap_model_contrasts.py first"
        )
    with bootstrap_path.open(encoding="utf-8") as handle:
        bootstrap = json.load(handle)
    (TABLES / "supplementary_table_s5_ordinal_clinical.md").write_text(
        build_table(summary, bootstrap), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
