# -*- coding: utf-8 -*-
"""Assemble the manuscript from frozen aggregate result artifacts."""

import json
import os
import re
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(os.environ.get("SEPSIS_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
RESULTS = ROOT / "results"
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)

DATABASES = (("mimic4", "MIMIC-IV"), ("eicu", "eICU-CRD"))
REFERENCE_PATH = ROOT / "manuscript" / "references.json"
CONFIG_LABELS = {
    "ordinal_notemporal": "Ordinal, no temporal penalty",
    "ordinal_temporal": "Ordinal + temporal penalty",
    "nominal_notemporal": "Nominal, no temporal penalty",
    "nominal_temporal": "Nominal + temporal penalty",
}
TABLE_DISPLAY_TO_SOURCE = {
    1: 1,
    2: 2,
    3: 4,
    4: 9,
    5: 10,
    6: 3,
    7: 8,
    8: 6,
    9: 7,
    10: 5,
}
FIGURE_DISPLAY_TO_SOURCE = {
    1: 1,
    2: 2,
    3: 5,
    4: 11,
    5: 3,
    6: 6,
    7: 4,
    8: 10,
    9: 7,
    10: 8,
    11: 9,
}
SUPPLEMENTARY_DISPLAY_TO_SOURCE = {
    1: 6,
    2: 7,
    3: 4,
    4: 1,
    5: 8,
    6: 2,
    7: 3,
    8: 5,
    9: 9,
    10: 10,
    11: 11,
    12: 12,
}


def read_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def fixed(value: float, digits: int = 2) -> str:
    quantum = Decimal(1).scaleb(-digits)
    rounded = Decimal(str(value)).quantize(quantum, rounding=ROUND_HALF_UP)
    return f"{rounded:.{digits}f}"


def adaptive_fixed(value: float) -> str:
    value_decimal = abs(Decimal(str(value)))
    digits = 4 if Decimal("0") < value_decimal < Decimal("0.01") else 2
    return fixed(value, digits)


def signed_fixed(value: float, digits: int = 2) -> str:
    rendered = fixed(value, digits)
    if Decimal(rendered) == 0:
        return fixed(0, digits)
    prefix = "+" if value > 0 else ""
    return prefix + rendered


def pct(value: float, digits: int = 2) -> str:
    return fixed(100 * value, digits) + "%"


def ci_text(record: dict, digits: int = 2) -> str:
    lower, upper = record["ci95"]
    return (
        f"{pct(record['estimate'], digits)}, with a 95% CI from "
        f"{fixed(100 * lower, digits)} to {fixed(100 * upper, digits)}"
    )


def mean_sd_text(record: dict, scale: float = 1.0, digits: int = 4) -> str:
    return (
        f"{fixed(scale * record['mean'], digits)}, "
        f"SD {fixed(scale * record['sd'], digits)}"
    )


def signed_pct(value: float, digits: int = 2) -> str:
    return signed_fixed(100 * value, digits)


def renumber_assets(text: str) -> str:
    """Renumber figure and table references in one atomic mapping pass."""
    table_source_to_display = {
        source: display
        for display, source in TABLE_DISPLAY_TO_SOURCE.items()
    }
    figure_source_to_display = {
        source: display
        for display, source in FIGURE_DISPLAY_TO_SOURCE.items()
    }

    def table_replacement(match: re.Match) -> str:
        source = int(match.group(1))
        return f"Table {table_source_to_display[source]}"

    def figure_replacement(match: re.Match) -> str:
        source = int(match.group(1))
        return f"Figure {figure_source_to_display[source]}"

    text = re.sub(
        r"(?<!Supplementary )Table\s+([1-9][0-9]*)\b",
        table_replacement,
        text,
    )
    text = re.sub(
        r"Figure\s+([1-9][0-9]*)(?![0-9])",
        figure_replacement,
        text,
    )
    supplementary_source_to_display = {
        source: display
        for display, source in SUPPLEMENTARY_DISPLAY_TO_SOURCE.items()
    }

    def supplementary_replacement(match: re.Match) -> str:
        source = int(match.group(1))
        return f"Supplementary Table S{supplementary_source_to_display[source]}"

    return re.sub(
        r"Supplementary Table S([1-9]|1[0-2])\b",
        supplementary_replacement,
        text,
    )


def normalize_reported_statistics(text: str) -> str:
    """Normalize reported statistics without changing Markdown heading numbers."""
    protected: dict[str, str] = {}

    def protect(value: str) -> str:
        token = f"\u0000{len(protected)}\u0000"
        protected[token] = value
        return token

    def normalize_p(match: re.Match) -> str:
        operator = match.group(1)
        value = Decimal(match.group(2))
        if value < Decimal("0.001") and operator in {"<", "≤", "="}:
            return protect("P<0.001")
        rounded = value.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
        return protect(f"P{operator}{rounded:.3f}")

    text = re.sub(
        r"\b[Pp]\s*(<|≤|=|>|≥)\s*(\d*\.\d+)",
        normalize_p,
        text,
    )
    text = text.replace("95% CI", protect("95% CI"))

    def normalize_percent(match: re.Match) -> str:
        rounded = Decimal(match.group(1)).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )
        return f"{rounded:.2f}%"

    text = re.sub(
        r"(?<![\w.])([+-]?\d+(?:\.\d+)?)\s*%",
        normalize_percent,
        text,
    )

    def normalize_decimal(match: re.Match) -> str:
        value = Decimal(f"{match.group(1)}.{match.group(2)}")
        # Preserve precision for non-zero contrasts below 0.01; rounding these
        # to 0.00 would erase the direction and magnitude of the estimate.
        digits = 4 if Decimal("0") < abs(value) < Decimal("0.01") else 2
        rounded = value.quantize(
            Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP
        )
        prefix = "+" if match.group(1).startswith("+") and rounded >= 0 else ""
        return f"{prefix}{rounded:.{digits}f}"

    normalized_lines = []
    for line in text.splitlines():
        if line.lstrip().startswith("#"):
            normalized_lines.append(line)
            continue
        normalized_lines.append(
            re.sub(
                r"(?<![\w.])([+-]?\d+)\.(\d+)(?![\w.])",
                normalize_decimal,
                line,
            )
        )
    text = "\n".join(normalized_lines)
    for token, value in protected.items():
        text = text.replace(token, value)
    return text


def formal_auc(formal: dict, database: str, configuration: str) -> list[float]:
    metrics = formal[database]["ensemble"][configuration]["metrics"]["macro_auc"]
    return [float(metrics[str(hour)]) for hour in (24, 48, 72)]


def formal_stability(formal: dict, database: str, configuration: str) -> dict:
    return formal[database]["ensemble"][configuration]["stability"]


def xgb_record(xgb: list[dict], database: str) -> dict:
    return next(item for item in xgb if item["database"] == database)


def cohort_records() -> dict:
    output = {}
    for database, identifier, duration in (
        ("mimic4", "stay_id", "los"),
        ("eicu", "patientunitstayid", "los_h"),
        ("mimic3", "icustay_id", "los"),
    ):
        columns = [identifier, "y3", duration]
        if database == "eicu":
            columns.append("hospitalid")
        frame = pd.read_parquet(
            ROOT / "data" / "processed" / database / "design_all.parquet",
            columns=columns,
        )
        mapped = frame.loc[frame["y3"].notna()].copy()
        complete = mapped.loc[mapped[duration] >= 72].copy()
        eligible24 = mapped.loc[mapped[duration] >= 24].copy()
        early24 = eligible24.loc[eligible24[duration] < 72].copy()
        record = {
            "full": int(len(frame)),
            "mapped": int(len(mapped)),
            "unmapped": int(frame["y3"].isna().sum()),
            "landmarks": {
                str(hour): int((mapped[duration] >= hour).sum())
                for hour in (24, 48, 72)
            },
            "complete": int(len(complete)),
            "early24": int(len(early24)),
            "early24_home": float((early24["y3"] == 0).mean()),
            "early24_death": float((early24["y3"] == 2).mean()),
            "complete_home": float((complete["y3"] == 0).mean()),
            "complete_death": float((complete["y3"] == 2).mean()),
            "mapped_outcome_retention": float(len(mapped) / len(frame)),
            "mapped_to_complete_retention": float(len(complete) / len(mapped)),
            "full_to_complete_retention": float(len(complete) / len(frame)),
        }
        if database == "eicu":
            per_hospital = complete.groupby("hospitalid").size()
            record["hospitals"] = int(len(per_hospital))
            record["hospital_median"] = int(per_hospital.median())
            record["hospital_max"] = int(per_hospital.max())
        output[database] = record
    return output


def first_appeared_at_72() -> dict[str, dict[str, float | int]]:
    """Return first-at-72 h change among 24-to-72 h class flips."""
    paths = {
        "mimic4": RESULTS / "shared_fold" / "mimic4_shared_fold.npz",
        "eicu": RESULTS / "shared_fold" / "eicu_shared_fold.npz",
        "mimic3": RESULTS / "mimic3_shared" / "mimic3_shared.npz",
    }
    output = {}
    for database, path in paths.items():
        archive = np.load(path, allow_pickle=True)
        class_24 = archive["p24"].argmax(axis=1)
        class_48 = archive["p48"].argmax(axis=1)
        class_72 = archive["p72"].argmax(axis=1)
        changed = class_24 != class_72
        numerator = int(((class_24 == class_48) & changed).sum())
        denominator = int(changed.sum())
        output[database] = {
            "n": numerator,
            "denominator": denominator,
            "fraction": numerator / denominator,
        }
    return output


def direction_word(value: float) -> str:
    return "lower" if value < 0 else "higher"


def all_same_direction(values: list[float]) -> bool:
    return all(value < 0 for value in values) or all(value > 0 for value in values)


def transition_direction_rates(matrix: list[list[int]]) -> dict[str, float]:
    denominator = sum(sum(row) for row in matrix)
    ordinal_upward = sum(
        count
        for source, row in enumerate(matrix)
        for target, count in enumerate(row)
        if target > source
    )
    ordinal_downward = sum(
        count
        for source, row in enumerate(matrix)
        for target, count in enumerate(row)
        if target < source
    )
    return {
        "ordinal_upward": ordinal_upward / denominator,
        "ordinal_downward": ordinal_downward / denominator,
    }


def load_reference_library() -> dict[str, dict]:
    entries = read_json(REFERENCE_PATH)
    library = {entry["key"]: entry for entry in entries}
    if len(library) != len(entries):
        raise ValueError("Duplicate reference key")
    if not 40 <= len(library) <= 50:
        raise ValueError(f"Expected 40-50 references, found {len(library)}")
    return library


def compact_citation_numbers(numbers: list[int]) -> str:
    ordered = sorted(set(numbers))
    groups = []
    start = previous = ordered[0]
    for number in ordered[1:]:
        if number == previous + 1:
            previous = number
            continue
        groups.append((start, previous))
        start = previous = number
    groups.append((start, previous))
    parts = []
    for start, end in groups:
        if start == end:
            parts.append(str(start))
        elif end == start + 1:
            parts.extend((str(start), str(end)))
        else:
            parts.append(f"{start}-{end}")
    return ",".join(parts)


def resolve_citations(
    text: str, library: dict[str, dict]
) -> tuple[str, list[str]]:
    citation_order: list[str] = []
    number_by_key: dict[str, int] = {}
    pattern = re.compile(r"\[@([a-z0-9_]+(?:\s*;\s*@[a-z0-9_]+)*)\]")

    def replace(match: re.Match) -> str:
        keys = re.findall(r"@?([a-z0-9_]+)", match.group(1))
        numbers = []
        for key in keys:
            if key not in library:
                raise KeyError(f"Unknown reference key: {key}")
            if key not in number_by_key:
                citation_order.append(key)
                number_by_key[key] = len(citation_order)
            numbers.append(number_by_key[key])
        return f"[{compact_citation_numbers(numbers)}]"

    resolved = pattern.sub(replace, text)
    unused = set(library) - set(citation_order)
    if unused:
        raise ValueError(f"Uncited references: {sorted(unused)}")
    return resolved, citation_order


def reference_list(citation_order: list[str], library: dict[str, dict]) -> str:
    return "\n\n".join(
        f"{number}. {library[key]['formatted']}"
        for number, key in enumerate(citation_order, 1)
    )


def build_abstract(
    formal: dict,
    shared: dict,
    clinical: dict,
    transport: dict,
    mimic3_shared: dict,
    hospital_validation: dict,
    transport_extended: dict,
    trajectory_patterns: dict,
    flip_risk: dict,
) -> str:
    cohorts = cohort_records()
    xgb_mimic = shared["databases"]["mimic4"]
    xgb_eicu = shared["databases"]["eicu"]
    mimic3_discrimination = mimic3_shared["discrimination"]["macro_auc"]
    dg_mimic = formal_auc(formal, "mimic4", "ordinal_temporal")
    dg_eicu = formal_auc(formal, "eicu", "ordinal_temporal")
    xgb_stab_m = xgb_mimic["stability"]
    xgb_stab_e = xgb_eicu["stability"]
    clinical_m = clinical["databases"]["mimic4"]["models"]["xgboost"]
    clinical_e = clinical["databases"]["eicu"]["models"]["xgboost"]
    mimic3_stability = mimic3_shared["stability"]["flip_24_72"]
    mimic3_stability_record = mimic3_shared["stability"]
    dg_stab_m = xgb_mimic["dg_comparison"]["stability"]
    dg_stab_e = xgb_eicu["dg_comparison"]["stability"]
    hospital_discrimination = hospital_validation["metrics"]
    hospital_comparison = hospital_validation[
        "comparison_with_patient_grouped"
    ]["hospital_minus_patient_grouped"]
    hospital_level = hospital_validation["hospital_level"]
    hospital_sites = hospital_level["distributions_across_primary_sites"]
    hospital_pooled = hospital_level["der_simonian_laird_72h_macro_auc"]
    transported = {
        direction: record["transport_landmarks"]["72"][
            "target_all_reaching_landmark"
        ]
        for direction, record in transport_extended["directions"].items()
    }
    transported_aucs = [record["macro_auc"] for record in transported.values()]
    transported_eces = [
        record["ece_top_label_10bin"] for record in transported.values()
    ]
    recalibration = transport_extended["recalibration"]
    recalibration_25 = [
        record["72"]["curves"]["25"]["ece_top_label_10bin"]["mean"]
        for record in recalibration.values()
    ]
    recalibration_200 = [
        record["72"]["curves"]["200"]["ece_top_label_10bin"]["mean"]
        for record in recalibration.values()
    ]
    temperature_25 = [
        record["72"]["curves_temperature_scaling"]["25"][
            "ece_top_label_10bin"
        ]["mean"]
        for record in recalibration.values()
    ]
    trigger_auc = {
        (record["database"], record["outcome"]): record["metrics"]["auc"]
        for record in flip_risk["within_database_oof"]
    }
    any_flip_simulation = flip_risk["simulations"]["instability_risk_trigger"][
        "within_database"
    ]
    threshold_records = {
        database: next(
            item
            for item in any_flip_simulation[database]["risk_thresholds"]
            if item["threshold"] == 0.1
        )
        for database in ("mimic4", "eicu", "mimic3")
    }
    four_of_six_calibration = sum(
        1
        for direction, record in recalibration.items()
        if record["72"]["curves_temperature_scaling"]["25"][
            "ece_top_label_10bin"
        ]["mean"]
        > transported[direction]["ece_top_label_10bin"]
    )
    if four_of_six_calibration != 4:
        calibration_comparison = {
            direction: {
                "temperature_25": record["72"][
                    "curves_temperature_scaling"
                ]["25"]["ece_top_label_10bin"]["mean"],
                "untreated": transported[direction]["ece_top_label_10bin"],
            }
            for direction, record in recalibration.items()
        }
        raise ValueError(
            "Expected temperature scaling to worsen calibration in four "
            f"directions; found {four_of_six_calibration}: "
            f"{calibration_comparison}"
        )
    return f"""## Abstract

**Background:** Sepsis discharge planning uses repeated predictions.
Single-landmark estimates cannot show predicted-class changes.

**Methods:** We studied complete three-landmark cohorts observed through 72 h
from MIMIC-IV, eICU-CRD, and MIMIC-III. Outcomes were home discharge, facility
discharge, and in-hospital death. XGBoost used shared folds and common
hyperparameters. eICU-CRD added hospital-held-out validation. We assessed class
changes, fixed-weight transport, local recalibration, and enhanced-monitoring
selection.

**Results:** Cohorts contained {cohorts['mimic4']['complete']:,},
{cohorts['eicu']['complete']:,}, and {mimic3_shared['cohort']['n']:,} stays
through 72 h. 72 h macro-AUC was
{fixed(xgb_mimic['metrics']['72']['macro_auc'])},
{fixed(xgb_eicu['metrics']['72']['macro_auc'])}, and
{fixed(mimic3_discrimination['72'])}. Predicted 24-to-72 h class changes were
{pct(xgb_stab_m['flip_24_72'])}, {pct(xgb_stab_e['flip_24_72'])}, and
{pct(mimic3_stability['estimate'])}. Predicted
non-death-to-death class conversions occurred in
{pct(clinical_m['clinical_major_upgrades']['major_upgrade_to_death']['estimate'])},
{pct(clinical_e['clinical_major_upgrades']['major_upgrade_to_death']['estimate'])},
and {pct(mimic3_shared['clinical_transitions']['major_upgrade_to_death']['estimate'])}.
At 25 labels, multinomial error rose. Temperature scaling lowered untreated-
transport error in two directions and raised it in four.
At 200 labels, error was 0.02 to 0.03. At a 0.10 endpoint-change
risk threshold, {pct(threshold_records['mimic4']['reassessments_reduced_vs_all'])},
{pct(threshold_records['eicu']['reassessments_reduced_vs_all'])}, and
{pct(threshold_records['mimic3']['reassessments_reduced_vs_all'])} fewer stays
were selected for enhanced monitoring. Selected stays included
{pct(threshold_records['mimic4']['major_death_upgrade_capture_fraction'])},
{pct(threshold_records['eicu']['major_death_upgrade_capture_fraction'])}, and
{pct(threshold_records['mimic3']['major_death_upgrade_capture_fraction'])} of
non-death-to-death class conversions. Selected stays with no such conversion
were
{pct(threshold_records['mimic4']['reassessment_without_major_death_upgrade_fraction'])},
{pct(threshold_records['eicu']['reassessment_without_major_death_upgrade_fraction'])},
and
{pct(threshold_records['mimic3']['reassessment_without_major_death_upgrade_fraction'])}.
Fewer DG-GRU class changes accompanied lower AUC and smaller movement.

**Conclusions:** Many model-predicted destination classes changed between 24 and
72 h. Three landmarks cannot estimate time to stability. Fewer flips did not
indicate a better model. Transport and monitoring selection did not establish
clinical benefit. Local recalibration required direction-specific assessment.

**Keywords:** sepsis; discharge destination; ordinal outcome; prediction stability;
transportability; recalibration; reassessment
"""


def build_results(
    formal: dict,
    xgb: list[dict],
    stability: dict,
    sensitivity: dict,
    sensitivity_v2: dict,
    shared: dict,
    clinical: dict,
    transport: dict,
    mimic3_shared: dict,
    hospital_validation: dict,
    transport_extended: dict,
    trajectory_patterns: dict,
    flip_risk: dict,
) -> str:
    cohorts = cohort_records()
    first_72 = first_appeared_at_72()
    mimic3_cohort = mimic3_shared["cohort"]
    mimic3_discrimination = mimic3_shared["discrimination"]["macro_auc"]
    mimic3_stability = mimic3_shared["stability"]
    mimic3_transitions = mimic3_shared["clinical_transitions"]
    mimic3_correctness = mimic3_shared["correctness"]
    mimic3_direction = transition_direction_rates(
        mimic3_shared["transition_24_72"]["counts"]
    )
    hospital_metrics = hospital_validation["metrics"]
    hospital_stability = hospital_validation["stability"]
    hospital_comparison = hospital_validation[
        "comparison_with_patient_grouped"
    ]
    hospital_level = hospital_validation["hospital_level"]
    hospital_site_distributions = hospital_level[
        "distributions_across_primary_sites"
    ]
    hospital_pooled = hospital_level["der_simonian_laird_72h_macro_auc"]
    hospital_patient_flip = (
        hospital_stability["flip"]["24_72"]["estimate"]
        - hospital_comparison["hospital_minus_patient_grouped"]["flip_24_72"][
            "estimate"
        ]
    )
    transport_directions = transport_extended["directions"]
    transport_72 = {
        direction: record["transport_landmarks"]["72"][
            "target_all_reaching_landmark"
        ]
        for direction, record in transport_directions.items()
    }
    transport_groups = {
        "cross_institution": (
            "mimic4_to_eicu",
            "eicu_to_mimic4",
        ),
        "same_institution_cross_era": (
            "mimic4_to_mimic3",
            "mimic3_to_mimic4",
        ),
        "cross_institution_and_era": (
            "mimic3_to_eicu",
            "eicu_to_mimic3",
        ),
    }
    transport_fixed_extended = {
        direction: record["fixed_24_to_72_target_complete"]
        for direction, record in transport_directions.items()
    }
    recalibration = transport_extended["recalibration"]
    recalibration_25 = [
        record["72"]["curves"]["25"]["ece_top_label_10bin"]["mean"]
        for record in recalibration.values()
    ]
    recalibration_200 = [
        record["72"]["curves"]["200"]["ece_top_label_10bin"]["mean"]
        for record in recalibration.values()
    ]
    temperature_25 = [
        record["72"]["curves_temperature_scaling"]["25"][
            "ece_top_label_10bin"
        ]["mean"]
        for record in recalibration.values()
    ]
    temperature_worse_directions = sum(
        1
        for direction, record in recalibration.items()
        if record["72"]["curves_temperature_scaling"]["25"][
            "ece_top_label_10bin"
        ]["mean"]
        > transport_72[direction]["ece_top_label_10bin"]
    )
    temperature_better_directions = sum(
        1
        for direction, record in recalibration.items()
        if record["72"]["curves_temperature_scaling"]["25"][
            "ece_top_label_10bin"
        ]["mean"]
        < transport_72[direction]["ece_top_label_10bin"]
    )
    trajectory = trajectory_patterns["databases"]
    trigger_oof = {
        (record["database"], record["outcome"]): record
        for record in flip_risk["within_database_oof"]
    }
    any_flip_simulation = flip_risk["simulations"]["instability_risk_trigger"][
        "within_database"
    ]
    death_upgrade_simulation = flip_risk["simulations"][
        "death_upgrade_risk_trigger"
    ]["within_database"]
    any_flip_10 = {
        database: next(
            item
            for item in any_flip_simulation[database]["risk_thresholds"]
            if item["threshold"] == 0.1
        )
        for database in ("mimic4", "eicu", "mimic3")
    }
    death_upgrade_10 = {
        database: next(
            item
            for item in death_upgrade_simulation[database]["risk_thresholds"]
            if item["threshold"] == 0.1
        )
        for database in ("mimic4", "eicu", "mimic3")
    }
    xgb_m = xgb_record(xgb, "mimic4")["definitions"]
    xgb_e = xgb_record(xgb, "eicu")["definitions"]
    shared_m = shared["databases"]["mimic4"]
    shared_e = shared["databases"]["eicu"]
    shared_stab_m = shared_m["stability"]
    shared_stab_e = shared_e["stability"]
    clinical_m = clinical["databases"]["mimic4"]
    clinical_e = clinical["databases"]["eicu"]
    dg_m = formal_auc(formal, "mimic4", "ordinal_temporal")
    dg_e = formal_auc(formal, "eicu", "ordinal_temporal")
    stab_m = formal_stability(formal, "mimic4", "ordinal_temporal")
    stab_e = formal_stability(formal, "eicu", "ordinal_temporal")
    temporal_m = formal["mimic4"]["paired_contrasts"]["temporal_vs_none_ordinal"]
    temporal_e = formal["eicu"]["paired_contrasts"]["temporal_vs_none_ordinal"]
    ordinal_m = formal["mimic4"]["paired_contrasts"]["ordinal_vs_nominal_no_temporal"]
    ordinal_e = formal["eicu"]["paired_contrasts"]["ordinal_vs_nominal_no_temporal"]
    ece_m = formal["mimic4"]["ensemble"]["ordinal_temporal"]["calibration"]["72"][
        "top_label"
    ]["ece_10bin"]
    ece_e = formal["eicu"]["ensemble"]["ordinal_temporal"]["calibration"]["72"][
        "top_label"
    ]["ece_10bin"]
    xgb_ece_m = shared_m["metrics"]["72"]["ece_10bin"]
    xgb_ece_e = shared_e["metrics"]["72"]["ece_10bin"]
    six = {item["database"]: item for item in sensitivity["six_vs_48"]}
    full_complete = {
        item["database"]: item for item in sensitivity["full_vs_complete"]
    }
    icu_death = sensitivity_v2["icu_death_exclusion"]
    icd9 = sensitivity_v2["icd_era"]["mimic4_icd9"]
    icd10 = sensitivity_v2["icd_era"]["mimic4_icd10"]
    reverse_m = sensitivity_v2["ordinal_order_reversal"]["mimic4"][
        "reversed_order_restored"
    ]
    reverse_e = sensitivity_v2["ordinal_order_reversal"]["eicu"][
        "reversed_order_restored"
    ]
    bootstrap_m = sensitivity_v2["model_contrast_bootstrap"]["mimic4"][
        "contrasts"
    ]
    bootstrap_e = sensitivity_v2["model_contrast_bootstrap"]["eicu"][
        "contrasts"
    ]
    ordinal_boot_m = bootstrap_m["ordinal_minus_nominal_no_temporal"]
    ordinal_boot_e = bootstrap_e["ordinal_minus_nominal_no_temporal"]
    mimic3 = sensitivity_v2["mimic3_comparison"]["complete_three_landmark_subset"]
    class_counts_m = shared_m["cohort"]["outcome_counts"]
    class_counts_e = shared_e["cohort"]["outcome_counts"]
    xgb_conf_m = shared_stab_m["confidence_24h"]
    xgb_conf_e = shared_stab_e["confidence_24h"]
    transition_m = shared_stab_m["transition_24_to_72_counts"]
    transition_e = shared_stab_e["transition_24_to_72_counts"]
    shared_direction_m = transition_direction_rates(transition_m)
    shared_direction_e = transition_direction_rates(transition_e)
    home_to_facility_m = transition_m[0][1] / sum(transition_m[0])
    facility_to_death_m = transition_m[1][2] / sum(transition_m[1])
    home_to_facility_e = transition_e[0][1] / sum(transition_e[0])
    facility_to_death_e = transition_e[1][2] / sum(transition_e[1])
    clinical_major_m = clinical_m["models"]["xgboost"]["clinical_major_upgrades"]
    clinical_major_e = clinical_e["models"]["xgboost"]["clinical_major_upgrades"]
    correctness_m = clinical_m["models"]["xgboost"][
        "correctness_relative_to_final_outcome"
    ]
    correctness_e = clinical_e["models"]["xgboost"][
        "correctness_relative_to_final_outcome"
    ]
    true_probability_m = clinical_m["models"]["xgboost"]["true_class_probability"]
    true_probability_e = clinical_e["models"]["xgboost"]["true_class_probability"]
    temporal_clinical_m = clinical_m["temporal_vs_notemporal"]
    temporal_clinical_e = clinical_e["temporal_vs_notemporal"]
    decomposition_m = temporal_clinical_m["flip_category_decomposition"]
    decomposition_e = temporal_clinical_e["flip_category_decomposition"]
    death_subgroup_m = temporal_clinical_m["true_death_subgroup"]
    death_subgroup_e = temporal_clinical_e["true_death_subgroup"]
    dg_stability_m = shared_m["dg_comparison"]["stability"]
    dg_stability_e = shared_e["dg_comparison"]["stability"]
    temporal_m_same = all_same_direction(temporal_m["flip_0_2"]["difference_values"])
    temporal_e_same = all_same_direction(temporal_e["flip_0_2"]["difference_values"])
    ordinal_m_same = all_same_direction(ordinal_m["auc_72"]["difference_values"])
    ordinal_e_same = all_same_direction(ordinal_e["auc_72"]["difference_values"])
    transport_m = transport["transport"]["mimic4_to_eicu"]["landmarks"]["72"][
        "target_all_reaching_landmark"
    ]
    transport_e = transport["transport"]["eicu_to_mimic4"]["landmarks"]["72"][
        "target_all_reaching_landmark"
    ]
    transport_landmarks_m = {
        hour: transport["transport"]["mimic4_to_eicu"]["landmarks"][str(hour)][
            "target_all_reaching_landmark"
        ]
        for hour in (24, 48, 72)
    }
    transport_landmarks_e = {
        hour: transport["transport"]["eicu_to_mimic4"]["landmarks"][str(hour)][
            "target_all_reaching_landmark"
        ]
        for hour in (24, 48, 72)
    }
    transport_fixed_m = transport["transport"]["mimic4_to_eicu"][
        "fixed_rule_24_to_72_target_complete72"
    ]
    transport_fixed_e = transport["transport"]["eicu_to_mimic4"][
        "fixed_rule_24_to_72_target_complete72"
    ]
    transport_features = {
        str(hour): transport["transport"]["mimic4_to_eicu"]["landmarks"][str(hour)][
            "feature_count"
        ]
        for hour in (24, 48, 72)
    }
    transport_slopes_m = transport_m["class_calibration"]
    transport_slopes_e = transport_e["class_calibration"]

    return f"""## 3. Results

### 3.1 Study population and outcome distribution

The MIMIC-IV source cohort contained {cohorts['mimic4']['full']:,} sepsis ICU
stays. A three-class outcome was mapped for {cohorts['mimic4']['mapped']:,}
stays, and {cohorts['mimic4']['complete']:,} were observed through 72 h. Mapped
outcome retention was
{pct(cohorts['mimic4']['mapped_outcome_retention'])}.
Complete-window retention was
{pct(cohorts['mimic4']['full_to_complete_retention'])} of source stays and
{pct(cohorts['mimic4']['mapped_to_complete_retention'])} of mapped stays.
The eICU-CRD source cohort contained {cohorts['eicu']['full']:,} stays. A mapped
outcome was available for {cohorts['eicu']['mapped']:,}, and
{cohorts['eicu']['complete']:,} were observed through 72 h. Mapped outcome
retention was {pct(cohorts['eicu']['mapped_outcome_retention'])}.
Complete-window retention was
{pct(cohorts['eicu']['full_to_complete_retention'])} of source stays and
{pct(cohorts['eicu']['mapped_to_complete_retention'])} of mapped stays.
MIMIC-III contained
{cohorts['mimic3']['mapped']:,} mapped stays, and
{mimic3_cohort['n']:,} from {mimic3_cohort['patient_count']:,} patients were
observed through 72 h. Complete-window retention was
{pct(cohorts['mimic3']['full_to_complete_retention'])}. MIMIC-IV contributed
{cohorts['mimic4']['landmarks']['24']:,},
{cohorts['mimic4']['landmarks']['48']:,}, and
{cohorts['mimic4']['landmarks']['72']:,} stays at 24, 48, and 72 h. eICU-CRD
contributed {cohorts['eicu']['landmarks']['24']:,},
{cohorts['eicu']['landmarks']['48']:,}, and
{cohorts['eicu']['landmarks']['72']:,} stays at the same landmarks. The stability
cohorts included {cohorts['mimic4']['complete']:,},
{cohorts['eicu']['complete']:,}, and {mimic3_cohort['n']:,} stays. At 72 h,
MIMIC-IV included {class_counts_m['0']:,} home, {class_counts_m['1']:,} facility,
and {class_counts_m['2']:,} death outcomes. eICU-CRD included
{class_counts_e['0']:,}, {class_counts_e['1']:,}, and {class_counts_e['2']:,}
outcomes at the same levels. MIMIC-III included
{mimic3_cohort['outcome_counts']['0']:,},
{mimic3_cohort['outcome_counts']['1']:,}, and
{mimic3_cohort['outcome_counts']['2']:,}. Figure 1A summarizes cohort
construction. Table 1, Table 2, and Figure 1B summarize these data.

Stays observed through 72 h differed from earlier ICU exits. In MIMIC-IV, early
exits had {pct(cohorts['mimic4']['early24_home'])} home discharge and
{pct(cohorts['mimic4']['early24_death'])} death. The complete-window group had
{pct(cohorts['mimic4']['complete_home'])} and
{pct(cohorts['mimic4']['complete_death'])}, respectively. In eICU-CRD, early
exits had {pct(cohorts['eicu']['early24_home'])} home and
{pct(cohorts['eicu']['early24_death'])} death. The corresponding complete-window
values were {pct(cohorts['eicu']['complete_home'])} and
{pct(cohorts['eicu']['complete_death'])}. The stability estimand
applies to stays observed through all three landmarks.

### 3.2 Landmark-to-landmark stability across three databases

The primary comparison used the complete cohort, shared patient folds, and common
hyperparameters. The XGBoost 24-to-72 h class flip was
{pct(shared_stab_m['flip_24_72'])} in MIMIC-IV, with a 95% CI from
{fixed(100 * shared_stab_m['flip_24_72_ci95'][0])} to
{fixed(100 * shared_stab_m['flip_24_72_ci95'][1])}. The eICU-CRD flip was
{pct(shared_stab_e['flip_24_72'])}, with a 95% CI from
{fixed(100 * shared_stab_e['flip_24_72_ci95'][0])} to
{fixed(100 * shared_stab_e['flip_24_72_ci95'][1])}. The MIMIC-III
same-institution replication flip
was {pct(mimic3_stability['flip_24_72']['estimate'])}, with a 95% CI from
{fixed(100 * mimic3_stability['flip_24_72']['ci95'][0])} to
{fixed(100 * mimic3_stability['flip_24_72']['ci95'][1])}. Mean probability total
variation was {fixed(shared_stab_m['tv_24_72'])},
{fixed(shared_stab_e['tv_24_72'])}, and
{fixed(mimic3_stability['total_variation_24_72']['estimate'])}. Table 4 and
Figure 5 summarize the stability estimates.

Return to the 24 h class after an intermediate change occurred in
{pct(shared_stab_m['return_to_24h_class'])},
{pct(shared_stab_e['return_to_24h_class'])}, and
{pct(mimic3_stability['return_to_24h_class']['estimate'])} of stays. Higher-coded
changes occurred in {pct(shared_direction_m['ordinal_upward'])} versus
{pct(shared_direction_m['ordinal_downward'])} lower-coded changes in
MIMIC-IV. The corresponding eICU-CRD values were
{pct(shared_direction_e['ordinal_upward'])} and
{pct(shared_direction_e['ordinal_downward'])}. MIMIC-III values were
{pct(mimic3_direction['ordinal_upward'])} and
{pct(mimic3_direction['ordinal_downward'])}. With landmark-specific populations
and landmark-specific folds, sensitivity estimates of the 24-to-72 h
class-flip rates in MIMIC-IV and eICU-CRD were
{pct(stability['mimic4']['flip']['24_72']['estimate'])} and
{pct(stability['eicu']['flip']['24_72']['estimate'])}.

The complete trajectory identified changes missed by two endpoints. Predictions
were unchanged at all three landmarks in
{pct(trajectory['mimic4']['patterns']['stable_all']['estimate'])},
{pct(trajectory['eicu']['patterns']['stable_all']['estimate'])}, and
{pct(trajectory['mimic3']['patterns']['stable_all']['estimate'])} of stays. Any
adjacent change occurred in
{pct(trajectory['mimic4']['clinical_trajectories']['any_adjacent_change']['estimate'])},
{pct(trajectory['eicu']['clinical_trajectories']['any_adjacent_change']['estimate'])},
and
{pct(trajectory['mimic3']['clinical_trajectories']['any_adjacent_change']['estimate'])}.
Transient 48 h excursions occurred in
{pct(trajectory['mimic4']['patterns']['transient_48h_excursion']['estimate'])},
{pct(trajectory['eicu']['patterns']['transient_48h_excursion']['estimate'])}, and
{pct(trajectory['mimic3']['patterns']['transient_48h_excursion']['estimate'])}.
Successive higher-coded trajectories occurred in
{pct(trajectory['mimic4']['patterns']['monotone_worsening']['estimate'])},
{pct(trajectory['eicu']['patterns']['monotone_worsening']['estimate'])}, and
{pct(trajectory['mimic3']['patterns']['monotone_worsening']['estimate'])}.
Figure 5E summarizes all trajectory categories. Table 4 reports the proportions
stable at all landmarks and with any adjacent change. The categories describe
fitted prediction sequences, not verified patient trajectories.

The eICU-CRD hospital validation included
{hospital_validation['cohort']['n']:,} stays from
{hospital_validation['cohort']['n_hospitals']:,} hospitals. It excluded
{hospital_validation['cohort']['rows_excluded_for_multiple_hospitals']:,} stays
from
{hospital_validation['cohort']['patients_excluded_for_multiple_hospitals']:,}
patients observed at multiple hospitals. The same-row patient-grouped 24-to-72 h
flip was {pct(hospital_patient_flip)}. The hospital-held-out flip was
{pct(hospital_stability['flip']['24_72']['estimate'])}. The difference was
{signed_pct(hospital_comparison['hospital_minus_patient_grouped']['flip_24_72']['estimate'], 2)}
percentage points. The 95% CI was
{signed_pct(hospital_comparison['hospital_minus_patient_grouped']['flip_24_72']['ci95'][0], 2)}
to
{signed_pct(hospital_comparison['hospital_minus_patient_grouped']['flip_24_72']['ci95'][1], 2)}.
Hospital-held-out validation showed a similar point estimate, without a
prespecified equivalence margin. Table 9 presents the paired validation
comparison.

### 3.3 Predicted transitions and movement relative to the final class

A predicted class conversion from non-death to death occurred in
{pct(clinical_major_m['major_upgrade_to_death']['estimate'])} of MIMIC-IV stays.
The corresponding eICU-CRD and MIMIC-III values were
{pct(clinical_major_e['major_upgrade_to_death']['estimate'])} and
{pct(mimic3_transitions['major_upgrade_to_death']['estimate'])}. Home-to-facility
movement occurred in {pct(clinical_major_m['home_to_facility']['estimate'])},
{pct(clinical_major_e['home_to_facility']['estimate'])}, and
{pct(mimic3_transitions['home_to_facility']['estimate'])}. Among stays predicted
home at 24 h, the 72 h predicted class changed to facility in
{pct(home_to_facility_m)} of MIMIC-IV stays and {pct(home_to_facility_e)} of
eICU-CRD stays. Among facility predictions, the 72 h predicted class changed to
death in {pct(facility_to_death_m)} and {pct(facility_to_death_e)}. Table 10 and
Figure 11A show these distinct transitions.

Changes were not uniformly errors. Across all stays, the predicted class moved
closer to the observed final class in
{pct(correctness_m['all']['gain']['estimate'])} of MIMIC-IV stays and
{pct(correctness_e['all']['gain']['estimate'])} of eICU-CRD stays. It moved
farther in {pct(correctness_m['all']['loss']['estimate'])} and
{pct(correctness_e['all']['loss']['estimate'])}. These values describe movement
toward or away from the observed final class, not verified clinical improvement.
Among changed predictions, Figure 11B reports the closer and farther shares
with denominators restricted to stays whose predicted class changed. The
probability assigned to the eventual class was higher in
{pct(true_probability_m['among_flips']['improved']['estimate'])} and
{pct(true_probability_e['among_flips']['improved']['estimate'])}. The mean
changes were
{signed_fixed(true_probability_m['among_flips']['pt_true_change_mean']['estimate'], 2)}
and
{signed_fixed(true_probability_e['among_flips']['pt_true_change_mean']['estimate'], 2)}.
Across all MIMIC-III stays, the predicted class moved closer in
{pct(mimic3_correctness['all']['gain']['estimate'])} and farther in
{pct(mimic3_correctness['all']['loss']['estimate'])}. Its eventual-class
probability was higher among flips in
{pct(mimic3_correctness['true_class_probability']['among_flips']['improved']['estimate'])}.
Even so, updates that moved away from the eventual label remained common.
Figure 11B-C show this mixed pattern. Figure 11A-D show the full transition and
movement relative to the observed final class. Supplementary Table S6 reports
the complete transition and probability decomposition.

### 3.4 Early confidence stratification

At 24 h confidence of at least 0.60, the subgroup included
{pct(xgb_conf_m['0.6']['coverage'])} of MIMIC-IV stays and
{pct(xgb_conf_e['0.6']['coverage'])} of eICU-CRD stays. It included
{pct(mimic3_stability['confidence_ge_0_6']['coverage']['estimate'])} of MIMIC-III
stays. Subsequent flips occurred in
{pct(xgb_conf_m['0.6']['flip_24_72'])},
{pct(xgb_conf_e['0.6']['flip_24_72'])}, and
{pct(mimic3_stability['confidence_ge_0_6']['subsequent_flip_24_72']['estimate'])}.
These rates were below the overall values of
{pct(shared_stab_m['flip_24_72'])},
{pct(shared_stab_e['flip_24_72'])}, and
{pct(mimic3_stability['flip_24_72']['estimate'])}. At 0.70, MIMIC-IV and
eICU-CRD retained {pct(xgb_conf_m['0.7']['coverage'])} and
{pct(xgb_conf_e['0.7']['coverage'])}. Their later flip rates were
{pct(xgb_conf_m['0.7']['flip_24_72'])} and
{pct(xgb_conf_e['0.7']['flip_24_72'])}. The MIMIC-III artifact lacked this
threshold. High early confidence defined a smaller stratum, but low confidence
did not establish a wrong prediction. Figure 5B
and Table 4 show the strata.

### 3.5 Discrimination and calibration

At 72 h, complete-cohort XGBoost macro-AUC was
{fixed(shared_m['metrics']['72']['macro_auc'])} in MIMIC-IV and
{fixed(shared_e['metrics']['72']['macro_auc'])} in eICU-CRD. MIMIC-III reached
{fixed(mimic3_discrimination['72'])}. The corresponding 24 and 48 h values
appear in Table 3, Table 8, and Figure 3A. Fold-level metrics appear in
Supplementary Table S7. Paired model contrasts appear in Figure 6.

At 72 h, XGBoost top-label calibration error was {fixed(xgb_ece_m)} in MIMIC-IV
and {fixed(xgb_ece_e)} in eICU-CRD. The regularized ordinal DG-GRU values were
{fixed(ece_m)} and {fixed(ece_e)}. Reliability curves appear in Figure 4.
The eICU-CRD hospital-held-out 72 h macro-AUC was
{fixed(hospital_metrics['72']['macro_auc'])}. Corresponding landmark values
appear in Figure 10A. This analysis is internal-external validation, not
prospective validation. Across evaluable hospitals, the P10-P90 site AUC was
{fixed(hospital_site_distributions['macro_auc_72']['p10'])} to
{fixed(hospital_site_distributions['macro_auc_72']['p90'])}. The observed range
was {fixed(hospital_site_distributions['macro_auc_72']['min'])} to
{fixed(hospital_site_distributions['macro_auc_72']['max'])}. Calibration slopes
and intercepts appear in Supplementary Table S4. Decision curves are reported in
Supplementary Table S1 as model behaviour only. Supplementary Table S8 reports
all hospital-level validation estimates.

### 3.6 Model comparison and temporal regularization

The regularized ordinal DG-GRU reached 72 h macro-AUC values of
{fixed(dg_m[2])} and {fixed(dg_e[2])}. These values were below XGBoost. Its
24-to-72 h flips were {pct(stab_m['flip_0_2'])} and {pct(stab_e['flip_0_2'])}.
Its total variation was {fixed(stab_m['tv_0_2'])} and
{fixed(stab_e['tv_0_2'])}. Lower flip counts accompanied lower
discrimination and smaller probability movement. Figure 3 and Figure 6 show
these contrasts. Supplementary Table S2 reports all paired seed contrasts, and
Supplementary Table S3 reports all model metrics.

Adding temporal regularization produced mean flip changes across three seeds of
{signed_pct(temporal_m['flip_0_2']['difference_mean'])} and
{signed_pct(temporal_e['flip_0_2']['difference_mean'])} percentage points. The
aligned ensemble changes were
{signed_fixed(100 * temporal_clinical_m['flip_rate_change']['estimate'], 2)} and
{signed_fixed(100 * temporal_clinical_e['flip_rate_change']['estimate'], 2)}
percentage points. Their paired 95% CIs were
{signed_fixed(100 * temporal_clinical_m['flip_rate_change']['ci95'][0], 2)} to
{signed_fixed(100 * temporal_clinical_m['flip_rate_change']['ci95'][1], 2)} and
{signed_fixed(100 * temporal_clinical_e['flip_rate_change']['ci95'][0], 2)} to
{signed_fixed(100 * temporal_clinical_e['flip_rate_change']['ci95'][1], 2)}.
Movements toward and away from the observed final class changed, as Figure 11D
shows. Point estimates suggested fewer new predicted non-death-to-death
conversions and more deaths not predicted at 72 h. The paired intervals did not
establish both directions. The penalty has no clinical validation.

Without temporal regularization, the ordinal readout had 72 h AUC values
{signed_fixed(ordinal_m['auc_72']['difference_mean'], 2)} higher in MIMIC-IV
and {signed_fixed(ordinal_e['auc_72']['difference_mean'], 2)} higher in
eICU-CRD than the nominal readout.
NRI and IDI results were exploratory. Supplementary Table S5 reports them.

### 3.7 Fixed-weight transport and local recalibration

At 72 h, fixed-weight transport macro-AUC ranged from
{fixed(min(record['macro_auc'] for record in transport_72.values()))} to
{fixed(max(record['macro_auc'] for record in transport_72.values()))} across all
six directions. The two cross-institution directions reached
{fixed(transport_72[transport_groups['cross_institution'][0]]['macro_auc'])} and
{fixed(transport_72[transport_groups['cross_institution'][1]]['macro_auc'])}.
The same-institution cross-era directions reached
{fixed(transport_72[transport_groups['same_institution_cross_era'][0]]['macro_auc'])}
and
{fixed(transport_72[transport_groups['same_institution_cross_era'][1]]['macro_auc'])}.
The joint cross-institution and cross-era directions reached
{fixed(transport_72[transport_groups['cross_institution_and_era'][0]]['macro_auc'])}
and
{fixed(transport_72[transport_groups['cross_institution_and_era'][1]]['macro_auc'])}.
Top-label calibration error ranged from
{fixed(min(record['ece_top_label_10bin'] for record in transport_72.values()))}
to
{fixed(max(record['ece_top_label_10bin'] for record in transport_72.values()))}.
The same source weights performed differently in different targets. Untreated
weights were fixed during target evaluation. Separate source models were fit at
each landmark. Table 6, Figure 7A, and Supplementary Table S9 report all six
directions.

With 25 labels, the multinomial calibrator produced a 72 h mean expected
calibration error from {fixed(min(recalibration_25))} to
{fixed(max(recalibration_25))}. Temperature scaling on the same splits produced
mean expected calibration error from {fixed(min(temperature_25))} to
{fixed(max(temperature_25))}. Its mean value was lower than multinomial
calibration in every direction. The paired-split
2.5th-97.5th percentile range crossed zero in five of six directions. Against
untreated transport, however, it had lower expected calibration error in
{temperature_better_directions} of six directions and higher error in
{temperature_worse_directions}. By 200 labels, multinomial mean expected
calibration error was
{fixed(min(recalibration_200))} to {fixed(max(recalibration_200))}. Neither
method supported a universal recalibration rule. Table 7 reports local
recalibration and the accompanying reassessment simulation. Figure 7B-D and
Supplementary Table S10 show the learning curves.

### 3.8 Retrospective reassessment-trigger simulation

At 24 h, a logistic model predicted whether the model-assigned destination class
would differ at 72 h. Out-of-fold AUC for that endpoint was
{fixed(trigger_oof[('mimic4', 'any_flip')]['metrics']['auc'])},
{fixed(trigger_oof[('eicu', 'any_flip')]['metrics']['auc'])}, and
{fixed(trigger_oof[('mimic3', 'any_flip')]['metrics']['auc'])}. Its AUPRC was
{fixed(trigger_oof[('mimic4', 'any_flip')]['metrics']['auprc'])},
{fixed(trigger_oof[('eicu', 'any_flip')]['metrics']['auprc'])}, and
{fixed(trigger_oof[('mimic3', 'any_flip')]['metrics']['auprc'])}. The
direct trigger for model-predicted non-death-to-death conversion had AUCs of
{fixed(trigger_oof[('mimic4', 'major_death_upgrade')]['metrics']['auc'])},
{fixed(trigger_oof[('eicu', 'major_death_upgrade')]['metrics']['auc'])}, and
{fixed(trigger_oof[('mimic3', 'major_death_upgrade')]['metrics']['auc'])}.
Its AUPRC was
{fixed(trigger_oof[('mimic4', 'major_death_upgrade')]['metrics']['auprc'])},
{fixed(trigger_oof[('eicu', 'major_death_upgrade')]['metrics']['auprc'])}, and
{fixed(trigger_oof[('mimic3', 'major_death_upgrade')]['metrics']['auprc'])}.
Brier scores were
{fixed(trigger_oof[('mimic4', 'any_flip')]['metrics']['brier'])},
{fixed(trigger_oof[('eicu', 'any_flip')]['metrics']['brier'])}, and
{fixed(trigger_oof[('mimic3', 'any_flip')]['metrics']['brier'])} for endpoint
change, and
{fixed(trigger_oof[('mimic4', 'major_death_upgrade')]['metrics']['brier'])},
{fixed(trigger_oof[('eicu', 'major_death_upgrade')]['metrics']['brier'])}, and
{fixed(trigger_oof[('mimic3', 'major_death_upgrade')]['metrics']['brier'])} for
model-predicted direct conversion. Top-label calibration errors were
{adaptive_fixed(trigger_oof[('mimic4', 'any_flip')]['metrics']['ece_10bin'])},
{adaptive_fixed(trigger_oof[('eicu', 'any_flip')]['metrics']['ece_10bin'])}, and
{adaptive_fixed(trigger_oof[('mimic3', 'any_flip')]['metrics']['ece_10bin'])} for
endpoint change, and
{adaptive_fixed(trigger_oof[('mimic4', 'major_death_upgrade')]['metrics']['ece_10bin'])},
{adaptive_fixed(trigger_oof[('eicu', 'major_death_upgrade')]['metrics']['ece_10bin'])}, and
{adaptive_fixed(trigger_oof[('mimic3', 'major_death_upgrade')]['metrics']['ece_10bin'])} for
model-predicted direct conversion.

At a predicted-risk threshold of 0.10 for model-predicted endpoint change,
enhanced monitoring was selected for
{pct(any_flip_10['mimic4']['trigger_coverage'])},
{pct(any_flip_10['eicu']['trigger_coverage'])}, and
{pct(any_flip_10['mimic3']['trigger_coverage'])} of stays. Thus,
{pct(any_flip_10['mimic4']['reassessments_reduced_vs_all'])},
{pct(any_flip_10['eicu']['reassessments_reduced_vs_all'])}, and
{pct(any_flip_10['mimic3']['reassessments_reduced_vs_all'])} fewer stays were
selected than when monitoring every stay. The selected set included
{pct(any_flip_10['mimic4']['major_death_upgrade_capture_fraction'])},
{pct(any_flip_10['eicu']['major_death_upgrade_capture_fraction'])}, and
{pct(any_flip_10['mimic3']['major_death_upgrade_capture_fraction'])} of predicted
non-death-to-death class conversions. However,
{pct(any_flip_10['mimic4']['reassessment_without_major_death_upgrade_fraction'])},
{pct(any_flip_10['eicu']['reassessment_without_major_death_upgrade_fraction'])},
and
{pct(any_flip_10['mimic3']['reassessment_without_major_death_upgrade_fraction'])}
of selected stays had no such predicted conversion. At the same 0.10 threshold, the
direct trigger for model-predicted conversion selected
{pct(1 - death_upgrade_10['mimic4']['reassessments_reduced_vs_all'])},
{pct(1 - death_upgrade_10['eicu']['reassessments_reduced_vs_all'])}, and
{pct(1 - death_upgrade_10['mimic3']['reassessments_reduced_vs_all'])} of stays.
This corresponded to
{pct(death_upgrade_10['mimic4']['reassessments_reduced_vs_all'])},
{pct(death_upgrade_10['eicu']['reassessments_reduced_vs_all'])}, and
{pct(death_upgrade_10['mimic3']['reassessments_reduced_vs_all'])} fewer stays
than monitoring every stay. It missed
{pct(death_upgrade_10['mimic4']['major_death_upgrade_miss_fraction'])},
{pct(death_upgrade_10['eicu']['major_death_upgrade_miss_fraction'])}, and
{pct(death_upgrade_10['mimic3']['major_death_upgrade_miss_fraction'])} of
such predicted conversions. Among 24-to-72 h model-predicted class changes,
the proportions first observed at the 72 h landmark were
{pct(first_72['mimic4']['fraction'])},
{pct(first_72['eicu']['fraction'])}, and
{pct(first_72['mimic3']['fraction'])}. Enhanced monitoring through 72 h was
therefore simulated directly.
Table 7 reports the reassessment rows. Figure 8A-F and Supplementary Table S11
show this tradeoff. The
simulation reused destination-model folds and did not assign care.

### 3.9 Sensitivity analyses

The six-to-48 h analysis used only information available by six hours. Its class
flip was {pct(six['mimic4']['six_to_48h_complete_subset']['flip'])} and
{pct(six['eicu']['six_to_48h_complete_subset']['flip'])}. Separating
rehabilitation as a fourth class changed 72 h macro-AUC from
{fixed(xgb_m['3class']['72']['macro_auc'])} to
{fixed(xgb_m['4class']['72']['macro_auc'])} in MIMIC-IV. The eICU-CRD change was
{fixed(xgb_e['3class']['72']['macro_auc'])} to
{fixed(xgb_e['4class']['72']['macro_auc'])}. Complete-window and full-population
results are reported in Table 5.

After excluding ICU deaths, the 24-to-72 h flip was
{pct(icu_death['mimic4']['complete_three_landmark_subset']['flip_24_72'])} in
{icu_death['mimic4']['complete_three_landmark_subset']['n']:,} MIMIC-IV stays.
The eICU-CRD value was
{pct(icu_death['eicu']['complete_three_landmark_subset']['flip_24_72'])} in
{icu_death['eicu']['complete_three_landmark_subset']['n']:,} stays. This
exclusion used final ICU survival status. Class changes therefore persisted
among ICU survivors in both databases.

MIMIC-IV stays coded with ICD-9 had a flip of
{pct(icd9['complete_three_landmark_subset']['flip_24_72'])}. Stays coded with
ICD-10 had a flip of {pct(icd10['complete_three_landmark_subset']['flip_24_72'])}.
These coding-era strata were descriptive and were not an isolated era effect.
After restoring the probability axis, reversing the ordinal order produced
flips of
{pct(reverse_m['flip_24_72']['mean'])} in
{cohorts['mimic4']['complete']:,} MIMIC-IV stays and
{pct(reverse_e['flip_24_72']['mean'])} in
{cohorts['eicu']['complete']:,} eICU-CRD stays. The standard-order estimates
for the same unregularized models were
{pct(sensitivity_v2['ordinal_order_reversal']['mimic4']['standard_ordinal_notemporal']['flip_24_72']['mean'])}
and
{pct(sensitivity_v2['ordinal_order_reversal']['eicu']['standard_ordinal_notemporal']['flip_24_72']['mean'])}.
The paired differences from the standard order were
{signed_pct(sensitivity_v2['ordinal_order_reversal']['mimic4']['paired_reversed_minus_standard']['flip_24_72']['mean'])}
and
{signed_pct(sensitivity_v2['ordinal_order_reversal']['eicu']['paired_reversed_minus_standard']['flip_24_72']['mean'])}
percentage points. This stress test does not validate the clinical ordering.
Figure 9A-F summarize these sensitivity analyses.
"""


def _legacy_build_discussion(
    formal: dict,
    shared: dict,
    clinical: dict,
    transport: dict,
    mimic3_shared: dict,
    hospital_validation: dict,
    transport_extended: dict,
    trajectory_patterns: dict,
    flip_risk: dict,
) -> str:
    temporal_m = formal["mimic4"]["paired_contrasts"]["temporal_vs_none_ordinal"]
    temporal_e = formal["eicu"]["paired_contrasts"]["temporal_vs_none_ordinal"]
    shared_m = shared["databases"]["mimic4"]["stability"]
    shared_e = shared["databases"]["eicu"]["stability"]
    flip_m = shared_m["flip_24_72"]
    flip_e = shared_e["flip_24_72"]
    return_m = shared_m["return_to_24h_class"]
    return_e = shared_e["return_to_24h_class"]
    confidence_m = shared_m["confidence_24h"]["0.6"]["flip_24_72"]
    confidence_e = shared_e["confidence_24h"]["0.6"]["flip_24_72"]
    direction_m = transition_direction_rates(shared_m["transition_24_to_72_counts"])
    direction_e = transition_direction_rates(shared_e["transition_24_to_72_counts"])
    upward_m = direction_m["ordinal_upward"]
    downward_m = direction_m["ordinal_downward"]
    upward_e = direction_e["ordinal_upward"]
    downward_e = direction_e["ordinal_downward"]
    direction_m = "more often" if upward_m > downward_m else "less often"
    direction_e = "more often" if upward_e > downward_e else "less often"
    clinical_m = clinical["databases"]["mimic4"]["models"]["xgboost"]
    clinical_e = clinical["databases"]["eicu"]["models"]["xgboost"]
    temporal_clinical_m = clinical["databases"]["mimic4"]["temporal_vs_notemporal"]
    temporal_clinical_e = clinical["databases"]["eicu"]["temporal_vs_notemporal"]
    dg_m = shared["databases"]["mimic4"]["dg_comparison"]["stability"][
        "dg_gru_ordinal_temporal_three_seed_ensemble"
    ]
    dg_e = shared["databases"]["eicu"]["dg_comparison"]["stability"][
        "dg_gru_ordinal_temporal_three_seed_ensemble"
    ]
    transport_m = transport["transport"]["mimic4_to_eicu"]["landmarks"]["72"][
        "target_all_reaching_landmark"
    ]
    transport_e = transport["transport"]["eicu_to_mimic4"]["landmarks"]["72"][
        "target_all_reaching_landmark"
    ]
    mimic3_flip = mimic3_shared["stability"]["flip_24_72"]["estimate"]
    mimic3_tv = mimic3_shared["stability"]["total_variation_24_72"]["estimate"]
    mimic3_transitions = mimic3_shared["clinical_transitions"]
    hospital_comparison = hospital_validation[
        "comparison_with_patient_grouped"
    ]
    hospital_flip_difference = hospital_comparison[
        "hospital_minus_patient_grouped"
    ]["flip_24_72"]
    hospital_level = hospital_validation["hospital_level"]
    hospital_sites = hospital_level["distributions_across_primary_sites"]
    transport_72 = {
        direction: record["transport_landmarks"]["72"][
            "target_all_reaching_landmark"
        ]
        for direction, record in transport_extended["directions"].items()
    }
    recalibration = transport_extended["recalibration"]
    recalibration_25 = [
        record["72"]["curves"]["25"]["ece_top_label_10bin"]["mean"]
        for record in recalibration.values()
    ]
    recalibration_200 = [
        record["72"]["curves"]["200"]["ece_top_label_10bin"]["mean"]
        for record in recalibration.values()
    ]
    trajectory = trajectory_patterns["databases"]
    trigger_oof = {
        (record["database"], record["outcome"]): record
        for record in flip_risk["within_database_oof"]
    }
    any_flip_10 = {
        database: next(
            item
            for item in flip_risk["simulations"]["instability_risk_trigger"][
                "within_database"
            ][database]["risk_thresholds"]
            if item["threshold"] == 0.1
        )
        for database in ("mimic4", "eicu", "mimic3")
    }
    return f"""## 4. Discussion

Using the complete cohort, one shared patient-grouped fold partition, and common
XGBoost hyperparameters, class changed between 24 and 72 h in {pct(flip_m)} of
MIMIC-IV stays, {pct(flip_e)} of eICU-CRD stays, and {pct(mimic3_flip)} of
MIMIC-III stays. Any change between adjacent landmarks occurred in
{pct(trajectory['mimic4']['clinical_trajectories']['any_adjacent_change']['estimate'])},
{pct(trajectory['eicu']['clinical_trajectories']['any_adjacent_change']['estimate'])},
and
{pct(trajectory['mimic3']['clinical_trajectories']['any_adjacent_change']['estimate'])},
whereas the three predictions were unchanged at all landmarks in
{pct(trajectory['mimic4']['patterns']['stable_all']['estimate'])},
{pct(trajectory['eicu']['patterns']['stable_all']['estimate'])}, and
{pct(trajectory['mimic3']['patterns']['stable_all']['estimate'])}. The
24-to-72 h comparison alone missed transient 48 h changes. The
probability of return to
the 24 h class after an intermediate change was {pct(return_m)} and
{pct(return_e)}, and the total-variation estimates were
{fixed(shared_m['tv_24_72'])}, {fixed(shared_e['tv_24_72'])}, and
{fixed(mimic3_tv)}. High confidence at 24 h selected a smaller subgroup, but the 0.6
stratum still had later flips in {pct(confidence_m)} and {pct(confidence_e)}.
The central finding is not that discharge destination is unpredictable. It is
that a single-landmark discrimination estimate does not describe the stability of
    the class that would be used for planning.

The estimand is deliberately bounded. Three fixed snapshots estimate how often
the predicted class differs between 24 and 72 h; they do not identify the exact
time at which a prediction becomes stable. The complete-cohort reanalysis also
removes a specific source of artefact from the earlier rates, with the same
patients assigned to one outer-fold partition at all landmarks.
Landmark-to-landmark changes can still reflect model updating, the changing
predictor window, or residual sampling variability. The confidence intervals
describe the evaluation sample conditional on the fitted models and folds; they
do not replace repeated training or prospective monitoring.

The hospital-held-out eICU-CRD analysis addresses a different leakage pathway.
The flip difference from patient-grouped folds was
{signed_pct(hospital_flip_difference['estimate'], 2)} percentage points, with a 95% CI from
{signed_pct(hospital_flip_difference['ci95'][0], 2)} to
    {signed_pct(hospital_flip_difference['ci95'][1], 2)}. The instability result
    persisted when hospitals were held out. Discrimination was modestly lower
    under hospital holding out, however, and site-level 72 h AUC ranged from
{fixed(hospital_sites['macro_auc_72']['p10'])} at the 10th percentile to
{fixed(hospital_sites['macro_auc_72']['p90'])} at the 90th percentile. This
between-site spread is important: a pooled estimate can conceal sites where the
same model performs close to chance.

Existing sepsis discharge models have mainly estimated destination at one fixed
time or have studied risk factors for non-home discharge [@cai2024;@zhang2026].
Ordinal clinical outcomes have also been evaluated with temporal validation
[@sodnomdavaa2026], and predictor modelling choices have been associated with
model stability [@phinyo2026;@jce2021]. Those analyses
ask whether a model performs consistently across settings or coding choices. The
present estimand is different: for the same stay, does the predicted destination
class change as the available ICU history grows? A model can rank stays well at
each landmark and still produce a different class for a substantial minority of
individuals.

Several features of the data may accompany this pattern. The first day contains limited
information about respiratory trajectory, neurological recovery, and the eventual
need for institutional care. As observations accumulate, the estimated probability
can move across a decision boundary even when the underlying risk has not changed
abruptly. Higher-coded changes were {direction_m} frequent in MIMIC-IV at
{pct(upward_m)} versus {pct(downward_m)} lower-coded changes. In eICU-CRD,
the corresponding values were {pct(upward_e)} and {pct(downward_e)}. This is an
interpretation of model behaviour, not a causal estimate.

Not all changes are equivalent. A predicted class conversion from non-death to
death occurred in
{pct(clinical_m['clinical_major_upgrades']['major_upgrade_to_death']['estimate'])}
of MIMIC-IV stays and
{pct(clinical_e['clinical_major_upgrades']['major_upgrade_to_death']['estimate'])}
of eICU-CRD stays and
{pct(mimic3_transitions['major_upgrade_to_death']['estimate'])} of MIMIC-III
stays, whereas home-to-facility movement occurred in
{pct(clinical_m['clinical_major_upgrades']['home_to_facility']['estimate'])},
{pct(clinical_e['clinical_major_upgrades']['home_to_facility']['estimate'])}, and
{pct(mimic3_transitions['home_to_facility']['estimate'])}.
These endpoints have different planning implications and should not be collapsed
into a generic flip rate. The later prediction was also not uniformly better:
among stays that changed class, the probability assigned to the eventual observed
class was higher for
{pct(clinical_m['true_class_probability']['among_flips']['improved']['estimate'])}
and
{pct(clinical_e['true_class_probability']['among_flips']['improved']['estimate'])}.
This mixed pattern is compatible with studying scheduled reassessment, but it does
not support a rule that every later prediction should replace an earlier one.

The confidence analysis has a narrower interpretation. A high-confidence
prediction at 24 h was less likely to change, but high-confidence predictions were
a minority. Low confidence is plausible as a trigger for reassessment; the present
data do not show that such a trigger is associated with better workflow, placement
decisions, or patient outcomes.

Compared with no penalty, temporal regularization had a DG-GRU flip rate lower by
{fixed(abs(100 * temporal_m['flip_0_2']['difference_mean']), 2)} percentage points in
MIMIC-IV and {fixed(abs(100 * temporal_e['flip_0_2']['difference_mean']), 2)} percentage
points in eICU-CRD. The aligned ensemble decomposition gave changes of
{signed_fixed(100 * temporal_clinical_m['flip_rate_change']['estimate'], 2)} and
{signed_fixed(100 * temporal_clinical_e['flip_rate_change']['estimate'], 2)}
    percentage points. The lower class-change rate reflected fewer movements
    toward or away from the observed final class.

A lower DG-GRU flip rate cannot be read as greater reliability. The regularized
DG-GRU had a 24-to-72 h flip rate of {pct(dg_m['flip_24_72'])} in MIMIC-IV and
{pct(dg_e['flip_24_72'])} in eICU-CRD, below XGBoost, but it also had lower
discrimination and much smaller total variation of {fixed(dg_m['tv_24_72'])} and
{fixed(dg_e['tv_24_72'])}. A model can appear stable by remaining near the same
decision boundary or by failing to move its probability mass when the patient
changes. Stability must therefore be reported jointly with AUC, calibration,
    probability movement, transition direction, and movement relative to the
observed final class.

The temporal penalty raises the same concern. Smoothing and non-deterioration
    pressure could coincide with fewer higher-coded changes if true deterioration is not
reflected in the score.
In both databases, the point estimates were compatible with fewer new predicted
    non-death-to-death conversions and more deaths not predicted at 72 h, although
paired intervals did not establish both directions. The appropriate conclusion is
uncertainty, not safety. Neither the
smoothness weight nor the monotonicity weight has clinical validation, and a
smaller flip count is not a clinical endpoint.

The ordinal ordering remains an assumption. Facility combines rehabilitation,
skilled nursing, long-term care, other institutions, and surviving hospice
discharges; it is not a measured point on a biological severity scale. The
ordinal-versus-nominal contrast and the reversed-order stress test show that
results depend partly on this encoding, but neither proves that facility lies on a
true continuum between home and death. The NRI/IDI analyses describe reclassification
under the fitted ordinal model and do not validate the clinical ordering.

The six-direction fixed-weight transport analysis contrasts repeated sampling
within one institution across eras with movement across institutions and source
periods. MIMIC-IV and MIMIC-III share an institution but not an era. MIMIC-IV
and eICU-CRD differ in institution, care system, case mix, and source period,
so their directions are not clean institution-only contrasts. MIMIC-III and
eICU-CRD change both institution and era. The same target can therefore be rank
ordered with different accuracy depending on the source weights, and the same
source can show different calibration depending on the target. At 72 h, transport macro-AUC
ranged from {fixed(min(record['macro_auc'] for record in transport_72.values()))}
to {fixed(max(record['macro_auc'] for record in transport_72.values()))}, while
top-label calibration error ranged from
{fixed(min(record['ece_top_label_10bin'] for record in transport_72.values()))}
to {fixed(max(record['ece_top_label_10bin'] for record in transport_72.values()))}.
This range shows why common feature names and a respectable same-institution AUC
are insufficient evidence of cross-site readiness.

The recalibration experiment adds a practical qualification. Twenty-five local
labels were too few for the multinomial calibrator: mean calibration error rose
to {fixed(min(recalibration_25))}-{fixed(max(recalibration_25))}. A
one-parameter temperature scaler fitted on the same splits kept calibration
error at {fixed(min(
    recalibration[direction]['72']['curves_temperature_scaling']['25'][
        'ece_top_label_10bin'
    ]['mean']
    for direction in recalibration
))}-{fixed(max(
    recalibration[direction]['72']['curves_temperature_scaling']['25'][
        'ece_top_label_10bin'
    ]['mean']
    for direction in recalibration
))}, with lower mean error in every direction and paired-split percentile ranges
crossing zero in five of six directions. By 200 labels, the
multinomial calibrator caught up, with calibration error of
{fixed(min(recalibration_200))}-{fixed(max(recalibration_200))}. The appropriate
decision is therefore to monitor transported calibration and test a deliberately
simple local update before considering a more flexible calibrator, not to
recalibrate every model by default. Macro-AUC did not show a consistent gain; a
monotone probability transform does not repair ranking errors.

The reassessment-trigger simulation provides an operational bridge without
establishing clinical utility. The 24 h endpoint-change trigger had AUCs of
{fixed(trigger_oof[('mimic4', 'any_flip')]['metrics']['auc'])},
{fixed(trigger_oof[('eicu', 'any_flip')]['metrics']['auc'])}, and
{fixed(trigger_oof[('mimic3', 'any_flip')]['metrics']['auc'])}, while the
direct conversion trigger had AUCs of
{fixed(trigger_oof[('mimic4', 'major_death_upgrade')]['metrics']['auc'])},
{fixed(trigger_oof[('eicu', 'major_death_upgrade')]['metrics']['auc'])}, and
{fixed(trigger_oof[('mimic3', 'major_death_upgrade')]['metrics']['auc'])}. This
discrimination is stronger than the simulated workflow evidence. At a
predicted-risk threshold of 0.10 for endpoint class change, the selected set
included
{pct(any_flip_10['mimic4']['major_death_upgrade_capture_fraction'])},
{pct(any_flip_10['eicu']['major_death_upgrade_capture_fraction'])}, and
{pct(any_flip_10['mimic3']['major_death_upgrade_capture_fraction'])} of predicted
non-death-to-death class conversions. At that threshold,
{pct(any_flip_10['mimic4']['reassessments_reduced_vs_all'])},
{pct(any_flip_10['eicu']['reassessments_reduced_vs_all'])}, and
{pct(any_flip_10['mimic3']['reassessments_reduced_vs_all'])} fewer stays were
selected than when monitoring every stay. The direct conversion strategy missed
{pct(death_upgrade_10['mimic4']['major_death_upgrade_miss_fraction'])},
{pct(death_upgrade_10['eicu']['major_death_upgrade_miss_fraction'])}, and
{pct(death_upgrade_10['mimic3']['major_death_upgrade_miss_fraction'])} of such
conversions and included a smaller proportion of non-death class changes.
The simulation thus exposes a tradeoff but cannot choose a
threshold for care.

Strengths include explicit landmark populations, a prespecified conditional
stability estimand, complete-cohort shared patient folds, patient-aligned model
contrasts, complete three-landmark trajectory categories, major predicted-class
transition definitions, movements toward or away from the observed final class,
hospital-held-out eICU-CRD validation, hospital-clustered uncertainty,
six-direction fixed-weight transport, paired low-label calibrator comparison,
trigger discrimination with confidence intervals, and a transparent reassessment
simulation. The analysis keeps discrimination, probability movement, stability,
and the simulated selected fraction separate rather than selecting a model on class-change rate alone.

Several limitations are important. The stability cohort required observation through
72 h and differed from patients who left the ICU earlier; the estimates are
conditional and should not be extrapolated to all sepsis admissions. The study was
retrospective, sepsis was identified from diagnosis codes, and the destination
labels came from hospital administrative fields. Facility includes heterogeneous
destinations. MIMIC-IV is a single-center database, whereas eICU-CRD has
heterogeneous site documentation and substantial missingness in temperature and
neurological observations. Fairness across race, ethnicity, language, and
socioeconomic groups was not formally assessed because these variables were not
available consistently across the three databases. XGBoost used one fixed seed, and the XGBoost-to-DG-GRU
comparison combines architecture, input, and seed averaging. The ICU-death
exclusion conditioned on final survival status and should not be used as the
primary estimand. The NRI/IDI analyses were exploratory. MIMIC-III shared the
institution but not the era and therefore cannot separate historical practice from
    fixed-weight transport. Hospital-held-out eICU-CRD validation keeps hospitals
    separate across folds but remains retrospective internal-external validation, not prospective
deployment. Six-direction transport confounds institution, era, case mix,
documentation, and outcome prevalence. Recalibration used at most 200 labelled
target outcomes over deterministic splits and does not establish a universal label
budget. Temperature scaling was prespecified as a simpler comparator, but both
calibrators were evaluated only over finite local samples. The instability-risk
and reassessment analysis reused the destination model fold assignments, and its
trigger AUC and simulation intervals are conditional on fitted cross-fitted
predictions. It did not independently repeat trigger-model training or use a
prospective trigger cohort.

Most importantly, no analysis assigned reassessment or prediction to a care
pathway. Decision-curve coordinates describe threshold-specific model behaviour.
    Stability, movement toward or away from the observed final class, and transport performance do not show that a
reassessment trigger is associated with better discharge planning, resource use, or patient
outcomes. A clinical-utility claim would require a prospective workflow study with
prespecified actions and outcome endpoints.

Future work should evaluate whether scheduled reassessment based on early
instability risk is associated with better planning metrics without higher alarm burden, and whether
prospective outcome and placement data can distinguish a genuine trajectory
change from documentation or coding noise. External prospective validation should
freeze weights before transport and preregister a direction-specific calibration
plan. The stability estimand can be added to model reports without changing the
underlying prediction task.
"""


def build_discussion(
    formal: dict,
    shared: dict,
    clinical: dict,
    transport: dict,
    mimic3_shared: dict,
    hospital_validation: dict,
    transport_extended: dict,
    trajectory_patterns: dict,
    flip_risk: dict,
) -> str:
    shared_m = shared["databases"]["mimic4"]["stability"]
    shared_e = shared["databases"]["eicu"]["stability"]
    clinical_m = clinical["databases"]["mimic4"]["models"]["xgboost"]
    clinical_e = clinical["databases"]["eicu"]["models"]["xgboost"]
    temporal_m = formal["mimic4"]["paired_contrasts"]["temporal_vs_none_ordinal"]
    temporal_e = formal["eicu"]["paired_contrasts"]["temporal_vs_none_ordinal"]
    temporal_clinical_m = clinical["databases"]["mimic4"]["temporal_vs_notemporal"]
    temporal_clinical_e = clinical["databases"]["eicu"]["temporal_vs_notemporal"]
    dg_m = shared["databases"]["mimic4"]["dg_comparison"]["stability"][
        "dg_gru_ordinal_temporal_three_seed_ensemble"
    ]
    dg_e = shared["databases"]["eicu"]["dg_comparison"]["stability"][
        "dg_gru_ordinal_temporal_three_seed_ensemble"
    ]
    mimic3_stability = mimic3_shared["stability"]
    mimic3_transitions = mimic3_shared["clinical_transitions"]
    hospital_comparison = hospital_validation[
        "comparison_with_patient_grouped"
    ]["hospital_minus_patient_grouped"]
    hospital_flip_difference = hospital_comparison["flip_24_72"]
    hospital_sites = hospital_validation["hospital_level"][
        "distributions_across_primary_sites"
    ]
    trajectory = trajectory_patterns["databases"]
    transport_72 = {
        direction: record["transport_landmarks"]["72"][
            "target_all_reaching_landmark"
        ]
        for direction, record in transport_extended["directions"].items()
    }
    recalibration = transport_extended["recalibration"]
    recalibration_25 = [
        record["72"]["curves"]["25"]["ece_top_label_10bin"]["mean"]
        for record in recalibration.values()
    ]
    recalibration_200 = [
        record["72"]["curves"]["200"]["ece_top_label_10bin"]["mean"]
        for record in recalibration.values()
    ]
    temperature_25 = [
        record["72"]["curves_temperature_scaling"]["25"][
            "ece_top_label_10bin"
        ]["mean"]
        for record in recalibration.values()
    ]
    temperature_worse_directions = sum(
        1
        for direction, record in recalibration.items()
        if record["72"]["curves_temperature_scaling"]["25"][
            "ece_top_label_10bin"
        ]["mean"]
        > transport_72[direction]["ece_top_label_10bin"]
    )
    temperature_better_directions = sum(
        1
        for direction, record in recalibration.items()
        if record["72"]["curves_temperature_scaling"]["25"][
            "ece_top_label_10bin"
        ]["mean"]
        < transport_72[direction]["ece_top_label_10bin"]
    )
    trigger_oof = {
        (record["database"], record["outcome"]): record
        for record in flip_risk["within_database_oof"]
    }
    any_flip_10 = {
        database: next(
            item
            for item in flip_risk["simulations"]["instability_risk_trigger"][
                "within_database"
            ][database]["risk_thresholds"]
            if item["threshold"] == 0.1
        )
        for database in ("mimic4", "eicu", "mimic3")
    }
    return f"""## 4. Discussion

The central finding is repeated class change in the model-predicted destination
for the same stay. With complete cohorts and shared patient folds, the 24-to-72 h flip
was {pct(shared_m['flip_24_72'])}, {pct(shared_e['flip_24_72'])}, and
{pct(mimic3_stability['flip_24_72']['estimate'])}. Any adjacent change occurred in
{pct(trajectory['mimic4']['clinical_trajectories']['any_adjacent_change']['estimate'])},
{pct(trajectory['eicu']['clinical_trajectories']['any_adjacent_change']['estimate'])},
and
{pct(trajectory['mimic3']['clinical_trajectories']['any_adjacent_change']['estimate'])}.
Predictions were stable at all landmarks in
{pct(trajectory['mimic4']['patterns']['stable_all']['estimate'])},
{pct(trajectory['eicu']['patterns']['stable_all']['estimate'])}, and
{pct(trajectory['mimic3']['patterns']['stable_all']['estimate'])}. Transient 48 h
excursions occurred in
{pct(trajectory['mimic4']['patterns']['transient_48h_excursion']['estimate'])},
{pct(trajectory['eicu']['patterns']['transient_48h_excursion']['estimate'])}, and
{pct(trajectory['mimic3']['patterns']['transient_48h_excursion']['estimate'])}.
The two-endpoint comparison missed changes visible in the full
trajectory. These results describe fitted prediction sequences, not verified
patient-state paths.

The predicted transitions involved different disposition categories and should
not be treated as interchangeable. Predicted class conversions from non-death
to death occurred in
{pct(clinical_m['clinical_major_upgrades']['major_upgrade_to_death']['estimate'])},
{pct(clinical_e['clinical_major_upgrades']['major_upgrade_to_death']['estimate'])},
and {pct(mimic3_transitions['major_upgrade_to_death']['estimate'])} of stays.
Home-to-facility movement occurred in
{pct(clinical_m['clinical_major_upgrades']['home_to_facility']['estimate'])},
{pct(clinical_e['clinical_major_upgrades']['home_to_facility']['estimate'])}, and
{pct(mimic3_transitions['home_to_facility']['estimate'])}. Movement relative to
the observed final class was also mixed. The final-class probability was higher in
{pct(clinical_m['true_class_probability']['among_flips']['improved']['estimate'])}
and
{pct(clinical_e['true_class_probability']['among_flips']['improved']['estimate'])}
of changed predictions. Eventual-class probability decreased in
{pct(1 - clinical_m['true_class_probability']['among_flips']['improved']['estimate'])}
and
{pct(1 - clinical_e['true_class_probability']['among_flips']['improved']['estimate'])}
of changed predictions. The MIMIC-III value was
{pct(mimic3_shared['correctness']['true_class_probability']['among_flips']['worsened']['estimate'])}.
A later prediction was not automatically better. The trigger performance
warrants prospective evaluation of a prespecified monitoring pathway, but it
does not support routine replacement of every earlier prediction.

The high-confidence stratum was smaller and had fewer later class changes. At
the 0.60 threshold, 24-to-72 h class changes occurred in
{pct(shared_m['confidence_24h']['0.6']['flip_24_72'])},
{pct(shared_e['confidence_24h']['0.6']['flip_24_72'])}, and
{pct(mimic3_stability['confidence_ge_0_6']['subsequent_flip_24_72']['estimate'])}.
The endpoint-change
trigger reached AUCs of
{fixed(trigger_oof[('mimic4', 'any_flip')]['metrics']['auc'])},
{fixed(trigger_oof[('eicu', 'any_flip')]['metrics']['auc'])}, and
{fixed(trigger_oof[('mimic3', 'any_flip')]['metrics']['auc'])}. The direct
trigger for model-predicted conversion reached
{fixed(trigger_oof[('mimic4', 'major_death_upgrade')]['metrics']['auc'])},
{fixed(trigger_oof[('eicu', 'major_death_upgrade')]['metrics']['auc'])}, and
{fixed(trigger_oof[('mimic3', 'major_death_upgrade')]['metrics']['auc'])}. At a
predicted-risk threshold of 0.10 for endpoint class change, the selected set
included
{pct(any_flip_10['mimic4']['major_death_upgrade_capture_fraction'])},
{pct(any_flip_10['eicu']['major_death_upgrade_capture_fraction'])}, and
{pct(any_flip_10['mimic3']['major_death_upgrade_capture_fraction'])} of
predicted non-death-to-death class conversions. Enhanced monitoring was still
selected for
{pct(any_flip_10['mimic4']['trigger_coverage'])},
{pct(any_flip_10['eicu']['trigger_coverage'])}, and
{pct(any_flip_10['mimic3']['trigger_coverage'])} of stays. Most selected stays
did not have such a predicted conversion. This tradeoff is a hypothesis for prospective
workflow study, not evidence of clinical utility.

The estimand is deliberately bounded. Three fixed snapshots estimate class
instability through 72 h. They do not estimate an exact time to stability. The
complete-cohort analysis removed fold-assignment variation from the main
comparison. Changes can still reflect the expanding predictor window, model
updating, or sampling variation. Evaluation-sample intervals are conditional on
the fitted models and folds. Hospital-held-out eICU-CRD validation produced a
{signed_pct(hospital_flip_difference['estimate'], 2)} percentage-point
difference. Its 95% CI was
{signed_pct(hospital_flip_difference['ci95'][0], 2)} to
{signed_pct(hospital_flip_difference['ci95'][1], 2)}. Hospital-held-out
validation produced a similar point estimate without a prespecified equivalence
margin. Site-level 72 h AUC had a P10-P90 range
from {fixed(hospital_sites['macro_auc_72']['p10'])} to
{fixed(hospital_sites['macro_auc_72']['p90'])}. The observed range was
{fixed(hospital_sites['macro_auc_72']['min'])} to
{fixed(hospital_sites['macro_auc_72']['max'])}.

A lower DG-GRU flip count cannot be read as greater reliability. Its flips were
{pct(dg_m['flip_24_72'])} and {pct(dg_e['flip_24_72'])}. Its total variation was
only {fixed(dg_m['tv_24_72'])} and {fixed(dg_e['tv_24_72'])}. Discrimination was
also lower. Temporal regularization produced mean flip changes across three
seeds of
{signed_pct(temporal_m['flip_0_2']['difference_mean'])} and
{signed_pct(temporal_e['flip_0_2']['difference_mean'])} percentage points. The
aligned changes were
{signed_fixed(100 * temporal_clinical_m['flip_rate_change']['estimate'], 2)} and
{signed_fixed(100 * temporal_clinical_e['flip_rate_change']['estimate'], 2)}.
This lower count included fewer movements toward or away from the observed
final class. Point estimates suggested fewer new predicted non-death-to-death
conversions and more deaths not predicted at 72 h. The paired intervals did not
establish both directions. The penalty has no clinical validation. A lower
class-change count is not a clinical endpoint.

Fixed-weight transport and local recalibration remain supporting evidence.
Transport macro-AUC ranged from
{fixed(min(record['macro_auc'] for record in transport_72.values()))} to
{fixed(max(record['macro_auc'] for record in transport_72.values()))}. Top-label
calibration error ranged from
{fixed(min(record['ece_top_label_10bin'] for record in transport_72.values()))}
to
{fixed(max(record['ece_top_label_10bin'] for record in transport_72.values()))}.
The same source-fixed weights performed differently across targets. With 25
labels, multinomial mean expected calibration error was
{fixed(min(recalibration_25))} to {fixed(max(recalibration_25))}. Mean expected
calibration error was lower for temperature scaling than for multinomial
calibration in all six directions. Paired percentile ranges crossed zero in
five directions. Temperature scaling nevertheless had lower expected
calibration error than untreated transport in only
{temperature_better_directions} of six directions. It had higher error in
{temperature_worse_directions} directions. At 200 labels, multinomial mean
expected calibration error was
{fixed(min(recalibration_200))} to {fixed(max(recalibration_200))}. The direction
of change still differed by setting. Monitoring and simple local updates require
direction-specific evaluation. A universal recalibration rule was not supported.

The ordinal order remains an assumption. Facility includes rehabilitation,
skilled nursing, long-term care, and surviving hospice discharge. It is not a
measured point on a biological severity scale. The nominal comparison and
reversed-order stress test show partial dependence on this encoding. Neither
analysis proves that facility lies between home and death clinically. The NRI and
IDI results are exploratory and do not validate the ordering.

Several limitations bound the interpretation. The stability cohort required
observation through 72 h and differed from earlier ICU exits. Sepsis was
identified from diagnosis codes. eICU-CRD also used diagnosis text containing
`sepsis` or `septic`. This text rule added 234 stays without a qualifying ICD-9
code. Nineteen were supported only by aseptic or septic-arthritis text.
Diagnosis timing was not restricted to the first 24 h. Destination labels came
from administrative fields. Surviving hospice discharge was mapped to facility
except MIMIC-III `HOSPICE-HOME`, which was mapped to home. Against-advice
discharge was mapped to home. Absolute home and facility shares were therefore
not used as cross-database descriptive evidence. Transport estimates retain this
mapping heterogeneity as a limitation. Discharge destination also reflects social
support, insurance, and local bed availability. Facility combines heterogeneous
placement settings. A small probability change across a class boundary could
trigger a flip as no minimum movement was required. No class-imbalance
resampling was applied, so observed flip frequency may partly reflect class
prevalence at the decision boundary. MIMIC-IV is single-center.
eICU-CRD has heterogeneous documentation and missing temperature or
neurological observations. Ninety-eight patients contributing 216 rows appeared
at multiple eICU-CRD hospitals. Hospital bootstrap could therefore give these
patients repeated representation. Hospital-held-out validation excluded them.
The XGBoost-to-DG-GRU contrast combines architecture, inputs, and seed averaging.
ICU-death exclusion used final survival status. MIMIC-III is a same-institution
replication. Its source period overlaps MIMIC-IV from 2008 through 2012, so the
analysis is not fully independent cross-era validation. Hospital-held-out
eICU-CRD analysis is internal-external validation. Transport combines
institution, era, case mix, documentation, and prevalence. Recalibration used
at most 200 labelled outcomes. The reassessment model reused destination fold
assignments.

The study has several strengths. It uses explicit landmark populations
and one conditional stability estimand. Shared patient folds align the three
landmark models. Complete trajectories distinguish transient and successive
changes from other changes. Transition-specific and movement-based results
show why a single flip rate is insufficient. Hospital-clustered uncertainty
and hospital-held-out validation address clustering. Paired model contrasts
aligned the temporal comparisons but did not separate architecture from
input representation. Transport, recalibration, and reassessment analyses
remain transparently secondary.

Future work should test enhanced monitoring prospectively. Such work needs
prespecified actions, workload measures, and patient or placement endpoints.
Prospective monitoring should distinguish true state change from documentation
noise. Transport studies should freeze weights and report calibration by
direction. Stability should be added to model reports together with
discrimination, probability movement, and movement relative to the observed final class.
"""


def build_conclusions(
    shared: dict,
    clinical: dict,
    mimic3_shared: dict,
    transport_extended: dict,
    trajectory_patterns: dict,
    flip_risk: dict,
) -> str:
    shared_m = shared["databases"]["mimic4"]["stability"]
    shared_e = shared["databases"]["eicu"]["stability"]
    shared_3 = mimic3_shared["stability"]
    clinical_m = clinical["databases"]["mimic4"]["models"]["xgboost"]
    clinical_e = clinical["databases"]["eicu"]["models"]["xgboost"]
    mimic3_transitions = mimic3_shared["clinical_transitions"]
    trajectory = trajectory_patterns["databases"]
    transport_72 = [
        record["transport_landmarks"]["72"][
            "target_all_reaching_landmark"
        ]["macro_auc"]
        for record in transport_extended["directions"].values()
    ]
    any_flip = {
        database: next(
            item
            for item in flip_risk["simulations"]["instability_risk_trigger"][
                "within_database"
            ][database]["risk_thresholds"]
            if item["threshold"] == 0.1
        )
        for database in ("mimic4", "eicu", "mimic3")
    }
    recalibration_25 = [
        record["72"]["curves"]["25"]["ece_top_label_10bin"]["mean"]
        for record in transport_extended["recalibration"].values()
    ]
    temperature_25 = [
        record["72"]["curves_temperature_scaling"]["25"][
            "ece_top_label_10bin"
        ]["mean"]
        for record in transport_extended["recalibration"].values()
    ]
    temperature_ci_cross_zero = sum(
        1
        for record in transport_extended["recalibration"].values()
        if record["72"]["temperature_scaling_minus_multinomial_logistic"]["25"][
            "ece_top_label_10bin"
        ]["p2_5"]
        <= 0
        <= record["72"]["temperature_scaling_minus_multinomial_logistic"]["25"][
            "ece_top_label_10bin"
        ]["p97_5"]
    )
    trigger_auc = {
        (record["database"], record["outcome"]): record["metrics"]["auc"]
        for record in flip_risk["within_database_oof"]
    }
    return f"""## 5. Conclusions

Among sepsis ICU stays observed through 72 h, a substantial minority of
destination predictions changed class. The 24-to-72 h class-change rate was
{pct(shared_m['flip_24_72'])} in MIMIC-IV,
{pct(shared_e['flip_24_72'])} in eICU-CRD, and
{pct(shared_3['flip_24_72']['estimate'])} in MIMIC-III. Any adjacent change occurred in
{pct(trajectory['mimic4']['clinical_trajectories']['any_adjacent_change']['estimate'])},
{pct(trajectory['eicu']['clinical_trajectories']['any_adjacent_change']['estimate'])},
and
{pct(trajectory['mimic3']['clinical_trajectories']['any_adjacent_change']['estimate'])}.
Predictions were stable at all landmarks in
{pct(trajectory['mimic4']['patterns']['stable_all']['estimate'])},
{pct(trajectory['eicu']['patterns']['stable_all']['estimate'])}, and
{pct(trajectory['mimic3']['patterns']['stable_all']['estimate'])}. One aggregate
flip rate would conceal transitions across different disposition categories.

Predicted class conversions from non-death to death occurred in
{pct(clinical_m['clinical_major_upgrades']['major_upgrade_to_death']['estimate'])},
{pct(clinical_e['clinical_major_upgrades']['major_upgrade_to_death']['estimate'])},
and {pct(mimic3_transitions['major_upgrade_to_death']['estimate'])} of stays.
Home-to-facility movement occurred in
{pct(clinical_m['clinical_major_upgrades']['home_to_facility']['estimate'])},
{pct(clinical_e['clinical_major_upgrades']['home_to_facility']['estimate'])}, and
{pct(mimic3_transitions['home_to_facility']['estimate'])}. Three landmarks
characterize the
proportion that remains unstable. They do not estimate an exact time to
stability.

The high-confidence stratum was smaller and had fewer later class changes.
Endpoint class change was also discriminable, with AUCs of
{fixed(trigger_auc[('mimic4', 'any_flip')])},
{fixed(trigger_auc[('eicu', 'any_flip')])}, and
{fixed(trigger_auc[('mimic3', 'any_flip')])}. At a predicted-risk threshold of
0.10, the selected set included
{pct(any_flip['mimic4']['major_death_upgrade_capture_fraction'])},
{pct(any_flip['eicu']['major_death_upgrade_capture_fraction'])}, and
{pct(any_flip['mimic3']['major_death_upgrade_capture_fraction'])} of predicted
non-death-to-death conversions. Fewer stays were selected than when monitoring
every stay. Most selected stays did not have such a predicted conversion. This tradeoff
remains a hypothesis for prospective workflow evaluation.

Lower class-change counts were not evidence of a better model. The DG-GRU also
had lower discrimination and smaller probability movement. Temporal
regularization showed movements toward or away from the observed final class
and has no clinical validation. Transport and
local recalibration were direction-specific supporting analyses.

Fixed-weight transport produced 72 h macro-AUCs from
{fixed(min(transport_72))} to {fixed(max(transport_72))}. With 25 labels,
multinomial mean expected calibration error was
{fixed(min(recalibration_25))} to {fixed(max(recalibration_25))}. Temperature
scaling point estimates were lower than the multinomial calibrator in all six
directions. Paired percentile intervals crossed zero in
{temperature_ci_cross_zero} of six directions. At the same label count, temperature scaling had higher expected
calibration error than untreated transport in four of six directions. Transport
and recalibration did not establish cross-site readiness. No analysis
established clinical benefit.
Stability, predicted transitions, probability movement, and movement relative
to the observed final class should accompany discrimination in model reports.
"""


def build_highlights(
    formal: dict,
    shared: dict,
    clinical: dict,
    mimic3_shared: dict,
    hospital_validation: dict,
    transport_extended: dict,
    trajectory_patterns: dict,
    flip_risk: dict,
) -> str:
    shared_m = shared["databases"]["mimic4"]["stability"]
    shared_e = shared["databases"]["eicu"]["stability"]
    clinical_m = clinical["databases"]["mimic4"]["models"]["xgboost"]
    clinical_e = clinical["databases"]["eicu"]["models"]["xgboost"]
    compact_m = shared_m["flip_24_72"]
    compact_e = shared_e["flip_24_72"]
    compact_3 = mimic3_shared["stability"]["flip_24_72"]["estimate"]
    hospital_difference = hospital_validation[
        "comparison_with_patient_grouped"
    ]["hospital_minus_patient_grouped"]["flip_24_72"]
    transport_72 = {
        direction: record["transport_landmarks"]["72"][
            "target_all_reaching_landmark"
        ]
        for direction, record in transport_extended["directions"].items()
    }
    trajectory = trajectory_patterns["databases"]
    any_flip_10 = {
        database: next(
            item
            for item in flip_risk["simulations"]["instability_risk_trigger"][
                "within_database"
            ][database]["risk_thresholds"]
            if item["threshold"] == 0.1
        )
        for database in ("mimic4", "eicu", "mimic3")
    }
    temporal_m = formal["mimic4"]["paired_contrasts"][
        "temporal_vs_none_ordinal"
    ]["flip_0_2"]["difference_mean"]
    temporal_e = formal["eicu"]["paired_contrasts"][
        "temporal_vs_none_ordinal"
    ]["flip_0_2"]["difference_mean"]
    return f"""# Highlights

- In complete three-landmark cohorts with shared patient folds, discharge-destination class changed between 24 and 72 h in {pct(compact_m)}, {pct(compact_e)}, and {pct(compact_3)} of MIMIC-IV, eICU-CRD, and MIMIC-III stays.
- The prediction class was unchanged at all three landmarks in {pct(trajectory['mimic4']['patterns']['stable_all']['estimate'])}, {pct(trajectory['eicu']['patterns']['stable_all']['estimate'])}, and {pct(trajectory['mimic3']['patterns']['stable_all']['estimate'])} of stays; transient 48 h excursions occurred in {pct(trajectory['mimic4']['patterns']['transient_48h_excursion']['estimate'])}, {pct(trajectory['eicu']['patterns']['transient_48h_excursion']['estimate'])}, and {pct(trajectory['mimic3']['patterns']['transient_48h_excursion']['estimate'])}.
- Hospital-held-out minus patient-grouped flip was {signed_pct(hospital_difference['estimate'], 2)} percentage points, with a 95% CI from {signed_pct(hospital_difference['ci95'][0], 2)} to {signed_pct(hospital_difference['ci95'][1], 2)}; the instability result persisted when hospitals were held out.
- Predicted non-death-to-death class conversions occurred in {pct(clinical_m['clinical_major_upgrades']['major_upgrade_to_death']['estimate'])} and {pct(clinical_e['clinical_major_upgrades']['major_upgrade_to_death']['estimate'])} of MIMIC-IV and eICU-CRD stays.
- Confidence available at 24 h identified a smaller subgroup with fewer later class changes.
- A lower DG-GRU class-change rate accompanied lower discrimination and smaller probability movement, so fewer changes did not indicate a better model.
- Temporal regularization changed class-change rates by {signed_pct(temporal_m)} and {signed_pct(temporal_e)} percentage points without closing the discrimination gap.
- Fixed-weight 72 h transport AUC ranged from {fixed(min(record['macro_auc'] for record in transport_72.values()))} to {fixed(max(record['macro_auc'] for record in transport_72.values()))} across six institution-era directions, with direction-specific calibration.
- At a predicted-risk threshold of 0.10, the selected set included {pct(any_flip_10['mimic4']['major_death_upgrade_capture_fraction'])}, {pct(any_flip_10['eicu']['major_death_upgrade_capture_fraction'])}, and {pct(any_flip_10['mimic3']['major_death_upgrade_capture_fraction'])} of predicted non-death-to-death class conversions in retrospective simulation; this is not clinical utility.
"""


def replace_between(text: str, start: str, end: str, replacement: str) -> str:
    pattern = re.compile(
        re.escape(start) + r".*?(?=" + re.escape(end) + r")", re.DOTALL
    )
    updated, count = pattern.subn(replacement.rstrip() + "\n\n", text, count=1)
    if count != 1:
        raise ValueError(f"Could not replace section between {start!r} and {end!r}")
    return updated


def main() -> None:
    formal = read_json(RESULTS / "formal_summary" / "summary.json")
    xgb = read_json(RESULTS / "landmarks" / "xgb_summary.json")
    stability = read_json(RESULTS / "stability_v3" / "primary_stability.json")
    shared = read_json(RESULTS / "shared_fold" / "summary.json")
    clinical = read_json(RESULTS / "clinical_flips" / "summary.json")
    transport = read_json(RESULTS / "transport" / "summary.json")
    sensitivity = read_json(RESULTS / "sensitivity" / "sensitivity_summary.json")
    sensitivity_v2 = read_json(
        RESULTS / "sensitivity_v2" / "sensitivity_v2_summary.json"
    )
    mimic3_shared = read_json(RESULTS / "mimic3_shared" / "summary.json")
    hospital_validation = read_json(
        RESULTS / "eicu_hospital_validation" / "summary.json"
    )
    transport_extended = read_json(
        RESULTS / "transport_extended" / "summary.json"
    )
    trajectory_patterns = read_json(
        RESULTS / "trajectory_patterns" / "summary.json"
    )
    flip_risk = read_json(RESULTS / "flip_risk" / "summary.json")
    reference_library = load_reference_library()
    template = (ROOT / "manuscript" / "manuscript.md").read_text(encoding="utf-8")

    assembled = replace_between(
        template,
        "## Abstract",
        "## 1. Introduction",
        build_abstract(
            formal,
            shared,
            clinical,
            transport,
            mimic3_shared,
            hospital_validation,
            transport_extended,
            trajectory_patterns,
            flip_risk,
        ),
    )
    results_text = normalize_reported_statistics(build_results(
        formal,
        xgb,
        stability,
        sensitivity,
        sensitivity_v2,
        shared,
        clinical,
        transport,
        mimic3_shared,
        hospital_validation,
        transport_extended,
        trajectory_patterns,
        flip_risk,
    ))
    assembled = assembled.replace(
        "{{RESULTS}}", results_text.replace("## 3. Results\n\n", "")
    )
    discussion_text = normalize_reported_statistics(build_discussion(
        formal,
        shared,
        clinical,
        transport,
        mimic3_shared,
        hospital_validation,
        transport_extended,
        trajectory_patterns,
        flip_risk,
    ))
    assembled = assembled.replace(
        "{{DISCUSSION}}", discussion_text.replace("## 4. Discussion\n\n", "")
    )
    conclusions_text = normalize_reported_statistics(build_conclusions(
        shared,
        clinical,
        mimic3_shared,
        transport_extended,
        trajectory_patterns,
        flip_risk,
    ))
    assembled = assembled.replace(
        "{{CONCLUSIONS}}",
        conclusions_text.replace("## 5. Conclusions\n\n", ""),
    )
    assembled, citation_order = resolve_citations(assembled, reference_library)
    assembled = assembled.replace(
        "{{REFERENCES}}", reference_list(citation_order, reference_library)
    )
    assembled = renumber_assets(assembled)
    assembled = re.sub(r"\n{3,}", "\n\n", assembled).strip() + "\n"

    if "{{" in assembled or "}}" in assembled:
        raise ValueError("Unresolved manuscript placeholder")

    output = OUTPUTS / "manuscript_sepsis_stability.md"
    output.write_text(assembled, encoding="utf-8")
    (OUTPUTS / "highlights.md").write_text(
        build_highlights(
            formal,
            shared,
            clinical,
            mimic3_shared,
            hospital_validation,
            transport_extended,
            trajectory_patterns,
            flip_risk,
        ),
        encoding="utf-8",
    )

    abstract = assembled.split("## Abstract", 1)[1].split("## 1. Introduction", 1)[0]
    word_count = len(re.findall(r"\b[\w'-]+\b", abstract))
    audit = {
        "manuscript": str(output),
        "abstract_word_count": word_count,
        "reference_count": len(citation_order),
        "formal_configurations": sum(
            len(formal[database]["configurations"]) for database, _ in DATABASES
        ),
        "title": assembled.splitlines()[0].lstrip("# ").strip(),
        "first_visible_at_72h": first_appeared_at_72(),
    }
    with (OUTPUTS / "manuscript_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2)
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
