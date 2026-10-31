# -*- coding: utf-8 -*-
"""Check formal run completeness, row alignment, and manuscript consistency."""

import json
import os
import re
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from docx import Document
from docx.oxml.ns import qn
from sklearn.metrics import roc_auc_score


def read_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


ROOT = Path(os.environ.get("SEPSIS_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
FORMAL = ROOT / "results" / "formal"
CONFIGS = (
    "ordinal_notemporal",
    "ordinal_temporal",
    "nominal_notemporal",
    "nominal_temporal",
)
SEEDS = (1, 2, 3)
LANDMARKS = (24, 48, 72)
TITLE = (
    "Predicted-Class Stability and Transport of Discharge-Disposition Models "
    "Among ICU Stays With Sepsis at 24, 48, and 72 Hours: A Three-Database "
    "Study of Recalibration and Enhanced-Monitoring Trigger Simulation"
)
AUTHOR_LINE = (
    "Zhipeng Wang1*, Luning Wang2*, Changsong Wang1, Xiyu Xu3, "
    "Pengli Zhai4, Zejun Liu1, Hui Feng1, Qian Hou1, Ming Guo1"
)
AUTHOR_AFFILIATIONS = (
    "1 Department of TCM, Zhongda Hospital, Southeast University, Nanjing, China",
    "2 Department of Rehabilitation Medicine, Zhongda Hospital, Southeast University, Nanjing, China",
    "3 School of Medicine, Southeast University, Nanjing, China",
    "4 Jiangbei Campus, Jiangsu Provincial Traditional Chinese Medicine Hospital, Nanjing, China",
)
AUTHOR_METADATA = "Zhipeng Wang; Luning Wang; Changsong Wang; Xiyu Xu; Pengli Zhai; Zejun Liu; Hui Feng; Qian Hou; Ming Guo"
COFIRST_TEXT = (
    "* These authors contributed equally to this work and are co-first authors."
)
CORRESPONDING_TEXT = (
    "Corresponding author: Zhipeng Wang, Department of TCM, Zhongda Hospital, "
    "Southeast University, China. Email: 311000687@seu.edu.cn"
)
DECLARATION_TEXT = (
    "6. Declarations",
    "Data availability",
    "MIMIC-IV version 3.1",
    "Code availability",
    "Reporting guidelines",
    "TRIPOD+AI",
    "Ethics approval and consent to participate",
    "Authors' contributions",
    "Study protocol and patient involvement",
    "Higher Education Teaching Reform Research Project of Xuzhou Medical University",
    "Clinical trial number",
    "Not applicable.",
    "Competing interests",
    "The authors declare no competing interests.",
    "Declaration of generative AI and AI-assisted technologies",
    "Consent for publication",
)
SOURCE_ROOT = Path(
    os.environ.get("SEPSIS_SOURCE_ROOT", r"D:\TT paper\0811Temporal Pathway")
)
MAIN_FIGURE_STEMS = (
    "figure1_cohort_outcome",
    "figure2_model",
    "figure3_discrimination",
    "figure4_calibration",
    "figure5_stability",
    "figure6_ablation",
    "figure7_transport_reassessment",
    "figure8_reassessment_simulation",
    "figure9_sensitivity_analyses",
    "figure10_hospital_validation",
    "figure11_clinical_transitions_safety",
)
MAIN_TABLE_NAMES = (
    "table1_cohort.md",
    "table2_landmarks.md",
    "table3_discrimination.md",
    "table4_stability.md",
    "table5_sensitivity.md",
    "table6_transport.md",
    "table7_recalibration_reassessment.md",
    "table8_shared_fold_metrics.md",
    "table9_hospital_validation.md",
    "table10_clinical_transitions.md",
)
SUPPLEMENTARY_TABLE_NAMES = (
    "supplementary_table_s1_decision_curves.md",
    "supplementary_table_s2_contrasts.md",
    "supplementary_table_s3_model_metrics.md",
    "supplementary_table_s4_calibration.md",
    "supplementary_table_s5_ordinal_clinical.md",
    "supplementary_table_s6_clinical_flips.md",
    "supplementary_table_s7_shared_fold.md",
    "supplementary_table_s8_hospital_validation.md",
    "supplementary_table_s9_transport_full.md",
    "supplementary_table_s10_recalibration.md",
    "supplementary_table_s11_reassessment.md",
    "supplementary_table_s12_tripod_ai.md",
)
TRANSPORT_DIRECTIONS = {
    "mimic4_to_eicu",
    "eicu_to_mimic4",
    "mimic4_to_mimic3",
    "mimic3_to_mimic4",
    "mimic3_to_eicu",
    "eicu_to_mimic3",
}
TITLE_DIRECTION_TERMS = (
    "Predicted-Class Stability",
    "Transport",
    "Sepsis",
    "Three-Database",
    "Recalibration",
    "Enhanced-Monitoring",
)


def canonical_database_name(value: str) -> str:
    """Normalize database labels used in Markdown transport tables."""
    token = re.sub(r"[^a-z0-9]+", "", value.lower())
    aliases = {
        "mimiciv": "mimic4",
        "mimic4": "mimic4",
        "mimiciii": "mimic3",
        "mimic3": "mimic3",
        "eicu": "eicu",
        "eicucrd": "eicu",
    }
    return aliases.get(token, token)


def parse_transport_direction(cell: str) -> tuple[str, str] | None:
    """Parse source and target from a Markdown direction cell."""
    normalized = cell.strip().lower().replace("→", "->")
    parts = re.split(r"->|\bto\b", normalized, maxsplit=1)
    if len(parts) != 2:
        return None
    source, target = (canonical_database_name(part) for part in parts)
    if source not in {"mimic3", "mimic4", "eicu"}:
        return None
    if target not in {"mimic3", "mimic4", "eicu"}:
        return None
    return source, target


def markdown_rows(path: Path) -> list[list[str]]:
    """Return cell contents for Markdown pipe-table rows."""
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if cells and not all(re.fullmatch(r":?-+:?", cell) for cell in cells):
            rows.append(cells)
    return rows


def first_asset_citations(text: str, label: str) -> list[int]:
    """Return unique display numbers in first-citation order."""
    body = text.split("## 7. References", 1)[0]
    if label == "Table":
        pattern = re.compile(r"(?<!Supplementary )Table\s+([1-9][0-9]*)\b")
    else:
        pattern = re.compile(r"Figure\s+([1-9][0-9]*)(?![0-9])")
    order = []
    seen = set()
    for value in pattern.findall(body):
        number = int(value)
        if number not in seen:
            seen.add(number)
            order.append(number)
    return order


def citation_numbers(text: str) -> list[int]:
    """Expand numeric citation groups in first-appearance order."""
    numbers: list[int] = []
    pattern = re.compile(r"\[(\d+(?:-\d+)?(?:,\d+(?:-\d+)?)*)\]")
    for group in pattern.findall(text):
        for part in group.split(","):
            if "-" in part:
                start, end = (int(value) for value in part.split("-", 1))
                numbers.extend(range(start, end + 1))
            else:
                numbers.append(int(part))
    return numbers


def citation_endpoint_numbers(text: str) -> list[int]:
    """Return every directly cited number, using range endpoints only."""
    numbers: list[int] = []
    pattern = re.compile(r"\[(\d+(?:-\d+)?(?:,\d+(?:-\d+)?)*)\]")
    for group in pattern.findall(text):
        for part in group.split(","):
            if "-" in part:
                start, end = (int(value) for value in part.split("-", 1))
                numbers.extend((start, end))
            else:
                numbers.append(int(part))
    return numbers


def p_value_failures(text: str) -> list[str]:
    failures = []
    for operator, raw_value in re.findall(
        r"\bP\s*(<|≤|=|>|≥)\s*(\d*\.\d+)", text
    ):
        value = float(raw_value)
        if operator in {"<", "≤", "="} and value < 0.001:
            if raw_value != "0.001":
                failures.append(f"P{operator}{raw_value}")
            continue
        decimals = len(raw_value.split(".", 1)[1])
        if decimals != 3:
            failures.append(f"P{operator}{raw_value}")
    return failures


def decimal_format_failures(text: str) -> list[str]:
    """Allow two decimals generally and four only for non-zero values below 0.01."""
    failures = []
    reporting_lines = [
        line for line in text.splitlines() if not line.lstrip().startswith("#")
    ]
    reporting_text = "\n".join(reporting_lines)
    reporting_text = re.sub(
        r"\bSections?\s+\d+(?:\.\d+)?"
        r"(?:(?:\s*(?:,|and|to)\s*)\d+(?:\.\d+)?)*",
        "",
        reporting_text,
    )
    for raw_value in re.findall(
        r"(?<![\w.])([+-]?\d+\.\d+)(?![\w.])", reporting_text
    ):
        decimals = len(raw_value.split(".", 1)[1])
        value = float(raw_value)
        expected = 4 if 0 < abs(value) < 0.01 else 2
        if decimals != expected:
            failures.append(raw_value)
    return sorted(set(failures))


def body_sentence_length_failures(
    manuscript_text: str, max_words: int = 30
) -> list[dict]:
    """Return narrative body sentences longer than the configured limit."""
    body = manuscript_text.split("## 7. References", 1)[0]
    body = re.sub(r"```.*?```", "", body, flags=re.DOTALL)
    sections = {
        part.splitlines()[0].strip(): "\n".join(part.splitlines()[1:])
        for part in re.split(r"(?m)^## ", body)[1:]
    }
    narrative_sections = (
        "Abstract",
        "1. Introduction",
        "2. Methods",
        "3. Results",
        "4. Discussion",
        "5. Conclusions",
    )
    failures: list[dict] = []
    for section_name in narrative_sections:
        section_text = sections.get(section_name, "")
        section_text = re.sub(r"(?m)^\s*#{2,6}\s+.*$", "", section_text)
        section_text = re.sub(r"(?m)^\s*\|.*$", "", section_text)
        section_text = re.sub(r"`[^`\n]*`", "", section_text)
        section_text = re.sub(r"\[(\d+(?:-\d+)?(?:,\d+(?:-\d+)?)*)\]", "", section_text)
        section_text = section_text.replace("**", "")
        for paragraph in re.split(r"\n\s*\n", section_text):
            paragraph = " ".join(paragraph.split())
            if not paragraph:
                continue
            paragraph = re.sub(r"^(?:\d+\.|[-*])\s+", "", paragraph)
            sentences = re.split(
                r"(?<=[.!?])\s+(?=(?:[A-Z0-9]|eICU|iPhone))",
                paragraph,
            )
            for sentence in sentences:
                word_count = len(
                    re.findall(r"\b[\w'’\-]+\b", sentence, flags=re.UNICODE)
                )
                if word_count > max_words:
                    failures.append(
                        {
                            "section": section_name,
                            "word_count": word_count,
                            "sentence": sentence,
                        }
                    )
    return failures


def outcome_mapping_checks() -> list[dict]:
    checks = []
    audit_path = ROOT / "work" / "mimic4_outcome_relabel_audit.json"
    if not audit_path.exists():
        checks.append(
            {
                "check": "MIMIC-IV HOSPICE outcome audit",
                "status": "fail",
                "detail": "missing relabel audit",
            }
        )
    else:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        survivor_classes = audit["validation"].get(
            "cohort_hospice_survivor_classes", {}
        )
        passed = (
            audit["validation"]["cohort_hospice_survivors"] > 0
            and set(survivor_classes) == {"2.0"}
        )
        checks.append(
            {
                "check": "MIMIC-IV surviving HOSPICE mapped to facility",
                "source_hospice_survivors": audit["validation"][
                    "source_hospice_survivors"
                ],
                "cohort_hospice_survivors": audit["validation"][
                    "cohort_hospice_survivors"
                ],
                "cohort_classes": survivor_classes,
                "status": "pass" if passed else "fail",
            }
        )

    eicu = pd.read_parquet(
        ROOT / "data" / "processed" / "eicu" / "cohort_v2_all.parquet",
        columns=["age"],
    )
    processed_gt89 = int((eicu["age"] == 90).sum())
    connection = duckdb.connect()
    connection.execute("PRAGMA threads=4")
    source = str(SOURCE_ROOT / "eicu").replace("\\", "/").replace("'", "''")
    source_gt89 = int(
        connection.execute(
            f"""
            WITH SEP AS (
                SELECT DISTINCT patientunitstayid
                FROM read_csv_auto('{source}/diagnosis.csv.gz')
                WHERE replace(icd9code, '.', '') LIKE '038%'
                   OR replace(icd9code, '.', '') LIKE '9959%'
                   OR replace(icd9code, '.', '') LIKE '78552%'
                   OR lower(diagnosisstring) LIKE '%sepsis%'
                   OR lower(diagnosisstring) LIKE '%septic%'
            )
            SELECT COUNT(*)
            FROM read_csv_auto('{source}/patient.csv.gz') p
            JOIN SEP USING (patientunitstayid)
            WHERE p.unitvisitnumber = 1 AND trim(p.age) = '> 89'
            """
        ).fetchone()[0]
    )
    checks.append(
        {
            "check": "eICU age >89 included as 90",
            "source_count": source_gt89,
            "processed_count": processed_gt89,
            "status": "pass" if source_gt89 == processed_gt89 else "fail",
        }
    )

    mimic3 = pd.read_parquet(
        ROOT / "data" / "processed" / "mimic3" / "cohort.parquet",
        columns=["DISCHARGE_LOCATION", "y3"],
    )
    psych = mimic3.loc[
        mimic3["DISCHARGE_LOCATION"].eq("DISCH-TRAN TO PSYCH HOSP")
    ]
    checks.append(
        {
            "check": "MIMIC-III psych-transfer spelling mapped to facility",
            "n": int(len(psych)),
            "n_mapped_facility": int((psych["y3"] == 1).sum()),
            "status": (
                "pass"
                if len(psych) > 0 and (psych["y3"] == 1).all()
                else "fail"
            ),
        }
    )

    eicu_design = pd.read_parquet(
        ROOT / "data" / "processed" / "eicu" / "design_v3.parquet",
        columns=["patientunitstayid", "y3"],
    )
    eicu_sequence = np.load(ROOT / "data" / "processed" / "eicu" / "seq72.npz")
    eicu_aligned = (
        np.array_equal(
            eicu_design["patientunitstayid"].to_numpy(dtype=np.int64),
            eicu_sequence["patientunitstayid"].astype(np.int64),
        )
        and np.array_equal(
            eicu_design["y3"].to_numpy(dtype=np.int64),
            eicu_sequence["y3"].astype(np.int64),
        )
        and len(eicu_design) == 9215
    )
    checks.append(
        {
            "check": "eICU corrected model view and sequence alignment",
            "design_rows": int(len(eicu_design)),
            "sequence_rows": int(len(eicu_sequence["X"])),
            "status": "pass" if eicu_aligned else "fail",
        }
    )
    return checks


def main() -> None:
    checks = outcome_mapping_checks()
    expected = []
    for database in ("mimic4", "eicu"):
        for configuration in CONFIGS:
            for seed in SEEDS:
                stem = FORMAL / database / f"{configuration}_s{seed}"
                expected.append(stem.with_suffix(".json"))
                expected.append(stem.with_suffix(".npz"))
    missing = [str(path) for path in expected if not path.exists()]
    checks.append(
        {
            "check": "formal run files complete",
            "expected": len(expected),
            "missing": missing,
            "status": "pass" if not missing else "fail",
        }
    )
    if missing:
        raise FileNotFoundError(f"Missing formal artifacts: {missing[:5]}")

    sensitivity_v2 = (
        ROOT / "results" / "sensitivity_v2" / "sensitivity_v2_summary.json"
    )
    if not sensitivity_v2.exists():
        raise FileNotFoundError(sensitivity_v2)
    sensitivity = json.loads(sensitivity_v2.read_text(encoding="utf-8"))
    sensitivity_required = [
        ROOT / record for record in sensitivity["artifacts"].values()
    ]
    sensitivity_missing = [
        str(path) for path in sensitivity_required if not path.exists()
    ]
    reverse_expected = [
        ROOT
        / "results"
        / "sensitivity_v2"
        / "formal_reverse"
        / database
        / f"reverse_notemporal_s{seed}.{extension}"
        for database in ("mimic4", "eicu")
        for seed in SEEDS
        for extension in ("json", "npz")
    ]
    sensitivity_missing.extend(
        str(path) for path in reverse_expected if not path.exists()
    )
    checks.append(
        {
            "check": "source-recovered and reverse-order sensitivity artifacts",
            "expected": len(sensitivity_required) + len(reverse_expected),
            "missing": sensitivity_missing,
            "status": "pass" if not sensitivity_missing else "fail",
        }
    )

    corrected_paths = {
        "shared_fold": ROOT / "results" / "shared_fold" / "summary.json",
        "clinical_flips": ROOT / "results" / "clinical_flips" / "summary.json",
        "transport": ROOT / "results" / "transport" / "summary.json",
    }
    corrected_missing = [
        str(path) for path in corrected_paths.values() if not path.exists()
    ]
    corrected_payloads = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in corrected_paths.items()
        if path.exists()
    }
    corrected_failures = []
    if not corrected_missing:
        shared_summary = corrected_payloads["shared_fold"]
        clinical_summary = corrected_payloads["clinical_flips"]
        transport_summary = corrected_payloads["transport"]
        for database in ("mimic4", "eicu"):
            shared_record = shared_summary["databases"][database]
            clinical_record = clinical_summary["databases"][database]
            if shared_record["cohort"]["n"] != clinical_record["n"]:
                corrected_failures.append(f"{database}: corrected cohort size mismatch")
            if (
                abs(
                    shared_record["stability"]["flip_24_72"]
                    - clinical_record["models"]["xgboost"]["flip_24_72"]["estimate"]
                )
                > 1e-9
            ):
                corrected_failures.append(f"{database}: XGBoost flip mismatch")
        for direction in ("mimic4_to_eicu", "eicu_to_mimic4"):
            if direction not in transport_summary["transport"]:
                corrected_failures.append(f"transport direction missing: {direction}")
    checks.append(
        {
            "check": "corrected shared-fold, clinical and transport artifacts",
            "missing": corrected_missing,
            "failures": corrected_failures,
            "status": (
                "pass"
                if not corrected_missing and not corrected_failures
                else "fail"
            ),
        }
    )

    extended_required = {
        "MIMIC-III shared-fold summary": (
            ROOT / "results" / "mimic3_shared" / "summary.json"
        ),
        "MIMIC-III shared-fold predictions": (
            ROOT / "results" / "mimic3_shared" / "mimic3_shared.npz"
        ),
        "MIMIC-III shared-fold report": (
            ROOT / "results" / "mimic3_shared" / "report.md"
        ),
        "eICU hospital-held-out summary": (
            ROOT / "results" / "eicu_hospital_validation" / "summary.json"
        ),
        "eICU hospital-held-out predictions": (
            ROOT / "results" / "eicu_hospital_validation" / "eicu_hospital_fold.npz"
        ),
        "eICU hospital-level metrics": (
            ROOT / "results" / "eicu_hospital_validation" / "hospital_metrics.csv"
        ),
        "eICU hospital-held-out report": (
            ROOT / "results" / "eicu_hospital_validation" / "report.md"
        ),
        "six-direction transport summary": (
            ROOT / "results" / "transport_extended" / "summary.json"
        ),
        "transport recalibration learning curves": (
            ROOT
            / "results"
            / "transport_extended"
            / "recalibration_learning_curves.csv"
        ),
        "transport recalibration replicates": (
            ROOT / "results" / "transport_extended" / "recalibration_replicates.csv"
        ),
        "six-direction transport report": (
            ROOT / "results" / "transport_extended" / "report.md"
        ),
        "flip-risk summary": ROOT / "results" / "flip_risk" / "summary.json",
        "flip-risk predictions": (
            ROOT / "results" / "flip_risk" / "predictions.npz"
        ),
        "flip-risk report": ROOT / "results" / "flip_risk" / "report.md",
    }
    extended_missing = [
        str(path) for path in extended_required.values() if not path.exists()
    ]
    checks.append(
        {
            "check": "three-database extension artifacts complete",
            "expected": len(extended_required),
            "missing": extended_missing,
            "status": "pass" if not extended_missing else "fail",
        }
    )

    extension_payloads = {}
    extension_parse_failures = []
    for name, path in (
        ("mimic3", extended_required["MIMIC-III shared-fold summary"]),
        (
            "eicu_hospital",
            extended_required["eICU hospital-held-out summary"],
        ),
        (
            "transport_extended",
            extended_required["six-direction transport summary"],
        ),
        ("flip_risk", extended_required["flip-risk summary"]),
    ):
        if not path.exists():
            continue
        try:
            extension_payloads[name] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            extension_parse_failures.append(f"{name}: {error}")

    mimic3_failures = []
    mimic3 = extension_payloads.get("mimic3")
    if mimic3 is None:
        mimic3_failures.append("summary unavailable")
    else:
        cohort = mimic3.get("cohort", {})
        fold = mimic3.get("training", {}).get("fold_integrity", {})
        if cohort.get("n") != 3638 or cohort.get("patient_count") != 3330:
            mimic3_failures.append(
                "cohort is not the complete 3,638-stay/3,330-patient landmark cohort"
            )
        discrimination = mimic3.get("discrimination", {})
        if (
            set(discrimination.get("macro_auc", {})) != {"24", "48", "72"}
            or set(discrimination.get("landmarks", {})) != {"24", "48", "72"}
        ):
            mimic3_failures.append("24/48/72 h discrimination records incomplete")
        if fold.get("n_splits") != 5:
            mimic3_failures.append("expected five patient-grouped folds")
        if fold.get("same_fold_partition_reused_at_all_landmarks") is not True:
            mimic3_failures.append("landmark models do not reuse the same fold partition")
        if any(fold.get("held_out_subject_overlap_counts", [1])):
            mimic3_failures.append("held-out subjects overlap across folds")
        if fold.get("test_partitions_cover_every_row_once") is not True:
            mimic3_failures.append("test folds do not cover every row exactly once")
    mimic3_npz = extended_required["MIMIC-III shared-fold predictions"]
    if mimic3_npz.exists():
        try:
            with np.load(mimic3_npz) as bundle:
                required_keys = {"ids", "y3", "fold_id", "p24", "p48", "p72"}
                absent = sorted(required_keys.difference(bundle.files))
                if absent:
                    mimic3_failures.append(f"NPZ keys missing: {absent}")
                elif len(bundle["y3"]) != 3638:
                    mimic3_failures.append(
                        f"NPZ rows={len(bundle['y3'])}, expected 3638"
                    )
        except (OSError, ValueError) as error:
            mimic3_failures.append(f"NPZ unreadable: {error}")
    checks.append(
        {
            "check": "MIMIC-III cross-era replication uses complete landmark shared folds",
            "failures": mimic3_failures,
            "status": "pass" if not mimic3_failures else "fail",
        }
    )

    eicu_hospital_failures = []
    eicu_hospital = extension_payloads.get("eicu_hospital")
    if eicu_hospital is None:
        eicu_hospital_failures.append("summary unavailable")
    else:
        design = eicu_hospital.get("design", {})
        cohort = eicu_hospital.get("cohort", {})
        if design.get("validation_type") != "hospital-grouped internal-external validation":
            eicu_hospital_failures.append("validation type is not hospital-grouped")
        if design.get("strictly_external_validation") is not False:
            eicu_hospital_failures.append(
                "hospital-held-out analysis must not claim strict external validation"
            )
        if design.get("prospective_validation") is not False:
            eicu_hospital_failures.append(
                "hospital-held-out analysis must not claim prospective validation"
            )
        if not cohort.get("n") or not cohort.get("n_hospitals"):
            eicu_hospital_failures.append(
                "hospital-held-out cohort size metadata is missing"
            )
        if cohort.get("patients_observed_in_more_than_one_hospital") != 0:
            eicu_hospital_failures.append(
                "validation cohort still contains patients observed at multiple hospitals"
            )
        if cohort.get("patients_excluded_for_multiple_hospitals", 0) <= 0:
            eicu_hospital_failures.append(
                "cross-hospital patient exclusions were not recorded"
            )
        fold_integrity = eicu_hospital.get("training", {}).get(
            "fold_integrity", {}
        )
        if fold_integrity.get("patient_membership_is_whole_fold") is not True:
            eicu_hospital_failures.append(
                "patient-level fold integrity was not demonstrated"
            )
        if set(eicu_hospital.get("metrics", {})) != {"24", "48", "72"}:
            eicu_hospital_failures.append("hospital-held-out metrics incomplete")
        if set(eicu_hospital.get("fold_metrics", {})) != {"24", "48", "72"}:
            eicu_hospital_failures.append("hospital-level fold metrics incomplete")
        comparison = eicu_hospital.get(
            "comparison_with_patient_grouped", {}
        ).get("hospital_minus_patient_grouped", {})
        if "flip_24_72" not in comparison:
            eicu_hospital_failures.append(
                "paired patient-grouped versus hospital-held-out flip contrast missing"
            )
    eicu_hospital_npz = extended_required[
        "eICU hospital-held-out predictions"
    ]
    if eicu_hospital_npz.exists():
        try:
            with np.load(eicu_hospital_npz) as bundle:
                required_keys = {
                    "ids",
                    "hospital_ids",
                    "uniquepids",
                    "fold_id",
                    "y3",
                    "p24",
                    "p48",
                    "p72",
                }
                absent = sorted(required_keys.difference(bundle.files))
                if absent:
                    eicu_hospital_failures.append(f"NPZ keys missing: {absent}")
                elif len(bundle["y3"]) != cohort.get("n"):
                    eicu_hospital_failures.append(
                        f"NPZ rows={len(bundle['y3'])}, expected {cohort.get('n')}"
                    )
                elif len(np.unique(bundle["hospital_ids"])) != cohort.get(
                    "n_hospitals"
                ):
                    eicu_hospital_failures.append(
                        "NPZ hospital-group count does not match the summary"
                    )
                else:
                    patient_folds = pd.DataFrame(
                        {
                            "patient": bundle["uniquepids"].astype(str),
                            "fold": bundle["fold_id"],
                        }
                    )
                    cross_fold_patients = int(
                        patient_folds.groupby("patient")["fold"].nunique().gt(1).sum()
                    )
                    if cross_fold_patients:
                        eicu_hospital_failures.append(
                            f"{cross_fold_patients} patients cross hospital folds"
                        )
        except (OSError, ValueError) as error:
            eicu_hospital_failures.append(f"NPZ unreadable: {error}")
    hospital_csv = extended_required["eICU hospital-level metrics"]
    if hospital_csv.exists():
        hospital_metrics = pd.read_csv(hospital_csv)
        if len(hospital_metrics) != cohort.get("n_hospitals"):
            eicu_hospital_failures.append(
                "hospital CSV row count does not match the summary"
            )
        if "macro_auc_72" not in hospital_metrics.columns:
            eicu_hospital_failures.append("hospital CSV lacks 72 h macro-AUC")
    checks.append(
        {
            "check": "eICU hospital-held-out internal-external validation",
            "hospitals": (
                eicu_hospital.get("cohort", {}).get("n_hospitals")
                if eicu_hospital
                else None
            ),
            "failures": eicu_hospital_failures,
            "status": "pass" if not eicu_hospital_failures else "fail",
        }
    )

    transport_failures = []
    transport_extended = extension_payloads.get("transport_extended")
    if transport_extended is None:
        transport_failures.append("summary unavailable")
    else:
        directions = set(transport_extended.get("directions", {}))
        taxonomy = set(transport_extended.get("direction_taxonomy", {}))
        recalibration = set(transport_extended.get("recalibration", {}))
        if directions != TRANSPORT_DIRECTIONS:
            transport_failures.append(
                f"direction set mismatch: {sorted(directions)}"
            )
        if taxonomy != TRANSPORT_DIRECTIONS:
            transport_failures.append("transport taxonomy does not cover all six directions")
        if recalibration != TRANSPORT_DIRECTIONS:
            transport_failures.append(
                "target recalibration does not cover all six directions"
            )
        types = [
            record.get("type")
            for record in transport_extended.get("direction_taxonomy", {}).values()
        ]
        expected_type_counts = {
            "cross_institution": 2,
            "same_institution_cross_era": 2,
            "cross_institution_and_era": 2,
        }
        for type_name, count in expected_type_counts.items():
            if types.count(type_name) != count:
                transport_failures.append(
                    f"{type_name} directions={types.count(type_name)}, expected {count}"
                )
        if transport_extended.get("direction_taxonomy", {}).get(
            "mimic4_to_mimic3", {}
        ).get("type") != "same_institution_cross_era":
            transport_failures.append(
                "MIMIC-IV to MIMIC-III is not labeled same-institution cross-era"
            )
        for direction in sorted(TRANSPORT_DIRECTIONS):
            for milestone in ("24", "48", "72"):
                block = (
                    transport_extended.get("recalibration", {})
                    .get(direction, {})
                    .get(milestone, {})
                )
                target_n = block.get("target_n")
                target_patients = block.get("target_unique_patient_n")
                calibration_n = block.get("calibration_population_n")
                calibration_patients = block.get(
                    "calibration_population_unique_patient_n"
                )
                if (
                    target_n is None
                    or target_patients is None
                    or calibration_n is None
                    or calibration_patients is None
                    or calibration_n != target_patients
                    or calibration_n != calibration_patients
                    or target_patients > target_n
                ):
                    transport_failures.append(
                        f"{direction}/{milestone} h recalibration is not "
                        "one-stay-per-patient"
                    )
    checks.append(
        {
            "check": "six-direction fixed-weight transport and recalibration",
            "expected_directions": 6,
            "failures": transport_failures,
            "status": "pass" if not transport_failures else "fail",
        }
    )

    flip_risk_failures = []
    flip_risk = extension_payloads.get("flip_risk")
    if flip_risk is None:
        flip_risk_failures.append("summary unavailable")
    else:
        source_databases = {
            cohort.get("database") for cohort in flip_risk.get("cohorts", [])
        }
        if source_databases != {"mimic3", "mimic4", "eicu"}:
            flip_risk_failures.append(
                "flip-risk cohorts do not cover all three databases"
            )
        expected_triggers = {
            "instability_risk_trigger",
            "death_upgrade_risk_trigger",
        }
        if set(flip_risk.get("simulations", {})) != expected_triggers:
            flip_risk_failures.append(
                "both any-flip and death-upgrade reassessment simulations are required"
            )
        if flip_risk.get("simulation_definition", {}).get("clinical_utility") is not False:
            flip_risk_failures.append(
                "retrospective reassessment simulation must not claim clinical utility"
            )
        if (
            flip_risk.get("model", {}).get("estimator")
            != "L2-regularized binary logistic regression"
        ):
            flip_risk_failures.append(
                "instability-risk estimator is not the documented logistic model"
            )
    flip_npz = extended_required["flip-risk predictions"]
    if flip_npz.exists():
        try:
            with np.load(flip_npz) as bundle:
                required_keys = {
                    "mimic4__any_flip__event",
                    "eicu__any_flip__event",
                    "mimic3__any_flip__event",
                    "mimic4__major_death_upgrade__event",
                    "eicu__major_death_upgrade__event",
                    "mimic3__major_death_upgrade__event",
                }
                absent = sorted(required_keys.difference(bundle.files))
                if absent:
                    flip_risk_failures.append(f"NPZ keys missing: {absent}")
        except (OSError, ValueError) as error:
            flip_risk_failures.append(f"NPZ unreadable: {error}")
    checks.append(
        {
            "check": "flip-risk transport and retrospective reassessment simulation",
            "failures": flip_risk_failures,
            "status": "pass" if not flip_risk_failures else "fail",
        }
    )

    checks.append(
        {
            "check": "three-database extension summaries parse",
            "failures": extension_parse_failures,
            "status": "pass" if not extension_parse_failures else "fail",
        }
    )

    summary_path = ROOT / "results" / "formal_summary" / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    alignment_failures = []
    auc_failures = []
    flip_failures = []
    for database in ("mimic4", "eicu"):
        reference_ids = None
        reference_y = None
        for configuration in CONFIGS:
            seed_probabilities = []
            for seed in SEEDS:
                bundle = np.load(
                    FORMAL / database / f"{configuration}_s{seed}.npz"
                )
                ids = bundle["ids"]
                y = bundle["y"]
                if reference_ids is None:
                    reference_ids = ids.copy()
                    reference_y = y.copy()
                elif not np.array_equal(reference_ids, ids):
                    alignment_failures.append(
                        f"{database}/{configuration}/s{seed}: ids"
                    )
                elif not np.array_equal(reference_y, y):
                    alignment_failures.append(
                        f"{database}/{configuration}/s{seed}: y"
                    )
                seed_probabilities.append(bundle["probability"])
            probability = np.mean(seed_probabilities, axis=0)
            y = reference_y
            for index, hour in enumerate(LANDMARKS):
                recomputed_auc = float(
                    roc_auc_score(
                        y,
                        probability[:, index, :],
                        multi_class="ovr",
                        average="macro",
                        labels=np.arange(3),
                    )
                )
                reported_auc = summary[database]["ensemble"][configuration][
                    "metrics"
                ]["macro_auc"][str(hour)]
                if abs(recomputed_auc - reported_auc) > 0.00011:
                    auc_failures.append(
                        {
                            "database": database,
                            "configuration": configuration,
                            "hour": hour,
                            "reported": reported_auc,
                            "recomputed": round(recomputed_auc, 4),
                        }
                    )
            prediction = probability.argmax(2)
            recomputed_flip = float((prediction[:, 0] != prediction[:, 2]).mean())
            reported_flip = summary[database]["ensemble"][configuration][
                "stability"
            ]["flip_0_2"]
            if abs(recomputed_flip - reported_flip) > 0.00011:
                flip_failures.append(
                    {
                        "database": database,
                        "configuration": configuration,
                        "reported": reported_flip,
                        "recomputed": round(recomputed_flip, 4),
                    }
                )
    checks.extend(
        [
            {
                "check": "seed and configuration row alignment",
                "n_failures": len(alignment_failures),
                "failures": alignment_failures,
                "status": "pass" if not alignment_failures else "fail",
            },
            {
                "check": "ensemble AUC reproduces summary",
                "n_failures": len(auc_failures),
                "failures": auc_failures,
                "status": "pass" if not auc_failures else "fail",
            },
            {
                "check": "ensemble class flip reproduces summary",
                "n_failures": len(flip_failures),
                "failures": flip_failures,
                "status": "pass" if not flip_failures else "fail",
            },
        ]
    )

    title_files = (
        ROOT / "outline_sepsis.md",
        ROOT / "README.md",
        ROOT / "configs" / "study.json",
        ROOT / "manuscript" / "manuscript.md",
    )
    title_failures = [
        str(path) for path in title_files if TITLE not in path.read_text(encoding="utf-8")
    ]
    checks.append(
        {
            "check": "title synchronized",
            "n_failures": len(title_failures),
            "failures": title_failures,
            "status": "pass" if not title_failures else "fail",
        }
    )
    missing_title_terms = [
        term for term in TITLE_DIRECTION_TERMS if term.lower() not in TITLE.lower()
    ]
    checks.append(
        {
            "check": "title retains stability, transport, recalibration and reassessment direction",
            "required_terms": list(TITLE_DIRECTION_TERMS),
            "missing_terms": missing_title_terms,
            "status": "pass" if not missing_title_terms else "fail",
        }
    )
    outline_text = (ROOT / "outline_sepsis.md").read_text(encoding="utf-8")
    outline_required = (
        "23.32%",
        "transient 48 h",
        "0.047 to 0.060",
        "AUPRC",
        "Main narrative hierarchy",
        "limited to 30 words",
        "Figure 10A-F",
    )
    outline_forbidden = (
        "324 words",
        "Figure 7C-D",
        "0.134 to 0.181",
    )
    outline_missing = [term for term in outline_required if term not in outline_text]
    outline_stale = [term for term in outline_forbidden if term in outline_text]
    checks.append(
        {
            "check": "final outline is synchronized with trajectory, paired calibration, and reassessment evidence",
            "missing_terms": outline_missing,
            "stale_terms": outline_stale,
            "status": (
                "pass"
                if not outline_missing and not outline_stale
                else "fail"
            ),
        }
    )

    manuscript_path = ROOT / "outputs" / "manuscript_sepsis_stability.md"
    if manuscript_path.exists():
        manuscript = manuscript_path.read_text(encoding="utf-8")
        unresolved = "{{" in manuscript or "}}" in manuscript
        checks.append(
            {
                "check": "assembled manuscript has no unresolved placeholder",
                "status": "pass" if not unresolved else "fail",
            }
        )

    three_database_text_paths = (
        ROOT / "manuscript" / "manuscript.md",
        ROOT / "outputs" / "manuscript_sepsis_stability.md",
    )
    three_database_failures = [
        str(path)
        for path in three_database_text_paths
        if not path.exists()
        or "three-database" not in path.read_text(encoding="utf-8").lower()
    ]
    checks.append(
        {
            "check": "manuscript source and assembled output state Three-Database design",
            "failures": three_database_failures,
            "status": "pass" if not three_database_failures else "fail",
        }
    )

    audit_path = ROOT / "outputs" / "manuscript_audit.json"
    abstract_word_count = None
    abstract_failures = []
    if not audit_path.exists():
        abstract_failures.append("missing manuscript audit")
    else:
        try:
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            abstract_word_count = int(audit["abstract_word_count"])
            if abstract_word_count > 250:
                abstract_failures.append(
                    f"abstract word count={abstract_word_count}, expected <=250"
                )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            abstract_failures.append(f"abstract audit unreadable: {error}")
    checks.append(
        {
            "check": "abstract at or below 250 words",
            "abstract_word_count": abstract_word_count,
            "failures": abstract_failures,
            "status": "pass" if not abstract_failures else "fail",
        }
    )

    manuscript_text_for_assets = (
        manuscript_path.read_text(encoding="utf-8")
        if manuscript_path.exists()
        else ""
    )
    abstract_text = (
        manuscript_text_for_assets.split("## Abstract", 1)[1]
        .split("## 1. Introduction", 1)[0]
        if "## Abstract" in manuscript_text_for_assets
        else ""
    )
    abstract_required = (
        "observed through 72 h",
        "fixed-weight transport",
        "local recalibration",
        "enhanced-monitoring",
        "no such conversion",
        "87.24%",
        "91.78%",
        "88.53%",
        "raised it in four",
        "cannot estimate time to stability",
    )
    abstract_missing = [
        phrase for phrase in abstract_required if phrase not in abstract_text
    ]
    checks.append(
        {
            "check": "abstract states conditioning, transport, recalibration and monitoring burden",
            "missing_phrases": abstract_missing,
            "status": "pass" if not abstract_missing else "fail",
        }
    )

    figure_citation_order = first_asset_citations(manuscript_text_for_assets, "Figure")
    table_citation_order = first_asset_citations(manuscript_text_for_assets, "Table")
    asset_order_failures = []
    if figure_citation_order != list(range(1, 12)):
        asset_order_failures.append(
            f"first Figure citation order={figure_citation_order}"
        )
    if table_citation_order != list(range(1, 11)):
        asset_order_failures.append(
            f"first Table citation order={table_citation_order}"
        )
    checks.append(
        {
            "check": "main figures and tables are first cited in display order",
            "figure_order": figure_citation_order,
            "table_order": table_citation_order,
            "failures": asset_order_failures,
            "status": "pass" if not asset_order_failures else "fail",
        }
    )

    supplementary_assets = sorted(
        {
            int(value)
            for value in re.findall(
                r"Supplementary Table S([1-9][0-9]*)",
                manuscript_text_for_assets,
            )
        }
    )
    missing_supplementary_assets = sorted(
        set(range(1, 13)) - set(supplementary_assets)
    )
    supplementary_citation_sequence = []
    for value in re.findall(
        r"Supplementary Table S([1-9][0-9]*)",
        manuscript_text_for_assets,
    ):
        number = int(value)
        if number not in supplementary_citation_sequence:
            supplementary_citation_sequence.append(number)
    if supplementary_citation_sequence != list(range(1, 13)):
        missing_supplementary_assets.append(
            "first-citation order="
            f"{supplementary_citation_sequence}"
        )
    checks.append(
        {
            "check": (
                "supplementary tables S1-S12 are cited in first-appearance order"
            ),
            "cited": supplementary_assets,
            "first_citation_order": supplementary_citation_sequence,
            "missing": missing_supplementary_assets,
            "status": "pass" if not missing_supplementary_assets else "fail",
        }
    )

    context_required = (
        "8,999",
        "182 hospitals",
        "216",
        "98",
        "20.28%",
        "21.32%",
        "P10-P90",
        "0.64",
        "0.77",
        "0.58",
        "0.83",
        "fixed during target evaluation",
    )
    context_missing = [
        phrase
        for phrase in context_required
        if phrase not in manuscript_text_for_assets
    ]
    checks.append(
        {
            "check": "hospital cohort, site range and fixed transport context are explicit",
            "missing_phrases": context_missing,
            "status": "pass" if not context_missing else "fail",
        }
    )

    citation_failures = []
    citation_order: list[int] = []
    citation_endpoints: set[int] = set()
    reference_numbers: list[int] = []
    reporting_text = ""
    if manuscript_path.exists():
        manuscript_text = manuscript_path.read_text(encoding="utf-8")
        body_text, separator, reference_text = manuscript_text.partition(
            "## 7. References"
        )
        if not separator:
            citation_failures.append("missing reference section")
        reference_numbers = [
            int(number)
            for number in re.findall(r"(?m)^(\d+)\.\s", reference_text)
        ]
        citation_order = citation_numbers(body_text)
        citation_endpoints = set(citation_endpoint_numbers(body_text))
        cited = set(citation_order)
        reference_set = set(reference_numbers)
        if not 40 <= len(reference_numbers) <= 50:
            citation_failures.append(
                f"reference count={len(reference_numbers)}, expected 40-50"
            )
        if reference_numbers != list(range(1, len(reference_numbers) + 1)):
            citation_failures.append("reference numbering is not continuous")
        if cited != set(range(1, len(reference_numbers) + 1)):
            citation_failures.append(
                f"uncited reference numbers={sorted(set(range(1, len(reference_numbers) + 1)) - cited)}"
            )
        if not cited.issubset(reference_set):
            citation_failures.append(
                f"citation numbers without references={sorted(cited - reference_set)}"
            )
        seen = set()
        first_appearance = []
        for number in citation_order:
            if number not in seen:
                first_appearance.append(number)
                seen.add(number)
        if first_appearance != list(range(1, len(first_appearance) + 1)):
            citation_failures.append(
                f"first-appearance citation order={first_appearance[:20]}"
            )
        abstract_text, _, after_abstract = body_text.partition("## 1. Introduction")
        results_through_conclusions = (
            after_abstract.partition("## 3. Results")[2].partition(
                "## 6. Declarations"
            )[0]
        )
        reporting_text = abstract_text + results_through_conclusions
    else:
        citation_failures.append("missing assembled manuscript")
    checks.append(
        {
            "check": "all references are cited in first-appearance numeric order",
            "reference_count": len(reference_numbers),
            "unique_cited_numbers": len(set(citation_order)),
            "failures": citation_failures,
            "status": "pass" if not citation_failures else "fail",
        }
    )

    reference_doi_failures = []
    reference_path = ROOT / "manuscript" / "references.json"
    verification_path = ROOT / "outputs" / "reference_verification.json"
    if not reference_path.exists():
        reference_doi_failures.append("missing references.json")
    if not verification_path.exists():
        reference_doi_failures.append("missing DOI verification record")
    if reference_path.exists() and verification_path.exists():
        references = read_json(reference_path)
        verification = read_json(verification_path)
        reference_dois = {entry["doi"].lower() for entry in references}
        verified_entries = verification.get("entries", [])
        verified_dois = {
            entry["doi"].lower()
            for entry in verified_entries
            if entry.get("verified")
        }
        if not 40 <= len(references) <= 50:
            reference_doi_failures.append(
                f"reference count={len(references)}, expected 40-50"
            )
        if verification.get("count") != len(references):
            reference_doi_failures.append("DOI verification count mismatch")
        if not verification.get("all_verified"):
            reference_doi_failures.append("not every DOI passed verification")
        if verification.get("title_mismatch_count") != 0:
            reference_doi_failures.append("registered DOI title mismatch")
        if reference_dois != verified_dois:
            reference_doi_failures.append("DOI verification coverage mismatch")
        if any(
            not entry.get("doi_handle_registered")
            or not entry.get("provider_record_found")
            for entry in verified_entries
        ):
            reference_doi_failures.append("DOI handle or agency record failed")
    checks.append(
        {
            "check": "40-50 references verified through DOI Handle and Crossref/DataCite",
            "verification_file": str(verification_path),
            "failures": reference_doi_failures,
            "status": "pass" if not reference_doi_failures else "fail",
        }
    )

    p_failures = p_value_failures(reporting_text)
    decimal_failures = decimal_format_failures(reporting_text)
    main_docx_for_numbers = ROOT / "outputs" / "1029TP_Sepsis_Manuscript.docx"
    if main_docx_for_numbers.exists():
        main_for_numbers = Document(main_docx_for_numbers)
        table_text = "\n".join(
            cell.text
            for table in main_for_numbers.tables
            for row in table.rows
            for cell in row.cells
        )
        p_failures.extend(p_value_failures(table_text))
        decimal_failures.extend(decimal_format_failures(table_text))
    checks.append(
        {
            "check": "reported decimals and P values follow the manuscript rule",
            "p_value_failures": sorted(set(p_failures)),
            "decimal_failures": sorted(set(decimal_failures)),
            "status": (
                "pass"
                if not set(p_failures) and not set(decimal_failures)
                else "fail"
            ),
        }
    )

    language_failures = []
    if manuscript_path.exists():
        manuscript_text = manuscript_path.read_text(encoding="utf-8")
        sections = {
            part.splitlines()[0]: part
            for part in re.split(r"(?m)^## ", manuscript_text)[1:]
        }
        for section_name in ("1. Introduction", "4. Discussion"):
            section_text = sections.get(section_name, "")
            if section_text.count("(") or section_text.count(")"):
                language_failures.append(
                    f"{section_name} contains parentheses: "
                    f"{section_text.count('(')} open, {section_text.count(')')} close"
                )
        narrative_text = manuscript_text.split("## 6. Declarations", 1)[0]
        narrative_text = re.sub(r"```.*?```", "", narrative_text, flags=re.DOTALL)
        narrative_text = re.sub(r"`[^`\n]*`", "", narrative_text)
        if narrative_text.count("(") or narrative_text.count(")"):
            language_failures.append(
                "narrative text outside formulas contains parentheses: "
                f"{narrative_text.count('(')} open, "
                f"{narrative_text.count(')')} close"
            )
        strong_causal = re.compile(
            r"\b(?:cause|caused|causes|causing|lead to|led to|result in|"
            r"resulted in|determine|determined|drive|drives|drove|driven|"
            r"because|improve|improved|improves|improving|reduce|reduced|"
            r"reduces|reducing|increase|increased|increases|increasing|"
            r"worsen|worsened|worsening|effect|effects|impact|influence)\b",
            re.IGNORECASE,
        )
        for section_name in ("1. Introduction", "4. Discussion"):
            hits = sorted(
                {
                    match.group(0).lower()
                    for match in strong_causal.finditer(sections.get(section_name, ""))
                }
            )
            if hits:
                language_failures.append(
                    f"{section_name} strong causal terms={hits}"
                )
    checks.append(
        {
            "check": "plain non-causal language with minimal parentheses",
            "failures": language_failures,
            "status": "pass" if not language_failures else "fail",
        }
    )

    sentence_length_failures = []
    if manuscript_path.exists():
        sentence_length_failures = body_sentence_length_failures(
            manuscript_path.read_text(encoding="utf-8")
        )
    else:
        sentence_length_failures = [
            {
                "section": "assembled manuscript",
                "word_count": 0,
                "sentence": "missing assembled manuscript",
            }
        ]
    checks.append(
        {
            "check": "main narrative sentences contain at most 30 words",
            "max_words": 30,
            "n_long_sentences": len(sentence_length_failures),
            "failures": sentence_length_failures,
            "status": "pass" if not sentence_length_failures else "fail",
        }
    )

    figure_missing = [
        str(ROOT / "results" / "figures" / f"{stem}.{extension}")
        for stem in MAIN_FIGURE_STEMS
        for extension in ("png", "pdf", "svg")
        if not (ROOT / "results" / "figures" / f"{stem}.{extension}").exists()
    ]
    checks.append(
        {
            "check": "eleven main figures in PNG, PDF and SVG",
            "expected": len(MAIN_FIGURE_STEMS) * 3,
            "missing": figure_missing,
            "status": "pass" if not figure_missing else "fail",
        }
    )
    figure7_files = [
        ROOT / "results" / "figures" / f"figure7_transport_reassessment.{extension}"
        for extension in ("png", "pdf", "svg")
    ]
    checks.append(
        {
            "check": "Figure 7 transport and recalibration artifact exists",
            "missing": [str(path) for path in figure7_files if not path.exists()],
            "status": "pass" if all(path.exists() for path in figure7_files) else "fail",
        }
    )
    for figure_number, stem in (
        (8, "figure8_reassessment_simulation"),
        (9, "figure9_sensitivity_analyses"),
        (10, "figure10_hospital_validation"),
        (11, "figure11_clinical_transitions_safety"),
    ):
        figure_files = [
            ROOT / "results" / "figures" / f"{stem}.{extension}"
            for extension in ("png", "pdf", "svg")
        ]
        checks.append(
            {
                "check": f"Figure {figure_number} section-specific artifact exists",
                "missing": [
                    str(path) for path in figure_files if not path.exists()
                ],
                "status": (
                    "pass"
                    if all(path.exists() for path in figure_files)
                    else "fail"
                ),
            }
        )

    table_names = MAIN_TABLE_NAMES + SUPPLEMENTARY_TABLE_NAMES
    table_missing = [
        str(ROOT / "results" / "tables" / name)
        for name in table_names
        if not (ROOT / "results" / "tables" / name).exists()
    ]
    checks.append(
        {
            "check": "ten main and twelve supplementary tables",
            "expected": len(table_names),
            "missing": table_missing,
            "status": "pass" if not table_missing else "fail",
        }
    )
    table7_path = ROOT / "results" / "tables" / "table7_recalibration_reassessment.md"
    checks.append(
        {
            "check": "Table 7 recalibration and reassessment artifact exists",
            "path": str(table7_path),
            "status": "pass" if table7_path.exists() else "fail",
        }
    )

    table7_failures = []
    if not table7_path.exists():
        table7_failures.append("missing Table 7")
    else:
        transport = read_json(
            ROOT / "results" / "transport_extended" / "summary.json"
        )
        risk = read_json(ROOT / "results" / "flip_risk" / "summary.json")
        rows = markdown_rows(table7_path)
        for direction, record in transport["recalibration"].items():
            direction_label = " to ".join(
                {
                    "mimic4": "MIMIC-IV",
                    "eicu": "eICU-CRD",
                    "mimic3": "MIMIC-III",
                }[item]
                for item in direction.split("_to_")
            )
            matched = [
                row for row in rows if row and row[0] == direction_label
            ]
            if len(matched) != 2:
                table7_failures.append(
                    f"{direction}: expected two recalibration rows"
                )
                continue
            record_72 = record["72"]
            baseline_ece = record_72["curves"]["0"][
                "ece_top_label_10bin"
            ]["mean"]
            for row in matched:
                label_count = row[1]
                multinomial = record_72["curves"][label_count]
                temperature = record_72["curves_temperature_scaling"][
                    label_count
                ]
                paired = record_72[
                    "temperature_scaling_minus_multinomial_logistic"
                ][label_count]["ece_top_label_10bin"]
                wanted = (
                    f"{baseline_ece:.2f}",
                    f"{multinomial['ece_top_label_10bin']['mean']:.2f}",
                    f"{temperature['ece_top_label_10bin']['mean']:.2f}",
                    (
                        f"{(0.0 if abs(paired['mean']) < 0.005 else paired['mean']):+.2f} "
                        f"({paired['p2_5']:+.2f} to "
                        f"{paired['p97_5']:+.2f})"
                    ),
                )
                wanted = tuple(
                    value.replace("+0.00", "0.00").replace("-0.00", "0.00")
                    for value in wanted
                )
                observed = tuple(row[2:6])
                if observed != wanted:
                    table7_failures.append(
                        f"{direction} at {label_count} labels: "
                        "paired ECE values do not match source"
                    )
        for database in ("mimic4", "eicu", "mimic3"):
            label = {
                "mimic4": "MIMIC-IV",
                "eicu": "eICU-CRD",
                "mimic3": "MIMIC-III",
            }[database]
            matched = [
                row
                for row in rows
                if row and row[0] == label
            ]
            if len(matched) != 1:
                table7_failures.append(
                    f"{database}: expected one monitoring row"
                )
                continue
            threshold = next(
                item
                for item in risk["simulations"]["instability_risk_trigger"][
                    "within_database"
                ][database]["risk_thresholds"]
                if item["threshold"] == 0.1
            )
            oof = next(
                item
                for item in risk["within_database_oof"]
                if item["database"] == database
                and item["outcome"] == "any_flip"
            )
            wanted = (
                f"{oof['metrics']['auc']:.2f}",
                f"{100 * threshold['reassessments_reduced_vs_all']:.2f}",
                f"{100 * threshold['major_death_upgrade_capture_fraction']:.2f}",
                f"{threshold['major_death_upgrade_capture_enrichment']:.2f}",
            )
            observed = tuple(matched[0][6:10])
            if observed != wanted:
                table7_failures.append(
                    f"{database}: monitoring values do not match source"
                )
    checks.append(
        {
            "check": "Table 7 uses equal-label paired recalibration and monitoring sources",
            "failures": table7_failures,
            "status": "pass" if not table7_failures else "fail",
        }
    )
    for table_number, filename in (
        (8, "table8_shared_fold_metrics.md"),
        (9, "table9_hospital_validation.md"),
        (10, "table10_clinical_transitions.md"),
    ):
        path = ROOT / "results" / "tables" / filename
        checks.append(
            {
                "check": f"Table {table_number} new main-table artifact exists",
                "path": str(path),
                "status": "pass" if path.exists() else "fail",
            }
        )

    main_shared_path = ROOT / "results" / "tables" / "table8_shared_fold_metrics.md"
    supplementary_shared_path = (
        ROOT / "results" / "tables" / "supplementary_table_s7_shared_fold.md"
    )
    shared_fold_failures = []
    if not supplementary_shared_path.exists():
        shared_fold_failures.append("missing Supplementary Table S2")
    else:
        supplementary_text = supplementary_shared_path.read_text(encoding="utf-8")
        if supplementary_text == main_shared_path.read_text(encoding="utf-8"):
            shared_fold_failures.append("S2 duplicates main Table 7")
        supplementary_rows = markdown_rows(supplementary_shared_path)
        if not supplementary_rows or "Fold" not in supplementary_rows[0]:
            shared_fold_failures.append("S2 is not a fold-level table")
        else:
            body_rows = supplementary_rows[1:]
            if len(body_rows) != 30:
                shared_fold_failures.append(
                    f"S2 has {len(body_rows)} rows; expected 30"
                )
            fold_groups = {
                (row[0], row[1], int(row[2]))
                for row in body_rows
                if len(row) >= 3 and row[2].isdigit()
            }
            expected_groups = {
                (database, str(hour), fold)
                for database in ("MIMIC-IV", "eICU-CRD")
                for hour in (24, 48, 72)
                for fold in range(1, 6)
            }
            if fold_groups != expected_groups:
                shared_fold_failures.append(
                    "S2 database-landmark-fold combinations are incomplete"
                )
    checks.append(
        {
            "check": "Supplementary Table S2 is distinct and fold-specific",
            "failures": shared_fold_failures,
            "status": "pass" if not shared_fold_failures else "fail",
        }
    )

    table5_path = ROOT / "results" / "tables" / "table5_sensitivity.md"
    table5_failures = []
    if not table5_path.exists():
        table5_failures.append("missing Table 5")
    else:
        mimic3_summary = read_json(
            ROOT / "results" / "mimic3_shared" / "summary.json"
        )
        mimic3_rows = [
            row for row in markdown_rows(table5_path) if row and row[0] == "MIMIC-III"
        ]
        if len(mimic3_rows) != 1:
            table5_failures.append("expected exactly one MIMIC-III row")
        else:
            row = mimic3_rows[0]
            expected_n = f"{int(mimic3_summary['cohort']['n']):,}"
            expected_flip = (
                f"{100 * float(mimic3_summary['stability']['flip_24_72']['estimate']):.2f}"
            )
            expected_tv = f"{float(mimic3_summary['stability']['total_variation_24_72']['estimate']):.2f}"
            expected_auc = [
                (
                    f"{milestone} h "
                    f"{float(mimic3_summary['discrimination']['macro_auc'][str(milestone)]):.2f}"
                )
                for milestone in (24, 48, 72)
            ]
            for value in (
                expected_n,
                expected_flip,
                expected_tv,
                *expected_auc,
            ):
                if not any(value in cell for cell in row):
                    table5_failures.append(
                        f"current MIMIC-III value {value} missing"
                    )
        sensitivity_v2_summary = read_json(
            ROOT / "results" / "sensitivity_v2" / "sensitivity_v2_summary.json"
        )
        rows = markdown_rows(table5_path)
        source_specs = (
            (
                "MIMIC-IV",
                "ICU-death exclusion",
                sensitivity_v2_summary["icu_death_exclusion"]["mimic4"][
                    "complete_three_landmark_subset"
                ],
            ),
            (
                "eICU-CRD",
                "ICU-death exclusion",
                sensitivity_v2_summary["icu_death_exclusion"]["eicu"][
                    "complete_three_landmark_subset"
                ],
            ),
            (
                "MIMIC-IV",
                "ICD-9",
                sensitivity_v2_summary["icd_era"]["mimic4_icd9"][
                    "complete_three_landmark_subset"
                ],
            ),
            (
                "MIMIC-IV",
                "ICD-10",
                sensitivity_v2_summary["icd_era"]["mimic4_icd10"][
                    "complete_three_landmark_subset"
                ],
            ),
        )
        for database, marker, record in source_specs:
            matched = [
                row
                for row in rows
                if row
                and row[0] == database
                and marker in row[1]
            ]
            if len(matched) != 1:
                table5_failures.append(
                    f"{database}/{marker}: expected one row"
                )
                continue
            expected_values = (
                f"{record['n']:,}",
                f"{100 * record['flip_24_72']:.2f}",
                f"{record['tv_24_72']:.2f}",
                f"{record['macro_auc']['72']:.2f}",
            )
            if any(value not in matched[0] for value in expected_values):
                table5_failures.append(
                    f"{database}/{marker}: values do not match source"
                )
    checks.append(
        {
            "check": "Table 5 matches complete-cohort and sensitivity sources",
            "failures": table5_failures,
            "status": "pass" if not table5_failures else "fail",
        }
    )

    table6_path = ROOT / "results" / "tables" / "table6_transport.md"
    table6_failures = []
    if not table6_path.exists():
        table6_failures.append("missing Table 6")
    else:
        observed_directions = {
            direction
            for row in markdown_rows(table6_path)
            if row
            for direction in [parse_transport_direction(row[0])]
            if direction is not None
        }
        expected_direction_pairs = {
            ("mimic4", "eicu"),
            ("eicu", "mimic4"),
            ("mimic4", "mimic3"),
            ("mimic3", "mimic4"),
            ("mimic3", "eicu"),
            ("eicu", "mimic3"),
        }
        missing_directions = sorted(
            expected_direction_pairs.difference(observed_directions)
        )
        if missing_directions:
            table6_failures.append(
                f"missing transport directions: {missing_directions}"
            )
    checks.append(
        {
            "check": "Table 6 covers all six fixed-weight transport directions",
            "n_directions": (
                len(observed_directions) if table6_path.exists() else 0
            ),
            "failures": table6_failures,
            "status": "pass" if not table6_failures else "fail",
        }
    )

    leaked_separator_rows = []
    for name in table_names:
        path = ROOT / "results" / "tables" / name
        if not path.exists():
            continue
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip().startswith("|"):
                continue
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            if cells and all(re.fullmatch(r":?-+:?", cell) for cell in cells):
                if line_number > 2:
                    leaked_separator_rows.append(f"{name}:{line_number}")
    checks.append(
        {
            "check": "Markdown table separators do not become data rows",
            "leaked": leaked_separator_rows,
            "status": "pass" if not leaked_separator_rows else "fail",
        }
    )

    clinical_table_path = (
        ROOT / "results" / "tables" / "supplementary_table_s6_clinical_flips.md"
    )
    clinical_table_failures = []
    if clinical_table_path.exists():
        clinical_lines = [
            line
            for line in clinical_table_path.read_text(encoding="utf-8").splitlines()
            if line.strip().startswith("|")
        ]
        if len(clinical_lines) != 10:
            clinical_table_failures.append(
                f"expected 10 Markdown rows, found {len(clinical_lines)}"
            )
        for index, line in enumerate(clinical_lines, start=1):
            width = len(line.strip().strip("|").split("|"))
            if width != 9:
                clinical_table_failures.append(
                    f"row {index} has {width} columns instead of 9"
                )
    checks.append(
        {
            "check": "clinical-flip supplementary table has aligned 9-column rows",
            "failures": clinical_table_failures,
            "status": "pass" if not clinical_table_failures else "fail",
        }
    )

    main_docx = ROOT / "outputs" / "1029TP_Sepsis_Manuscript.docx"
    mirrored_docx = ROOT / "1029Manuscript.docx"
    supplement_docx = ROOT / "outputs" / "1029TP_Sepsis_Supplementary.docx"
    docx_failures = []
    if (
        not main_docx.exists()
        or not mirrored_docx.exists()
        or not supplement_docx.exists()
    ):
        docx_failures.append("missing DOCX file")
    else:
        main = Document(main_docx)
        mirrored = Document(mirrored_docx)
        supplement = Document(supplement_docx)
        main_xml = main.element.xml
        bookmark_names = re.findall(r'w:name="(_Ref\d+)"', main_xml)
        field_targets = set(re.findall(r"REF (_Ref\d+)", main_xml))
        expected_bookmarks = {
            f"_Ref{number}" for number in range(1, len(reference_numbers) + 1)
        }
        expected_field_targets = {
            f"_Ref{number}" for number in citation_endpoints
        }
        if set(bookmark_names) != expected_bookmarks:
            docx_failures.append("main DOCX reference bookmarks mismatch")
        if len(bookmark_names) != len(expected_bookmarks):
            docx_failures.append("main DOCX duplicate or missing reference bookmarks")
        if field_targets != expected_field_targets:
            docx_failures.append("main DOCX citation REF fields mismatch")
        if main_xml.count('w:fldCharType="begin"') != main_xml.count(
            'w:fldCharType="end"'
        ):
            docx_failures.append("main DOCX unbalanced REF fields")
        main_text = "\n".join(paragraph.text for paragraph in main.paragraphs)
        mirrored_text = "\n".join(
            paragraph.text for paragraph in mirrored.paragraphs
        )
        supplement_text = "\n".join(
            paragraph.text for paragraph in supplement.paragraphs
        )
        for label, document, text in (
            ("main", main, main_text),
            ("mirrored 1029Manuscript", mirrored, mirrored_text),
        ):
            if document.paragraphs[0].text.strip() != TITLE:
                docx_failures.append(f"{label} DOCX title")
            expected_front_matter = (
                TITLE,
                AUTHOR_LINE,
                *AUTHOR_AFFILIATIONS,
                COFIRST_TEXT,
                CORRESPONDING_TEXT,
                "Abstract",
            )
            actual_front_matter = tuple(
                paragraph.text.strip() for paragraph in document.paragraphs[:9]
            )
            if actual_front_matter != expected_front_matter:
                docx_failures.append(f"{label} DOCX author block order")
            for marker in DECLARATION_TEXT:
                if marker not in text:
                    docx_failures.append(
                        f"{label} DOCX missing declaration marker: {marker}"
                    )
            if "ICH-TemporalRegularization" in text:
                docx_failures.append(
                    f"{label} DOCX contains ICH-specific code repository"
                )
            if document.core_properties.author != AUTHOR_METADATA:
                docx_failures.append(f"{label} DOCX author metadata")
            if len(document.tables) != 10:
                docx_failures.append(
                    f"{label} DOCX tables={len(document.tables)}, expected 10"
                )
            if len(document.inline_shapes) != 11:
                docx_failures.append(
                    f"{label} DOCX figures={len(document.inline_shapes)}, "
                    "expected 11"
                )
            noncollapsible_titles = [
                paragraph.text
                for paragraph in document.paragraphs
                if paragraph.style.name.startswith("Heading")
                or (
                    paragraph._p.pPr is not None
                    and paragraph._p.pPr.outlineLvl is not None
                )
            ]
            if noncollapsible_titles:
                docx_failures.append(
                    f"{label} DOCX contains collapsible outline headings: "
                    f"{noncollapsible_titles[:3]}"
                )
            for style_name in ("SectionTitle", "SubsectionTitle"):
                try:
                    title_style = document.styles[style_name]
                except KeyError:
                    docx_failures.append(
                        f"{label} DOCX missing noncollapsible style {style_name}"
                    )
                    continue
                style_properties = title_style.element.pPr
                if (
                    style_properties is not None
                    and style_properties.outlineLvl is not None
                ):
                    docx_failures.append(
                        f"{label} DOCX {style_name} still has an outline level"
                    )
        if mirrored.paragraphs[8].text.strip() != "Abstract":
            docx_failures.append("mirrored 1029Manuscript author block before Abstract")
        if len(main.tables) != 10:
            docx_failures.append(
                f"main DOCX tables={len(main.tables)}, expected 10"
            )
        if len(main.inline_shapes) != 11:
            docx_failures.append(
                f"main DOCX figures={len(main.inline_shapes)}, expected 11"
            )
        if len(supplement.tables) != 12:
            docx_failures.append(
                f"supplementary DOCX tables={len(supplement.tables)}, expected 12"
            )
        for table_number in range(7, 11):
            if f"Table {table_number}" not in main_text:
                docx_failures.append(
                    f"main DOCX Table {table_number} caption"
                )
        for figure_number in range(7, 12):
            if f"Figure {figure_number}" not in main_text:
                docx_failures.append(
                    f"main DOCX Figure {figure_number} caption"
                )
        for reference in (
            "Figure 9A",
            "Figure 9B-D",
            "Figure 10A-F",
            "Figure 11A-F",
            "Figure 8A",
            "Figure 4A-D",
        ):
            if reference not in main_text:
                docx_failures.append(
                    f"main DOCX missing section-specific reference {reference}"
                )
        if "Figure 9C-D" in main_text:
            docx_failures.append(
                "main DOCX still uses the obsolete recalibration panel citation"
            )
        if "Three-Database" not in main_text:
            docx_failures.append("main DOCX does not state Three-Database design")
        for table_number in range(8, 12):
            if f"Table S{table_number}" not in supplement_text:
                docx_failures.append(
                    f"supplementary DOCX Table S{table_number} caption"
                )
        for label, text in (
            ("main DOCX", main_text),
            ("supplementary DOCX", supplement_text),
        ):
            if "{{" in text or "}}" in text:
                docx_failures.append(f"{label} unresolved placeholder")
    checks.append(
        {
            "check": "Word artifacts and embedded tables/figures",
            "failures": docx_failures,
            "status": "pass" if not docx_failures else "fail",
        }
    )

    layout_failures = []
    if main_docx.exists() and supplement_docx.exists():
        for label, document in (("main", main), ("supplement", supplement)):
            section = document.sections[0]
            if (
                round(section.page_width.inches, 2) != 8.5
                or round(section.page_height.inches, 2) != 11.0
            ):
                layout_failures.append(f"{label} page size")
            if any(
                round(value, 2) != 1.0
                for value in (
                    section.left_margin.inches,
                    section.right_margin.inches,
                    section.top_margin.inches,
                    section.bottom_margin.inches,
                )
            ):
                layout_failures.append(f"{label} margins")
            if any(
                paragraph.text.strip()
                for part in (section.header, section.footer)
                for paragraph in part.paragraphs
            ):
                layout_failures.append(f"{label} header or footer text")
            normal = document.styles["Normal"]
            if (
                normal.font.name != "Times New Roman"
                or normal.font.size.pt != 12
                or normal.paragraph_format.line_spacing != 1.5
            ):
                layout_failures.append(f"{label} Normal style")
            if (
                document.paragraphs[0].runs
                and (
                    document.paragraphs[0].runs[0].font.size.pt != 18
                    or document.paragraphs[0].runs[0].bold is not True
                )
            ):
                layout_failures.append(f"{label} title typography")
            for table_number, table in enumerate(document.tables, start=1):
                width = sum(cell.width.inches for cell in table.rows[0].cells)
                if width > 6.55:
                    layout_failures.append(
                        f"{label} table {table_number} width={width:.2f} in"
                    )
                borders = table._tbl.tblPr.find(qn("w:tblBorders"))
                values = (
                    {
                        child.tag.rsplit("}", 1)[-1]: child.get(qn("w:val"))
                        for child in borders
                    }
                    if borders is not None
                    else {}
                )
                expected = {
                    "top": "single",
                    "bottom": "single",
                    "left": "none",
                    "right": "none",
                    "insideH": "none",
                    "insideV": "none",
                }
                if any(values.get(edge) != value for edge, value in expected.items()):
                    layout_failures.append(
                        f"{label} table {table_number} borders"
                    )
            if any(shape.width.inches > 6.45 for shape in document.inline_shapes):
                layout_failures.append(f"{label} embedded figure width")
    else:
        layout_failures.append("missing DOCX for layout audit")
    checks.append(
        {
            "check": "Word layout matches 1025 reference",
            "failures": layout_failures,
            "status": "pass" if not layout_failures else "fail",
        }
    )

    stale_phrases = (
        "not estimable",
        "could not be estimated",
        "exact ICU death timing absent",
        "diagnosis-era flag absent",
        "the same folds were shared across landmarks",
        "fixed after five pilot runs",
        "selected from five pilot settings",
        "when do ordinal discharge-destination predictions stabilize",
        "more than one quarter of xgboost",
        "8,855",
        "21,981",
        "applying the same weights at 24 and 72 h",
        "probabilities but not the ranking",
        "mandatory 48 h reassessment",
        "mandatory 48-hour reassessment",
        "single 48 h reassessment",
        "retrospective 48 h reassessment",
        "make instability clinically relevant",
    )
    stale_hits = [
        phrase
        for path in (
            ROOT / "outline_sepsis.md",
            ROOT / "README.md",
            ROOT / "outputs" / "manuscript_sepsis_stability.md",
            ROOT / "results" / "flip_risk" / "report.md",
        )
        for phrase in stale_phrases
        if phrase.lower() in path.read_text(encoding="utf-8").lower()
    ]
    checks.append(
        {
            "check": "no stale or unsupported manuscript claims",
            "stale_hits": stale_hits,
            "status": "pass" if not stale_hits else "fail",
        }
    )

    failures = [check for check in checks if check["status"] == "fail"]
    report = {
        "status": "pass" if not failures else "fail",
        "checks": checks,
    }
    output = ROOT / "work" / "result_integrity.json"
    with output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
