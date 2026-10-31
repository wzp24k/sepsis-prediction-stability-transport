# -*- coding: utf-8 -*-
"""Consolidate the completed v2 sensitivity artifacts."""

from __future__ import annotations

import json
import os
from pathlib import Path


ROOT = Path(os.environ.get("SEPSIS_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
BASE = ROOT / "results" / "sensitivity_v2"


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    required = {
        "flags": BASE / "flags" / "sensitivity_flags_summary.json",
        "mimic4_icu_death": BASE / "xgb" / "mimic4_exclude_icu_death.json",
        "eicu_icu_death": BASE / "xgb" / "eicu_exclude_icu_death.json",
        "mimic4_icd9": BASE / "xgb" / "mimic4_icd9_era.json",
        "mimic4_icd10": BASE / "xgb" / "mimic4_icd10_era.json",
        "reverse_order": BASE / "reverse_order_summary.json",
        "ordinal_clinical": BASE / "ordinal_clinical_summary.json",
        "model_contrast_bootstrap": BASE / "model_contrast_bootstrap.json",
        "mimic3": BASE / "mimic3" / "mimic3_xgb_summary.json",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(missing)
    summary = {
        "status": "complete",
        "artifacts": {key: str(path.relative_to(ROOT)) for key, path in required.items()},
        "flag_counts": read_json(required["flags"]),
        "icu_death_exclusion": {
            "mimic4": read_json(required["mimic4_icu_death"]),
            "eicu": read_json(required["eicu_icu_death"]),
        },
        "icd_era": {
            "mimic4_icd9": read_json(required["mimic4_icd9"]),
            "mimic4_icd10": read_json(required["mimic4_icd10"]),
            "eicu_status": (
                "not applicable: diagnosis.icd9code is a harmonised field and "
                "does not identify admission-level ICD-10 coding"
            ),
        },
        "ordinal_order_reversal": read_json(required["reverse_order"]),
        "ordinal_clinical_metrics": read_json(required["ordinal_clinical"]),
        "model_contrast_bootstrap": read_json(
            required["model_contrast_bootstrap"]
        ),
        "mimic3_comparison": read_json(required["mimic3"]),
        "completed_outline_items": [
            "exclude_in_icu_death",
            "era_stratified_icd9_icd10 (MIMIC-IV; eICU not applicable)",
            "ordinal_order_swapped",
            "ordinal_vs_nominal_nri_idi",
            "optional_mimic3_same_institution_comparison",
        ],
        "not_used_as_primary_evidence": [
            (
                "ordinal order reversal: a coding-order stress test, with the "
                "reversed probability axis restored before comparison"
            ),
            (
                "MIMIC-III: different era and data schema; reported as a "
                "same-institution comparison rather than external validation "
                "of fixed MIMIC-IV weights"
            ),
            (
                "ICD-era strata: models were retrained within each era, so "
                "differences include secular changes in practice and coding"
            ),
        ],
        "not_run": {
            "negative_controls": (
                "time reversal, time permutation and generic Laplace smoothing "
                "were listed in an earlier configuration but were not part of "
                "the final v3 estimand or result set"
            ),
            "full_lambda_grid": (
                "formal runs used fixed weights 0.04/0.06; exploratory pilot "
                "settings were not retained as comparative evidence and the full "
                "lambda grid was not run"
            ),
            "no_graph_ablation": (
                "not part of the final v3 four-configuration ablation grid"
            ),
        },
    }
    output = BASE / "sensitivity_v2_summary.json"
    with output.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(output, flush=True)


if __name__ == "__main__":
    main()
