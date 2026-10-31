# -*- coding: utf-8 -*-
"""Apply the audited MIMIC-IV discharge mapping without rescanning chart data."""

from __future__ import annotations

import json
import os
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


ROOT = Path(os.environ.get("SEPSIS_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
SOURCE_ROOT = Path(
    os.environ.get("SEPSIS_SOURCE_ROOT", r"D:\TT paper\0811Temporal Pathway")
)
ADMISSIONS = SOURCE_ROOT / "mimic-iv-3.1" / "hosp" / "admissions.csv.gz"
OUT = ROOT / "data" / "processed" / "mimic4"
AUDIT = ROOT / "work" / "mimic4_outcome_relabel_audit.json"


def sql_path(path: Path) -> str:
    return str(path).replace("\\", "/").replace("'", "''")


def source_mapping() -> pd.DataFrame:
    connection = duckdb.connect()
    connection.execute("PRAGMA threads=16")
    frame = connection.execute(
        f"""
        SELECT CAST(hadm_id AS BIGINT) hadm_id,
               upper(trim(discharge_location)) discharge_location,
               TRY_CAST(hospital_expire_flag AS INTEGER) hospital_expire_flag
        FROM read_csv_auto('{sql_path(ADMISSIONS)}')
        """
    ).fetchdf()
    frame["y"] = np.select(
        [
            frame["hospital_expire_flag"].eq(1)
            | frame["discharge_location"].fillna("").str.startswith("DIED"),
            frame["discharge_location"].isin(
                ["HOME", "HOME HEALTH CARE", "AGAINST ADVICE"]
            ),
            frame["discharge_location"].isin(["REHAB", "ASSISTED LIVING"]),
            frame["discharge_location"].isin(
                [
                    "SKILLED NURSING FACILITY",
                    "CHRONIC/LONG TERM ACUTE CARE",
                    "OTHER FACILITY",
                    "ACUTE HOSPITAL",
                    "PSYCH FACILITY",
                    "HEALTHCARE FACILITY",
                    "HOSPICE",
                ]
            ),
        ],
        [3, 0, 1, 2],
        default=-1,
    ).astype(np.int64)
    frame["y3"] = frame["y"].map({0: 0, 1: 1, 2: 1, 3: 2}).fillna(-1).astype(np.int64)
    return frame[["hadm_id", "discharge_location", "hospital_expire_flag", "y", "y3"]]


def update_parquet(path: Path, mapping: pd.DataFrame) -> dict:
    frame = pd.read_parquet(path)
    before = frame["y"].value_counts(dropna=False).sort_index().to_dict()
    added_hadm_id = "hadm_id" not in frame.columns
    if "hadm_id" not in frame.columns:
        cohort = pd.read_parquet(OUT / "cohort.parquet", columns=["stay_id", "hadm_id"])
        lookup = dict(
            zip(cohort["stay_id"].astype(np.int64), cohort["hadm_id"].astype(np.int64))
        )
        frame["hadm_id"] = frame["stay_id"].astype(np.int64).map(lookup)
    target = mapping.set_index("hadm_id")
    new_y = frame["hadm_id"].astype(np.int64).map(target["y"]).fillna(-1).astype(np.int64)
    frame["y"] = new_y.where(new_y >= 0, np.nan)
    frame["y3"] = pd.Series(new_y, index=frame.index).map(
        {0: 0, 1: 1, 2: 1, 3: 2}
    )
    if added_hadm_id:
        frame = frame.drop(columns="hadm_id")
    frame.to_parquet(path, index=False)
    after = frame["y"].value_counts(dropna=False).sort_index().to_dict()
    return {
        "path": str(path.relative_to(ROOT)),
        "rows": int(len(frame)),
        "before": {str(key): int(value) for key, value in before.items()},
        "after": {str(key): int(value) for key, value in after.items()},
    }


def update_npz(path: Path, mapping: pd.DataFrame) -> dict:
    bundle = np.load(path, allow_pickle=True)
    arrays = {key: bundle[key] for key in bundle.files}
    ids = arrays["stay_id"].astype(np.int64)
    cohort = pd.read_parquet(OUT / "cohort.parquet", columns=["stay_id", "hadm_id"])
    hadm_lookup = dict(
        zip(cohort["stay_id"].astype(np.int64), cohort["hadm_id"].astype(np.int64))
    )
    hadm = np.asarray([hadm_lookup[int(identifier)] for identifier in ids], dtype=np.int64)
    target = mapping.set_index("hadm_id")
    old_y = arrays["y"].copy()
    y = pd.Series(hadm).map(target["y"]).fillna(-1).to_numpy(dtype=np.int64)
    arrays["y"] = y
    arrays["y3"] = pd.Series(y).map({0: 0, 1: 1, 2: 1, 3: 2}).fillna(-1).to_numpy(
        dtype=np.int64
    )
    np.savez_compressed(path, **arrays)
    return {
        "path": str(path.relative_to(ROOT)),
        "rows": int(len(y)),
        "changed": int((old_y != y).sum()),
        "after": {str(key): int(value) for key, value in pd.Series(y).value_counts().sort_index().items()},
    }


def validate(mapping: pd.DataFrame) -> dict:
    hospice_survivors = mapping[
        mapping["discharge_location"].eq("HOSPICE")
        & mapping["hospital_expire_flag"].eq(0)
    ]
    hospice_deaths = mapping[
        mapping["discharge_location"].eq("HOSPICE")
        & mapping["hospital_expire_flag"].eq(1)
    ]
    if not hospice_survivors["y"].eq(2).all():
        raise AssertionError("Surviving HOSPICE discharges were not mapped to facility")
    if not hospice_deaths["y"].eq(3).all():
        raise AssertionError("HOSPICE with hospital_expire_flag=1 was not mapped to death")
    cohort = pd.read_parquet(OUT / "cohort.parquet", columns=["hadm_id", "y"])
    merged = cohort.merge(
        mapping[["hadm_id", "discharge_location", "hospital_expire_flag"]],
        on="hadm_id",
        how="left",
    )
    survivors = merged[
        merged["discharge_location"].eq("HOSPICE")
        & merged["hospital_expire_flag"].eq(0)
    ]
    if not survivors["y"].eq(2).all():
        raise AssertionError("Derived cohort still maps surviving HOSPICE to death")
    return {
        "source_hospice_survivors": int(len(hospice_survivors)),
        "source_hospice_with_expire_flag": int(len(hospice_deaths)),
        "cohort_hospice_survivors": int(len(survivors)),
        "cohort_hospice_survivor_classes": {
            str(key): int(value)
            for key, value in survivors["y"].value_counts().sort_index().items()
        },
    }


def main() -> None:
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    mapping = source_mapping()
    records = []
    for path in (
        OUT / "cohort.parquet",
        OUT / "design_all.parquet",
        OUT / "design.parquet",
    ):
        if path.exists():
            records.append(update_parquet(path, mapping))
    for path in (OUT / "seq_all72.npz", OUT / "seq72.npz"):
        if path.exists():
            records.append(update_npz(path, mapping))
    audit = {"updates": records, "validation": validate(mapping)}
    with AUDIT.open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, ensure_ascii=False)
    print(json.dumps(audit, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
