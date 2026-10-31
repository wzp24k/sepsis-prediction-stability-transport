# -*- coding: utf-8 -*-
"""Recover ICU death and diagnosis-era flags for sensitivity analyses."""

from __future__ import annotations

import json
import os
from pathlib import Path

import duckdb
import pandas as pd


ROOT = Path(os.environ.get("SEPSIS_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
SOURCE_ROOT = Path(
    os.environ.get("SEPSIS_SOURCE_ROOT", r"D:\TT paper\0811Temporal Pathway")
)
DATA = ROOT / "data" / "processed"
OUT = ROOT / "results" / "sensitivity_v2" / "flags"
OUT.mkdir(parents=True, exist_ok=True)


def q(path: Path) -> str:
    return str(path).replace("\\", "/").replace("'", "''")


def mimic4_flags(connection: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    source = SOURCE_ROOT / "mimic-iv-3.1"
    design = pd.read_parquet(
        DATA / "mimic4" / "design_all.parquet",
        columns=["stay_id", "subject_id", "los", "y3"],
    )
    query = f"""
    WITH I AS (
        SELECT CAST(stay_id AS BIGINT) stay_id,
               CAST(hadm_id AS BIGINT) hadm_id,
               CAST(intime AS TIMESTAMP) intime,
               CAST(outtime AS TIMESTAMP) outtime
        FROM read_csv_auto('{q(source / "icu" / "icustays.csv.gz")}')
    ),
    A AS (
        SELECT CAST(hadm_id AS BIGINT) hadm_id,
               TRY_CAST(deathtime AS TIMESTAMP) deathtime,
               TRY_CAST(dischtime AS TIMESTAMP) dischtime,
               discharge_location,
               TRY_CAST(hospital_expire_flag AS INTEGER) hospital_expire_flag
        FROM read_csv_auto('{q(source / "hosp" / "admissions.csv.gz")}')
    ),
    D AS (
        SELECT CAST(hadm_id AS BIGINT) hadm_id,
               max(CASE
                   WHEN CAST(icd_version AS INTEGER) = 9
                    AND (
                        upper(trim(icd_code)) LIKE '038%'
                        OR upper(trim(icd_code)) LIKE '9959%'
                        OR upper(trim(icd_code)) LIKE '78552%'
                    ) THEN 1 ELSE 0
               END) sepsis_icd9,
               max(CASE
                   WHEN CAST(icd_version AS INTEGER) = 10
                    AND (
                        upper(trim(icd_code)) LIKE 'A40%'
                        OR upper(trim(icd_code)) LIKE 'A41%'
                        OR upper(trim(icd_code)) LIKE 'R652%'
                    ) THEN 1 ELSE 0
               END) sepsis_icd10
        FROM read_csv_auto(
            '{q(source / "hosp" / "diagnoses_icd.csv.gz")}',
            all_varchar = true
        )
        GROUP BY 1
    )
    SELECT I.stay_id, I.hadm_id, I.intime, I.outtime,
           A.deathtime, A.dischtime, A.discharge_location,
           A.hospital_expire_flag, D.sepsis_icd9, D.sepsis_icd10
    FROM I
    LEFT JOIN A USING (hadm_id)
    LEFT JOIN D USING (hadm_id)
    """
    flags = connection.execute(query).fetchdf()
    flags["stay_id"] = flags["stay_id"].astype("int64")
    for column in ("intime", "outtime", "deathtime", "dischtime"):
        flags[column] = pd.to_datetime(flags[column], errors="coerce")
    flags["icu_death"] = (
        flags["deathtime"].notna()
        & flags["intime"].notna()
        & flags["outtime"].notna()
        & (flags["deathtime"] >= flags["intime"])
        & (flags["deathtime"] <= flags["outtime"])
    )
    flags["death_after_icu_discharge"] = (
        flags["deathtime"].notna()
        & flags["outtime"].notna()
        & (flags["deathtime"] > flags["outtime"])
    )
    flags["icd_era"] = "unclassified"
    flags.loc[
        (flags["sepsis_icd9"] == 1) & (flags["sepsis_icd10"] == 0), "icd_era"
    ] = "ICD-9"
    flags.loc[
        (flags["sepsis_icd9"] == 0) & (flags["sepsis_icd10"] == 1), "icd_era"
    ] = "ICD-10"
    flags.loc[
        (flags["sepsis_icd9"] == 1) & (flags["sepsis_icd10"] == 1), "icd_era"
    ] = "mixed"
    flags["icu_death_within_72h"] = flags["icu_death"] & (
        (flags["deathtime"] - flags["intime"]).dt.total_seconds() < 72 * 3600
    )
    flags = design.merge(
        flags.drop(columns=["los", "y3"], errors="ignore"),
        on="stay_id",
        how="left",
        validate="one_to_one",
    )
    flags.to_parquet(OUT / "mimic4_sensitivity_flags.parquet", index=False)
    return flags


def eicu_flags(connection: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    design = pd.read_parquet(
        DATA / "eicu" / "design_all.parquet",
        columns=["patientunitstayid", "uniquepid", "hospitalid", "los_h", "y3"],
    )
    source = SOURCE_ROOT / "eicu" / "patient.csv.gz"
    flags = connection.execute(
        f"""
        SELECT CAST(patientunitstayid AS BIGINT) patientunitstayid,
               unitdischargestatus,
               unitdischargelocation,
               TRY_CAST(unitdischargeoffset AS INTEGER) unitdischargeoffset,
               hospitaldischargestatus,
               hospitaldischargelocation,
               TRY_CAST(hospitaldischargeoffset AS INTEGER) hospitaldischargeoffset,
               TRY_CAST(hospitaldischargeyear AS INTEGER) hospitaldischargeyear
        FROM read_csv_auto('{q(source)}')
        """
    ).fetchdf()
    flags["icu_death"] = (
        flags["unitdischargestatus"].fillna("").str.strip().str.lower() == "expired"
    )
    flags["hospital_death"] = (
        flags["hospitaldischargestatus"].fillna("").str.strip().str.lower()
        == "expired"
    )
    flags["icu_death_within_72h"] = flags["icu_death"] & (
        flags["unitdischargeoffset"].fillna(10**12) < 72 * 60
    )
    flags["icd_era"] = "not_applicable"
    flags = design.merge(
        flags,
        on="patientunitstayid",
        how="left",
        validate="one_to_one",
    )
    flags.to_parquet(OUT / "eicu_sensitivity_flags.parquet", index=False)
    return flags


def summarize(
    mimic4: pd.DataFrame, eicu: pd.DataFrame
) -> dict:
    def counts(frame: pd.DataFrame, era: bool = False) -> dict:
        eligible = frame.loc[frame["y3"].notna()]
        output = {
            "all_outcome_mapped": int(len(eligible)),
            "icu_death": int(eligible["icu_death"].fillna(False).sum()),
            "icu_death_within_72h": int(
                eligible["icu_death_within_72h"].fillna(False).sum()
            ),
            "icd_era": eligible["icd_era"].fillna("missing").value_counts().to_dict(),
        }
        for hour, duration in ((24, "los"), (48, "los"), (72, "los")):
            landmark = eligible.loc[eligible[duration] >= hour]
            output[f"landmark_{hour}h"] = {
                "n": int(len(landmark)),
                "icu_death": int(landmark["icu_death"].fillna(False).sum()),
            }
        if era:
            for label in ("ICD-9", "ICD-10"):
                subset = eligible.loc[eligible["icd_era"] == label]
                output[f"{label.lower()}_landmarks"] = {
                    str(hour): int((subset["los"] >= hour).sum())
                    for hour in (24, 48, 72)
                }
        return output

    return {
        "mimic4": counts(mimic4, era=True),
        "eicu": {
            **counts(eicu.assign(los=eicu["los_h"])),
            "hospital_death": int(
                eicu.loc[eicu["y3"].notna(), "hospital_death"].fillna(False).sum()
            ),
            "unit_discharge_status": eicu["unitdischargestatus"]
            .fillna("missing")
            .value_counts()
            .to_dict(),
        },
        "definitions": {
            "mimic4_icu_death": (
                "admissions.deathtime lies within the ICU intime-outtime interval"
            ),
            "eicu_icu_death": "patient.unitdischargestatus == Expired",
            "mimic4_icd_era": (
                "sepsis code version in diagnoses_icd; admissions with both ICD-9 "
                "and ICD-10 sepsis codes are labelled mixed and excluded from era "
                "strata"
            ),
            "eicu_icd_era": (
                "not applicable because eICU diagnosis.icd9code is a harmonised "
                "field and does not provide ICD-10 admission coding"
            ),
        },
    }


def main() -> None:
    connection = duckdb.connect()
    connection.execute("PRAGMA threads=12")
    connection.execute("PRAGMA memory_limit='10GB'")
    mimic4 = mimic4_flags(connection)
    eicu = eicu_flags(connection)
    summary = summarize(mimic4, eicu)
    with (OUT / "sensitivity_flags_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
