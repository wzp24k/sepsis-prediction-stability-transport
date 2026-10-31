# -*- coding: utf-8 -*-
"""Build a same-institution MIMIC-III era comparison with XGBoost."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier


ROOT = Path(os.environ.get("SEPSIS_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
SOURCE = (
    Path(os.environ.get("SEPSIS_SOURCE_ROOT", r"D:\TT paper\0811Temporal Pathway"))
    / "mimic-iii-clinical-database-1.4_1"
    / "mimic-iii-clinical-database-1.4"
)
OUT = ROOT / "data" / "processed" / "mimic3"
RESULT = ROOT / "results" / "sensitivity_v2" / "mimic3"
OUT.mkdir(parents=True, exist_ok=True)
RESULT.mkdir(parents=True, exist_ok=True)


def q(path: Path) -> str:
    return str(path).replace("\\", "/").replace("'", "''")


def build_cohort(connection: duckdb.DuckDBPyConnection) -> None:
    connection.execute(
        f"""
        CREATE TEMP TABLE SEP AS
        SELECT DISTINCT CAST(hadm_id AS BIGINT) hadm_id
        FROM read_csv_auto(
            '{q(SOURCE / "DIAGNOSES_ICD.csv.gz")}',
            all_varchar = true
        )
        WHERE regexp_matches(
            upper(trim(icd9_code)),
            '^(038|9959|78552)'
        )
        """
    )
    connection.execute(
        f"""
        COPY (
        WITH I AS (
            SELECT CAST(i.icustay_id AS BIGINT) icustay_id,
                   CAST(i.hadm_id AS BIGINT) hadm_id,
                   CAST(i.subject_id AS BIGINT) subject_id,
                   CAST(i.intime AS TIMESTAMP) intime,
                   CAST(i.outtime AS TIMESTAMP) outtime,
                   i.first_careunit,
                   TRY_CAST(i.los AS DOUBLE) * 24.0 los,
                   p.gender,
                   date_diff(
                       'year',
                       CAST(p.dob AS TIMESTAMP),
                       CAST(i.intime AS TIMESTAMP)
                   ) raw_age
            FROM read_csv_auto('{q(SOURCE / "ICUSTAYS.csv.gz")}') i
            JOIN SEP USING (hadm_id)
            JOIN read_csv_auto('{q(SOURCE / "PATIENTS.csv.gz")}') p
              USING (subject_id)
            QUALIFY row_number() OVER (
                PARTITION BY i.hadm_id ORDER BY i.intime, i.icustay_id
            ) = 1
        ),
        A AS (
            SELECT CAST(hadm_id AS BIGINT) hadm_id,
                   admission_type,
                   discharge_location,
                   TRY_CAST(hospital_expire_flag AS INTEGER) hospital_expire_flag
            FROM read_csv_auto('{q(SOURCE / "ADMISSIONS.csv.gz")}')
        ),
        C AS (
            SELECT I.*, A.admission_type, A.discharge_location,
                   A.hospital_expire_flag,
                   least(I.raw_age, 90) age,
                   CASE
                       WHEN A.hospital_expire_flag = 1
                         OR upper(A.discharge_location) LIKE 'DEAD%'
                       THEN 2
                       WHEN upper(A.discharge_location) IN (
                           'HOME', 'HOME HEALTH CARE',
                           'LEFT AGAINST MEDICAL ADVI', 'LEFT AMA',
                           'HOME WITH HOME IV PROVIDR',
                           'HOSPICE-HOME'
                       ) THEN 0
                       WHEN upper(A.discharge_location) IN (
                           'SNF', 'SNF-MEDICAID ONLY CERTIF',
                           'REHAB/DISTINCT PART HOSP',
                           'LONG TERM CARE HOSPITAL', 'SHORT TERM HOSPITAL',
                           'DISC-TRAN CANCER/CHLDRN H',
                           'DISCH-TRAN TO PSYCH HOSP', 'OTHER FACILITY',
                           'ICF', 'DISC-TRAN TO FEDERAL HC',
                           'HOSPICE-MEDICAL FACILITY'
                       ) THEN 1
                       ELSE NULL
                   END y3
            FROM I JOIN A USING (hadm_id)
        )
        SELECT * FROM C WHERE age >= 18
        ) TO '{q(OUT / "cohort.parquet")}' (FORMAT PARQUET)
        """
    )
    print(
        connection.execute(
            f"""
            SELECT COUNT(*) n,
                   COUNT(y3) mapped,
                   COUNT(DISTINCT subject_id) patients,
                   SUM(CASE WHEN los >= 72 THEN 1 ELSE 0 END) los72
            FROM read_parquet('{q(OUT / "cohort.parquet")}')
            """
        ).fetchdf().to_string(index=False),
        flush=True,
    )


def build_hourly(connection: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    cohort = q(OUT / "cohort.parquet")
    connection.execute(
        f"""
        CREATE TEMP TABLE V AS
        SELECT c.icustay_id, CAST(c.charttime AS TIMESTAMP) charttime,
               CASE c.itemid
                   WHEN 211 THEN 'hr' WHEN 220045 THEN 'hr'
                   WHEN 615 THEN 'rr' WHEN 618 THEN 'rr'
                   WHEN 220210 THEN 'rr' WHEN 224690 THEN 'rr'
                   WHEN 646 THEN 'spo2' WHEN 220277 THEN 'spo2'
                   WHEN 51 THEN 'sbp' WHEN 442 THEN 'sbp'
                   WHEN 455 THEN 'sbp' WHEN 220050 THEN 'sbp'
                   WHEN 220179 THEN 'sbp'
                   WHEN 8368 THEN 'dbp' WHEN 8440 THEN 'dbp'
                   WHEN 8441 THEN 'dbp' WHEN 220051 THEN 'dbp'
                   WHEN 220180 THEN 'dbp'
                   WHEN 52 THEN 'mbp' WHEN 443 THEN 'mbp'
                   WHEN 456 THEN 'mbp' WHEN 220052 THEN 'mbp'
                   WHEN 220181 THEN 'mbp'
                   WHEN 676 THEN 'temp' WHEN 677 THEN 'temp'
                   WHEN 678 THEN 'temp' WHEN 679 THEN 'temp'
                   WHEN 223761 THEN 'temp' WHEN 223762 THEN 'temp'
               END channel,
               CASE
                   WHEN c.itemid IN (676, 677, 223762)
                   THEN c.valuenum * 9.0 / 5.0 + 32.0
                   ELSE c.valuenum
               END AS value
        FROM read_csv_auto('{q(SOURCE / "CHARTEVENTS.csv.gz")}') c
        JOIN read_parquet('{cohort}') co USING (icustay_id)
        WHERE c.itemid IN (
            211, 220045, 615, 618, 220210, 224690, 646, 220277,
            51, 442, 455, 220050, 220179, 8368, 8440, 8441,
            220051, 220180, 52, 443, 456, 220052, 220181,
            676, 677, 678, 679, 223761, 223762
        )
          AND c.valuenum IS NOT NULL
          AND c.charttime IS NOT NULL
        """
    )
    connection.execute(
        f"""
        CREATE TEMP TABLE G AS
        SELECT c.icustay_id, CAST(c.charttime AS TIMESTAMP) charttime,
               max(CASE WHEN c.itemid = 198 THEN c.valuenum END) total,
               max(CASE WHEN c.itemid = 220739 THEN c.valuenum END) eye,
               max(CASE WHEN c.itemid = 223900 THEN c.valuenum END) verbal,
               max(CASE WHEN c.itemid = 223901 THEN c.valuenum END) motor
        FROM read_csv_auto('{q(SOURCE / "CHARTEVENTS.csv.gz")}') c
        JOIN read_parquet('{cohort}') co USING (icustay_id)
        WHERE c.itemid IN (198, 220739, 223900, 223901)
          AND c.valuenum IS NOT NULL
          AND c.charttime IS NOT NULL
        GROUP BY 1, 2
        """
    )
    connection.execute(
        """
        CREATE TEMP TABLE O AS
        SELECT icustay_id, charttime, channel, value FROM V
        UNION ALL
        SELECT icustay_id, charttime, 'gcs',
               coalesce(total, eye + verbal + motor)
        FROM G
        WHERE total IS NOT NULL
           OR (eye IS NOT NULL AND verbal IS NOT NULL AND motor IS NOT NULL)
        """
    )
    connection.execute(
        f"""
        COPY (
        SELECT o.icustay_id,
               CAST(floor(
                   date_diff('second', c.intime, o.charttime) / 3600.0
               ) AS INTEGER) hh,
               median(o.value) FILTER (WHERE o.channel = 'hr') hr,
               median(o.value) FILTER (WHERE o.channel = 'rr') rr,
               median(o.value) FILTER (WHERE o.channel = 'spo2') spo2,
               median(o.value) FILTER (WHERE o.channel = 'sbp') sbp,
               median(o.value) FILTER (WHERE o.channel = 'dbp') dbp,
               median(o.value) FILTER (WHERE o.channel = 'mbp') mbp,
               median(o.value) FILTER (WHERE o.channel = 'temp') tempf,
               median(o.value) FILTER (WHERE o.channel = 'gcs') gcs
        FROM O o
        JOIN read_parquet('{cohort}') c USING (icustay_id)
        WHERE date_diff('second', c.intime, o.charttime) >= 0
          AND date_diff('second', c.intime, o.charttime) < 72 * 3600
        GROUP BY 1, 2
        ) TO '{q(OUT / "hourly72.parquet")}' (FORMAT PARQUET)
        """
    )
    hourly = pd.read_parquet(OUT / "hourly72.parquet")
    # MIMIC-III source columns are uppercase while the derived design uses
    # lowercase names. DuckDB preserves that spelling in the hourly export.
    hourly.columns = [column.lower() for column in hourly.columns]
    print("MIMIC-III hourly", hourly.shape, flush=True)
    return hourly


def build_labs(connection: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    connection.execute(
        f"""
        COPY (
        SELECT c.icustay_id,
               min(l.valuenum) FILTER (WHERE l.itemid = 50912) crea_min,
               max(l.valuenum) FILTER (WHERE l.itemid = 50912) crea_max,
               min(l.valuenum) FILTER (WHERE l.itemid = 50813) lac_min,
               max(l.valuenum) FILTER (WHERE l.itemid = 50813) lac_max,
               min(l.valuenum) FILTER (WHERE l.itemid = 51301) wbc_min,
               max(l.valuenum) FILTER (WHERE l.itemid = 51301) wbc_max,
               min(l.valuenum) FILTER (WHERE l.itemid = 51265) plt_min,
               max(l.valuenum) FILTER (WHERE l.itemid = 51265) plt_max,
               min(l.valuenum) FILTER (WHERE l.itemid = 50885) bili_min,
               max(l.valuenum) FILTER (WHERE l.itemid = 50885) bili_max,
               min(l.valuenum) FILTER (WHERE l.itemid = 50983) na_min,
               max(l.valuenum) FILTER (WHERE l.itemid = 50983) na_max,
               min(l.valuenum) FILTER (WHERE l.itemid = 51006) bun_min,
               max(l.valuenum) FILTER (WHERE l.itemid = 51006) bun_max,
               min(l.valuenum) FILTER (WHERE l.itemid = 51222) hgb_min,
               max(l.valuenum) FILTER (WHERE l.itemid = 51222) hgb_max,
               max(l.valuenum) FILTER (WHERE l.itemid = 51237) inr_max
        FROM read_csv_auto('{q(SOURCE / "LABEVENTS.csv.gz")}') l
        JOIN read_parquet('{q(OUT / "cohort.parquet")}') c USING (hadm_id)
        WHERE l.itemid IN (
            50912, 50813, 51301, 51265, 50885, 50983, 51006, 51222, 51237
        )
          AND l.valuenum IS NOT NULL
          AND CAST(l.charttime AS TIMESTAMP) >= c.intime
          AND CAST(l.charttime AS TIMESTAMP) < c.intime + INTERVAL 24 HOUR
        GROUP BY 1
        ) TO '{q(OUT / "labs24.parquet")}' (FORMAT PARQUET)
        """
    )
    labs = pd.read_parquet(OUT / "labs24.parquet")
    print("MIMIC-III labs", labs.shape, flush=True)
    return labs


def build_features(hourly: pd.DataFrame) -> pd.DataFrame:
    channels = ["hr", "rr", "spo2", "sbp", "dbp", "mbp", "tempf", "gcs"]
    work = hourly.sort_values(["icustay_id", "hh"], kind="mergesort").copy()
    work["window"] = (work["hh"] // 24).astype(int)
    parts = []
    for window in range(3):
        grouped = (
            work.loc[work["window"] == window]
            .groupby("icustay_id")[channels]
            .agg(["mean", "min", "max", "last", "count"])
        )
        grouped.columns = [
            f"{channel}_w{window + 1}_{stat}"
            for channel, stat in grouped.columns
        ]
        parts.append(grouped)
    features = pd.concat(parts, axis=1).reset_index()
    features.to_parquet(OUT / "vitals24_72.parquet", index=False)
    return features


def build_design(
    connection: duckdb.DuckDBPyConnection,
    features: pd.DataFrame,
    labs: pd.DataFrame,
) -> pd.DataFrame:
    del connection
    cohort = pd.read_parquet(OUT / "cohort.parquet")
    cohort.columns = [column.lower() for column in cohort.columns]
    design = cohort.merge(features, on="icustay_id", how="left").merge(
        labs, on="icustay_id", how="left"
    )
    design["male"] = (design["gender"].astype(str).str.upper() == "M").astype(int)
    design = pd.get_dummies(
        design,
        columns=["admission_type", "first_careunit"],
        prefix=["adm", "unit"],
        dummy_na=False,
    )
    design.to_parquet(OUT / "design_all.parquet", index=False)
    print("MIMIC-III design", design.shape, flush=True)
    return design


def feature_columns(frame: pd.DataFrame, milestone: int) -> list[str]:
    labs = [
        "crea_min",
        "crea_max",
        "lac_min",
        "lac_max",
        "wbc_min",
        "wbc_max",
        "plt_min",
        "plt_max",
        "bili_min",
        "bili_max",
        "na_min",
        "na_max",
        "bun_min",
        "bun_max",
        "hgb_min",
        "hgb_max",
        "inr_max",
    ]
    static = ["age", "male"] + labs
    static += [
        column
        for column in frame.columns
        if column.startswith(("adm_", "unit_"))
    ]
    tokens = [f"_w{index}_" for index in range(1, milestone // 24 + 1)]
    dynamic = [
        column for column in frame.columns if any(token in column for token in tokens)
    ]
    return list(dict.fromkeys(static + dynamic))


def train_xgboost() -> dict:
    frame = pd.read_parquet(OUT / "design_all.parquet")
    probabilities = {}
    ids = {}
    outcomes = {}
    summary = {
        "database": "MIMIC-III",
        "run_scope": "single fixed-seed patient-grouped out-of-fold run",
        "inference": "no confidence intervals; historical descriptive comparison only",
        "role": (
            "same-institution different-era comparison; not pooled into the "
            "primary estimate and not external validation of fixed weights"
        ),
        "landmarks": {},
    }
    parameters = {
        "objective": "multi:softprob",
        "num_class": 3,
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.06,
        "subsample": 0.85,
        "colsample_bytree": 0.8,
        "tree_method": "hist",
        "n_jobs": int(os.environ.get("SEPSIS_XGB_THREADS", "4")),
        "random_state": 20261029,
        "eval_metric": "mlogloss",
    }
    for milestone in (24, 48, 72):
        eligible = frame["y3"].notna() & (frame["los"] >= milestone)
        subset = frame.loc[eligible].copy()
        y = subset["y3"].to_numpy(dtype=int)
        groups = subset["subject_id"].astype(str).to_numpy()
        columns = feature_columns(subset, milestone)
        x = subset[columns].astype(float)
        folds = list(GroupKFold(n_splits=5).split(x, y, groups))
        probability = np.zeros((len(subset), 3), dtype=float)
        for train_index, test_index in folds:
            model = XGBClassifier(**parameters)
            model.fit(x.iloc[train_index], y[train_index])
            probability[test_index] = model.predict_proba(x.iloc[test_index])
        probabilities[milestone] = probability
        ids[milestone] = subset["icustay_id"].to_numpy(dtype=np.int64)
        outcomes[milestone] = y
        summary["landmarks"][str(milestone)] = {
            "n": int(len(subset)),
            "class_counts": {
                str(value): int((y == value).sum()) for value in range(3)
            },
            "macro_auc": round(
                float(
                    roc_auc_score(
                        y,
                        probability,
                        multi_class="ovr",
                        average="macro",
                        labels=np.arange(3),
                    )
                ),
                4,
            ),
        }
        print("MIMIC-III", milestone, summary["landmarks"][str(milestone)], flush=True)

    common = np.intersect1d(
        np.intersect1d(ids[24], ids[48]),
        ids[72],
    )
    aligned = {}
    for milestone in (24, 48, 72):
        lookup = {
            int(identifier): index for index, identifier in enumerate(ids[milestone])
        }
        index = np.asarray(
            [lookup[int(identifier)] for identifier in common], dtype=int
        )
        aligned[milestone] = {
            "p": probabilities[milestone][index],
            "y": outcomes[milestone][index],
        }
    prediction_24 = aligned[24]["p"].argmax(1)
    prediction_72 = aligned[72]["p"].argmax(1)
    summary["complete_three_landmark_subset"] = {
        "n": int(len(common)),
        "macro_auc": {
            str(milestone): round(
                float(
                    roc_auc_score(
                        aligned[milestone]["y"],
                        aligned[milestone]["p"],
                        multi_class="ovr",
                        average="macro",
                        labels=np.arange(3),
                    )
                ),
                4,
            )
            for milestone in (24, 48, 72)
        },
        "flip_24_72": round(
            float((prediction_24 != prediction_72).mean()), 4
        ),
        "tv_24_72": round(
            float(
                (
                    0.5
                    * np.abs(aligned[24]["p"] - aligned[72]["p"]).sum(1)
                ).mean()
            ),
            4,
        ),
    }
    np.savez_compressed(
        RESULT / "mimic3_xgb.npz",
        p24=aligned[24]["p"],
        p48=aligned[48]["p"],
        p72=aligned[72]["p"],
        y24=aligned[24]["y"],
        y48=aligned[48]["y"],
        y72=aligned[72]["y"],
        ids=common,
    )
    return summary


def main() -> None:
    started = time.time()
    connection = duckdb.connect()
    connection.execute(
        f"PRAGMA threads={int(os.environ.get('SEPSIS_DUCKDB_THREADS', '12'))}"
    )
    connection.execute("PRAGMA memory_limit='12GB'")
    measurement_paths = (OUT / "hourly72.parquet", OUT / "labs24.parquet")
    reuse_derived = os.environ.get("SEPSIS_REUSE_DERIVED", "0") == "1"
    if reuse_derived and all(path.exists() for path in measurement_paths):
        # Outcome labels can change without changing source ICU identifiers.
        # Rebuild the cohort mapping but reuse the expensive measurements.
        build_cohort(connection)
        hourly = pd.read_parquet(OUT / "hourly72.parquet")
        hourly.columns = [column.lower() for column in hourly.columns]
        labs = pd.read_parquet(OUT / "labs24.parquet")
        print("Rebuilt cohort; reusing MIMIC-III hourly and laboratory tables", flush=True)
    else:
        build_cohort(connection)
        hourly = build_hourly(connection)
        labs = build_labs(connection)
    features = build_features(hourly)
    build_design(connection, features, labs)
    summary = train_xgboost()
    with (RESULT / "mimic3_xgb_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(
        json.dumps(
            {
                "summary": summary,
                "elapsed_seconds": round(time.time() - started, 1),
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
