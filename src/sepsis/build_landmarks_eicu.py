# -*- coding: utf-8 -*-
"""Build eICU hourly, landmark-feature, and 72-hour sequence artifacts."""

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import duckdb


ROOT = Path(os.environ.get("SEPSIS_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
SOURCE_ROOT = Path(
    os.environ.get("SEPSIS_SOURCE_ROOT", r"D:\TT paper\0811Temporal Pathway")
)
EICU = SOURCE_ROOT / "eicu"
OUT = ROOT / "data" / "processed" / "eicu"
OUT.mkdir(parents=True, exist_ok=True)


def q(path: Path) -> str:
    return str(path).replace("\\", "/").replace("'", "''")


def build_hourly(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    cohort = q(OUT / "cohort_v2_all.parquet")
    start = time.time()
    con.execute(
        f"""
        CREATE TEMP TABLE P AS
        SELECT patientunitstayid,
               CAST(floor(observationoffset / 60.0) AS INT) hh,
               CASE WHEN heartrate BETWEEN 20 AND 250 THEN heartrate END hr,
               CASE WHEN respiration BETWEEN 4 AND 60 THEN respiration END rr,
               CASE WHEN sao2 BETWEEN 50 AND 100 THEN sao2 END spo2,
               CASE WHEN systemicsystolic BETWEEN 40 AND 300
                    THEN systemicsystolic END sbp_i,
               CASE WHEN systemicdiastolic BETWEEN 20 AND 200
                    THEN systemicdiastolic END dbp_i,
               CASE WHEN systemicmean BETWEEN 30 AND 200
                    THEN systemicmean END mbp_i
        FROM read_csv_auto('{q(EICU / "vitalPeriodic.csv.gz")}')
        WHERE patientunitstayid IN (
            SELECT patientunitstayid FROM read_parquet('{cohort}')
        )
          AND observationoffset >= 0
          AND observationoffset < 72 * 60
        """
    )
    print(
        "eICU periodic rows:",
        con.execute("SELECT COUNT(*) FROM P").fetchone()[0],
        f"{time.time() - start:.0f}s",
        flush=True,
    )

    con.execute(
        f"""
        CREATE TEMP TABLE A AS
        SELECT patientunitstayid,
               CAST(floor(observationoffset / 60.0) AS INT) hh,
               CASE WHEN noninvasivesystolic BETWEEN 40 AND 300
                    THEN noninvasivesystolic END sbp_n,
               CASE WHEN noninvasivediastolic BETWEEN 20 AND 200
                    THEN noninvasivediastolic END dbp_n,
               CASE WHEN noninvasivemean BETWEEN 30 AND 200
                    THEN noninvasivemean END mbp_n
        FROM read_csv_auto('{q(EICU / "vitalAperiodic.csv.gz")}')
        WHERE patientunitstayid IN (
            SELECT patientunitstayid FROM read_parquet('{cohort}')
        )
          AND observationoffset >= 0
          AND observationoffset < 72 * 60
        """
    )
    print(
        "eICU aperiodic rows:",
        con.execute("SELECT COUNT(*) FROM A").fetchone()[0],
        f"{time.time() - start:.0f}s",
        flush=True,
    )

    con.execute(
        f"""
        CREATE TEMP TABLE N AS
        SELECT patientunitstayid,
               CAST(floor(nursingchartoffset / 60.0) AS INT) hh,
               CASE
                   WHEN nursingchartcelltypevalname = 'Temperature (C)'
                        AND TRY_CAST(nursingchartvalue AS DOUBLE) BETWEEN 25 AND 45
                   THEN TRY_CAST(nursingchartvalue AS DOUBLE) * 9.0 / 5.0 + 32.0
               END AS tempf,
               CASE
                   WHEN nursingchartcelltypevalname = 'GCS Total'
                        AND TRY_CAST(nursingchartvalue AS DOUBLE) BETWEEN 3 AND 15
                   THEN TRY_CAST(nursingchartvalue AS DOUBLE)
               END AS gcs
        FROM read_csv_auto('{q(EICU / "nurseCharting.csv.gz")}')
        WHERE patientunitstayid IN (
            SELECT patientunitstayid FROM read_parquet('{cohort}')
        )
          AND nursingchartoffset >= 0
          AND nursingchartoffset < 72 * 60
          AND nursingchartcelltypevalname IN ('Temperature (C)', 'GCS Total')
        """
    )
    print(
        "eICU nurse-chart rows:",
        con.execute("SELECT COUNT(*) FROM N").fetchone()[0],
        f"{time.time() - start:.0f}s",
        flush=True,
    )

    con.execute(
        """
        CREATE TEMP TABLE H AS
        WITH PP AS (
            SELECT patientunitstayid, hh,
                   median(hr) hr,
                   median(rr) rr,
                   median(spo2) spo2,
                   median(sbp_i) sbp_i,
                   median(dbp_i) dbp_i,
                   median(mbp_i) mbp_i
            FROM P
            GROUP BY 1, 2
        ),
        AA AS (
            SELECT patientunitstayid, hh,
                   median(sbp_n) sbp_n,
                   median(dbp_n) dbp_n,
                   median(mbp_n) mbp_n
            FROM A
            GROUP BY 1, 2
        ),
        NN AS (
            SELECT patientunitstayid, hh,
                   median(tempf) tempf,
                   median(gcs) gcs
            FROM N
            GROUP BY 1, 2
        ),
        K AS (
            SELECT patientunitstayid, hh FROM PP
            UNION
            SELECT patientunitstayid, hh FROM AA
            UNION
            SELECT patientunitstayid, hh FROM NN
        )
        SELECT K.patientunitstayid, K.hh,
               PP.hr, PP.rr, PP.spo2,
               coalesce(PP.sbp_i, AA.sbp_n) AS sbp,
               coalesce(PP.dbp_i, AA.dbp_n) AS dbp,
               coalesce(PP.mbp_i, AA.mbp_n) AS mbp,
               NN.tempf, NN.gcs
        FROM K
        LEFT JOIN PP USING (patientunitstayid, hh)
        LEFT JOIN AA USING (patientunitstayid, hh)
        LEFT JOIN NN USING (patientunitstayid, hh)
        """
    )
    hourly = con.execute("SELECT * FROM H").fetchdf()
    hourly.to_parquet(OUT / "hourly_all.parquet", index=False)
    channel_cols = ["hr", "rr", "spo2", "sbp", "dbp", "mbp", "tempf", "gcs"]
    print(
        "eICU hourly:",
        hourly.shape,
        "coverage %:",
        (hourly[channel_cols].notna().mean() * 100).round(1).to_dict(),
        f"{time.time() - start:.0f}s",
        flush=True,
    )
    return hourly


def build_features(hourly: pd.DataFrame) -> pd.DataFrame:
    channels = ["hr", "rr", "spo2", "sbp", "dbp", "mbp", "tempf", "gcs"]
    work = hourly.sort_values(
        ["patientunitstayid", "hh"], kind="mergesort"
    ).copy()
    work["window"] = (work["hh"] // 24).astype(int)
    parts = []
    for window in (0, 1, 2):
        grouped = (
            work.loc[work["window"] == window]
            .groupby("patientunitstayid")[channels]
            .agg(["mean", "min", "max", "last", "count"])
        )
        grouped.columns = [
            f"{channel}_w{window + 1}_{stat}"
            for channel, stat in grouped.columns
        ]
        parts.append(grouped)
    features = pd.concat(parts, axis=1).reset_index()
    features.to_parquet(OUT / "vitals_all.parquet", index=False)
    return features


def build_labs(con: duckdb.DuckDBPyConnection) -> None:
    cohort = q(OUT / "cohort_v2_all.parquet")
    con.execute(
        f"""
        COPY (
            WITH L AS (
                SELECT c.patientunitstayid, l.labname, l.labresult
                FROM read_csv_auto('{q(EICU / "lab.csv.gz")}') l
                JOIN read_parquet('{cohort}') c USING (patientunitstayid)
                WHERE l.labname IN (
                    'creatinine', 'lactate', 'WBC x 1000',
                    'platelets x 1000', 'total bilirubin', 'sodium',
                    'BUN', 'Hgb', 'PT - INR'
                )
                  AND l.labresult IS NOT NULL
                  AND l.labresultoffset >= 0
                  AND l.labresultoffset < 24 * 60
            )
            SELECT patientunitstayid,
                   min(labresult) FILTER (WHERE labname = 'creatinine') crea_min,
                   max(labresult) FILTER (WHERE labname = 'creatinine') crea_max,
                   min(labresult) FILTER (WHERE labname = 'lactate') lac_min,
                   max(labresult) FILTER (WHERE labname = 'lactate') lac_max,
                   min(labresult) FILTER (WHERE labname = 'WBC x 1000') wbc_min,
                   max(labresult) FILTER (WHERE labname = 'WBC x 1000') wbc_max,
                   min(labresult) FILTER (WHERE labname = 'platelets x 1000') plt_min,
                   max(labresult) FILTER (WHERE labname = 'platelets x 1000') plt_max,
                   min(labresult) FILTER (WHERE labname = 'total bilirubin') bili_min,
                   max(labresult) FILTER (WHERE labname = 'total bilirubin') bili_max,
                   min(labresult) FILTER (WHERE labname = 'sodium') na_min,
                   max(labresult) FILTER (WHERE labname = 'sodium') na_max,
                   min(labresult) FILTER (WHERE labname = 'BUN') bun_min,
                   max(labresult) FILTER (WHERE labname = 'BUN') bun_max,
                   min(labresult) FILTER (WHERE labname = 'Hgb') hgb_min,
                   max(labresult) FILTER (WHERE labname = 'Hgb') hgb_max,
                   max(labresult) FILTER (WHERE labname = 'PT - INR') inr_max
            FROM L
            GROUP BY 1
        ) TO '{q(OUT / "labs_all.parquet")}' (FORMAT PARQUET)
        """
    )


def build_design(features: pd.DataFrame) -> pd.DataFrame:
    cohort = pd.read_parquet(OUT / "cohort_v2_all.parquet")
    labs = pd.read_parquet(OUT / "labs_all.parquet")
    design = cohort.merge(features, on="patientunitstayid", how="left").merge(
        labs, on="patientunitstayid", how="left"
    )
    design["male"] = design["gender"].isin(["M", "Male"]).astype(int)
    design = pd.get_dummies(
        design,
        columns=["unitadmitsource", "unittype"],
        prefix=["src", "ut"],
        dummy_na=False,
    )
    design["y3"] = design["y"].map({0: 0, 1: 1, 2: 1, 3: 2})
    design.to_parquet(OUT / "design_all.parquet", index=False)
    complete = design.loc[
        design["y"].notna() & (design["los_h"] >= 72)
    ].copy()
    complete.to_parquet(OUT / "design_v3.parquet", index=False)
    print("eICU complete-window design", complete.shape, flush=True)
    return design


def build_sequences(hourly: pd.DataFrame, design: pd.DataFrame) -> None:
    channels = ["hr", "rr", "spo2", "sbp", "dbp", "mbp", "tempf", "gcs"]
    ids = design["patientunitstayid"].astype(int).to_numpy()
    index = {value: i for i, value in enumerate(ids)}
    x = np.full((len(design), 72, len(channels)), np.nan, dtype=np.float32)
    for row in hourly.itertuples(index=False):
        i = index.get(int(row.patientunitstayid))
        if i is None or not 0 <= int(row.hh) < 72:
            continue
        values = [row.hr, row.rr, row.spo2, row.sbp, row.dbp, row.mbp, row.tempf, row.gcs]
        x[i, int(row.hh), :] = np.asarray(values, dtype=np.float32)
    mask = (~np.isnan(x)).any(axis=2).astype(np.float32)
    y4 = design["y"].fillna(-1).to_numpy(dtype=np.int64)
    y3 = design["y3"].fillna(-1).to_numpy(dtype=np.int64)
    np.savez_compressed(
        OUT / "seq_all72.npz",
        X=x,
        M=mask,
        y=y4,
        y3=y3,
        patientunitstayid=ids,
        uniquepid=design["uniquepid"].astype(str).to_numpy(),
        hospitalid=design["hospitalid"].astype(np.int64).to_numpy(),
    )

    current = pd.read_parquet(ROOT / "data" / "processed" / "eicu" / "design_v3.parquet")
    current_ids = current["patientunitstayid"].astype(int).to_numpy()
    lookup = {int(pid): i for i, pid in enumerate(ids)}
    positions = np.asarray([lookup[int(pid)] for pid in current_ids], dtype=np.int64)
    np.savez_compressed(
        OUT / "seq72.npz",
        X=x[positions],
        M=mask[positions],
        y4=current["y"].to_numpy(dtype=np.int64),
        y3=current["y"].map({0: 0, 1: 1, 2: 1, 3: 2}).to_numpy(dtype=np.int64),
        patientunitstayid=current_ids,
        uniquepid=current["uniquepid"].astype(str).to_numpy(),
        hospitalid=current["hospitalid"].astype(np.int64).to_numpy(),
    )


def main() -> None:
    con = duckdb.connect()
    con.execute("PRAGMA threads=8")
    con.execute("PRAGMA memory_limit='12GB'")
    start = time.time()
    hourly = build_hourly(con)
    features = build_features(hourly)
    build_labs(con)
    design = build_design(features)
    build_sequences(hourly, design)
    print(
        "eICU landmark artifacts:",
        {"design_all": design.shape, "hourly": hourly.shape},
        f"{time.time() - start:.0f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
