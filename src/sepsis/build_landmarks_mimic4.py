# -*- coding: utf-8 -*-
"""Build MIMIC-IV hourly, landmark-feature, and sequence artifacts."""

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
MIMIC = SOURCE_ROOT / "mimic-iv-3.1"
OUT = ROOT / "data" / "processed" / "mimic4"
OUT.mkdir(parents=True, exist_ok=True)


def q(path: Path) -> str:
    return str(path).replace("\\", "/").replace("'", "''")


def build_hourly(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    cohort = q(OUT / "cohort.parquet")
    start = time.time()
    con.execute(
        f"""
        CREATE TEMP TABLE V AS
        SELECT c.stay_id, c.charttime,
               CASE c.itemid
                    WHEN 220045 THEN 'hr'
                    WHEN 220210 THEN 'rr'
                    WHEN 224690 THEN 'rr'
                    WHEN 220277 THEN 'spo2'
                    WHEN 220179 THEN 'sbp'
                    WHEN 220050 THEN 'sbp'
                    WHEN 220180 THEN 'dbp'
                    WHEN 220051 THEN 'dbp'
                    WHEN 220181 THEN 'mbp'
                    WHEN 220052 THEN 'mbp'
                    WHEN 223761 THEN 'temp'
                    WHEN 223762 THEN 'temp'
               END AS channel,
               CASE WHEN c.itemid = 223762 THEN c.valuenum * 9.0 / 5.0 + 32.0
                    ELSE c.valuenum END AS value
        FROM read_csv_auto('{q(MIMIC / "icu" / "chartevents.csv.gz")}') c
        JOIN read_parquet('{cohort}') co USING (stay_id)
        WHERE c.itemid IN (
            220045, 220210, 224690, 220277, 220179, 220050,
            220180, 220051, 220181, 220052, 223761, 223762
        )
          AND c.valuenum IS NOT NULL
        """
    )
    con.execute(
        f"""
        CREATE TEMP TABLE G AS
        SELECT c.stay_id, c.charttime,
               max(CASE WHEN c.itemid = 220739 THEN c.valuenum END) eye,
               max(CASE WHEN c.itemid = 223900 THEN c.valuenum END) verbal,
               max(CASE WHEN c.itemid = 223901 THEN c.valuenum END) motor
        FROM read_csv_auto('{q(MIMIC / "icu" / "chartevents.csv.gz")}') c
        JOIN read_parquet('{cohort}') co USING (stay_id)
        WHERE c.itemid IN (220739, 223900, 223901)
          AND c.valuenum IS NOT NULL
        GROUP BY 1, 2
        """
    )
    con.execute(
        f"""
        CREATE TEMP TABLE OBS AS
        SELECT V.stay_id, V.charttime, V.channel, V.value FROM V
        UNION ALL
        SELECT G.stay_id, G.charttime, 'gcs',
               G.eye + G.verbal + G.motor
        FROM G
        WHERE G.eye IS NOT NULL AND G.verbal IS NOT NULL AND G.motor IS NOT NULL
        """
    )
    con.execute(
        f"""
        CREATE TEMP TABLE H AS
        SELECT o.stay_id,
               CAST(floor(date_diff('second', c.intime, o.charttime) / 3600.0) AS INT) hh,
               median(o.value) FILTER (WHERE o.channel = 'hr') hr,
               median(o.value) FILTER (WHERE o.channel = 'rr') rr,
               median(o.value) FILTER (WHERE o.channel = 'spo2') spo2,
               median(o.value) FILTER (WHERE o.channel = 'sbp') sbp,
               median(o.value) FILTER (WHERE o.channel = 'dbp') dbp,
               median(o.value) FILTER (WHERE o.channel = 'mbp') mbp,
               median(o.value) FILTER (WHERE o.channel = 'temp') tempf,
               median(o.value) FILTER (WHERE o.channel = 'gcs') gcs
        FROM OBS o
        JOIN read_parquet('{cohort}') c USING (stay_id)
        WHERE date_diff('second', c.intime, o.charttime) >= 0
          AND date_diff('second', c.intime, o.charttime) < 72 * 3600
        GROUP BY 1, 2
        """
    )
    hourly = con.execute("SELECT * FROM H").fetchdf()
    hourly.to_parquet(OUT / "hourly_all.parquet", index=False)
    channels = ["hr", "rr", "spo2", "sbp", "dbp", "mbp", "tempf", "gcs"]
    print(
        "MIMIC hourly:",
        hourly.shape,
        "coverage %:",
        (hourly[channels].notna().mean() * 100).round(1).to_dict(),
        f"{time.time() - start:.0f}s",
        flush=True,
    )
    return hourly


def build_features(hourly: pd.DataFrame) -> pd.DataFrame:
    channels = ["hr", "rr", "spo2", "sbp", "dbp", "mbp", "tempf", "gcs"]
    work = hourly.sort_values(["stay_id", "hh"], kind="mergesort").copy()
    work["window"] = (work["hh"] // 24).astype(int)
    parts = []
    for window in (0, 1, 2):
        grouped = (
            work.loc[work["window"] == window]
            .groupby("stay_id")[channels]
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


def build_design(features: pd.DataFrame) -> pd.DataFrame:
    cohort = pd.read_parquet(OUT / "cohort.parquet")
    labs = pd.read_parquet(OUT / "labs.parquet")
    static = cohort[
        [
            "stay_id",
            "subject_id",
            "age",
            "gender",
            "admission_type",
            "first_careunit",
            "los",
            "y",
        ]
    ]
    design = static.merge(features, on="stay_id", how="left").merge(
        labs, on="stay_id", how="left"
    )
    design["male"] = (design["gender"] == "M").astype(int)
    design = pd.get_dummies(
        design,
        columns=["admission_type", "first_careunit"],
        prefix=["adm", "unit"],
        dummy_na=False,
    )
    design["y3"] = design["y"].map({0: 0, 1: 1, 2: 1, 3: 2})
    design.to_parquet(OUT / "design_all.parquet", index=False)
    complete = design.loc[design["y"].notna() & (design["los"] >= 72)].copy()
    zero_complete_dummies = [
        column
        for column in complete.columns
        if column.startswith(("adm_", "unit_")) and not complete[column].any()
    ]
    complete = complete.drop(columns=["los", *zero_complete_dummies])
    complete.to_parquet(OUT / "design.parquet", index=False)
    print("MIMIC-IV complete-window design", complete.shape, flush=True)
    return design


def build_sequences(hourly: pd.DataFrame, design: pd.DataFrame) -> None:
    channels = ["hr", "rr", "spo2", "sbp", "dbp", "mbp", "tempf", "gcs"]
    ids = design["stay_id"].astype(int).to_numpy()
    index = {value: i for i, value in enumerate(ids)}
    x = np.full((len(design), 72, len(channels)), np.nan, dtype=np.float32)
    for row in hourly.itertuples(index=False):
        i = index.get(int(row.stay_id))
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
        stay_id=ids,
        subject_id=design["subject_id"].astype(np.int64).to_numpy(),
    )

    current = pd.read_parquet(OUT / "design.parquet")
    current_ids = current["stay_id"].astype(int).to_numpy()
    lookup = {int(sid): i for i, sid in enumerate(ids)}
    positions = np.asarray([lookup[int(sid)] for sid in current_ids], dtype=np.int64)
    np.savez_compressed(
        OUT / "seq72.npz",
        X=x[positions],
        M=mask[positions],
        y=current["y"].to_numpy(dtype=np.int64),
        y3=current["y"].map({0: 0, 1: 1, 2: 1, 3: 2}).to_numpy(dtype=np.int64),
        stay_id=current_ids,
        subject_id=current["subject_id"].astype(np.int64).to_numpy(),
    )


def main() -> None:
    con = duckdb.connect()
    con.execute("PRAGMA threads=8")
    con.execute("PRAGMA memory_limit='12GB'")
    start = time.time()
    hourly = build_hourly(con)
    features = build_features(hourly)
    design = build_design(features)
    build_sequences(hourly, design)
    print(
        "MIMIC landmark artifacts:",
        {"design_all": design.shape, "hourly": hourly.shape},
        f"{time.time() - start:.0f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
