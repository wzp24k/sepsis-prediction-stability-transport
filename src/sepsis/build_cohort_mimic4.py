import duckdb, pandas as pd, numpy as np, time, os
from pathlib import Path
ROOT = Path(os.environ.get("SEPSIS_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
SOURCE_ROOT = Path(os.environ.get("SEPSIS_SOURCE_ROOT", r"D:\TT paper\0811Temporal Pathway"))
iv = str(SOURCE_ROOT / "mimic-iv-3.1")
out = str(ROOT / "data" / "processed" / "mimic4")
os.makedirs(out, exist_ok=True)
con=duckdb.connect(); con.execute("PRAGMA threads=8"); con.execute("PRAGMA memory_limit='10GB'")
t0=time.time()

con.execute(f"""
CREATE TEMP TABLE sep AS
SELECT DISTINCT CAST(hadm_id AS BIGINT) hadm_id FROM read_csv_auto('{iv}/hosp/diagnoses_icd.csv.gz', all_varchar=true)
WHERE upper(trim(icd_code)) LIKE 'A40%' OR upper(trim(icd_code)) LIKE 'A41%' OR upper(trim(icd_code)) LIKE 'R652%'
   OR upper(trim(icd_code)) LIKE '038%' OR upper(trim(icd_code)) LIKE '9959%' OR upper(trim(icd_code)) LIKE '78552%'
""")
con.execute(f"""
COPY (
WITH i AS (
  SELECT i.stay_id, i.hadm_id, i.subject_id, i.intime, i.outtime, i.first_careunit,
         p.gender, p.anchor_age + (year(CAST(i.intime AS TIMESTAMP)) - p.anchor_year) AS age
  FROM read_csv_auto('{iv}/icu/icustays.csv.gz') i
  JOIN sep USING(hadm_id)
  JOIN read_csv_auto('{iv}/hosp/patients.csv.gz') p USING(subject_id)
  QUALIFY row_number() OVER (PARTITION BY i.hadm_id ORDER BY i.intime)=1
),
a AS (
  SELECT i.*, a.admission_type, a.discharge_location, a.hospital_expire_flag,
         date_diff('second', i.intime, i.outtime)/3600.0 AS los
  FROM i JOIN read_csv_auto('{iv}/hosp/admissions.csv.gz') a USING(hadm_id)
)
SELECT stay_id, hadm_id, subject_id, intime, outtime, first_careunit, gender, age,
       admission_type, los,
       CASE WHEN hospital_expire_flag=1 OR discharge_location LIKE 'DIED%' THEN 3
            WHEN discharge_location IN ('HOME','HOME HEALTH CARE','AGAINST ADVICE') THEN 0
            WHEN discharge_location IN ('REHAB','ASSISTED LIVING') THEN 1
            WHEN discharge_location IN ('SKILLED NURSING FACILITY','CHRONIC/LONG TERM ACUTE CARE','OTHER FACILITY','ACUTE HOSPITAL','PSYCH FACILITY','HEALTHCARE FACILITY','HOSPICE') THEN 2
            ELSE NULL END AS y
FROM a WHERE age>=18
) TO '{out}/cohort.parquet' (FORMAT PARQUET)
""")
print("cohort", con.execute(f"SELECT COUNT(*), SUM(CASE WHEN y IS NOT NULL THEN 1 ELSE 0 END), SUM(CASE WHEN los>=72 THEN 1 ELSE 0 END) FROM read_parquet('{out}/cohort.parquet')").fetchone(), f"{time.time()-t0:.0f}s", flush=True)

con.execute(f"""
CREATE TEMP TABLE c72 AS
SELECT stay_id, intime FROM read_parquet('{out}/cohort.parquet') WHERE los>=72 AND y IS NOT NULL
""")
print("c72", con.execute("SELECT COUNT(*) FROM c72").fetchone()[0], flush=True)

con.execute(f"""
COPY (
WITH vit AS (
  SELECT c.stay_id, c.charttime,
         CASE c.itemid
              WHEN 220045 THEN 'hr'
              WHEN 220210 THEN 'rr' WHEN 224690 THEN 'rr'
              WHEN 220277 THEN 'spo2'
              WHEN 220179 THEN 'sbp' WHEN 220050 THEN 'sbp'
              WHEN 220180 THEN 'dbp' WHEN 220051 THEN 'dbp'
              WHEN 220181 THEN 'mbp' WHEN 220052 THEN 'mbp'
              WHEN 223761 THEN 'temp' WHEN 223762 THEN 'temp'
         END AS ch,
         CASE WHEN c.itemid=223762 THEN c.valuenum*9.0/5.0+32.0 ELSE c.valuenum END AS v
  FROM read_csv_auto('{iv}/icu/chartevents.csv.gz') c
  JOIN c72 ON c72.stay_id=c.stay_id
  WHERE c.itemid IN (220045,220210,224690,220277,220179,220050,220180,220051,220181,220052,223761,223762)
    AND c.valuenum IS NOT NULL
),
gcs AS (
  SELECT c.stay_id, c.charttime,
         max(CASE WHEN c.itemid=220739 THEN c.valuenum END) eye,
         max(CASE WHEN c.itemid=223900 THEN c.valuenum END) verb,
         max(CASE WHEN c.itemid=223901 THEN c.valuenum END) mot
  FROM read_csv_auto('{iv}/icu/chartevents.csv.gz') c
  JOIN c72 ON c72.stay_id=c.stay_id
  WHERE c.itemid IN (220739,223900,223901) AND c.valuenum IS NOT NULL
  GROUP BY 1,2
)
SELECT stay_id, charttime, ch, v FROM vit
UNION ALL
SELECT stay_id, charttime, 'gcs' AS ch, eye+verb+mot AS v FROM gcs
WHERE eye IS NOT NULL AND verb IS NOT NULL AND mot IS NOT NULL
) TO '{out}/vitals_raw.parquet' (FORMAT PARQUET)
""")
print("vitals rows", con.execute(f"SELECT COUNT(*) FROM read_parquet('{out}/vitals_raw.parquet')").fetchone()[0], f"{time.time()-t0:.0f}s", flush=True)
