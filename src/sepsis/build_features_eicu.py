import duckdb, pandas as pd, numpy as np, time, json
import os
from pathlib import Path
ROOT = Path(os.environ.get("SEPSIS_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
SOURCE_ROOT = Path(os.environ.get("SEPSIS_SOURCE_ROOT", r"D:\TT paper\0811Temporal Pathway"))
E = str(SOURCE_ROOT / "eicu")
OUT = str(ROOT / "data" / "processed" / "eicu")
os.makedirs(OUT, exist_ok=True)
con=duckdb.connect(); con.execute("PRAGMA threads=8"); con.execute("PRAGMA memory_limit='10GB'")
t0=time.time()

con.execute(f"""CREATE TEMP TABLE V AS
SELECT patientunitstayid, observationoffset/60.0 AS h,
       CASE WHEN heartrate BETWEEN 20 AND 250 THEN heartrate END hr,
       CASE WHEN respiration BETWEEN 4 AND 60 THEN respiration END rr,
       CASE WHEN sao2 BETWEEN 50 AND 100 THEN sao2 END spo2,
       CASE WHEN systemicsystolic BETWEEN 40 AND 300 THEN systemicsystolic END sbp,
       CASE WHEN systemicdiastolic BETWEEN 20 AND 200 THEN systemicdiastolic END dbp,
       CASE WHEN systemicmean BETWEEN 30 AND 200 THEN systemicmean END mbp,
       CASE WHEN temperature IS NULL THEN NULL WHEN temperature<60 THEN temperature*9.0/5.0+32.0 ELSE temperature END tempf
FROM read_csv_auto('{E}/vitalPeriodic.csv.gz')
WHERE patientunitstayid IN (SELECT patientunitstayid FROM read_parquet('{OUT}/cohort_v2.parquet'))
  AND observationoffset>=0 AND observationoffset<4320""")
con.execute("""CREATE TEMP TABLE hourly AS
SELECT patientunitstayid, CAST(floor(h) AS INT) hh,
       median(hr) hr, median(rr) rr, median(spo2) spo2, median(sbp) sbp,
       median(dbp) dbp, median(mbp) mbp,
       median(CASE WHEN tempf BETWEEN 92 AND 110 THEN tempf END) tempf
FROM V GROUP BY 1,2""")
h=con.execute("SELECT * FROM hourly").fetchdf()
chans=["hr","rr","spo2","sbp","dbp","mbp","tempf"]
h["w"]=(h["hh"]//24).astype(int); h=h.sort_values(["patientunitstayid","hh"])
feats=[]
for w in (0,1,2):
    g=h[h["w"]==w].groupby("patientunitstayid")[chans].agg(["mean","min","max","last","count"])
    g.columns=[f"{a}_w{w+1}_{b}" for a,b in g.columns]; feats.append(g)
Xv=pd.concat(feats,axis=1).reset_index()
Xv.to_parquet(f"{OUT}/vitals_v2.parquet",index=False)
print("vitals", Xv.shape, f"{time.time()-t0:.0f}s", flush=True)

con.execute(f"""COPY (
WITH l AS (SELECT c.patientunitstayid, l.labname, l.labresult
  FROM read_csv_auto('{E}/lab.csv.gz') l JOIN read_parquet('{OUT}/cohort_v2.parquet') c USING(patientunitstayid)
  WHERE l.labname IN ('creatinine','lactate','WBC x 1000','platelets x 1000','total bilirubin','sodium','BUN','Hgb','PT - INR')
    AND l.labresult IS NOT NULL AND l.labresultoffset>=0 AND l.labresultoffset<1440)
SELECT patientunitstayid,
  min(labresult) FILTER (WHERE labname='creatinine') crea_min, max(labresult) FILTER (WHERE labname='creatinine') crea_max,
  min(labresult) FILTER (WHERE labname='lactate') lac_min, max(labresult) FILTER (WHERE labname='lactate') lac_max,
  min(labresult) FILTER (WHERE labname='WBC x 1000') wbc_min, max(labresult) FILTER (WHERE labname='WBC x 1000') wbc_max,
  min(labresult) FILTER (WHERE labname='platelets x 1000') plt_min, max(labresult) FILTER (WHERE labname='platelets x 1000') plt_max,
  min(labresult) FILTER (WHERE labname='total bilirubin') bili_min, max(labresult) FILTER (WHERE labname='total bilirubin') bili_max,
  min(labresult) FILTER (WHERE labname='sodium') na_min, max(labresult) FILTER (WHERE labname='sodium') na_max,
  min(labresult) FILTER (WHERE labname='BUN') bun_min, max(labresult) FILTER (WHERE labname='BUN') bun_max,
  min(labresult) FILTER (WHERE labname='Hgb') hgb_min, max(labresult) FILTER (WHERE labname='Hgb') hgb_max,
  max(labresult) FILTER (WHERE labname='PT - INR') inr_max
FROM l GROUP BY 1) TO '{OUT}/labs_v2.parquet' (FORMAT PARQUET)""")
print("saved feature tables", f"{time.time()-t0:.0f}s", flush=True)
