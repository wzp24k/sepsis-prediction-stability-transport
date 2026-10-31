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
CREATE TEMP TABLE hourly AS
SELECT stay_id, CAST(floor(date_diff('second', intime, charttime)/3600.0) AS INT) h,
       median(v) FILTER (WHERE ch='hr') hr, median(v) FILTER (WHERE ch='rr') rr,
       median(v) FILTER (WHERE ch='spo2') spo2, median(v) FILTER (WHERE ch='sbp') sbp,
       median(v) FILTER (WHERE ch='dbp') dbp, median(v) FILTER (WHERE ch='mbp') mbp,
       median(v) FILTER (WHERE ch='temp') AS tempf, median(v) FILTER (WHERE ch='gcs') gcs
FROM read_parquet('{out}/vitals_raw.parquet') v JOIN read_parquet('{out}/cohort.parquet') c USING(stay_id)
WHERE date_diff('second', intime, charttime) >= 0 AND date_diff('second', intime, charttime) < 72*3600
GROUP BY 1,2
""")
print("hourly", con.execute("SELECT COUNT(*) FROM hourly").fetchone()[0], f"{time.time()-t0:.0f}s", flush=True)
h=con.execute("SELECT * FROM hourly").fetchdf()
print("pandas rows", len(h), flush=True)

chans=["hr","rr","spo2","sbp","dbp","mbp","tempf","gcs"]
h["w"]=(h["h"]//24).astype(int)
h=h.sort_values(["stay_id","h"])
feats=[]
for w in (0,1,2):
    sub=h[h["w"]==w]
    g=sub.groupby("stay_id")[chans].agg(["mean","min","max","last","count"])
    g.columns=[f"{a}_w{w+1}_{b}" for a,b in g.columns]
    feats.append(g)
X=pd.concat(feats,axis=1).reset_index()
print("feature table", X.shape, f"{time.time()-t0:.0f}s", flush=True)
X.to_parquet(f"{out}/vitals_feats.parquet", index=False)

# labs first 24h
con.execute(f"""
COPY (
WITH l AS (
  SELECT c.stay_id, l.itemid, l.valuenum, l.charttime
  FROM read_csv_auto('{iv}/hosp/labevents.csv.gz') l
  JOIN read_parquet('{out}/cohort.parquet') c USING(hadm_id)
  WHERE l.itemid IN (50912,50813,51301,51265,50885,50983,51006,51222,51237)
    AND l.valuenum IS NOT NULL
    AND l.charttime >= c.intime AND l.charttime < c.intime + INTERVAL 24 HOUR
)
SELECT stay_id,
  min(valuenum) FILTER (WHERE itemid=50912) crea_min, max(valuenum) FILTER (WHERE itemid=50912) crea_max,
  min(valuenum) FILTER (WHERE itemid=50813) lac_min,  max(valuenum) FILTER (WHERE itemid=50813) lac_max,
  min(valuenum) FILTER (WHERE itemid=51301) wbc_min,  max(valuenum) FILTER (WHERE itemid=51301) wbc_max,
  min(valuenum) FILTER (WHERE itemid=51265) plt_min,  max(valuenum) FILTER (WHERE itemid=51265) plt_max,
  min(valuenum) FILTER (WHERE itemid=50885) bili_min, max(valuenum) FILTER (WHERE itemid=50885) bili_max,
  min(valuenum) FILTER (WHERE itemid=50983) na_min,   max(valuenum) FILTER (WHERE itemid=50983) na_max,
  min(valuenum) FILTER (WHERE itemid=51006) bun_min,  max(valuenum) FILTER (WHERE itemid=51006) bun_max,
  min(valuenum) FILTER (WHERE itemid=51222) hgb_min,  max(valuenum) FILTER (WHERE itemid=51222) hgb_max,
  min(valuenum) FILTER (WHERE itemid=51237) inr_max
FROM l GROUP BY 1
) TO '{out}/labs.parquet' (FORMAT PARQUET)
""")
print("labs", con.execute(f"SELECT COUNT(*) FROM read_parquet('{out}/labs.parquet')").fetchone()[0], f"{time.time()-t0:.0f}s", flush=True)

coh=con.execute(f"SELECT stay_id, subject_id, age, gender, admission_type, first_careunit, y FROM read_parquet('{out}/cohort.parquet') WHERE los>=72 AND y IS NOT NULL").fetchdf()
labs=pd.read_parquet(f"{out}/labs.parquet")
D=coh.merge(X,on="stay_id",how="left").merge(labs,on="stay_id",how="left")
D["male"]=(D["gender"]=="M").astype(int)
D=pd.get_dummies(D,columns=["admission_type","first_careunit"],prefix=["adm","unit"],dummy_na=False)
D.to_parquet(f"{out}/design.parquet",index=False)
print("design", D.shape, "labels", D.y.value_counts().sort_index().to_dict(), f"{time.time()-t0:.0f}s", flush=True)
