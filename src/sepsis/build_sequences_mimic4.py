import duckdb, pandas as pd, numpy as np, time
import os
from pathlib import Path
ROOT = Path(os.environ.get("SEPSIS_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
OUT = str(ROOT / "data" / "processed" / "mimic4")
os.makedirs(OUT, exist_ok=True)
con=duckdb.connect(); con.execute("PRAGMA threads=8"); con.execute("PRAGMA memory_limit='10GB'")
t0=time.time()

con.execute(f"""CREATE TEMP TABLE H AS
SELECT v.stay_id, CAST(floor(date_diff('second', c.intime, v.charttime)/3600.0) AS INT) hh,
       median(CASE WHEN v.ch='hr' THEN v.v END) hr,
       median(CASE WHEN v.ch='rr' THEN v.v END) rr,
       median(CASE WHEN v.ch='spo2' THEN v.v END) spo2,
       median(CASE WHEN v.ch='sbp' THEN v.v END) sbp,
       median(CASE WHEN v.ch='dbp' THEN v.v END) dbp,
       median(CASE WHEN v.ch='mbp' THEN v.v END) mbp,
       median(CASE WHEN v.ch='temp' THEN v.v END) tempf,
       median(CASE WHEN v.ch='gcs' THEN v.v END) gcs
FROM read_parquet('{OUT}/vitals_raw.parquet') v
JOIN read_parquet('{OUT}/cohort.parquet') c USING(stay_id)
WHERE date_diff('second', c.intime, v.charttime)>=0 AND date_diff('second', c.intime, v.charttime)<72*3600
GROUP BY 1,2""")
H=con.execute("SELECT * FROM H").fetchdf()
print("hourly rows", len(H), f"{time.time()-t0:.0f}s", flush=True)

D=pd.read_parquet(f"{OUT}/design.parquet")
stay=list(D["stay_id"])
hidx={s:i for i,s in enumerate(stay)}
chans=["hr","rr","spo2","sbp","dbp","mbp","tempf","gcs"]
X=np.full((len(stay),72,8), np.nan, dtype=np.float32)
for row in H.itertuples(index=False):
    i=hidx.get(row.stay_id)
    if i is None: continue
    if 0<=row.hh<72:
        X[i,row.hh,0]=row.hr; X[i,row.hh,1]=row.rr; X[i,row.hh,2]=row.spo2
        X[i,row.hh,3]=row.sbp; X[i,row.hh,4]=row.dbp; X[i,row.hh,5]=row.mbp
        X[i,row.hh,6]=row.tempf; X[i,row.hh,7]=row.gcs
M=(~np.isnan(X)).any(2).astype(np.float32)
print("X", X.shape, "feature missing %", np.round(np.isnan(X).mean((0,1))*100,1), flush=True)
print("timesteps with any obs: mean", M.sum(1).mean().round(1), "min", int(M.sum(1).min()), flush=True)
print("stays with 0 obs:", int((M.sum(1)==0).sum()), flush=True)
np.savez_compressed(f"{OUT}/seq72.npz", X=X, M=M, y=D["y"].values.astype(np.int64),
                    subject_id=D["subject_id"].values, stay_id=D["stay_id"].values)
print("saved", f"{time.time()-t0:.0f}s")

