import duckdb, pandas as pd, numpy as np, time, json
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, accuracy_score
from xgboost import XGBClassifier
import os
from pathlib import Path
ROOT = Path(os.environ.get("SEPSIS_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
SOURCE_ROOT = Path(os.environ.get("SEPSIS_SOURCE_ROOT", r"D:\TT paper\0811Temporal Pathway"))
E = str(SOURCE_ROOT / "eicu")
OUT = str(ROOT / "data" / "processed" / "eicu")
RES = ROOT / "results" / "eicu"
RES.mkdir(parents=True, exist_ok=True)
con=duckdb.connect(); con.execute("PRAGMA threads=8"); con.execute("PRAGMA memory_limit='10GB'")
t0=time.time()
C=f"read_parquet('{OUT}/cohort_v2.parquet')"

con.execute(f"""CREATE TEMP TABLE VP AS
SELECT patientunitstayid, CAST(floor(observationoffset/60.0) AS INT) hh,
  CASE WHEN heartrate BETWEEN 20 AND 250 THEN heartrate END hr,
  CASE WHEN respiration BETWEEN 4 AND 60 THEN respiration END rr,
  CASE WHEN sao2 BETWEEN 50 AND 100 THEN sao2 END spo2,
  CASE WHEN systemicsystolic BETWEEN 40 AND 300 THEN systemicsystolic END sbp_i,
  CASE WHEN systemicdiastolic BETWEEN 20 AND 200 THEN systemicdiastolic END dbp_i,
  CASE WHEN systemicmean BETWEEN 30 AND 200 THEN systemicmean END mbp_i
FROM read_csv_auto('{E}/vitalPeriodic.csv.gz')
WHERE patientunitstayid IN (SELECT patientunitstayid FROM {C})
  AND observationoffset>=0 AND observationoffset<4320""")
print("periodic", con.execute("SELECT COUNT(*) FROM VP").fetchone()[0], f"{time.time()-t0:.0f}s", flush=True)

con.execute(f"""CREATE TEMP TABLE VA AS
SELECT patientunitstayid, CAST(floor(observationoffset/60.0) AS INT) hh,
  CASE WHEN noninvasivesystolic BETWEEN 40 AND 300 THEN noninvasivesystolic END sbp_n,
  CASE WHEN noninvasivediastolic BETWEEN 20 AND 200 THEN noninvasivediastolic END dbp_n,
  CASE WHEN noninvasivemean BETWEEN 30 AND 200 THEN noninvasivemean END mbp_n
FROM read_csv_auto('{E}/vitalAperiodic.csv.gz')
WHERE patientunitstayid IN (SELECT patientunitstayid FROM {C})
  AND observationoffset>=0 AND observationoffset<4320""")
print("aperiodic", con.execute("SELECT COUNT(*) FROM VA").fetchone()[0], f"{time.time()-t0:.0f}s", flush=True)

con.execute(f"""CREATE TEMP TABLE NC AS
SELECT patientunitstayid, CAST(floor(nursingchartoffset/60.0) AS INT) hh,
  CASE WHEN nursingchartcelltypevalname='Temperature (C)' THEN
            CASE WHEN TRY_CAST(nursingchartvalue AS DOUBLE) BETWEEN 25 AND 45
                 THEN TRY_CAST(nursingchartvalue AS DOUBLE)*9.0/5.0+32.0 END
       ELSE NULL END tempf,
  CASE WHEN nursingchartcelltypevalname='GCS Total' THEN
            CASE WHEN TRY_CAST(nursingchartvalue AS DOUBLE) BETWEEN 3 AND 15
                 THEN TRY_CAST(nursingchartvalue AS DOUBLE) END
       ELSE NULL END gcs
FROM read_csv_auto('{E}/nurseCharting.csv.gz')
WHERE patientunitstayid IN (SELECT patientunitstayid FROM {C})
  AND nursingchartoffset>=0 AND nursingchartoffset<4320
  AND nursingchartcelltypevalname IN ('Temperature (C)','GCS Total')""")
print("nurse", con.execute("SELECT COUNT(*) FROM NC").fetchone()[0], f"{time.time()-t0:.0f}s", flush=True)

con.execute("""CREATE TEMP TABLE H AS
SELECT patientunitstayid, hh,
  median(hr) hr, median(rr) rr, median(spo2) spo2,
  coalesce(median(sbp_i), median(sbp_n)) sbp,
  coalesce(median(dbp_i), median(dbp_n)) dbp,
  coalesce(median(mbp_i), median(mbp_n)) mbp,
  median(tempf) tempf, median(gcs) gcs
FROM (
  SELECT patientunitstayid,hh,hr,rr,spo2,sbp_i,dbp_i,mbp_i,
         CAST(NULL AS DOUBLE) sbp_n, CAST(NULL AS DOUBLE) dbp_n, CAST(NULL AS DOUBLE) mbp_n,
         CAST(NULL AS DOUBLE) tempf, CAST(NULL AS DOUBLE) gcs FROM VP
  UNION ALL
  SELECT patientunitstayid,hh,CAST(NULL AS DOUBLE),CAST(NULL AS DOUBLE),CAST(NULL AS DOUBLE),
         CAST(NULL AS DOUBLE),CAST(NULL AS DOUBLE),CAST(NULL AS DOUBLE),
         sbp_n,dbp_n,mbp_n, CAST(NULL AS DOUBLE), CAST(NULL AS DOUBLE) FROM VA
  UNION ALL
  SELECT patientunitstayid,hh,CAST(NULL AS DOUBLE),CAST(NULL AS DOUBLE),CAST(NULL AS DOUBLE),
         CAST(NULL AS DOUBLE),CAST(NULL AS DOUBLE),CAST(NULL AS DOUBLE),
         CAST(NULL AS DOUBLE),CAST(NULL AS DOUBLE),CAST(NULL AS DOUBLE), tempf, gcs FROM NC
) GROUP BY 1,2""")
n=con.execute("SELECT COUNT(*) FROM H").fetchone()[0]
print("hourly", n, f"{time.time()-t0:.0f}s", flush=True)

h=con.execute("SELECT * FROM H").fetchdf()
chans=["hr","rr","spo2","sbp","dbp","mbp","tempf","gcs"]
print("hourly coverage per channel (%):", (h[chans].notna().mean()*100).round(1).to_dict(), flush=True)
h["w"]=(h["hh"]//24).astype(int); h=h.sort_values(["patientunitstayid","hh"])
feats=[]
for w in (0,1,2):
    g=h[h["w"]==w].groupby("patientunitstayid")[chans].agg(["mean","min","max","last","count"])
    g.columns=[f"{a}_w{w+1}_{b}" for a,b in g.columns]; feats.append(g)
Xv=pd.concat(feats,axis=1).reset_index()
Xv.to_parquet(f"{OUT}/vitals_v2.parquet",index=False)
print("vitals feats", Xv.shape, "missing%", round(float(Xv.drop(columns='patientunitstayid').isna().mean().mean()*100),1), flush=True)

coh=con.execute(f"SELECT * FROM {C} WHERE y IS NOT NULL").fetchdf()
L=pd.read_parquet(f"{OUT}/labs_v2.parquet")
D=coh.merge(Xv,on="patientunitstayid",how="left").merge(L,on="patientunitstayid",how="left")
D["male"]=D["gender"].isin(["M","Male"]).astype(int)
D=pd.get_dummies(D,columns=["unitadmitsource","unittype"],prefix=["src","ut"],dummy_na=False)
D.to_parquet(f"{OUT}/design_v3.parquet",index=False)
print("design", D.shape, f"{time.time()-t0:.0f}s", flush=True)

static=[c for c in D.columns if c in ("age","male") or c.startswith(("src_","ut_")) or (c.endswith(("_min","_max")) and "_w" not in c)]
w1=[c for c in D.columns if "_w1_" in c]; w2=[c for c in D.columns if "_w2_" in c]; w3=[c for c in D.columns if "_w3_" in c]
print("blocks", len(static),len(w1),len(w2),len(w3), flush=True)
p4=dict(objective="multi:softprob", num_class=4, n_estimators=300, max_depth=4, learning_rate=0.06,
        subsample=0.85, colsample_bytree=0.8, tree_method="hist", n_jobs=8, random_state=42, eval_metric="mlogloss")
y=D["y"].astype(int).values
gkf=list(GroupKFold(n_splits=5).split(D,y,D["uniquepid"].values))
P={}; res={}
for tag,cols in (("24",static+w1),("48",static+w1+w2),("72",static+w1+w2+w3)):
    X=D[cols].astype(float); Pr=np.zeros((len(D),4))
    for tr,te in gkf:
        m=XGBClassifier(**p4); m.fit(X.iloc[tr],y[tr]); Pr[te]=m.predict_proba(X.iloc[te])
    P[tag]=Pr
    res[tag]={"macro_auc":round(float(roc_auc_score(y,Pr,multi_class="ovr",average="macro")),4),
              "acc":round(float(accuracy_score(y,Pr.argmax(1))),4),
              "mean_maxprob":round(float(Pr.max(1).mean()),4)}
    print(tag,res[tag], flush=True)
a={k:P[k].argmax(1) for k in P}
M=np.zeros((4,4),int)
for i,j in zip(a["24"],a["72"]): M[i,j]+=1
st={"n":int(len(y))}
for x,z in (("24","48"),("48","72"),("24","72")):
    st[f"flip_{x}_{z}"]=round(float((a[x]!=a[z]).mean()),4)
    st[f"tv_{x}_{z}"]=round(float((0.5*np.abs(P[x]-P[z]).sum(1)).mean()),4)
st["revert"]=round(float(((a["24"]==a["72"])&(a["24"]!=a["48"])).mean()),4)
st["worse"]=round(float((a["72"]>a["24"]).mean()),4); st["better"]=round(float((a["72"]<a["24"]).mean()),4)
st["dist24"]=[int((a["24"]==k).sum()) for k in range(4)]; st["dist72"]=[int((a["72"]==k).sum()) for k in range(4)]
st["trans_24_72"]=M.tolist()
st["fac_death_share"]=round(float((M[2,3]+M[3,2])/max(int((a["24"]!=a["72"]).sum()),1)),4)
q=P["24"].max(1)
for thr in (0.5,0.6,0.7):
    m=q>=thr
    st[f"by24_{thr}"]={"cov":round(float(m.mean()),4),"n":int(m.sum()),
       "flip":round(float((a["24"][m]!=a["72"][m]).mean()),4),
       "acc24":round(float(accuracy_score(y[m],a["24"][m])),4)}
print(json.dumps({"auc":res,"stability":st},indent=1,ensure_ascii=False))
json.dump({"auc":res,"stability":st},open(RES / "xgb_4class.json","w"),indent=1)
np.savez(RES / "oof_4class.npz",p24=P["24"],p48=P["48"],p72=P["72"],y=y)
print(f"done {time.time()-t0:.0f}s")
