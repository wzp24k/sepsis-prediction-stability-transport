import pandas as pd, numpy as np, json
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from xgboost import XGBClassifier
import os
from pathlib import Path
ROOT = Path(os.environ.get("SEPSIS_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
DATA = ROOT / "data" / "processed" / "eicu"
RES = ROOT / "results" / "eicu"
D = pd.read_parquet(DATA / "design_v3.parquet")
d = np.load(RES / "oof_4class.npz")
y4=d["y"]; a72=d["p72"].argmax(1)
cm=confusion_matrix(y4,a72,labels=[0,1,2,3])
print("4-class 72h confusion (row=true):"); print(cm)
print("rehab recall 72h:", round(cm[1,1]/cm[1].sum(),3), " n true rehab:", int(cm[1].sum()))
D["y3"]=D["y"].map({0:0,1:1,2:1,3:2})
static=[c for c in D.columns if c in ("age","male") or c.startswith(("src_","ut_")) or (c.endswith(("_min","_max")) and "_w" not in c)]
w1=[c for c in D.columns if "_w1_" in c]; w2=[c for c in D.columns if "_w2_" in c]; w3=[c for c in D.columns if "_w3_" in c]
p3=dict(objective="multi:softprob", num_class=3, n_estimators=300, max_depth=4, learning_rate=0.06,
        subsample=0.85, colsample_bytree=0.8, tree_method="hist", n_jobs=8, random_state=42, eval_metric="mlogloss")
y=D["y3"].astype(int).values
gkf=list(GroupKFold(n_splits=5).split(D,y,D["uniquepid"].values))
P={}; res={}
for tag,cols in (("24",static+w1),("48",static+w1+w2),("72",static+w1+w2+w3)):
    X=D[cols].astype(float); Pr=np.zeros((len(D),3))
    for tr,te in gkf:
        m=XGBClassifier(**p3); m.fit(X.iloc[tr],y[tr]); Pr[te]=m.predict_proba(X.iloc[te])
    P[tag]=Pr
    res[tag]={"macro_auc":round(float(roc_auc_score(y,Pr,multi_class="ovr",average="macro")),4),
              "acc":round(float(accuracy_score(y,Pr.argmax(1))),4)}
    print(tag,res[tag], flush=True)
a={k:P[k].argmax(1) for k in P}
st={}
for x,z in (("24","48"),("48","72"),("24","72")):
    st[f"flip_{x}_{z}"]=round(float((a[x]!=a[z]).mean()),4)
st["tv_24_72"]=round(float((0.5*np.abs(P["24"]-P["72"]).sum(1)).mean()),4)
st["revert"]=round(float(((a["24"]==a["72"])&(a["24"]!=a["48"])).mean()),4)
q=P["24"].max(1)
for thr in (0.5,0.6):
    m=q>=thr
    st[f"by24_{thr}"]={"cov":round(float(m.mean()),4),"flip":round(float((a["24"][m]!=a["72"][m]).mean()),4)}
print(json.dumps({"auc":res,"stability":st},indent=1))
json.dump({"auc":res,"stability":st},open(RES / "xgb_3class.json","w"),indent=1)
