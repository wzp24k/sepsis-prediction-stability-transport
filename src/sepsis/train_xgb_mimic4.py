import pandas as pd, numpy as np, time, json
import os
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, accuracy_score
from xgboost import XGBClassifier

ROOT = Path(os.environ.get("SEPSIS_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
DATA = ROOT / "data" / "processed" / "mimic4"
RES = ROOT / "results" / "mimic4"
RES.mkdir(parents=True, exist_ok=True)
D = pd.read_parquet(DATA / "design.parquet")
y=D["y"].astype(int).values
groups=D["subject_id"].values
static=[c for c in D.columns if c in ("age","male") or c.startswith("adm_") or c.startswith("unit_") or c in
        ("crea_min","crea_max","lac_min","lac_max","wbc_min","wbc_max","plt_min","plt_max","bili_min","bili_max","na_min","na_max","bun_min","bun_max","hgb_min","hgb_max","inr_max")]
w1=[c for c in D.columns if c.endswith("_w1_mean") or c.endswith("_w1_min") or c.endswith("_w1_max") or c.endswith("_w1_last") or c.endswith("_w1_count")]
w2=[c for c in D.columns if "_w2_" in c]
w3=[c for c in D.columns if "_w3_" in c]
print("feature blocks", len(static), len(w1), len(w2), len(w3), flush=True)

params=dict(objective="multi:softprob", num_class=4, n_estimators=300, max_depth=4, learning_rate=0.06,
            subsample=0.85, colsample_bytree=0.8, tree_method="hist", n_jobs=8, random_state=42,
            eval_metric="mlogloss")

gkf=list(GroupKFold(n_splits=5).split(D, y, groups))
res={}; probs={}
t0=time.time()
for tag, cols in (("24", static+w1), ("48", static+w1+w2), ("72", static+w1+w2+w3)):
    X=D[cols].astype(float)
    P=np.zeros((len(D),4))
    for k,(tr,te) in enumerate(gkf):
        m=XGBClassifier(**params)
        m.fit(X.iloc[tr], y[tr])
        P[te]=m.predict_proba(X.iloc[te])
    probs[tag]=P
    auc=roc_auc_score(y, P, multi_class="ovr", average="macro")
    acc=accuracy_score(y, P.argmax(1))
    res[tag]={"macro_auc":round(float(auc),4),"acc":round(float(acc),4)}
    print(tag, res[tag], f"{time.time()-t0:.0f}s", flush=True)
np.savez(RES / "oof_4class.npz", p24=probs["24"], p48=probs["48"], p72=probs["72"], y=y)

P={k:probs[k] for k in probs}
pred={k:P[k].argmax(1) for k in P}
def flip(a,b): return float((pred[a]!=pred[b]).mean())
tv=lambda a,b: float((0.5*np.abs(P[a]-P[b]).sum(1)).mean())
agree72={k: float((pred[k]==pred["72"]).mean()) for k in ("24","48")}
settle=np.where(pred["24"]==pred["72"], 24, np.where(pred["48"]==pred["72"], 48, 72))
res_stats={
 "n": int(len(y)),
 "flip_24_48": round(flip("24","48"),4),
 "flip_48_72": round(flip("48","72"),4),
 "flip_24_72": round(flip("24","72"),4),
 "tv_24_72": round(tv("24","72"),4),
 "tv_24_48": round(tv("24","48"),4),
 "tv_48_72": round(tv("48","72"),4),
 "agree24_72": round(agree72["24"],4),
 "agree48_72": round(agree72["48"],4),
 "settle24": round(float((settle==24).mean()),4),
 "settle48": round(float((settle==48).mean()),4),
 "settle72": round(float((settle==72).mean()),4),
 "mean_maxprob_24": round(float(P["24"].max(1).mean()),4),
 "mean_maxprob_72": round(float(P["72"].max(1).mean()),4),
 "pred_dist_24": {int(k): int(v) for k,v in zip(*np.unique(pred["24"],return_counts=True))},
 "pred_dist_72": {int(k): int(v) for k,v in zip(*np.unique(pred["72"],return_counts=True))},
 "true_dist": {int(k): int(v) for k,v in zip(*np.unique(y,return_counts=True))},
}
print(json.dumps({"auc":res,"stability":res_stats}, indent=1, ensure_ascii=False))
json.dump({"auc":res,"stability":res_stats}, open(RES / "xgb_4class.json","w"), indent=1)
