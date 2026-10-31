import numpy as np, pandas as pd, json
import os
from pathlib import Path
ROOT = Path(os.environ.get("SEPSIS_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
out = ROOT / "results" / "mimic4"
d = np.load(out / "oof_4class.npz")
p24,p48,p72,y=d["p24"],d["p48"],d["p72"],d["y"]
a24,a48,a72=p24.argmax(1),p48.argmax(1),p72.argmax(1)
conf=p72.max(1)
for thr in (0.0,0.5,0.6,0.7):
    m=conf>=thr
    print(f"maxprob>={thr}: n={m.sum()} flip24_72={float((a24[m]!=a72[m]).mean()):.3f} agree48_72={float((a48[m]==a72[m]).mean()):.3f}")
dist=np.abs(a24-a72)
print("ordinal shift 24->72:", {int(k): round(float((dist==k).mean()),3) for k in range(4)})
print("shift among flippers:", {int(k): round(float((dist[dist>0]==k).mean()),3) for k in range(1,4)})
print("shift direction (24<72 worse):", round(float((a72>a24).mean()),3), " 24>72 better:", round(float((a72<a24).mean()),3))
ct=pd.crosstab(a24,a72,normalize=True)
print("confusion 24->72 (row=24, col=72):"); print((ct*100).round(1).to_string())
# per true class flip
for c in range(4):
    m=y==c
    print(f"true class {c}: n={m.sum()} flip24_72={float((a24[m]!=a72[m]).mean()):.3f} flip24_48={float((a24[m]!=a48[m]).mean()):.3f}")
# entropy
def ent(p):
    q=np.clip(p,1e-9,1); return float((-q*np.log(q)).sum(1).mean())
print("entropy 24/48/72:", round(ent(p24),3), round(ent(p48),3), round(ent(p72),3))
