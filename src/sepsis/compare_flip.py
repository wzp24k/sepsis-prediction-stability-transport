import numpy as np, json, duckdb
import os
from pathlib import Path
ROOT = Path(os.environ.get("SEPSIS_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
RES = ROOT / "results"
RES.mkdir(parents=True, exist_ok=True)
rng=np.random.default_rng(20261029)
def ci(d,B=4000):
    d=np.asarray(d,float); n=len(d); x=np.empty(B)
    for b in range(B):
        x[b]=d[rng.integers(0,n,n)].mean()
    return round(float(np.percentile(x,2.5)),4), round(float(np.percentile(x,97.5)),4)

out={}
for tag,path in (("MIMIC-IV", RES / "mimic4" / "oof_4class.npz"),
                 ("eICU", RES / "eicu" / "oof_4class.npz")):
    d=np.load(path); p24,p48,p72,y=d["p24"],d["p48"],d["p72"],d["y"]
    a24,a48,a72=p24.argmax(1),p48.argmax(1),p72.argmax(1)
    r={"n":int(len(y))}
    r["flip_24_48"]=[round(float((a24!=a48).mean()),4), ci(a24!=a48)]
    r["flip_48_72"]=[round(float((a48!=a72).mean()),4), ci(a48!=a72)]
    r["flip_24_72"]=[round(float((a24!=a72).mean()),4), ci(a24!=a72)]
    r["tv_24_72"]=round(float((0.5*np.abs(p24-p72).sum(1)).mean()),4)
    r["revert"]=round(float(((a24==a72)&(a24!=a48)).mean()),4)
    r["worse"]=round(float((a72>a24).mean()),4); r["better"]=round(float((a72<a24).mean()),4)
    M=np.zeros((4,4),int)
    for i,j in zip(a24,a72): M[i,j]+=1
    r["trans"]=M.tolist()
    tot=int((a24!=a72).sum())
    r["fac_death_share"]=round(float((M[2,3]+M[3,2])/tot),4)
    r["adjacent_share"]=round(float((np.abs(a24-a72)==1).sum()/tot),4)
    q=p24.max(1)
    r["strata"]={}
    for thr in (0.5,0.6,0.7):
        m=q>=thr
        r["strata"][thr]={"cov":round(float(m.mean()),4),"flip":[round(float((a24[m]!=a72[m]).mean()),4), ci(a24[m]!=a72[m])]}
    r["true"]=[int((y==k).sum()) for k in range(4)]
    r["pred24"]=[int((a24==k).sum()) for k in range(4)]
    r["pred72"]=[int((a72==k).sum()) for k in range(4)]
    out[tag]=r

json.dump(out, open(RES / "comparison" / "flip_compare.json", "w"), indent=1, ensure_ascii=False)
L=[]
L.append("| 指标 | MIMIC-IV | eICU |")
L.append("|---|---:|---:|")
L.append(f"| 例数（72 h 停留、结局可映射） | {out['MIMIC-IV']['n']:,} | {out['eICU']['n']:,} |")
L.append(f"| 24→48 h 翻转率 | {out['MIMIC-IV']['flip_24_48'][0]*100:.1f}% ({out['MIMIC-IV']['flip_24_48'][1][0]*100:.1f}–{out['MIMIC-IV']['flip_24_48'][1][1]*100:.1f}) | {out['eICU']['flip_24_48'][0]*100:.1f}% ({out['eICU']['flip_24_48'][1][0]*100:.1f}–{out['eICU']['flip_24_48'][1][1]*100:.1f}) |")
L.append(f"| 48→72 h 翻转率 | {out['MIMIC-IV']['flip_48_72'][0]*100:.1f}% | {out['eICU']['flip_48_72'][0]*100:.1f}% |")
L.append(f"| 24→72 h 翻转率 | {out['MIMIC-IV']['flip_24_72'][0]*100:.1f}% ({out['MIMIC-IV']['flip_24_72'][1][0]*100:.1f}–{out['MIMIC-IV']['flip_24_72'][1][1]*100:.1f}) | {out['eICU']['flip_24_72'][0]*100:.1f}% ({out['eICU']['flip_24_72'][1][0]*100:.1f}–{out['eICU']['flip_24_72'][1][1]*100:.1f}) |")
L.append(f"| 概率总变差 | {out['MIMIC-IV']['tv_24_72']:.3f} | {out['eICU']['tv_24_72']:.3f} |")
L.append(f"| 往返回退（24=72≠48） | {out['MIMIC-IV']['revert']*100:.1f}% | {out['eICU']['revert']*100:.1f}% |")
L.append(f"| 机构↔死亡占全部翻转 | {out['MIMIC-IV']['fac_death_share']*100:.1f}% | {out['eICU']['fac_death_share']*100:.1f}% |")
L.append(f"| 相邻一级占全部翻转 | {out['MIMIC-IV']['adjacent_share']*100:.1f}% | {out['eICU']['adjacent_share']*100:.1f}% |")
for thr in (0.5,0.6,0.7):
    a=out['MIMIC-IV']['strata'][thr]; b=out['eICU']['strata'][thr]
    L.append(f"| 24 h 置信度 ≥{thr}：覆盖 / 翻转率 | {a['cov']*100:.1f}% / {a['flip'][0]*100:.1f}% | {b['cov']*100:.1f}% / {b['flip'][0]*100:.1f}% |")
open(RES / "comparison" / "flip_compare.md", "w", encoding="utf-8").write("\n".join(L))
print("\n".join(L))

