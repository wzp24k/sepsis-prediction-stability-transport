# -*- coding: utf-8 -*-
"""DG-GRU with multi-landmark ordinal readout for sepsis discharge disposition."""
import argparse, json, math, time, os
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score

DATA = Path(os.environ.get("SEPSIS_PROJECT_ROOT", Path(__file__).resolve().parents[2])) / "data" / "processed" / "mimic4"
RES = Path(os.environ.get("SEPSIS_PROJECT_ROOT", Path(__file__).resolve().parents[2])) / "results" / "mimic4"
RES.mkdir(parents=True, exist_ok=True)
LM = [23, 47, 71]           # landmark indices for 24 / 48 / 72 h

class DynamicFeatureGraph(nn.Module):
    def __init__(self, n_feat, d_edge=16):
        super().__init__()
        self.d_edge = d_edge
        self.e = nn.Parameter(torch.randn(n_feat, d_edge) * 0.1)
        self.Wq = nn.Linear(d_edge + 1, d_edge)
        self.Wk = nn.Linear(d_edge + 1, d_edge)
    def forward(self, x):
        B, T, Fd = x.shape
        e = self.e[None, None].expand(B, T, Fd, -1)
        v = x.unsqueeze(-1)
        q = self.Wq(torch.cat([e, v], -1)); k = self.Wk(torch.cat([e, v], -1))
        A = torch.softmax((q @ k.transpose(-2, -1)) / math.sqrt(self.d_edge), -1)
        return (A @ x.unsqueeze(-1)).squeeze(-1) + x

class LandmarkDGGRU(nn.Module):
    def __init__(self, n_seq, n_static, K, hidden=64, dropout=0.2, use_graph=True):
        super().__init__()
        self.use_graph = use_graph
        if use_graph: self.dfg = DynamicFeatureGraph(n_seq)
        self.drop = nn.Dropout(dropout)
        self.gru = nn.GRU(n_seq, hidden, batch_first=True)
        self.ln = nn.LayerNorm(hidden)
        self.proj = nn.Linear(hidden, 64); self.act = nn.GELU()
        self.sev = nn.Linear(64, 1)
        self.semb = nn.Sequential(nn.Linear(n_static, 32), nn.GELU(), nn.Dropout(dropout))
        self.thr = nn.Linear(32, K - 1)
    def forward(self, x, static):
        if self.use_graph: x = self.dfg(x)
        out, _ = self.gru(self.drop(x))
        h = self.act(self.proj(self.ln(out)))
        s = self.sev(h).squeeze(-1)                      # (B,T) severity, higher = worse
        cum = torch.cumsum(F.softplus(self.thr(self.semb(static))), 1)   # (B,K-1) shared across landmarks
        return s, cum

def ordinal_logprob(logits):
    ls = F.logsigmoid(logits.double())
    lp0 = ls[:, :1]
    d = (ls[:, :-1] - ls[:, 1:]).clamp(max=-1e-8)
    lpk = ls[:, 1:] + torch.log(-torch.expm1(d))
    lplast = F.logsigmoid(-logits.double()[:, -1:])
    return torch.cat([lp0, lpk, lplast], 1).float()

def traj_penalty(s, lam_smooth, lam_mono):
    d = s[:, 1:] - s[:, :-1]
    smooth = (d ** 2).mean()
    mono = F.relu(d).mean()          # penalise worsening (severity rising)
    return lam_smooth * smooth + lam_mono * mono

def load():
    z = np.load(DATA / "seq72.npz")
    D = pd.read_parquet(DATA / "design.parquet")
    static = [c for c in D.columns if c in ("age", "male") or c.startswith(("adm_", "unit_"))
              or (c.endswith(("_min", "_max")) and "_w" not in c)]
    assert len(static) == len(set(static))
    assert (D["stay_id"].values == z["stay_id"]).all()
    return z["X"], z["M"], D[static].astype(np.float32).values, z["y"], z["subject_id"], static

def run(args):
    X, M, S, y, groups, static_names = load()
    K = int(y.max()) + 1
    n, T, Fd = X.shape
    print(f"n={n} T={T} F={Fd} K={K} static={S.shape[1]} mode={args.mode}", flush=True)
    use_graph = args.mode.startswith("dggru")
    lam_s = args.lam_smooth if args.mode.endswith("tpr") else 0.0
    lam_m = args.lam_mono if args.mode.endswith("tpr") else 0.0
    gkf = list(GroupKFold(n_splits=args.folds).split(X, y, groups))
    P = np.zeros((n, 3, K)); ev = []; t0 = time.time()
    for f, (tr, te) in enumerate(gkf):
        if args.folds_test and f >= args.folds_test: break
        mu = np.nanmean(X[tr], axis=(0, 1)); sd = np.nanstd(X[tr], axis=(0, 1)) + 1e-6
        def prep(idx):
            A = (X[idx] - mu) / sd
            A = np.where(np.isnan(A), 0.0, A)
            return A.astype(np.float32)
        Xtr, Xte = prep(tr), prep(te)
        med = np.nanmedian(S[tr], axis=0)
        med = np.where(np.isnan(med), 0.0, med)
        S_imp = np.where(np.isnan(S), med, S)
        sc = StandardScaler().fit(S_imp[tr])
        Str = sc.transform(S_imp[tr]).astype(np.float32)
        Ste = sc.transform(S_imp[te]).astype(np.float32)
        assert np.isfinite(Str).all() and np.isfinite(Ste).all()
        torch.manual_seed(args.seed * 100 + f); np.random.seed(args.seed * 100 + f)
        model = LandmarkDGGRU(Fd, S.shape[1], K, use_graph=use_graph)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        cnt = np.bincount(y[tr], minlength=K).astype(np.float32)
        w = torch.from_numpy((len(tr) / (K * cnt)).astype(np.float32))
        Xtr_t = torch.from_numpy(Xtr); Str_t = torch.from_numpy(Str)
        Ytr_t = torch.from_numpy(y[tr].astype(np.int64))
        for ep in range(args.epochs):
            model.train(); perm = torch.randperm(len(tr)); tot = 0.0; nb = 0
            for i in range(0, len(tr), args.batch):
                idx = perm[i:i + args.batch]
                s, cum = model(Xtr_t[idx], Str_t[idx])
                loss = 0.0
                for k, li in enumerate(LM):
                    lp = ordinal_logprob(cum - s[:, li].unsqueeze(-1))
                    loss = loss + F.nll_loss(lp, Ytr_t[idx], weight=w) / 3.0
                if lam_s or lam_m:
                    loss = loss + traj_penalty(s, lam_s, lam_m)
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
                tot += float(loss); nb += 1
            if args.verbose and (ep + 1) % 5 == 0:
                print(f"  fold{f} ep{ep+1} loss={tot/nb:.4f} {time.time()-t0:.0f}s", flush=True)
        model.eval()
        with torch.no_grad():
            s, cum = model(torch.from_numpy(Xte), torch.from_numpy(Ste))
            for k, li in enumerate(LM):
                P[te, k, :] = ordinal_logprob(cum - s[:, li].unsqueeze(-1)).exp().numpy()
            ev.append(te)
        print(f"  fold{f} done {time.time()-t0:.0f}s", flush=True)
    ev = np.unique(np.concatenate(ev))
    return P, y, K, ev

def stability(P, y):
    a = P.argmax(2)
    out = {"n": int(len(y)), "acc": {}, "auc": {}}
    for k, nm in enumerate(("24", "48", "72")):
        out["acc"][nm] = round(float(accuracy_score(y, a[:, k])), 4)
        out["auc"][nm] = round(float(roc_auc_score(y, P[:, k, :], multi_class="ovr", average="macro")), 4)
    for i, j, nm in ((0, 1, "24_48"), (1, 2, "48_72"), (0, 2, "24_72")):
        out[f"flip_{nm}"] = round(float((a[:, i] != a[:, j]).mean()), 4)
        out[f"tv_{nm}"] = round(float((0.5 * np.abs(P[:, i, :] - P[:, j, :]).sum(1)).mean()), 4)
    out["revert"] = round(float(((a[:, 0] == a[:, 2]) & (a[:, 0] != a[:, 1])).mean()), 4)
    out["worse"] = round(float((a[:, 2] > a[:, 0]).mean()), 4)
    out["better"] = round(float((a[:, 2] < a[:, 0]).mean()), 4)
    out["mean_maxprob"] = [round(float(P[:, k, :].max(1).mean()), 4) for k in range(3)]
    q = P[:, 0, :].max(1)
    for thr in (0.5, 0.6, 0.7):
        m = q >= thr
        out[f"by24_{thr}"] = {"cov": round(float(m.mean()), 4), "n": int(m.sum()),
                              "flip": round(float((a[m, 0] != a[m, 2]).mean()), 4)}
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="dggru_tpr")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lam_smooth", type=float, default=0.5)
    ap.add_argument("--lam_mono", type=float, default=0.1)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--folds_test", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    P, y, K, ev = run(args)
    P, y = P[ev], y[ev]
    st = stability(P, y)
    tag = f"{args.mode}_e{args.epochs}_s{args.seed}" + (f"_test{args.folds_test}" if args.folds_test else "")
    print(json.dumps(st, indent=1), flush=True)
    np.savez(RES / f"dggru_{tag}.npz", P=P, y=y, idx=ev)
    json.dump(st, open(RES / f"dggru_{tag}.json", "w"), indent=1)

