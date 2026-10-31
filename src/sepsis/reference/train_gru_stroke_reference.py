# -*- coding: utf-8 -*-
"""GRU + ordinal regression (cumulative link) on ICU sequences + static features."""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, cohen_kappa_score, roc_auc_score
import warnings
warnings.filterwarnings("ignore")

OUTDIR = r"D:\TT paper\0811Temporal Pathway\data\processed\mimic_stroke"
RESULTS = r"D:\TT paper\0811Temporal Pathway\results"
torch.manual_seed(42)
np.random.seed(42)

N_CLASSES = 6
N_THRESH = N_CLASSES - 1


class SeqOrdinal(nn.Module):
    def __init__(self, n_seq=14, hidden=64, n_static=0, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(n_seq, hidden, batch_first=True, num_layers=1)
        self.ln = nn.LayerNorm(hidden)
        self.proj = nn.Linear(hidden, 64)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(64 + n_static, N_THRESH + 1)

    def forward(self, x, mask, static=None):
        # x: (B, T, F), mask: (B, T); valid steps are the first `length` steps (left-aligned)
        lengths = mask.sum(1).clamp(min=1).long()
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True,
                                                   enforce_sorted=False)
        _, h = self.gru(packed)                 # h: (1, B, hidden)
        rep = self.act(self.proj(self.ln(h[-1])))
        rep = self.drop(rep)
        if static is not None:
            rep = torch.cat([rep, static], dim=1)
        out = self.head(rep)                    # (B, 6): 5 thresholds + 1 score
        thresh = out[:, :N_THRESH]
        s = out[:, N_THRESH:]
        cum = torch.cumsum(F.softplus(thresh), dim=1)   # increasing thresholds
        logits = cum - s                        # (B, 5) increasing
        # log P(y=0)=logsigmoid(l0); log P(y=k)=log(sigmoid(lk)-sigmoid(lk-1)); log P(y=5)=logsigmoid(-l4)
        # compute in float64 to avoid log(1-e^d) -inf when logit gaps underflow in float32
        ls = F.logsigmoid(logits.double())      # (B, 5)
        lp0 = ls[:, 0:1]
        d = (ls[:, :-1] - ls[:, 1:]).clamp(max=-1e-8)   # < 0
        lpk = ls[:, 1:] + torch.log(-torch.expm1(d))
        lplast = F.logsigmoid(-logits.double()[:, -1:])
        log_proba = torch.cat([lp0, lpk, lplast], dim=1).float()   # (B, 6)
        return log_proba


def load_data(use_ts_static: bool = False):
    z = np.load(os.path.join(OUTDIR, "icu_sequences.npz"))
    seqs = {int(k[4:]): z[k] for k in z.files if k.startswith("seq_")}
    masks = {int(k[5:]): z[k] for k in z.files if k.startswith("mask_")}
    fname = "static_plus_ts.csv" if use_ts_static else "static_features.csv"
    stat = pd.read_csv(os.path.join(OUTDIR, fname))
    stat = stat[stat["outcome_ordinal"].notna()].copy()
    stat = stat[stat["hadm_id"].isin(seqs.keys())].copy()
    y = stat["outcome_ordinal"].astype(int).values
    drop = ["hadm_id", "subject_id", "discharge_location", "outcome_ordinal", "hospital_expire_flag"]
    Xs = stat.drop(columns=[c for c in drop if c in stat.columns])
    Xs = Xs.select_dtypes(include=[np.number])
    Xs = Xs.loc[:, Xs.median() == Xs.median()]          # drop all-NaN columns
    Xs = Xs.fillna(Xs.median())
    hadms = stat["hadm_id"].values
    return seqs, masks, Xs, y, hadms


def to_tensor(seqs, masks, hadms):
    X = np.stack([seqs[h] for h in hadms]).astype(np.float32)
    M = np.stack([masks[h] for h in hadms]).astype(np.float32)
    return X, M


def fold_stats(Xseq, M, idx):
    """Per-feature mean/std from valid entries of the training fold (raw scale)."""
    sub = Xseq[idx]
    m = M[idx]
    vals = sub[m == 1]
    mean = np.nanmean(vals, axis=0)
    std = np.nanstd(vals, axis=0)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def impute_seq(Xraw, M, mean, std):
    """Standardize (train-fold stats), ffill/bfill within valid range, then mean-fill (0)."""
    X = (Xraw - mean) / std
    X = X.astype(np.float32)
    N, T, F = X.shape
    for i in range(N):
        t = int(M[i].sum())
        if t <= 1:
            continue
        seg = X[i, :t]
        if np.isnan(seg).any():
            X[i, :t] = pd.DataFrame(seg).ffill().bfill().values
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def train_fold(Xseq_tr_raw, M_tr, Xs_tr, y_tr, Xseq_te_raw, M_te, Xs_te,
               n_epochs=30, batch=128, lr=1e-3, seed=0, verbose=False, use_static=True):
    mean, std = fold_stats(Xseq_tr_raw, M_tr, np.arange(len(y_tr)))
    Xseq_tr = impute_seq(Xseq_tr_raw, M_tr, mean, std)
    Xseq_te = impute_seq(Xseq_te_raw, M_te, mean, std)

    torch.manual_seed(seed)
    n_static = Xs_tr.shape[1] if use_static else 0
    model = SeqOrdinal(n_seq=Xseq_tr.shape[2], hidden=64, n_static=n_static)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    counts = np.bincount(y_tr, minlength=N_CLASSES).astype(np.float32)
    w = (len(y_tr) / (N_CLASSES * counts)).astype(np.float32)
    weight = torch.from_numpy(w)

    n = len(y_tr)
    Xtr_t = torch.from_numpy(Xseq_tr)
    Mtr_t = torch.from_numpy(M_tr)
    S_tr = torch.from_numpy(Xs_tr.astype(np.float32))
    Y_tr = torch.from_numpy(y_tr.astype(np.int64))

    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(n)
        tot, nb = 0.0, 0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            sb = S_tr[idx] if use_static else None
            log_proba = model(Xtr_t[idx], Mtr_t[idx], sb)
            loss = F.nll_loss(log_proba, Y_tr[idx], weight=weight)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss.item()); nb += 1
        if verbose and (epoch + 1) % 5 == 0:
            print(f"  epoch {epoch + 1}/{n_epochs} loss={tot / nb:.4f}")

    model.eval()
    with torch.no_grad():
        sb_te = torch.from_numpy(Xs_te.astype(np.float32)) if use_static else None
        log_proba = model(torch.from_numpy(Xseq_te), torch.from_numpy(M_te), sb_te)
    return log_proba.exp().numpy()


def multiclass_auc(y_true, proba, n_classes=6):
    aucs = []
    for k in range(n_classes):
        try:
            aucs.append(roc_auc_score((y_true == k).astype(int), proba[:, k]))
        except Exception:
            pass
    return float(np.mean(aucs)) if aucs else 0.0


def run(n_epochs=30, verbose=False, use_static=True, tag="gru", use_ts_static=False):
    seqs, masks, Xs, y, hadms = load_data(use_ts_static=use_ts_static)
    Xseq, M = to_tensor(seqs, masks, hadms)
    print(f"[{tag}] n={len(y)}, seq={Xseq.shape}, static={Xs.shape}", flush=True)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs, kaps, aucs = [], [], []
    oof = np.zeros((len(y), N_CLASSES))
    for f, (tr, te) in enumerate(skf.split(Xs, y)):
        sc = StandardScaler().fit(Xs.iloc[tr])
        Xtr_s = sc.transform(Xs.iloc[tr])
        Xte_s = sc.transform(Xs.iloc[te])
        proba = train_fold(Xseq[tr], M[tr], Xtr_s, y[tr],
                           Xseq[te], M[te], Xte_s,
                           n_epochs=n_epochs, seed=42 + f, verbose=verbose,
                           use_static=use_static)
        oof[te] = proba
        pred = proba.argmax(axis=1)
        accs.append(accuracy_score(y[te], pred))
        kaps.append(cohen_kappa_score(y[te], pred))
        aucs.append(multiclass_auc(y[te], proba))
        print(f"  fold {f}: acc={accs[-1]:.4f} kappa={kaps[-1]:.4f} auc={aucs[-1]:.4f}", flush=True)
    line = (f"{tag} ordinal | acc: {np.mean(accs):.4f}+/-{np.std(accs):.4f} | "
            f"kappa: {np.mean(kaps):.4f} | auc: {np.mean(aucs):.4f}+/-{np.std(aucs):.4f}")
    print(line, flush=True)
    with open(os.path.join(RESULTS, f"{tag}_result.txt"), "w", encoding="utf-8") as f:
        f.write(line + "\n")
    np.savez(os.path.join(RESULTS, f"{tag}_oof.npz"),
             y=y, proba=oof, hadm_id=hadms)


if __name__ == "__main__":
    import sys
    n_ep = 30
    mode = "full"
    if len(sys.argv) > 1:
        n_ep = int(sys.argv[1])
    if len(sys.argv) > 2:
        mode = sys.argv[2]
    if mode == "seqonly":
        run(n_epochs=n_ep, verbose=True, use_static=False, tag="gru_seqonly")
    elif mode == "tsstatic":
        run(n_epochs=n_ep, verbose=True, use_static=True, tag="gru_tsstatic",
            use_ts_static=True)
    elif mode == "long":
        run(n_epochs=n_ep, verbose=True, use_static=True, tag="gru_long")
    elif mode == "full":
        run(n_epochs=n_ep, verbose=True, use_static=True, tag="gru")
