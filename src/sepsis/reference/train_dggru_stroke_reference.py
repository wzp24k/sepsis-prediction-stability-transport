# -*- coding: utf-8 -*-
"""Dynamic feature-graph GRU (DG-GRU) with Temporal Pathway Regularization (TPR).

Architecture:
  ICU time series x_t -> DynamicFeatureGraph (value-aware attention over the
  14 physiological/neurological features, adjacency re-computed per step)
  -> GRU -> per-step latent h_t -> recovery score trajectory s_t
  -> ordinal cumulative-link head (discharge disposition, 6 classes).

TPR (Temporal Pathway Regularization): a pathway-motivated prior on the
trajectory - the recovery score should progress monotonically (resolution of
the acute-phase program, per the GEO three-phase pathway atlas) with bounded
drift.  Modes: dggru_tpr / dggru / gru_tpr / gru (ablations);
dggru_tpr_anti / dggru_tpr_perm (negative controls: reversed-direction and
time-permuted priors).
"""
from __future__ import annotations
import math
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, cohen_kappa_score, roc_auc_score
import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from train_gru import (load_data, to_tensor, fold_stats, impute_seq,
                       multiclass_auc, RESULTS, N_CLASSES, N_THRESH)

torch.manual_seed(42)
np.random.seed(42)


class DynamicFeatureGraph(nn.Module):
    def __init__(self, n_feat: int, d_edge: int = 16):
        super().__init__()
        self.d_edge = d_edge
        self.e = nn.Parameter(torch.randn(n_feat, d_edge) * 0.1)
        self.Wq = nn.Linear(d_edge + 1, d_edge)
        self.Wk = nn.Linear(d_edge + 1, d_edge)

    def forward(self, x):
        B, T, F = x.shape
        e = self.e[None, None, :, :].expand(B, T, F, -1)
        v = x.unsqueeze(-1)
        q = self.Wq(torch.cat([e, v], dim=-1))
        k = self.Wk(torch.cat([e, v], dim=-1))
        A = torch.softmax((q @ k.transpose(-2, -1)) / math.sqrt(self.d_edge), dim=-1)
        y = A @ x.unsqueeze(-1)
        return y.squeeze(-1) + x, A


class DGGRUOrdinal(nn.Module):
    def __init__(self, n_seq: int, hidden: int = 64, n_static: int = 0,
                 dropout: float = 0.2, use_graph: bool = True):
        super().__init__()
        self.use_graph = use_graph
        if use_graph:
            self.dfg = DynamicFeatureGraph(n_seq)
        self.gru = nn.GRU(n_seq, hidden, batch_first=True, num_layers=1)
        self.ln = nn.LayerNorm(hidden)
        self.proj = nn.Linear(hidden, 64)
        self.act = nn.GELU()
        self.score = nn.Linear(64, 1)
        self.thresh = nn.Linear(64 + n_static, N_THRESH)

    def forward(self, x, mask, static=None, return_traj=False):
        if self.use_graph:
            x, _ = self.dfg(x)
        lengths = mask.sum(1).clamp(min=1).long()
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True,
                                                   enforce_sorted=False)
        out, _ = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=x.shape[1])  # (B,T,H)
        h = self.act(self.proj(self.ln(out)))                              # (B,T,64)
        s = self.score(h).squeeze(-1)                                      # (B,T)
        idx = (lengths - 1).clamp(min=0).unsqueeze(1)                      # last valid step
        s_final = s.gather(1, idx).squeeze(1)                              # (B,)
        rep = h.gather(1, idx.unsqueeze(-1).expand(-1, -1, 64)).squeeze(1)  # (B,64)
        if static is not None:
            rep = torch.cat([rep, static], dim=1)
        cum = torch.cumsum(F.softplus(self.thresh(rep)), dim=1)            # increasing
        logits = cum - s_final.unsqueeze(1)                                # (B,5)
        ls = F.logsigmoid(logits.double())
        lp0 = ls[:, 0:1]
        d = (ls[:, :-1] - ls[:, 1:]).clamp(max=-1e-8)
        lpk = ls[:, 1:] + torch.log(-torch.expm1(d))
        lplast = F.logsigmoid(-logits.double()[:, -1:])
        log_proba = torch.cat([lp0, lpk, lplast], dim=1).float()
        if return_traj:
            return log_proba, s, mask
        return log_proba


def tpr_loss(s, mask, lam_mono: float, lam_drift: float, delta: float = 0.05,
             variant: str = "mono"):
    """Monotone-progress + bounded-drift regularizers on the recovery score.

    variant: 'mono' - pathway prior (recovery score progresses monotonically);
             'anti' - negative control: prior direction reversed (the prior is
                      applied to -s, i.e. monotone decrease);
             'perm' - negative control: prior applied to a per-sequence random
                      time permutation of the valid trajectory.
    """
    if variant == "anti":
        return tpr_loss(-s, mask, lam_mono, lam_drift, delta=delta, variant="mono")
    if variant == "perm":
        B, T = s.shape
        lens = mask.sum(1).clamp(min=1).long()
        r = torch.rand(B, T, device=s.device)
        r = r.masked_fill(torch.arange(T, device=s.device).unsqueeze(0) >= lens.unsqueeze(1), -1.0)
        order = r.argsort(dim=1, stable=True)
        s = s.gather(1, order)
    valid = (mask[:, 1:] * mask[:, :-1]).float()
    diffs = s[:, 1:] - s[:, :-1]
    r_mono = (F.relu(-diffs) * valid).sum() / valid.sum().clamp(min=1.0)
    # drift: last valid step score - first valid step score
    first_idx = (mask.cumsum(1) == 1).float().argmax(1).unsqueeze(1)
    s_first = s.gather(1, first_idx).squeeze(1)
    last_idx = (mask.sum(1).clamp(min=1).long() - 1).unsqueeze(1)
    s_last = s.gather(1, last_idx).squeeze(1)
    r_drift = ((s_last - s_first) - delta).pow(2).mean()
    return lam_mono * r_mono + lam_drift * r_drift


def train_fold(Xseq_tr_raw, M_tr, Xs_tr, y_tr, Xseq_te_raw, M_te, Xs_te,
               n_epochs=30, batch=128, lr=1e-3, seed=0, verbose=False,
               use_graph=True, lam_mono=0.2, lam_drift=0.05,
               tpr_variant="mono"):
    mean, std = fold_stats(Xseq_tr_raw, M_tr, np.arange(len(y_tr)))
    Xseq_tr = impute_seq(Xseq_tr_raw, M_tr, mean, std)
    Xseq_te = impute_seq(Xseq_te_raw, M_te, mean, std)

    torch.manual_seed(seed)
    model = DGGRUOrdinal(n_seq=Xseq_tr.shape[2], hidden=64,
                         n_static=Xs_tr.shape[1], use_graph=use_graph)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    counts = np.bincount(y_tr, minlength=N_CLASSES).astype(np.float32)
    w = (len(y_tr) / (N_CLASSES * counts)).astype(np.float32)
    weight = torch.from_numpy(w)

    n = len(y_tr)
    Xtr_t = torch.from_numpy(Xseq_tr); Mtr_t = torch.from_numpy(M_tr)
    S_tr = torch.from_numpy(Xs_tr.astype(np.float32))
    Y_tr = torch.from_numpy(y_tr.astype(np.int64))
    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(n)
        tot, nb = 0.0, 0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            sb = S_tr[idx]
            log_proba, s, m = model(Xtr_t[idx], Mtr_t[idx], sb, return_traj=True)
            loss = F.nll_loss(log_proba, Y_tr[idx], weight=weight)
            loss = loss + tpr_loss(s, m, lam_mono, lam_drift, variant=tpr_variant)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss.item()); nb += 1
        if verbose and (epoch + 1) % 5 == 0:
            print(f"  epoch {epoch + 1}/{n_epochs} loss={tot / nb:.4f}")
    model.eval()
    with torch.no_grad():
        log_proba = model(torch.from_numpy(Xseq_te), torch.from_numpy(M_te),
                          torch.from_numpy(Xs_te.astype(np.float32)))
    return log_proba.exp().numpy()


def run(n_epochs=30, mode="dggru_tpr", verbose=True, lam1=None, lam2=None):
    tpr_variant = "mono"
    base_mode = mode
    for suf, v in (("_anti", "anti"), ("_perm", "perm")):
        if mode.endswith(suf):
            tpr_variant = v
            base_mode = mode[: -len(suf)]
            break
    use_graph = base_mode in ("dggru_tpr", "dggru")
    use_tpr = base_mode in ("dggru_tpr", "gru_tpr")
    if lam1 is None:
        lam1 = 0.2 if use_tpr else 0.0
    if lam2 is None:
        lam2 = 0.05 if use_tpr else 0.0
    tag = mode
    if use_tpr and (lam1, lam2) != (0.2, 0.05):
        tag = f"{mode}_m{lam1}_d{lam2}"
    seqs, masks, Xs, y, hadms = load_data()
    Xseq, M = to_tensor(seqs, masks, hadms)
    print(f"[{mode}] n={len(y)}, seq={Xseq.shape}, static={Xs.shape}", flush=True)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs, kaps, aucs = [], [], []
    oof = np.zeros((len(y), N_CLASSES))
    for f, (tr, te) in enumerate(skf.split(Xs, y)):
        sc = StandardScaler().fit(Xs.iloc[tr])
        Xtr_s = sc.transform(Xs.iloc[tr]); Xte_s = sc.transform(Xs.iloc[te])
        proba = train_fold(Xseq[tr], M[tr], Xtr_s, y[tr], Xseq[te], M[te], Xte_s,
                           n_epochs=n_epochs, seed=42 + f, verbose=verbose,
                           use_graph=use_graph, lam_mono=lam1, lam_drift=lam2,
                           tpr_variant=tpr_variant)
        oof[te] = proba
        pred = proba.argmax(1)
        accs.append(accuracy_score(y[te], pred))
        kaps.append(cohen_kappa_score(y[te], pred))
        aucs.append(multiclass_auc(y[te], proba))
        print(f"  fold {f}: acc={accs[-1]:.4f} kappa={kaps[-1]:.4f} auc={aucs[-1]:.4f}", flush=True)
    line = (f"{tag} | acc: {np.mean(accs):.4f}+/-{np.std(accs):.4f} | "
            f"kappa: {np.mean(kaps):.4f} | auc: {np.mean(aucs):.4f}+/-{np.std(aucs):.4f}")
    print(line, flush=True)
    with open(os.path.join(RESULTS, f"{tag}_result.txt"), "w", encoding="utf-8") as f:
        f.write(line + "\n")
    np.savez(os.path.join(RESULTS, f"{tag}_oof.npz"), y=y, proba=oof, hadm_id=hadms)


if __name__ == "__main__":
    n_ep = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    mode = sys.argv[2] if len(sys.argv) > 2 else "dggru_tpr"
    lam1 = float(sys.argv[3]) if len(sys.argv) > 3 else None
    lam2 = float(sys.argv[4]) if len(sys.argv) > 4 else None
    run(n_epochs=n_ep, mode=mode, lam1=lam1, lam2=lam2)
