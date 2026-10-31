# -*- coding: utf-8 -*-
"""Formal multi-landmark DG-GRU training for ordinal or nominal readouts."""

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler


ROOT = Path(os.environ.get("SEPSIS_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
LANDMARK_INDEX = (23, 47, 71)
CHANNELS = ("hr", "rr", "spo2", "sbp", "dbp", "mbp", "tempf", "gcs")


class DynamicFeatureGraph(nn.Module):
    def __init__(self, n_feature: int, edge_dimension: int = 16):
        super().__init__()
        self.edge_dimension = edge_dimension
        self.embedding = nn.Parameter(torch.randn(n_feature, edge_dimension) * 0.1)
        self.query = nn.Linear(edge_dimension + 1, edge_dimension)
        self.key = nn.Linear(edge_dimension + 1, edge_dimension)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        batch, steps, features = values.shape
        edge = self.embedding[None, None].expand(batch, steps, features, -1)
        observed = values.unsqueeze(-1)
        query = self.query(torch.cat([edge, observed], -1))
        key = self.key(torch.cat([edge, observed], -1))
        attention = torch.softmax(
            (query @ key.transpose(-2, -1)) / math.sqrt(self.edge_dimension), -1
        )
        return (attention @ values.unsqueeze(-1)).squeeze(-1) + values


class LandmarkModel(nn.Module):
    def __init__(
        self,
        n_sequence: int,
        n_static: int,
        classes: int,
        head: str,
        hidden: int = 64,
        dropout: float = 0.2,
        use_graph: bool = True,
    ):
        super().__init__()
        self.head = head
        self.graph = DynamicFeatureGraph(n_sequence) if use_graph else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(n_sequence, hidden, batch_first=True)
        self.norm = nn.LayerNorm(hidden)
        self.projection = nn.Linear(hidden, 64)
        self.activation = nn.GELU()
        self.severity = nn.Linear(64, 1)
        self.static_embedding = nn.Sequential(
            nn.Linear(n_static, 32), nn.GELU(), nn.Dropout(dropout)
        )
        self.threshold = nn.Linear(32, classes - 1)
        if head == "nominal":
            self.nominal = nn.Sequential(
                nn.Linear(64, 32),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(32, classes),
            )

    def forward(self, sequence: torch.Tensor, static: torch.Tensor) -> tuple:
        transformed = self.graph(sequence)
        output, _ = self.gru(self.dropout(transformed))
        hidden = self.activation(self.projection(self.norm(output)))
        severity = self.severity(hidden).squeeze(-1)
        if self.head == "ordinal":
            increments = F.softplus(self.threshold(self.static_embedding(static)))
            thresholds = torch.cumsum(increments, dim=1)
            return severity, thresholds
        nominal = self.nominal(hidden)
        return severity, nominal


def ordinal_log_probability(logits: torch.Tensor) -> torch.Tensor:
    logs = F.logsigmoid(logits.double())
    first = logs[:, :1]
    differences = (logs[:, :-1] - logs[:, 1:]).clamp(max=-1e-8)
    middle = logs[:, 1:] + torch.log(-torch.expm1(differences))
    last = F.logsigmoid(-logits.double()[:, -1:])
    return torch.cat([first, middle, last], 1).float()


def temporal_penalty(
    severity: torch.Tensor, smooth_weight: float, monotone_weight: float
) -> torch.Tensor:
    difference = severity[:, 1:] - severity[:, :-1]
    smooth = (difference**2).mean()
    monotone = F.relu(difference).mean()
    return smooth_weight * smooth + monotone_weight * monotone


def static_columns(frame: pd.DataFrame) -> list[str]:
    laboratory = [
        "crea_min",
        "crea_max",
        "lac_min",
        "lac_max",
        "wbc_min",
        "wbc_max",
        "plt_min",
        "plt_max",
        "bili_min",
        "bili_max",
        "na_min",
        "na_max",
        "bun_min",
        "bun_max",
        "hgb_min",
        "hgb_max",
        "inr_max",
    ]
    columns = ["age", "male"] + laboratory
    columns += [
        column
        for column in frame.columns
        if column.startswith(("adm_", "unit_", "src_", "ut_"))
    ]
    return [column for column in dict.fromkeys(columns) if column in frame.columns]


def load_data(database: str) -> dict:
    folder = ROOT / "data" / "processed" / database
    sequence = np.load(folder / "seq72.npz")
    design = pd.read_parquet(
        folder / ("design.parquet" if database == "mimic4" else "design_v3.parquet")
    )
    identifier = "stay_id" if database == "mimic4" else "patientunitstayid"
    group = "subject_id" if database == "mimic4" else "uniquepid"
    sequence_id = sequence[identifier]
    if not np.array_equal(design[identifier].to_numpy(), sequence_id):
        raise ValueError(f"{database}: design and sequence rows are not aligned")
    return {
        "x": sequence["X"].astype(np.float32),
        "mask": sequence["M"].astype(np.float32),
        "static": design[static_columns(design)].astype(np.float32).to_numpy(),
        "y": sequence["y3"].astype(np.int64),
        "groups": design[group].astype(str).to_numpy(),
        "ids": design[identifier].to_numpy(dtype=np.int64),
        "static_names": static_columns(design),
    }


def predict(model: nn.Module, x: torch.Tensor, static: torch.Tensor, head: str) -> np.ndarray:
    with torch.no_grad():
        severity, output = model(x, static)
        probabilities = []
        for position in LANDMARK_INDEX:
            if head == "ordinal":
                logits = output - severity[:, position].unsqueeze(-1)
                log_probability = ordinal_log_probability(logits)
            else:
                log_probability = F.log_softmax(output[:, position, :], dim=1)
            probabilities.append(log_probability.exp().cpu().numpy())
    return np.stack(probabilities, axis=1)


def metrics(probability: np.ndarray, y: np.ndarray) -> dict:
    prediction = probability.argmax(2)
    result = {"n": int(len(y)), "auc": {}, "accuracy": {}, "log_loss": {}}
    for index, landmark in enumerate((24, 48, 72)):
        result["auc"][str(landmark)] = round(
            float(
                roc_auc_score(
                    y,
                    probability[:, index, :],
                    multi_class="ovr",
                    average="macro",
                    labels=np.arange(3),
                )
            ),
            4,
        )
        result["accuracy"][str(landmark)] = round(
            float(accuracy_score(y, prediction[:, index])), 4
        )
        result["log_loss"][str(landmark)] = round(
            float(log_loss(y, probability[:, index, :], labels=np.arange(3))), 4
        )
    for first, second in ((0, 1), (1, 2), (0, 2)):
        result[f"flip_{first}_{second}"] = round(
            float((prediction[:, first] != prediction[:, second]).mean()), 4
        )
        result[f"tv_{first}_{second}"] = round(
            float(
                (
                    0.5
                    * np.abs(probability[:, first, :] - probability[:, second, :]).sum(1)
                ).mean()
            ),
            4,
        )
    result["return_to_24h_class"] = round(
        float(
            (
                (prediction[:, 0] == prediction[:, 2])
                & (prediction[:, 0] != prediction[:, 1])
            ).mean()
        ),
        4,
    )
    confidence = probability[:, 0, :].max(1)
    for threshold in (0.5, 0.6, 0.7):
        selected = confidence >= threshold
        result[f"confidence_{threshold}"] = {
            "coverage": round(float(selected.mean()), 4),
            "n": int(selected.sum()),
            "flip": round(
                float((prediction[selected, 0] != prediction[selected, 2]).mean()), 4
            )
            if selected.sum()
            else None,
        }
    return result


def run(args: argparse.Namespace) -> dict:
    data = load_data(args.database)
    x, mask, static, y, groups = (
        data["x"],
        data["mask"],
        data["static"],
        data["y"],
        data["groups"],
    )
    classes = int(y.max()) + 1
    if args.reverse_order:
        y = (classes - 1) - y
    folds = list(GroupKFold(n_splits=args.folds).split(x, y, groups))
    probability = np.zeros((len(y), 3, classes), dtype=np.float32)
    epochs_run = []
    started = time.time()
    print(
        args.database,
        {"n": len(y), "classes": classes, "static": static.shape[1]},
        vars(args),
        flush=True,
    )
    for fold, (train_index, test_index) in enumerate(folds):
        if args.fold_limit and fold >= args.fold_limit:
            break
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=0.15, random_state=args.seed * 100 + fold
        )
        fit_relative, validation_relative = next(
            splitter.split(
                x[train_index], y[train_index], groups[train_index]
            )
        )
        fit_index = train_index[fit_relative]
        validation_index = train_index[validation_relative]

        channel_mean = np.nanmean(x[fit_index], axis=(0, 1))
        channel_sd = np.nanstd(x[fit_index], axis=(0, 1)) + 1e-6

        def prepare_sequence(indices: np.ndarray) -> np.ndarray:
            values = (x[indices] - channel_mean) / channel_sd
            return np.where(np.isnan(values), 0.0, values).astype(np.float32)

        static_median = np.nanmedian(static[fit_index], axis=0)
        static_median = np.where(np.isnan(static_median), 0.0, static_median)
        static_imputed = np.where(np.isnan(static), static_median, static)
        scaler = StandardScaler().fit(static_imputed[fit_index])
        static_scaled = scaler.transform(static_imputed).astype(np.float32)

        torch.manual_seed(args.seed * 1000 + fold)
        np.random.seed(args.seed * 1000 + fold)
        model = LandmarkModel(
            n_sequence=x.shape[2],
            n_static=static.shape[1],
            classes=classes,
            head=args.head,
            hidden=args.hidden,
            dropout=args.dropout,
            use_graph=not args.no_graph,
        )
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
        fit_x = torch.from_numpy(prepare_sequence(fit_index))
        validation_x = torch.from_numpy(prepare_sequence(validation_index))
        fit_static = torch.from_numpy(static_scaled[fit_index])
        validation_static = torch.from_numpy(static_scaled[validation_index])
        fit_y = torch.from_numpy(y[fit_index])
        validation_y = torch.from_numpy(y[validation_index])
        best_loss = float("inf")
        best_state = None
        patience_counter = 0
        best_epoch = 0
        for epoch in range(args.epochs):
            model.train()
            permutation = torch.randperm(len(fit_index))
            for start in range(0, len(fit_index), args.batch_size):
                selected = permutation[start : start + args.batch_size]
                severity, output = model(fit_x[selected], fit_static[selected])
                loss = torch.zeros((), dtype=torch.float32)
                for position in LANDMARK_INDEX:
                    if args.head == "ordinal":
                        logits = output - severity[:, position].unsqueeze(-1)
                        log_probability = ordinal_log_probability(logits)
                        loss = loss + F.nll_loss(log_probability, fit_y[selected])
                    else:
                        loss = loss + F.cross_entropy(
                            output[:, position, :], fit_y[selected]
                        )
                loss = loss / len(LANDMARK_INDEX)
                if args.temporal:
                    loss = loss + temporal_penalty(
                        severity, args.smooth_weight, args.monotone_weight
                    )
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            model.eval()
            with torch.no_grad():
                validation_probability = predict(
                    model, validation_x, validation_static, args.head
                )
                validation_loss = float(
                    np.mean(
                        [
                            log_loss(
                                y[validation_index],
                                validation_probability[:, index, :],
                                labels=np.arange(classes),
                            )
                            for index in range(3)
                        ]
                    )
                )
            if validation_loss < best_loss - 1e-5:
                best_loss = validation_loss
                best_epoch = epoch + 1
                best_state = {
                    key: value.detach().clone()
                    for key, value in model.state_dict().items()
                }
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        probability[test_index] = predict(
            model,
            torch.from_numpy(prepare_sequence(test_index)),
            torch.from_numpy(static_scaled[test_index]),
            args.head,
        )
        epochs_run.append(best_epoch)
        print(
            f"fold {fold + 1}: best_epoch={best_epoch} validation_log_loss={best_loss:.4f} "
            f"elapsed={time.time() - started:.0f}s",
            flush=True,
        )
    evaluated = np.concatenate([test for _, test in folds[: len(epochs_run)]])
    configuration = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    result = {
        "database": args.database,
        "configuration": configuration,
        "metrics": metrics(probability[evaluated], y[evaluated]),
        "best_epochs": epochs_run,
        "elapsed_seconds": round(time.time() - started, 1),
    }
    return {
        "result": result,
        "probability": probability[evaluated],
        "y": y[evaluated],
        "ids": data["ids"][evaluated],
        "groups": data["groups"][evaluated],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", choices=("mimic4", "eicu"), required=True)
    parser.add_argument("--head", choices=("ordinal", "nominal"), default="ordinal")
    parser.add_argument("--temporal", action="store_true")
    parser.add_argument("--no_graph", action="store_true")
    parser.add_argument("--reverse_order", action="store_true")
    parser.add_argument("--smooth_weight", type=float, default=0.04)
    parser.add_argument("--monotone_weight", type=float, default=0.06)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold_limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--tag", required=True)
    parser.add_argument(
        "--output_folder",
        type=Path,
        default=None,
        help="Optional output directory; defaults to results/formal/<database>.",
    )
    args = parser.parse_args()
    torch.set_num_threads(int(os.environ.get("SEPSIS_TORCH_THREADS", "8")))
    output = run(args)
    output_folder = (
        args.output_folder
        if args.output_folder is not None
        else ROOT / "results" / "formal" / args.database
    )
    output_folder = output_folder.resolve()
    output_folder.mkdir(parents=True, exist_ok=True)
    stem = f"{args.tag}_s{args.seed}"
    np.savez_compressed(
        output_folder / f"{stem}.npz",
        probability=output["probability"],
        y=output["y"],
        ids=output["ids"],
        groups=output["groups"].astype("U"),
    )
    with (output_folder / f"{stem}.json").open("w", encoding="utf-8") as handle:
        json.dump(output["result"], handle, indent=2, ensure_ascii=False)
    print(json.dumps(output["result"], indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
