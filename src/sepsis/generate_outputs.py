# -*- coding: utf-8 -*-
"""Generate study tables and publication-style figures from frozen artifacts."""

import json
import os
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from sklearn.linear_model import LogisticRegression


ROOT = Path(os.environ.get("SEPSIS_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
TABLES = RESULTS / "tables"
FIGURES.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)

COLORS = {
    "mimic4": "#1F6F78",
    "eicu": "#C26D2E",
    "ordinal_notemporal": "#5B6770",
    "ordinal_temporal": "#1F6F78",
    "nominal_notemporal": "#B9A489",
    "nominal_temporal": "#A64B2A",
}
LABELS = {
    "mimic4": "MIMIC-IV",
    "eicu": "eICU-CRD",
    "ordinal_notemporal": "Ordinal, no temporal penalty",
    "ordinal_temporal": "Ordinal + temporal penalty",
    "nominal_notemporal": "Nominal, no temporal penalty",
    "nominal_temporal": "Nominal + temporal penalty",
}
CONFIGURATIONS = tuple(LABELS)[2:]


def fixed(value: float, digits: int | None = None) -> str:
    if digits is None:
        rounded_four = Decimal(str(value)).quantize(
            Decimal("0.0001"), rounding=ROUND_HALF_UP
        )
        digits = (
            4
            if Decimal("0") < abs(rounded_four) < Decimal("0.01")
            else 2
        )
    quantum = Decimal(1).scaleb(-digits)
    rounded = Decimal(str(value)).quantize(quantum, rounding=ROUND_HALF_UP)
    return f"{rounded:.{digits}f}"


def signed_fixed(value: float, digits: int | None = None) -> str:
    rounded = Decimal(fixed(value, digits))
    if rounded == 0:
        return fixed(0, digits)
    return ("+" if rounded > 0 else "") + fixed(value, digits)


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def load_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.08,
        1.08,
        label,
        transform=axis.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
        ha="left",
    )


def save_figure(figure: plt.Figure, stem: str) -> None:
    figure.savefig(FIGURES / f"{stem}.png", dpi=600, bbox_inches="tight")
    figure.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight")
    figure.savefig(FIGURES / f"{stem}.svg", bbox_inches="tight")
    plt.close(figure)


def figure_cohort_outcomes() -> None:
    mimic = pd.read_parquet(ROOT / "data/processed/mimic4/design_all.parquet")
    eicu = pd.read_parquet(ROOT / "data/processed/eicu/design_all.parquet")
    mimic_72 = int(((mimic["y3"].notna()) & (mimic["los"] >= 72)).sum())
    eicu_72 = int(((eicu["y3"].notna()) & (eicu["los_h"] >= 72)).sum())
    mimic_eligible_72 = int((mimic["los"] >= 72).sum())
    eicu_eligible_72 = int((eicu["los_h"] >= 72).sum())
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), gridspec_kw={"width_ratios": [1.4, 1]})
    axis = axes[0]
    axis.set_xlim(0, 10)
    axis.set_ylim(0, 10)
    axis.axis("off")
    boxes = [
        (0.4, 7.8, f"MIMIC-IV sepsis\nICU stays, {len(mimic):,}"),
        (0.4, 5.3, f"ICU stay >=72 h\n{mimic_eligible_72:,}"),
        (0.4, 2.8, f"Mapped outcome\n{mimic_72:,}"),
        (5.4, 7.8, f"eICU-CRD sepsis\nICU stays, {len(eicu):,}"),
        (5.4, 5.3, f"ICU stay >=72 h\n{eicu_eligible_72:,}"),
        (5.4, 2.8, f"Mapped outcome\n{eicu_72:,}"),
    ]
    for x, y, text in boxes:
        axis.add_patch(
            FancyBboxPatch(
                (x, y),
                4.0,
                1.35,
                boxstyle="round,pad=0.03,rounding_size=0.12",
                linewidth=0.8,
                edgecolor="#3D3D3D",
                facecolor="white",
            )
        )
        axis.text(x + 2.0, y + 0.68, text, ha="center", va="center")
    for x in (2.4, 7.4):
        for y in (7.8, 5.3):
            axis.add_patch(
                FancyArrowPatch(
                    (x, y - 0.1),
                    (x, y - 1.05),
                    arrowstyle="-|>",
                    mutation_scale=10,
                    linewidth=0.9,
                    color="#3D3D3D",
                )
            )
    panel_label(axis, "A")

    axis = axes[1]
    class_labels = ["Home", "Facility", "In-hospital death"]
    distribution = []
    for frame, duration in ((mimic, "los"), (eicu, "los_h")):
        subset = frame.loc[frame["y3"].notna() & (frame[duration] >= 72)]
        distribution.append(
            np.asarray([(subset["y3"] == index).mean() * 100 for index in range(3)])
        )
    positions = np.arange(3)
    width = 0.35
    for offset, values, database in (
        (-width / 2, distribution[0], "mimic4"),
        (width / 2, distribution[1], "eicu"),
    ):
        bars = axis.bar(
            positions + offset,
            values,
            width,
            color=COLORS[database],
            label=LABELS[database],
            linewidth=0,
        )
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + 1,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )
    axis.set_xticks(positions, class_labels, rotation=15, ha="right")
    axis.set_ylabel("Patients (%)")
    axis.set_ylim(0, 58)
    axis.legend(frameon=False, loc="upper right")
    panel_label(axis, "B")
    figure.tight_layout()
    save_figure(figure, "figure1_cohort_outcome")


def figure_architecture() -> None:
    figure, axis = plt.subplots(figsize=(7.2, 2.7))
    axis.set_xlim(0, 12)
    axis.set_ylim(0.45, 4.65)
    axis.axis("off")
    nodes = [
        (0.2, 1.80, 1.55, 1.15, "72-h ICU\n8 channels"),
        (2.40, 1.80, 1.55, 1.15, "Dynamic\nfeature\ngraph"),
        (4.60, 1.80, 1.35, 1.15, "GRU\n64 units"),
        (6.50, 1.80, 1.45, 1.15, "Landmark\nreadout"),
        (8.60, 2.65, 3.05, 1.05, "Ordinal shared thresholds\nHome < Facility < Death"),
        (8.60, 0.90, 3.05, 1.05, "Temporal penalty\nsmoothness +\nnon-worsening"),
    ]
    for x, y, width, height, text in nodes:
        axis.add_patch(
            FancyBboxPatch(
                (x, y),
                width,
                height,
                boxstyle="round,pad=0.03,rounding_size=0.10",
                linewidth=0.8,
                edgecolor="#3D3D3D",
                facecolor="white",
            )
        )
        axis.text(
            x + width / 2,
            y + height / 2,
            text,
            ha="center",
            va="center",
            fontsize=8,
        )
    for first, second in ((0, 1), (1, 2), (2, 3)):
        axis.add_patch(
            FancyArrowPatch(
            (nodes[first][0] + nodes[first][2], nodes[first][1] + 0.55),
            (nodes[second][0], nodes[second][1] + 0.55),
                arrowstyle="-|>",
                mutation_scale=10,
                linewidth=0.9,
                color="#3D3D3D",
            )
        )
    axis.add_patch(
        FancyArrowPatch(
            (7.97, 2.95),
            (8.55, 3.15),
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=0.9,
            color="#3D3D3D",
        )
    )
    axis.add_patch(
        FancyArrowPatch(
            (7.97, 1.80),
            (8.55, 1.45),
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=0.9,
            color="#3D3D3D",
        )
    )
    axis.text(
        7.30,
        3.75,
        "24 / 48 / 72 h",
        ha="center",
        va="center",
        color="#1F6F78",
        fontweight="bold",
    )
    axis.text(0.2, 4.35, "Input and model", fontsize=9, fontweight="bold")
    save_figure(figure, "figure2_model")


def figure_performance() -> None:
    shared = load_json(RESULTS / "shared_fold/summary.json")
    formal = load_json(RESULTS / "formal_summary/summary.json")
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
    axis = axes[0]
    label_offsets = {"mimic4": (0, 5), "eicu": (0, -10)}
    for database, label in (("mimic4", "MIMIC-IV"), ("eicu", "eICU-CRD")):
        record = shared["databases"][database]
        values = [
            record["metrics"][str(hour)]["macro_auc"]
            for hour in (24, 48, 72)
        ]
        axis.plot((24, 48, 72), values, marker="o", color=COLORS[database], label=label)
        for hour, value in zip((24, 48, 72), values):
            axis.annotate(
                f"{value:.3f}",
                (hour, value),
                xytext=label_offsets[database],
                textcoords="offset points",
                ha="center",
                fontsize=6.8,
            )
    axis.set_title("XGBoost, shared complete cohort")
    axis.set_xlabel("Landmark (hours)")
    axis.set_ylabel("Macro-AUC")
    axis.set_xticks((24, 48, 72))
    axis.set_ylim(0.64, 0.79)
    axis.legend(frameon=False, loc="lower right")
    panel_label(axis, "A")

    axis = axes[1]
    for database in ("mimic4", "eicu"):
        ensemble = [
            formal[database]["ensemble"]["ordinal_temporal"]["metrics"][
                "macro_auc"
            ][str(hour)]
            for hour in (24, 48, 72)
        ]
        seed_sd = [
            formal[database]["configurations"]["ordinal_temporal"]["aggregate"][
                f"auc_{hour}"
            ]["sd"]
            for hour in (24, 48, 72)
        ]
        axis.errorbar(
            (24, 48, 72),
            ensemble,
            yerr=seed_sd,
            marker="o",
            markersize=4,
            capsize=2,
            linewidth=0.9,
            color=COLORS[database],
            label=LABELS[database],
        )
        for hour, value in zip((24, 48, 72), ensemble):
            axis.text(hour, value + 0.006, f"{value:.3f}", ha="center", fontsize=6.8)
    axis.set_title("DG-GRU ordinal + temporal")
    axis.set_xlabel("Landmark (hours)")
    axis.set_xticks((24, 48, 72))
    axis.legend(frameon=False, loc="lower right")
    panel_label(axis, "B")
    figure.tight_layout()
    save_figure(figure, "figure3_discrimination")


def reliability(probability: np.ndarray, y: np.ndarray, class_index: int) -> tuple:
    observed = y == class_index
    predicted = probability[:, class_index]
    edges = np.linspace(0, 1, 11)
    x_values, y_values, sizes = [], [], []
    for lower, upper in zip(edges[:-1], edges[1:]):
        selected = (
            (predicted >= lower) & (predicted <= upper)
            if upper == 1
            else (predicted >= lower) & (predicted < upper)
        )
        if selected.sum() >= 10:
            x_values.append(float(predicted[selected].mean()))
            y_values.append(float(observed[selected].mean()))
            sizes.append(int(selected.sum()))
    return np.asarray(x_values), np.asarray(y_values), np.asarray(sizes)


def figure_calibration() -> None:
    figure, axes = plt.subplots(2, 3, figsize=(7.2, 4.5), sharex=True, sharey=True)
    class_names = ("Home", "Facility", "Death")
    for row, database in enumerate(("mimic4", "eicu")):
        xgb_bundle = np.load(
            RESULTS / "shared_fold" / f"{database}_shared_fold.npz"
        )
        dg_bundle = np.load(
            RESULTS / "formal_summary"
            / f"{database}_ordinal_temporal_ensemble.npz"
        )
        for column in range(3):
            axis = axes[row, column]
            axis.plot((0, 1), (0, 1), color="#8C8C8C", linewidth=0.7, linestyle="--")
            for probability, y, name, color in (
                (xgb_bundle["p72"], xgb_bundle["y3"], "XGBoost", "#5B6770"),
                (
                    dg_bundle["probability"][:, 2, :],
                    dg_bundle["y"],
                    "DG-GRU",
                    COLORS[database],
                ),
            ):
                x_values, y_values, sizes = reliability(probability, y, column)
                axis.plot(x_values, y_values, marker="o", markersize=2.8, color=color, label=name)
            axis.set_title(f"{LABELS[database]} · {class_names[column]}")
            axis.set_xlim(0, 0.85)
            axis.set_ylim(0, 0.85)
            if row == 1:
                axis.set_xlabel("Predicted probability")
            if column == 0:
                axis.set_ylabel("Observed proportion")
            if row == 0 and column == 0:
                axis.legend(frameon=False, loc="upper left")
    figure.tight_layout()
    save_figure(figure, "figure4_calibration")


def figure_stability() -> None:
    baseline = load_json(RESULTS / "shared_fold/summary.json")["databases"]
    formal = load_json(RESULTS / "formal_summary/summary.json")
    figure = plt.figure(figsize=(7.2, 5.8))
    grid = figure.add_gridspec(2, 3, height_ratios=(1.0, 1.0))
    axis = figure.add_subplot(grid[0, 0])
    methods = [
        ("XGBoost", "baseline"),
        ("DG-GRU", "ordinal_temporal"),
    ]
    x_positions = np.arange(2)
    for offset, database in ((-0.18, "mimic4"), (0.18, "eicu")):
        values = []
        errors = []
        for _, method in methods:
            if method == "baseline":
                record = baseline[database]["stability"]
                values.append(record["flip_24_72"] * 100)
                errors.append(
                    (
                        record["flip_24_72"] - record["flip_24_72_ci95"][0],
                        record["flip_24_72_ci95"][1] - record["flip_24_72"],
                    )
                )
            else:
                aggregate = formal[database]["configurations"][method]["aggregate"][
                    "flip_0_2"
                ]
                values.append(aggregate["mean"] * 100)
                errors.append((aggregate["sd"] * 100, aggregate["sd"] * 100))
        axis.bar(
            x_positions + offset,
            values,
            width=0.32,
            color=COLORS[database],
            label=LABELS[database],
        )
        axis.errorbar(
            x_positions + offset,
            values,
            yerr=np.asarray(errors).T,
            fmt="none",
            color="#2F2F2F",
            capsize=2,
            linewidth=0.8,
        )
    axis.set_xticks(x_positions, ("XGBoost", "DG-GRU"))
    axis.set_ylabel("24→72 h class flip (%)")
    axis.set_ylim(0, 34)
    axis.legend(frameon=False, loc="upper right")
    panel_label(axis, "A")

    axis = figure.add_subplot(grid[0, 1])
    thresholds = ("0.5", "0.6", "0.7")
    for database in ("mimic4", "eicu"):
        values = [
            baseline[database]["stability"]["confidence_24h"][threshold][
                "flip_24_72"
            ]
            * 100
            for threshold in thresholds
        ]
        axis.plot(thresholds, values, marker="o", color=COLORS[database], label=LABELS[database])
    axis.set_xlabel("24 h confidence threshold")
    axis.set_ylabel("24→72 h flip (%)")
    axis.set_ylim(0, 28)
    panel_label(axis, "B")

    axis = figure.add_subplot(grid[0, 2])
    for database in ("mimic4", "eicu"):
        aggregate = formal[database]["configurations"]["ordinal_temporal"]["aggregate"]
        points = (
            (
                baseline[database]["stability"]["flip_24_72"] * 100,
                baseline[database]["stability"]["tv_24_72"],
                "o",
                "XGB",
            ),
            (
                aggregate["flip_0_2"]["mean"] * 100,
                aggregate["tv_0_2"]["mean"],
                "s",
                "DG",
            ),
        )
        for flip, variation, marker, short_label in points:
            axis.scatter(
                flip,
                variation,
                color=COLORS[database],
                marker=marker,
                s=28,
                label=LABELS[database] if short_label == "XGB" else None,
            )
    axis.set_xlabel("Class flip (%)")
    axis.set_ylabel("Probability total variation")
    axis.set_xlim(6, 33)
    axis.set_ylim(0.04, 0.18)
    method_handles = [
        Line2D(
            [], [], marker="o", linestyle="None", color="#5B6770",
            markersize=4, label="XGBoost",
        ),
        Line2D(
            [], [], marker="s", linestyle="None", color="#5B6770",
            markersize=4, label="DG-GRU",
        ),
    ]
    database_handles = [
        Line2D(
            [], [], marker="o", linestyle="None", color=COLORS[database],
            markersize=4, label=LABELS[database],
        )
        for database in ("mimic4", "eicu")
    ]
    axis.legend(
        handles=method_handles + database_handles,
        frameon=False,
        loc="upper right",
        fontsize=6.4,
        ncol=2,
        handletextpad=0.3,
        columnspacing=0.8,
    )
    axis.text(
        -0.08,
        1.08,
        "C",
        transform=axis.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
        ha="left",
    )

    for column, database in enumerate(("mimic4", "eicu")):
        axis = figure.add_subplot(grid[1, column])
        matrix = np.asarray(
            baseline[database]["stability"]["transition_24_to_72_counts"],
            dtype=float,
        )
        matrix = matrix / matrix.sum(axis=1, keepdims=True) * 100
        image = axis.imshow(matrix, cmap="YlGnBu", vmin=0, vmax=100)
        for row in range(3):
            for col in range(3):
                axis.text(
                    col,
                    row,
                    f"{matrix[row, col]:.1f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if matrix[row, col] > 55 else "#222222",
                )
        axis.set_xticks(range(3), ("Home", "Facility", "Death"), rotation=20)
        axis.set_yticks(range(3), ("Home", "Facility", "Death"))
        axis.set_xlabel("Predicted at 72 h")
        axis.set_ylabel("Predicted at 24 h")
        axis.set_title(LABELS[database])
        for spine in axis.spines.values():
            spine.set_visible(False)
        if column == 0:
            panel_label(axis, "D")
    figure.subplots_adjust(
        left=0.07, right=0.87, bottom=0.12, top=0.94, wspace=0.55, hspace=0.55
    )
    color_axis = figure.add_axes((0.90, 0.12, 0.015, 0.33))
    figure.colorbar(image, cax=color_axis, label="Row-normalised transitions (%)")
    save_figure(figure, "figure5_stability")


def figure_ablation() -> None:
    formal = load_json(RESULTS / "formal_summary/summary.json")
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    contracts = (
        ("temporal_vs_none_ordinal", "Temporal − none\n(ordinal)"),
        ("temporal_vs_none_nominal", "Temporal − none\n(nominal)"),
        ("ordinal_vs_nominal_no_temporal", "Ordinal − nominal\n(no penalty)"),
        ("ordinal_vs_nominal_temporal", "Ordinal − nominal\n(with penalty)"),
    )
    labels = [item[1] for item in contracts]
    y_positions = np.arange(len(contracts))
    for axis, metric, title, scale in (
        (axes[0], "auc_72", "Δ 72 h macro-AUC", 1),
        (axes[1], "flip_0_2", "Δ 24→72 h flip", 100),
    ):
        for offset, database in ((-0.17, "mimic4"), (0.17, "eicu")):
            means, errors = [], []
            for key, _ in contracts:
                record = formal[database]["paired_contrasts"][key][metric]
                means.append(record["difference_mean"] * scale)
                errors.append(record["difference_sd"] * scale)
            axis.errorbar(
                np.asarray(means),
                y_positions + offset,
                xerr=errors,
                fmt="o",
                markersize=4,
                capsize=2,
                linewidth=0.8,
                color=COLORS[database],
                label=LABELS[database],
            )
        axis.axvline(0, color="#777777", linewidth=0.7, linestyle="--")
        axis.set_yticks(y_positions, labels)
        axis.set_xlabel(title + (" points" if metric == "flip_0_2" else ""))
        axis.invert_yaxis()
    axes[1].tick_params(axis="y", labelleft=False)
    panel_label(axes[0], "A")
    panel_label(axes[1], "B")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        legend_labels,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )
    figure.subplots_adjust(left=0.18, right=0.98, top=0.91, bottom=0.22, wspace=0.45)
    save_figure(figure, "figure6_ablation")


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines) + "\n"


def decision_points(
    y: np.ndarray, probability: np.ndarray, class_index: int
) -> list[list[str]]:
    observed = y == class_index
    prevalence = float(observed.mean())
    rows = []
    for threshold in np.arange(0.05, 0.61, 0.05):
        selected = probability[:, class_index] >= threshold
        true_positive = int((selected & observed).sum())
        false_positive = int((selected & ~observed).sum())
        net_benefit = (
            true_positive / len(y)
            - false_positive / len(y) * threshold / (1 - threshold)
        )
        treat_all = prevalence - (1 - prevalence) * threshold / (1 - threshold)
        rows.append(
            [
                f"{threshold:.2f}",
                fixed(net_benefit),
                fixed(treat_all),
                fixed(float(selected.mean())),
            ]
        )
    return rows


def calibration_slope_intercept(y: np.ndarray, probability: np.ndarray) -> list[list[str]]:
    rows = []
    for class_index, class_name in enumerate(("Home", "Facility", "Death")):
        observed = (y == class_index).astype(int)
        predicted = np.clip(probability[:, class_index], 1e-6, 1 - 1e-6)
        logit = np.log(predicted / (1 - predicted)).reshape(-1, 1)
        model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        model.fit(logit, observed)
        rows.append(
            [
                class_name,
                fixed(float(model.coef_[0, 0])),
                fixed(float(model.intercept_[0])),
            ]
        )
    return rows


def build_tables() -> None:
    xgb = load_json(RESULTS / "landmarks/xgb_summary.json")
    shared = load_json(RESULTS / "shared_fold/summary.json")
    clinical = load_json(RESULTS / "clinical_flips/summary.json")
    stability = load_json(RESULTS / "stability_v3/primary_stability.json")
    formal = load_json(RESULTS / "formal_summary/summary.json")
    transport = load_json(RESULTS / "transport/summary.json")
    sensitivity = load_json(RESULTS / "sensitivity/sensitivity_summary.json")
    sensitivity_v2 = load_json(
        RESULTS / "sensitivity_v2/sensitivity_v2_summary.json"
    )

    # Table 1: compact cohort description by three-class outcome.
    rows = []
    for database, folder in (("mimic4", "MIMIC-IV"), ("eicu", "eICU-CRD")):
        frame = pd.read_parquet(ROOT / f"data/processed/{database}/design_all.parquet")
        frame = frame.loc[frame["y3"].notna()]
        for outcome, outcome_name in enumerate(("Home", "Facility", "In-hospital death")):
            subset = frame.loc[frame["y3"] == outcome]
            rows.append(
                [
                    folder,
                    outcome_name,
                    f"{len(subset):,}",
                    f"{subset['age'].median():.0f} ({subset['age'].quantile(.25):.0f}–{subset['age'].quantile(.75):.0f})",
                    fixed(float(subset["male"].mean() * 100)),
                    (
                        f"{fixed(float(subset['lac_max'].median()))} "
                        f"({fixed(float(subset['lac_max'].quantile(.25)))}–"
                        f"{fixed(float(subset['lac_max'].quantile(.75)))})"
                    ),
                ]
            )
    table1 = markdown_table(
        ["Database", "Outcome", "n", "Age, median (IQR)", "Male, %", "Peak lactate, median (IQR)"],
        rows,
    )
    (TABLES / "table1_cohort.md").write_text(table1, encoding="utf-8")

    # Table 2: landmark populations and outcome distribution.
    rows = []
    for database, label, duration in (
        ("mimic4", "MIMIC-IV", "los"),
        ("eicu", "eICU-CRD", "los_h"),
    ):
        frame = pd.read_parquet(ROOT / f"data/processed/{database}/design_all.parquet")
        for hour in (24, 48, 72):
            subset = frame.loc[frame["y3"].notna() & (frame[duration] >= hour)]
            rows.append(
                [
                    label,
                    str(hour),
                    f"{len(subset):,}",
                    fixed(float((subset["y3"] == 0).mean() * 100)),
                    fixed(float((subset["y3"] == 1).mean() * 100)),
                    fixed(float((subset["y3"] == 2).mean() * 100)),
                ]
            )
    table2 = markdown_table(
        ["Database", "Landmark, h", "n", "Home, %", "Facility, %", "Death, %"],
        rows,
    )
    (TABLES / "table2_landmarks.md").write_text(table2, encoding="utf-8")

    # Table 3: discrimination.
    rows = []
    for database, label in (("mimic4", "MIMIC-IV"), ("eicu", "eICU-CRD")):
        record = next(item for item in xgb if item["database"] == database)
        for definition in ("3class", "4class"):
            values = [
                fixed(record["definitions"][definition][str(hour)]["macro_auc"])
                for hour in (24, 48, 72)
            ]
            rows.append(
                [label, f"XGBoost {definition}, landmark-specific cohort", *values]
            )
        shared_values = [
            fixed(shared["databases"][database]["metrics"][str(hour)]["macro_auc"])
            for hour in (24, 48, 72)
        ]
        rows.append(
            [label, "XGBoost 3class, shared complete cohort", *shared_values]
        )
        for configuration, method in (
            ("ordinal_notemporal", "DG-GRU ordinal"),
            ("ordinal_temporal", "DG-GRU ordinal + temporal"),
            ("nominal_notemporal", "DG-GRU nominal"),
            ("nominal_temporal", "DG-GRU nominal + temporal"),
        ):
            values = [
                fixed(
                    formal[database]['ensemble'][configuration]['metrics'][
                        'macro_auc'
                    ][str(hour)]
                )
                for hour in (24, 48, 72)
            ]
            rows.append([label, method, *values])
    table3 = markdown_table(
        ["Database", "Model", "24 h", "48 h", "72 h"],
        rows,
    )
    (TABLES / "table3_discrimination.md").write_text(table3, encoding="utf-8")

    # Table 4: stability.
    rows = []
    for database, label in (("mimic4", "MIMIC-IV"), ("eicu", "eICU-CRD")):
        for method, record in (
            ("XGBoost", shared["databases"][database]["stability"]),
            (
                "DG-GRU ordinal + temporal",
                formal[database]["ensemble"]["ordinal_temporal"]["stability"],
            ),
        ):
            if method == "XGBoost":
                flip = {
                    "estimate": record["flip_24_72"],
                    "ci95": record["flip_24_72_ci95"],
                }
                tv = record["tv_24_72"]
                confidence = {
                    "coverage": record["confidence_24h"]["0.6"]["coverage"],
                    "flip": record["confidence_24h"]["0.6"]["flip_24_72"],
                }
            else:
                flip = {
                    "estimate": record["flip_0_2"],
                    "ci95": record["flip_0_2_ci95"],
                }
                tv = record["tv_0_2"]
                confidence = record["confidence_24h"]["0.6"]
            rows.append(
                [
                    label,
                    method,
                    f"{fixed(flip['estimate'] * 100)} ({fixed(flip['ci95'][0] * 100)}–{fixed(flip['ci95'][1] * 100)})"
                    if flip["ci95"][0] is not None
                    else fixed(flip['estimate'] * 100),
                    fixed(tv),
                    fixed(confidence['coverage'] * 100),
                    fixed(confidence['flip'] * 100),
                ]
            )
    table4 = markdown_table(
        [
            "Database",
            "Model",
            "24→72 h flip, % (95% CI)",
            "Total variation",
            "24 h confidence ≥0.6, coverage %",
            "Flip within stratum, %",
        ],
        rows,
    )
    (TABLES / "table4_stability.md").write_text(table4, encoding="utf-8")

    # Table 5: paired contrasts.
    rows = []
    for database, label in (("mimic4", "MIMIC-IV"), ("eicu", "eICU-CRD")):
        for key, description in (
            ("temporal_vs_none_ordinal", "Temporal vs none, ordinal"),
            ("temporal_vs_none_nominal", "Temporal vs none, nominal"),
            ("ordinal_vs_nominal_no_temporal", "Ordinal vs nominal, no penalty"),
            ("ordinal_vs_nominal_temporal", "Ordinal vs nominal, temporal"),
        ):
            record = formal[database]["paired_contrasts"][key]
            rows.append(
                [
                    label,
                    description,
                    f"{signed_fixed(record['auc_72']['difference_mean'])} ± {fixed(record['auc_72']['difference_sd'])}",
                    f"{signed_fixed(record['flip_0_2']['difference_mean'] * 100, 2)} ± {fixed(record['flip_0_2']['difference_sd'] * 100, 2)}",
                    f"{signed_fixed(record['tv_0_2']['difference_mean'])} ± {fixed(record['tv_0_2']['difference_sd'])}",
                ]
            )
    supplementary_contrasts = markdown_table(
        [
            "Database",
            "Contrast",
            "Δ 72 h macro-AUC, mean ± SD",
            "Δ 24→72 h flip, percentage points",
            "Δ total variation",
        ],
        rows,
    )
    (TABLES / "supplementary_table_s2_contrasts.md").write_text(
        supplementary_contrasts, encoding="utf-8"
    )

    # Supplementary Table S3: discrimination, calibration, and probability scores.
    metric_rows = []
    for database, label in (("mimic4", "MIMIC-IV"), ("eicu", "eICU-CRD")):
        for configuration, method in (
            ("ordinal_notemporal", "DG-GRU ordinal, no temporal penalty"),
            ("ordinal_temporal", "DG-GRU ordinal + temporal penalty"),
            ("nominal_notemporal", "DG-GRU nominal, no temporal penalty"),
            ("nominal_temporal", "DG-GRU nominal + temporal penalty"),
        ):
            metrics = formal[database]["ensemble"][configuration]["metrics"]
            metric_rows.append(
                [
                    label,
                    method,
                    fixed(metrics['macro_auc']['72']),
                    fixed(metrics['accuracy']['72']),
                    fixed(metrics['log_loss']['72']),
                    fixed(metrics['brier']['72']),
                    fixed(
                        formal[database]['ensemble'][configuration][
                            'calibration'
                        ]['72']['top_label']['ece_10bin']
                    ),
                ]
            )
    metric_table = markdown_table(
        [
            "Database",
            "Model",
            "72 h macro-AUC",
            "Accuracy",
            "Log loss",
            "Brier score",
            "Top-label ECE",
        ],
        metric_rows,
    )
    (TABLES / "supplementary_table_s3_model_metrics.md").write_text(
        metric_table, encoding="utf-8"
    )

    # Supplementary Table S4: one-versus-rest calibration summaries.
    calibration_rows = []
    for database, label in (("mimic4", "MIMIC-IV"), ("eicu", "eICU-CRD")):
        xgb_bundle = np.load(
            RESULTS / "shared_fold" / f"{database}_shared_fold.npz"
        )
        dg_bundle = np.load(
            RESULTS / "formal_summary"
            / f"{database}_ordinal_temporal_ensemble.npz"
        )
        for model_name, probability, y in (
            ("XGBoost", xgb_bundle["p72"], xgb_bundle["y3"]),
            (
                "DG-GRU ordinal + temporal",
                dg_bundle["probability"][:, 2, :],
                dg_bundle["y"],
            ),
        ):
            for class_row in calibration_slope_intercept(y, probability):
                calibration_rows.append([label, model_name, *class_row])
    calibration_table = markdown_table(
        ["Database", "Model", "Outcome", "Calibration slope", "Intercept"],
        calibration_rows,
    )
    (TABLES / "supplementary_table_s4_calibration.md").write_text(
        calibration_table, encoding="utf-8"
    )

    # Table 5: prespecified and source-recovered sensitivity analyses.
    sensitivity_rows = []
    for record in sensitivity["six_vs_48"]:
        subset = record["six_to_48h_complete_subset"]
        sensitivity_rows.append(
            [
                "MIMIC-IV" if record["database"] == "mimic4" else "eICU-CRD",
                "6 to 48 h landmark; available early predictors",
                f"{subset['n']:,}",
                fixed(subset['flip'] * 100),
                fixed(subset['tv']),
                f"{fixed(subset['auc_6'])}/{fixed(subset['auc_48'])}",
            ]
        )
    for database, label in (("mimic4", "MIMIC-IV"), ("eicu", "eICU-CRD")):
        record = next(item for item in xgb if item["database"] == database)
        sensitivity_rows.append(
            [
                label,
                "Rehabilitation separated (4 classes), 72 h XGBoost",
                f"{record['definitions']['4class']['72']['n']:,}",
                "Not applicable",
                "Not applicable",
                fixed(record['definitions']['4class']['72']['macro_auc']),
            ]
        )
    for record in sensitivity["full_vs_complete"]:
        label = "MIMIC-IV" if record["database"] == "mimic4" else "eICU-CRD"
        for hour in ("24", "48"):
            full = record["landmarks"][hour]["full_landmark"]
            complete = record["landmarks"][hour]["complete_window_subset"]
            sensitivity_rows.append(
                [
                    label,
                    f"Full {hour} h population vs complete-window subset",
                    f"{full['n']:,} vs {complete['n']:,}",
                    "Not applicable",
                    "Not applicable",
                    f"{fixed(full['macro_auc'])} vs {fixed(complete['macro_auc'])}",
                ]
            )
    for database, label in (("mimic4", "MIMIC-IV"), ("eicu", "eICU-CRD")):
        record = sensitivity_v2["icu_death_exclusion"][database]
        complete = record["complete_three_landmark_subset"]
        sensitivity_rows.append(
            [
                label,
                "ICU-death exclusion, retrained XGBoost",
                f"{complete['n']:,}",
                fixed(complete['flip_24_72'] * 100),
                fixed(complete['tv_24_72']),
                fixed(record['landmarks']['72']['macro_auc']),
            ]
        )
    for key, label in (
        ("mimic4_icd9", "MIMIC-IV ICD-9 sepsis coding"),
        ("mimic4_icd10", "MIMIC-IV ICD-10 sepsis coding"),
    ):
        record = sensitivity_v2["icd_era"][key]
        complete = record["complete_three_landmark_subset"]
        sensitivity_rows.append(
            [
                "MIMIC-IV",
                label,
                f"{complete['n']:,}",
                fixed(complete['flip_24_72'] * 100),
                fixed(complete['tv_24_72']),
                fixed(record['landmarks']['72']['macro_auc']),
            ]
        )
    sensitivity_rows.append(
        [
            "eICU-CRD",
            "ICD-9 vs ICD-10 era stratification",
            "Not applicable",
            "Not applicable",
            "Not applicable",
            "Harmonised ICD-9-style diagnosis field",
        ]
    )
    for database, label in (("mimic4", "MIMIC-IV"), ("eicu", "eICU-CRD")):
        record = sensitivity_v2["ordinal_order_reversal"][database][
            "reversed_order_restored"
        ]
        sensitivity_rows.append(
            [
                label,
                "Ordinal order reversed; probability axis restored",
                f"{stability[database]['n']:,}",
                fixed(record['flip_24_72']['mean'] * 100),
                fixed(record['tv_24_72']['mean']),
                fixed(record['auc_72']['mean']),
            ]
        )
    mimic3 = sensitivity_v2["mimic3_comparison"]["complete_three_landmark_subset"]
    sensitivity_rows.append(
        [
            "MIMIC-III",
            "Same-institution different-era XGBoost comparison",
            f"{mimic3['n']:,}",
            fixed(mimic3['flip_24_72'] * 100),
            fixed(mimic3['tv_24_72']),
            fixed(mimic3['macro_auc']['72']),
        ]
    )
    table5 = markdown_table(
        [
            "Database",
            "Analysis",
            "n",
            "Flip, %",
            "Total variation",
            "Macro-AUC",
        ],
        sensitivity_rows,
    )
    (TABLES / "table5_sensitivity.md").write_text(table5, encoding="utf-8")

    # Table 6: fixed-weight cross-database transport performance.
    transport_rows = []
    for direction, label in (
        ("mimic4_to_eicu", "MIMIC-IV -> eICU-CRD"),
        ("eicu_to_mimic4", "eICU-CRD -> MIMIC-IV"),
    ):
        for hour in (24, 48, 72):
            record = transport["transport"][direction]["landmarks"][str(hour)][
                "target_all_reaching_landmark"
            ]
            transport_rows.append(
                [
                    label,
                    str(hour),
                    f"{record['n']:,}",
                    fixed(record["macro_auc"]),
                    fixed(record["accuracy"]),
                    fixed(record["log_loss"]),
                    fixed(record["brier_multiclass"]),
                    fixed(record["ece_top_label_10bin"]),
                ]
            )
    table6 = markdown_table(
        [
            "Transport direction",
            "Landmark, h",
            "n",
            "Macro-AUC",
            "Accuracy",
            "Log loss",
            "Brier score",
            "Top-label ECE",
        ],
        transport_rows,
    )
    (TABLES / "table6_transport.md").write_text(table6, encoding="utf-8")

    # Supplementary Table S1: decision-curve coordinates at 72 h.
    decision_rows = []
    for database, label in (("mimic4", "MIMIC-IV"), ("eicu", "eICU-CRD")):
        xgb_bundle = np.load(
            RESULTS / "shared_fold" / f"{database}_shared_fold.npz"
        )
        dg_bundle = np.load(
            RESULTS / "formal_summary"
            / f"{database}_ordinal_temporal_ensemble.npz"
        )
        for outcome_name, class_index in (("Home", 0), ("Death", 2)):
            for model_name, probability, y in (
                (
                    "XGBoost",
                    xgb_bundle["p72"],
                    xgb_bundle["y3"],
                ),
                (
                    "DG-GRU ordinal + temporal",
                    dg_bundle["probability"][:, 2, :],
                    dg_bundle["y"],
                ),
            ):
                for point in decision_points(y, probability, class_index):
                    decision_rows.append(
                        [label, model_name, outcome_name, *point]
                    )
    decision_table = markdown_table(
        [
            "Database",
            "Model",
            "Outcome",
            "Threshold",
            "Model net benefit",
            "Treat-all net benefit",
            "Selected fraction",
        ],
        decision_rows,
    )
    (TABLES / "supplementary_table_s1_decision_curves.md").write_text(
        decision_table, encoding="utf-8"
    )

    # Supplementary Table S6: clinically meaningful transitions and correctness.
    clinical_rows = []
    model_specs = (
        ("xgboost", "XGBoost, shared-fold"),
        ("ordinal_notemporal", "Ordinal DG-GRU, no temporal penalty"),
        ("ordinal_temporal", "Ordinal DG-GRU, temporal penalty"),
    )
    for database, label in (("mimic4", "MIMIC-IV"), ("eicu", "eICU-CRD")):
        clinical_record = clinical["databases"][database]
        for model_key, model_label in model_specs:
            model = clinical_record["models"][model_key]
            correctness = model["correctness_relative_to_final_outcome"]["all"]
            true_probability = model["true_class_probability"]
            major = model["clinical_major_upgrades"]
            clinical_rows.append(
                [
                    label,
                    model_label,
                    fixed(100 * model["flip_24_72"]["estimate"], 2),
                    fixed(100 * major["major_upgrade_to_death"]["estimate"], 2),
                    fixed(100 * major["home_to_facility"]["estimate"], 2),
                    (
                        f"{fixed(100 * correctness['gain']['estimate'], 2)} / "
                        f"{fixed(100 * correctness['loss']['estimate'], 2)} / "
                        f"{fixed(100 * correctness['tie']['estimate'], 2)}"
                    ),
                    signed_fixed(
                        true_probability["among_flips"]["pt_true_change_mean"][
                            "estimate"
                        ],
                        2,
                    ),
                    fixed(
                        100
                        * true_probability["among_flips"]["improved"]["estimate"],
                        2,
                    ),
                    "",
                ]
            )
        comparison = clinical_record["temporal_vs_notemporal"]
        decomposition = comparison["flip_category_decomposition"]
        death_subgroup = comparison["true_death_subgroup"]
        temporal_probability = clinical_record["models"]["ordinal_temporal"][
            "true_class_probability"
        ]["among_flips"]
        no_temporal_probability = clinical_record["models"]["ordinal_notemporal"][
            "true_class_probability"
        ]["among_flips"]
        clinical_rows.append(
            [
                label,
                "Temporal minus no temporal (pp)",
                signed_fixed(100 * comparison["flip_rate_change"]["estimate"], 2),
                signed_fixed(
                    100
                    * (
                        clinical_record["models"]["ordinal_temporal"][
                            "clinical_major_upgrades"
                        ]["major_upgrade_to_death"]["estimate"]
                        - clinical_record["models"]["ordinal_notemporal"][
                            "clinical_major_upgrades"
                        ]["major_upgrade_to_death"]["estimate"]
                    ),
                    2,
                ),
                signed_fixed(
                    100
                    * (
                        clinical_record["models"]["ordinal_temporal"][
                            "clinical_major_upgrades"
                        ]["home_to_facility"]["estimate"]
                        - clinical_record["models"]["ordinal_notemporal"][
                            "clinical_major_upgrades"
                        ]["home_to_facility"]["estimate"]
                    ),
                    2,
                ),
                (
                    f"{signed_fixed(100 * decomposition['correct_upgrade']['change_temporal_minus_notemporal']['estimate'], 2)} / "
                    f"{signed_fixed(100 * decomposition['incorrect_upgrade']['change_temporal_minus_notemporal']['estimate'], 2)} / "
                    f"{signed_fixed(100 * decomposition['correct_downgrade']['change_temporal_minus_notemporal']['estimate'], 2)} / "
                    f"{signed_fixed(100 * decomposition['incorrect_downgrade']['change_temporal_minus_notemporal']['estimate'], 2)} "
                    "(closer-higher / farther-higher / closer-lower / farther-lower)"
                ),
                (
                    signed_fixed(
                        100
                        * (
                            temporal_probability["pt_true_change_mean"]["estimate"]
                            - no_temporal_probability["pt_true_change_mean"][
                                "estimate"
                            ]
                        ),
                        2,
                    )
                ),
                signed_fixed(
                    100
                    * (
                        temporal_probability["improved"]["estimate"]
                        - no_temporal_probability["improved"]["estimate"]
                    ),
                    2,
                ),
                (
                    f"{signed_fixed(100 * death_subgroup['new_upgrade_to_death']['change_temporal_minus_notemporal']['estimate'], 2)} / "
                    f"{signed_fixed(100 * death_subgroup['missed_death_at72']['change_temporal_minus_notemporal']['estimate'], 2)}"
                ),
            ]
        )
    clinical_table = markdown_table(
        [
            "Database",
            "Analysis",
            "Class change, %",
            "Non-death to death conversion, % / pp",
            "home to facility, % / pp",
            "Closer / farther / equal among all stays, %; temporal category change, pp",
            "Flip-level mean true-class probability change",
            "Flip-level true-class probability improved, % / pp",
            "Death subgroup change, pp (new conversion / death not predicted)",
        ],
        clinical_rows,
    )
    (TABLES / "supplementary_table_s6_clinical_flips.md").write_text(
        clinical_table, encoding="utf-8"
    )

    # Supplementary Table S7: fold-level shared-fold XGBoost metrics.
    shared_rows = []
    for database, label in (("mimic4", "MIMIC-IV"), ("eicu", "eICU-CRD")):
        for hour in (24, 48, 72):
            for record in shared["databases"][database]["fold_metrics"][str(hour)]:
                shared_rows.append(
                    [
                        label,
                        str(hour),
                        str(record["fold"] + 1),
                        f"{record['n_test']:,}",
                        fixed(record["macro_auc"]),
                        fixed(record["class_auc"]["0"]),
                        fixed(record["class_auc"]["1"]),
                        fixed(record["class_auc"]["2"]),
                        fixed(record["log_loss"]),
                        fixed(record["brier"]),
                        fixed(record["ece_10bin"]),
                    ]
                )
    shared_table = markdown_table(
        [
            "Database",
            "Landmark, h",
            "Fold",
            "n",
            "Macro-AUC",
            "Home AUC",
            "Facility AUC",
            "Death AUC",
            "Log loss",
            "Brier score",
            "Top-label ECE",
        ],
        shared_rows,
    )
    (TABLES / "supplementary_table_s7_shared_fold.md").write_text(
        shared_table, encoding="utf-8"
    )


def main() -> None:
    configure_style()
    figure_cohort_outcomes()
    figure_architecture()
    figure_performance()
    figure_calibration()
    figure_stability()
    figure_ablation()
    build_tables()
    try:
        from sepsis.extend_three_database_outputs import (
            build as build_three_database,
        )
    except ModuleNotFoundError:
        from extend_three_database_outputs import build as build_three_database

    build_three_database()
    print(f"Outputs written to {FIGURES} and {TABLES}", flush=True)


if __name__ == "__main__":
    main()
