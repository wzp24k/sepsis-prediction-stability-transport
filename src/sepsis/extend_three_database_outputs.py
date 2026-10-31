# -*- coding: utf-8 -*-
"""Extend the frozen two-database outputs to the final three-database design."""

import json
import os
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(
    os.environ.get("SEPSIS_PROJECT_ROOT", Path(__file__).resolve().parents[2])
)
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
TABLES = RESULTS / "tables"
FIGURES.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)

COLORS = {
    "mimic4": "#1F6F78",
    "eicu": "#C26D2E",
    "mimic3": "#6B6F9C",
}
LABELS = {
    "mimic4": "MIMIC-IV",
    "eicu": "eICU-CRD",
    "mimic3": "MIMIC-III",
}
DATABASES = ("mimic4", "eicu", "mimic3")
DB_LABELS = tuple(LABELS[item] for item in DATABASES)
DIRECTIONS = (
    "mimic4_to_eicu",
    "eicu_to_mimic4",
    "mimic4_to_mimic3",
    "mimic3_to_mimic4",
    "mimic3_to_eicu",
    "eicu_to_mimic3",
)


def direction_label(direction: str) -> str:
    source, target = direction.split("_to_", 1)
    return f"{LABELS[source]} to {LABELS[target]}"


def fixed(value: float | None, digits: int | None = None) -> str:
    if value is None or pd.isna(value):
        return ""
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


def signed_fixed(value: float | None, digits: int | None = None) -> str:
    if value is None or pd.isna(value):
        return ""
    rendered = fixed(value, digits)
    rounded = Decimal(rendered)
    if rounded == 0:
        return fixed(0, digits)
    prefix = "+" if rounded > 0 else ""
    return prefix + rendered


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8.5,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 6.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


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


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    output = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    output.extend("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows)
    return "\n".join(output) + "\n"


def cohort_frames() -> dict[str, pd.DataFrame]:
    return {
        database: pd.read_parquet(
            ROOT / "data" / "processed" / database / "design_all.parquet"
        )
        for database in DATABASES
    }


def duration_column(database: str) -> str:
    return "los_h" if database == "eicu" else "los"


def figure1_three_databases() -> None:
    frames = cohort_frames()
    figure, axes = plt.subplots(
        1, 2, figsize=(7.4, 3.4), gridspec_kw={"width_ratios": [1.5, 1.0]}
    )
    axis = axes[0]
    axis.set_xlim(0, 12)
    axis.set_ylim(0, 10)
    axis.axis("off")
    for index, database in enumerate(DATABASES):
        frame = frames[database]
        duration = duration_column(database)
        eligible = int((frame[duration] >= 72).sum())
        complete = int((frame["y3"].notna() & (frame[duration] >= 72)).sum())
        x = 0.2 + index * 3.95
        boxes = (
            (8.0, f"{LABELS[database]}\nICU stays, {len(frame):,}"),
            (5.5, f"ICU stay >=72 h\n{eligible:,}"),
            (3.0, f"Mapped outcome\n{complete:,}"),
        )
        for y, text in boxes:
            axis.add_patch(
                FancyBboxPatch(
                    (x, y),
                    3.6,
                    1.25,
                    boxstyle="round,pad=0.03,rounding_size=0.10",
                    linewidth=0.8,
                    edgecolor="#3D3D3D",
                    facecolor="white",
                )
            )
            axis.text(
                x + 1.8,
                y + 0.63,
                text,
                ha="center",
                va="center",
                fontsize=7.2,
            )
        for y in (8.0, 5.5):
            axis.add_patch(
                FancyArrowPatch(
                    (x + 1.8, y - 0.08),
                    (x + 1.8, y - 1.05),
                    arrowstyle="-|>",
                    mutation_scale=9,
                    linewidth=0.8,
                    color="#3D3D3D",
                )
            )
    axis.text(
        0.2,
        9.65,
        "Patient-grouped replication; hospital-held-out eICU validation",
        fontsize=8,
        fontweight="bold",
    )
    panel_label(axis, "A")

    axis = axes[1]
    positions = np.arange(3)
    width = 0.24
    class_names = ("Home", "Facility", "Death")
    for offset, database in zip(
        (-width, 0, width), DATABASES
    ):
        frame = frames[database]
        duration = duration_column(database)
        subset = frame.loc[
            frame["y3"].notna() & (frame[duration] >= 72)
        ]
        values = np.asarray(
            [(subset["y3"] == index).mean() * 100 for index in range(3)]
        )
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
                value + 0.8,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=6.2,
            )
    axis.set_xticks(positions, class_names, rotation=15, ha="right")
    axis.set_ylabel("ICU stays (%)")
    axis.set_ylim(0, 64)
    axis.legend(frameon=False, loc="upper right")
    panel_label(axis, "B")
    figure.tight_layout()
    save_figure(figure, "figure1_cohort_outcome")


def figure3_three_databases() -> None:
    shared = load_json(RESULTS / "shared_fold" / "summary.json")
    hospital = load_json(
        RESULTS / "eicu_hospital_validation" / "summary.json"
    )
    mimic3 = load_json(RESULTS / "mimic3_shared" / "summary.json")
    formal = load_json(RESULTS / "formal_summary" / "summary.json")
    figure, axes = plt.subplots(1, 2, figsize=(7.4, 3.3), sharey=True)
    hours = np.asarray((24, 48, 72))

    axis = axes[0]
    label_offsets = {
        "mimic4": (0, 5),
        "eicu": (0, 5),
        "mimic3": (0, -11),
    }
    for database in DATABASES:
        if database == "mimic3":
            values = [
                mimic3["discrimination"]["macro_auc"][str(hour)]
                for hour in hours
            ]
            style = "-"
        else:
            values = [
                shared["databases"][database]["metrics"][str(hour)][
                    "macro_auc"
                ]
                for hour in hours
            ]
            style = "-"
        axis.plot(
            hours,
            values,
            style,
            marker="o",
            markersize=4,
            linewidth=1.0,
            color=COLORS[database],
            label=LABELS[database],
        )
        for hour, value in zip(hours, values):
            axis.annotate(
                f"{value:.3f}",
                (hour, value),
                xytext=label_offsets[database],
                textcoords="offset points",
                ha="center",
                fontsize=6.4,
            )
    hospital_values = [
        hospital["metrics"][str(hour)]["macro_auc"] for hour in hours
    ]
    axis.plot(
        hours,
        hospital_values,
        linestyle="--",
        marker="s",
        markersize=3.8,
        linewidth=0.9,
        color="#7B4F2C",
        label="eICU hospital-held-out",
    )
    for hour, value in zip(hours, hospital_values):
        axis.text(
            hour,
            value - 0.014,
            f"{value:.3f}",
            ha="center",
            fontsize=6.2,
            color="#7B4F2C",
        )
    axis.set_title("XGBoost by validation design")
    axis.set_xlabel("Landmark (hours)")
    axis.set_ylabel("Macro-AUC")
    axis.set_xticks(hours)
    axis.set_ylim(0.63, 0.78)
    axis.legend(frameon=False, loc="lower right", ncol=2)
    panel_label(axis, "A")

    axis = axes[1]
    for database in ("mimic4", "eicu"):
        values = [
            formal[database]["ensemble"]["ordinal_temporal"]["metrics"][
                "macro_auc"
            ][str(hour)]
            for hour in hours
        ]
        errors = [
            formal[database]["configurations"]["ordinal_temporal"]["aggregate"][
                f"auc_{hour}"
            ]["sd"]
            for hour in hours
        ]
        axis.errorbar(
            hours,
            values,
            yerr=errors,
            marker="o",
            markersize=4,
            capsize=2,
            linewidth=0.9,
            color=COLORS[database],
            label=LABELS[database],
        )
        for hour, value in zip(hours, values):
            axis.text(
                hour,
                value + 0.007,
                f"{value:.3f}",
                ha="center",
                fontsize=6.4,
            )
    axis.set_title("DG-GRU ordinal + temporal")
    axis.set_xlabel("Landmark (hours)")
    axis.set_xticks(hours)
    axis.legend(frameon=False, loc="lower right")
    panel_label(axis, "B")
    figure.tight_layout()
    save_figure(figure, "figure3_discrimination")


def reliability(
    probability: np.ndarray, y: np.ndarray, class_index: int
) -> tuple[np.ndarray, np.ndarray]:
    observed = y == class_index
    predicted = probability[:, class_index]
    edges = np.linspace(0, 1, 11)
    x_values, y_values = [], []
    for lower, upper in zip(edges[:-1], edges[1:]):
        selected = (
            (predicted >= lower) & (predicted <= upper)
            if upper == 1
            else (predicted >= lower) & (predicted < upper)
        )
        if selected.sum() >= 10:
            x_values.append(float(predicted[selected].mean()))
            y_values.append(float(observed[selected].mean()))
    return np.asarray(x_values), np.asarray(y_values)


def figure4_three_databases() -> None:
    figure, axes = plt.subplots(
        3, 3, figsize=(7.2, 6.1), sharex=True, sharey=True
    )
    class_names = ("Home", "Facility", "Death")
    for row, database in enumerate(DATABASES):
        if database == "mimic3":
            bundle = np.load(RESULTS / "mimic3_shared" / "mimic3_shared.npz")
            probability = bundle["p72"]
            y = bundle["y3"]
            model_specs = (("XGBoost", probability, y, "#5B6770"),)
        else:
            bundle = np.load(
                RESULTS / "shared_fold" / f"{database}_shared_fold.npz"
            )
            dg_bundle = np.load(
                RESULTS
                / "formal_summary"
                / f"{database}_ordinal_temporal_ensemble.npz"
            )
            model_specs = (
                (
                    "XGBoost",
                    bundle["p72"],
                    bundle["y3"],
                    "#5B6770",
                ),
                (
                    "DG-GRU",
                    dg_bundle["probability"][:, 2, :],
                    dg_bundle["y"],
                    COLORS[database],
                ),
            )
        for column in range(3):
            axis = axes[row, column]
            axis.plot(
                (0, 1),
                (0, 1),
                color="#8C8C8C",
                linewidth=0.7,
                linestyle="--",
            )
            for name, probs, labels, color in model_specs:
                x_values, y_values = reliability(probs, labels, column)
                axis.plot(
                    x_values,
                    y_values,
                    marker="o",
                    markersize=2.6,
                    linewidth=0.9,
                    color=color,
                    label=name,
                )
            axis.set_title(
                f"{LABELS[database]} - {class_names[column]}", fontsize=8
            )
            axis.set_xlim(0, 0.85)
            axis.set_ylim(0, 0.85)
            if row == 2:
                axis.set_xlabel("Predicted probability")
            if column == 0:
                axis.set_ylabel("Observed proportion")
            if row == 0 and column == 0:
                axis.legend(frameon=False, loc="upper left")
    figure.tight_layout()
    save_figure(figure, "figure4_calibration")


def figure5_three_databases() -> None:
    shared = load_json(RESULTS / "shared_fold" / "summary.json")["databases"]
    trajectory = load_json(
        RESULTS / "trajectory_patterns" / "summary.json"
    )["databases"]
    hospital = load_json(
        RESULTS / "eicu_hospital_validation" / "summary.json"
    )
    mimic3 = load_json(RESULTS / "mimic3_shared" / "summary.json")
    formal = load_json(RESULTS / "formal_summary" / "summary.json")
    figure = plt.figure(figsize=(7.4, 8.3))
    grid = figure.add_gridspec(
        3, 3, height_ratios=(1.0, 1.0, 0.86)
    )

    axis = figure.add_subplot(grid[0, 0])
    x_positions = np.arange(2)
    for offset, database in zip((-0.22, 0, 0.22), DATABASES):
        values, errors = [], []
        if database == "mimic3":
            record = mimic3["stability"]["flip_24_72"]
            xgb_value = record["estimate"]
            xgb_error = (
                xgb_value - record["ci95"][0],
                record["ci95"][1] - xgb_value,
            )
        else:
            record = shared[database]["stability"]
            xgb_value = record["flip_24_72"]
            xgb_error = (
                xgb_value - record["flip_24_72_ci95"][0],
                record["flip_24_72_ci95"][1] - xgb_value,
            )
        values.append(xgb_value * 100)
        errors.append(np.asarray(xgb_error) * 100)
        if database != "mimic3":
            aggregate = formal[database]["configurations"]["ordinal_temporal"][
                "aggregate"
            ]["flip_0_2"]
            values.append(aggregate["mean"] * 100)
            errors.append(
                np.asarray((aggregate["sd"] * 100, aggregate["sd"] * 100))
            )
            positions = x_positions + offset
        else:
            positions = x_positions[:1] + offset
        axis.bar(
            positions,
            values,
            width=0.19,
            color=COLORS[database],
            label=LABELS[database],
        )
        axis.errorbar(
            positions,
            values,
            yerr=np.asarray(errors).T,
            fmt="none",
            color="#2F2F2F",
            capsize=2,
            linewidth=0.7,
        )
    axis.set_xticks(x_positions, ("XGBoost", "DG-GRU"))
    axis.set_ylabel("24-to-72 h class flip (%)")
    axis.set_ylim(0, 34)
    axis.legend(frameon=False, loc="upper right")
    panel_label(axis, "A")

    axis = figure.add_subplot(grid[0, 1])
    thresholds = ("0.5", "0.6", "0.7")
    for database in DATABASES:
        if database == "mimic3":
            values = [
                mimic3["stability"]["confidence_ge_0_6"][
                    "subsequent_flip_24_72"
                ]["estimate"]
                * 100
            ]
            axis.scatter(
                ("0.6",),
                values,
                marker="s",
                s=30,
                color=COLORS[database],
                label=LABELS[database],
            )
            continue
        values = [
            shared[database]["stability"]["confidence_24h"][threshold][
                "flip_24_72"
            ]
            * 100
            for threshold in thresholds
        ]
        axis.plot(
            thresholds,
            values,
            marker="o",
            markersize=4,
            color=COLORS[database],
            label=LABELS[database],
        )
    hospital_values = [
        hospital["stability"]["confidence_24h"][threshold][
            "subsequent_flip_24_72"
        ]["estimate"]
        * 100
        for threshold in thresholds
    ]
    axis.plot(
        thresholds,
        hospital_values,
        marker="x",
        linestyle="--",
        linewidth=0.8,
        color="#7B4F2C",
        label="eICU hospital-held-out",
    )
    axis.set_xlabel("24 h confidence threshold")
    axis.set_ylabel("Subsequent 24-to-72 h flip (%)")
    axis.set_ylim(0, 28)
    axis.legend(frameon=False, loc="upper right", fontsize=6.1)
    panel_label(axis, "B")

    axis = figure.add_subplot(grid[0, 2])
    for database in DATABASES:
        if database == "mimic3":
            flip = mimic3["stability"]["flip_24_72"]["estimate"] * 100
            tv = mimic3["stability"]["total_variation_24_72"]["estimate"]
            axis.scatter(
                flip,
                tv,
                color=COLORS[database],
                marker="o",
                s=30,
                label=f"{LABELS[database]} XGB",
            )
            continue
        record = shared[database]["stability"]
        aggregate = formal[database]["configurations"]["ordinal_temporal"][
            "aggregate"
        ]
        points = (
            (
                record["flip_24_72"] * 100,
                record["tv_24_72"],
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
        for flip, tv, marker, short_label in points:
            axis.scatter(
                flip,
                tv,
                color=COLORS[database],
                marker=marker,
                s=28,
                label=f"{LABELS[database]} {short_label}",
            )
    axis.set_xlabel("Class flip (%)")
    axis.set_ylabel("Probability total variation")
    axis.set_xlim(6, 33)
    axis.set_ylim(0.04, 0.19)
    axis.legend(frameon=False, loc="upper left", fontsize=5.8)
    panel_label(axis, "C")

    matrices = []
    for database in DATABASES:
        if database == "mimic3":
            counts = np.asarray(
                mimic3["transition_24_72"]["counts"], dtype=float
            )
        else:
            counts = np.asarray(
                shared[database]["stability"]["transition_24_to_72_counts"],
                dtype=float,
            )
        matrices.append(counts / counts.sum(axis=1, keepdims=True) * 100)
    for column, database in enumerate(DATABASES):
        axis = figure.add_subplot(grid[1, column])
        matrix = matrices[column]
        image = axis.imshow(matrix, cmap="YlGnBu", vmin=0, vmax=100)
        for row in range(3):
            for col in range(3):
                axis.text(
                    col,
                    row,
                    f"{matrix[row, col]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6.8,
                    color="white" if matrix[row, col] > 55 else "#222222",
                )
        axis.set_xticks(
            range(3), ("Home", "Facility", "Death"), rotation=20
        )
        axis.set_yticks(range(3), ("Home", "Facility", "Death"))
        axis.set_xlabel("Predicted at 72 h")
        axis.set_ylabel("Predicted at 24 h")
        axis.set_title(LABELS[database])
        for spine in axis.spines.values():
            spine.set_visible(False)
        if column == 0:
            panel_label(axis, "D")

    axis = figure.add_subplot(grid[2, :])
    pattern_keys = (
        "stable_all",
        "transient_48h_excursion",
        "monotone_worsening",
        "monotone_improvement",
        "other_nonmonotone",
    )
    pattern_labels = (
        "Stable all",
        "Transient 48 h",
        "Successive\nhigher-coded",
        "Successive\nlower-coded",
        "Other",
    )
    positions = np.arange(len(pattern_keys))
    for offset, database in zip((-0.24, 0.0, 0.24), DATABASES):
        values = [
            100.0 * trajectory[database]["patterns"][key]["estimate"]
            for key in pattern_keys
        ]
        errors = np.asarray(
            [
                (
                    trajectory[database]["patterns"][key]["estimate"]
                    - trajectory[database]["patterns"][key]["ci95"][0],
                    trajectory[database]["patterns"][key]["ci95"][1]
                    - trajectory[database]["patterns"][key]["estimate"],
                )
                for key in pattern_keys
            ]
        )
        axis.bar(
            positions + offset,
            values,
            width=0.22,
            color=COLORS[database],
            label=LABELS[database],
        )
        axis.errorbar(
            positions + offset,
            values,
            yerr=errors.T * 100,
            fmt="none",
            ecolor="#2F2F2F",
            capsize=1.5,
            linewidth=0.6,
        )
    axis.set_xticks(positions, pattern_labels)
    axis.tick_params(axis="x", labelsize=6.6)
    axis.set_ylabel("Stays (%)")
    axis.set_ylim(0, 84)
    axis.grid(axis="y", color="#DDDDDD", linewidth=0.45, alpha=0.75)
    axis.set_axisbelow(True)
    axis.legend(frameon=False, loc="upper right", ncol=3)
    axis.set_title(
        "Complete 24/48/72 h trajectory patterns",
        loc="left",
        fontsize=8.5,
        pad=7,
    )
    panel_label(axis, "E")
    figure.subplots_adjust(
        left=0.07,
        right=0.88,
        bottom=0.08,
        top=0.95,
        wspace=0.55,
        hspace=0.58,
    )
    color_axis = figure.add_axes((0.91, 0.39, 0.015, 0.23))
    figure.colorbar(
        image, cax=color_axis, label="Row-normalised transitions (%)"
    )
    save_figure(figure, "figure5_stability")


def figure7_transport_recalibration_reassessment() -> None:
    transport = load_json(
        RESULTS / "transport_extended" / "summary.json"
    )
    figure, axes = plt.subplots(2, 2, figsize=(7.4, 6.2))
    axes = axes.ravel()

    axis = axes[0]
    direction_colors = {
        "cross_institution": "#1F6F78",
        "same_institution_cross_era": "#6B6F9C",
        "cross_institution_and_era": "#C26D2E",
    }
    direction_markers = {
        "cross_institution": "o",
        "same_institution_cross_era": "s",
        "cross_institution_and_era": "D",
    }
    label_offsets = {
        "mimic4_to_eicu": (5, 5, "left"),
        "mimic3_to_eicu": (5, 5, "left"),
        "mimic4_to_mimic3": (-8, 8, "right"),
        "mimic3_to_mimic4": (5, 5, "left"),
        "eicu_to_mimic4": (-10, 10, "right"),
        "eicu_to_mimic3": (8, -11, "left"),
    }
    display_y_offsets = {
        "eicu_to_mimic4": 0.00045,
        "eicu_to_mimic3": -0.00045,
    }
    short_directions = {
        "mimic4_to_eicu": "M4 -> eICU",
        "eicu_to_mimic4": "eICU -> M4",
        "mimic4_to_mimic3": "M4 -> M3",
        "mimic3_to_mimic4": "M3 -> M4",
        "mimic3_to_eicu": "M3 -> eICU",
        "eicu_to_mimic3": "eICU -> M3",
    }
    for direction in DIRECTIONS:
        record = transport["directions"][direction]
        metric = record["transport_landmarks"]["72"][
            "target_all_reaching_landmark"
        ]
        comparison_type = record["comparison_type"]
        x_value = metric["macro_auc"]
        y_value = metric["ece_top_label_10bin"]
        display_y = y_value + display_y_offsets.get(direction, 0.0)
        axis.scatter(
            x_value,
            display_y,
            s=30,
            color=direction_colors[comparison_type],
            marker=direction_markers[comparison_type],
            edgecolors="white",
            linewidths=0.35,
            zorder=3,
        )
        x_offset, y_offset, alignment = label_offsets[direction]
        arrow_properties = (
            {
                "arrowstyle": "-",
                "color": direction_colors[comparison_type],
                "linewidth": 0.45,
                "shrinkA": 1,
                "shrinkB": 2,
            }
            if direction in {"eicu_to_mimic4", "eicu_to_mimic3"}
            else None
        )
        axis.annotate(
            short_directions[direction],
            (x_value, display_y),
            xytext=(x_offset, y_offset),
            textcoords="offset points",
            fontsize=5.6,
            ha=alignment,
            va="center",
            clip_on=False,
            arrowprops=arrow_properties,
        )
    for comparison_type, color in direction_colors.items():
        axis.scatter(
            [],
            [],
            color=color,
            marker=direction_markers[comparison_type],
            edgecolors="white",
            linewidths=0.35,
            s=24,
            label=comparison_type.replace("_", " "),
        )
    axis.set_xlabel("72 h transported macro-AUC")
    axis.set_ylabel("72 h top-label ECE")
    axis.set_xlim(0.65, 0.84)
    axis.set_ylim(0.0, 0.105)
    axis.legend(frameon=False, loc="upper right", fontsize=5.8)
    panel_label(axis, "A")

    line_styles = {
        "mimic4_to_eicu": "-",
        "eicu_to_mimic4": "--",
        "mimic4_to_mimic3": "-.",
        "mimic3_to_mimic4": ":",
        "mimic3_to_eicu": (0, (5, 1)),
        "eicu_to_mimic3": (0, (1, 1)),
    }
    for axis, curve_key, title, label, y_limit in (
        (
            axes[1],
            "curves",
            "Multinomial logistic recalibration",
            "B",
            0.205,
        ),
        (
            axes[2],
            "curves_temperature_scaling",
            "One-parameter temperature scaling",
            "C",
            0.085,
        ),
    ):
        for direction in DIRECTIONS:
            curves = transport["recalibration"][direction]["72"][curve_key]
            x_values = [
                int(curves[str(count)]["requested_local_labels"])
                for count in (0, 25, 50, 100, 200)
            ]
            y_values = [
                curves[str(count)]["ece_top_label_10bin"]["mean"]
                for count in (0, 25, 50, 100, 200)
            ]
            comparison_type = transport["directions"][direction][
                "comparison_type"
            ]
            axis.plot(
                x_values,
                y_values,
                marker="o",
                markersize=3,
                linewidth=0.8,
                linestyle=line_styles[direction],
                color=direction_colors[comparison_type],
                alpha=0.9,
            )
        axis.set_title(title, loc="left", fontsize=8.5, pad=7)
        axis.set_xlabel("Local labelled outcomes")
        axis.set_ylabel("72 h top-label ECE")
        axis.set_xticks((0, 25, 50, 100, 200))
        axis.set_xlim(-5, 210)
        axis.set_ylim(0, y_limit)
        axis.grid(color="#DDDDDD", linewidth=0.45, alpha=0.75)
        axis.set_axisbelow(True)
        panel_label(axis, label)

    axis = axes[3]
    paired = []
    for direction in DIRECTIONS:
        difference = transport["recalibration"][direction]["72"][
            "temperature_scaling_minus_multinomial_logistic"
        ]["25"]["ece_top_label_10bin"]["mean"]
        paired.append(difference)
    positions = np.arange(len(DIRECTIONS))
    for position, direction, value in zip(positions, DIRECTIONS, paired):
        comparison_type = transport["directions"][direction][
            "comparison_type"
        ]
        axis.barh(
            position,
            value,
            color=direction_colors[comparison_type],
            height=0.58,
        )
    axis.axvline(0, color="#333333", linewidth=0.7)
    axis.set_yticks(
        positions,
        [short_directions[direction] for direction in DIRECTIONS],
        fontsize=6.4,
    )
    axis.invert_yaxis()
    axis.set_xlabel("Delta 72 h ECE at 25 labels")
    axis.set_title(
        "Temperature minus multinomial",
        loc="left",
        fontsize=8.5,
        pad=7,
    )
    axis.grid(axis="x", color="#DDDDDD", linewidth=0.45, alpha=0.75)
    axis.set_axisbelow(True)
    panel_label(axis, "D")
    figure.subplots_adjust(
        left=0.08,
        right=0.98,
        bottom=0.08,
        top=0.95,
        wspace=0.36,
        hspace=0.42,
    )
    save_figure(figure, "figure7_transport_reassessment")


def figure8_reassessment_simulation() -> None:
    risk = load_json(RESULTS / "flip_risk" / "summary.json")
    figure, axes = plt.subplots(
        3,
        2,
        figsize=(7.4, 8.8),
        gridspec_kw={"height_ratios": (1.12, 1.0, 0.92)},
    )
    short_labels = {"mimic4": "M4", "eicu": "eICU", "mimic3": "M3"}
    trigger_specs = (
        (
            axes[0, 0],
            "instability_risk_trigger",
            "o",
            "Endpoint-change risk trigger",
            (20, 92),
            (40, 102),
            "A",
        ),
        (
            axes[0, 1],
            "death_upgrade_risk_trigger",
            "s",
            "Direct conversion risk trigger",
            (2, 88),
            (10, 100),
            "B",
        ),
    )
    for axis, trigger, marker, title, x_limits, y_limits, label in trigger_specs:
        for database in DATABASES:
            simulation = risk["simulations"][trigger]["within_database"][
                database
            ]["risk_thresholds"]
            retained = [
                100 * (1 - item["reassessments_reduced_vs_all"])
                for item in simulation
            ]
            captured = [
                100 * item["major_death_upgrade_capture_fraction"]
                for item in simulation
            ]
            axis.plot(
                retained,
                captured,
                color=COLORS[database],
                marker=marker,
                markersize=4.0,
                linewidth=0.9,
                label=LABELS[database],
            )
            primary = next(
                item for item in simulation if item["threshold"] == 0.10
            )
            axis.scatter(
                100 * (1 - primary["reassessments_reduced_vs_all"]),
                100 * primary["major_death_upgrade_capture_fraction"],
                s=68,
                facecolors="none",
                edgecolors=COLORS[database],
                linewidths=1.0,
                zorder=5,
            )
        axis.set_title(title, loc="left", fontsize=8.5, pad=7)
        axis.set_xlabel("Stays selected (% of all)")
        axis.set_ylabel("Predicted conversions included (%)")
        axis.set_xlim(*x_limits)
        axis.set_ylim(*y_limits)
        axis.grid(color="#DDDDDD", linewidth=0.45, alpha=0.75)
        axis.set_axisbelow(True)
        axis.legend(frameon=False, loc="lower right", fontsize=6.5)
        panel_label(axis, label)

    axis = axes[1, 0]
    for database in DATABASES:
        any_primary = next(
            item
            for item in risk["simulations"]["instability_risk_trigger"][
                "within_database"
            ][database]["risk_thresholds"]
            if item["threshold"] == 0.10
        )
        death_primary = next(
            item
            for item in risk["simulations"]["death_upgrade_risk_trigger"][
                "within_database"
            ][database]["risk_thresholds"]
            if item["threshold"] == 0.10
        )
        points = (
            (
                any_primary["any_flip_capture_fraction"],
                any_primary["major_death_upgrade_capture_fraction"],
            ),
            (
                death_primary["any_flip_capture_fraction"],
                death_primary["major_death_upgrade_capture_fraction"],
            ),
        )
        axis.plot(
            [100 * points[0][0], 100 * points[1][0]],
            [100 * points[0][1], 100 * points[1][1]],
            color=COLORS[database],
            linewidth=0.9,
            alpha=0.45,
            zorder=1,
        )
        axis.scatter(
            100 * points[0][0],
            100 * points[0][1],
            s=38,
            color=COLORS[database],
            marker="o",
            zorder=3,
        )
        axis.scatter(
            100 * points[1][0],
            100 * points[1][1],
            s=35,
            facecolors="white",
            edgecolors=COLORS[database],
            marker="s",
            linewidths=1.0,
            zorder=3,
        )
        label_positions = {
            "mimic4": (5, 5, "left"),
            "eicu": (5, 5, "left"),
            "mimic3": (-5, -10, "right"),
        }
        offset_x, offset_y, alignment = label_positions[database]
        axis.annotate(
            short_labels[database],
            (100 * points[0][0], 100 * points[0][1]),
            xytext=(offset_x, offset_y),
            textcoords="offset points",
            ha=alignment,
            fontsize=6.2,
            color=COLORS[database],
        )
    axis.scatter([], [], color="#555555", marker="o", s=32, label="Endpoint-change trigger")
    axis.scatter(
        [],
        [],
        facecolors="white",
        edgecolors="#555555",
        marker="s",
        s=30,
        label="Direct conversion trigger",
    )
    axis.set_title("Inclusion at threshold 0.10", loc="left", fontsize=8.5, pad=7)
    axis.set_xlabel("Endpoint changes included (%)")
    axis.set_ylabel("Predicted conversions included (%)")
    axis.set_xlim(20, 100)
    axis.set_ylim(50, 96)
    axis.grid(color="#DDDDDD", linewidth=0.45, alpha=0.75)
    axis.set_axisbelow(True)
    axis.legend(frameon=False, loc="lower right", fontsize=6.2)
    panel_label(axis, "C")

    axis = axes[1, 1]
    for database in DATABASES:
        for trigger, marker in (
            ("instability_risk_trigger", "o"),
            ("death_upgrade_risk_trigger", "s"),
        ):
            primary = next(
                item
                for item in risk["simulations"][trigger]["within_database"][
                    database
                ]["risk_thresholds"]
                if item["threshold"] == 0.10
            )
            axis.scatter(
                100 * primary["reassessments_reduced_vs_all"],
                100 * primary[
                    "reassessment_without_major_death_upgrade_fraction"
                ],
                s=36,
                facecolors=COLORS[database] if marker == "o" else "white",
                edgecolors=COLORS[database],
                linewidths=0.9,
                marker=marker,
                zorder=3,
            )
        any_primary = next(
            item
            for item in risk["simulations"]["instability_risk_trigger"][
                "within_database"
            ][database]["risk_thresholds"]
            if item["threshold"] == 0.10
        )
        axis.annotate(
            short_labels[database],
            (
                100 * any_primary["reassessments_reduced_vs_all"],
                100 * any_primary[
                    "reassessment_without_major_death_upgrade_fraction"
                ],
            ),
            xytext=(-4, 5),
            textcoords="offset points",
            ha="right",
            fontsize=6.2,
            color=COLORS[database],
        )
    axis.set_title(
        "Selection and no conversion at threshold 0.10",
        loc="left",
        fontsize=8.5,
        pad=7,
    )
    axis.set_xlabel("Fewer stays selected than monitor all (%)")
    axis.set_ylabel("No predicted conversion in selected (%)")
    axis.set_xlim(15, 88)
    axis.set_ylim(68, 96)
    axis.grid(color="#DDDDDD", linewidth=0.45, alpha=0.75)
    axis.set_axisbelow(True)
    panel_label(axis, "D")

    oof = {
        (record["database"], record["outcome"]): record
        for record in risk["within_database_oof"]
    }
    outcome_specs = (
        ("any_flip", "o", "Class change"),
        ("major_death_upgrade", "s", "Conversion"),
    )
    for axis, metric_key, title, label, x_limits in (
        (
            axes[2, 0],
            "auc",
            "Trigger discrimination",
            "E",
            (0.68, 0.87),
        ),
        (
            axes[2, 1],
            "auprc",
            "Trigger precision-recall",
            "F",
            (0.18, 0.52),
        ),
    ):
        positions = []
        tick_labels = []
        position = 0
        for database in DATABASES:
            for outcome, marker, short_label in outcome_specs:
                record = oof[(database, outcome)]
                estimate = record["metrics"][metric_key]
                interval = record["metrics_ci95"][metric_key]
                positions.append(position)
                tick_labels.append(f"{short_labels[database]} {short_label}")
                axis.errorbar(
                    estimate,
                    position,
                    xerr=np.asarray(
                        [
                            [estimate - interval[0]],
                            [interval[1] - estimate],
                        ]
                    ),
                    fmt=marker,
                    markersize=4.2,
                    color=COLORS[database],
                    ecolor=COLORS[database],
                    capsize=2,
                    linewidth=0.8,
                )
                position += 1
            position += 0.35
        axis.set_yticks(positions, tick_labels, fontsize=6.4)
        axis.invert_yaxis()
        axis.set_xlim(*x_limits)
        axis.set_xlabel("AUC" if metric_key == "auc" else "AUPRC")
        axis.set_title(title, loc="left", fontsize=8.5, pad=7)
        axis.grid(axis="x", color="#DDDDDD", linewidth=0.45, alpha=0.75)
        axis.set_axisbelow(True)
        panel_label(axis, label)

    figure.subplots_adjust(
        left=0.11,
        right=0.985,
        bottom=0.06,
        top=0.965,
        wspace=0.40,
        hspace=0.48,
    )
    save_figure(figure, "figure8_reassessment_simulation")


def figure9_sensitivity_analyses() -> None:
    sensitivity = load_json(RESULTS / "sensitivity" / "sensitivity_summary.json")
    sensitivity_v2 = load_json(
        RESULTS / "sensitivity_v2" / "sensitivity_v2_summary.json"
    )
    xgb = {
        record["database"]: record
        for record in load_json(RESULTS / "landmarks" / "xgb_summary.json")
    }
    shared = load_json(RESULTS / "shared_fold" / "summary.json")
    full_complete = {
        record["database"]: record["landmarks"]
        for record in sensitivity["full_vs_complete"]
    }
    six = {
        record["database"]: record["six_to_48h_complete_subset"]
        for record in sensitivity["six_vs_48"]
    }
    icu_death = sensitivity_v2["icu_death_exclusion"]
    icd_era = sensitivity_v2["icd_era"]
    reversal = sensitivity_v2["ordinal_order_reversal"]
    ordinal_boot = {
        database: sensitivity_v2["model_contrast_bootstrap"][database][
            "contrasts"
        ]["ordinal_minus_nominal_no_temporal"]
        for database in ("mimic4", "eicu")
    }

    figure, axes = plt.subplots(2, 3, figsize=(7.4, 7.1))

    axis = axes[0, 0]
    offsets = {"mimic4": -0.07, "eicu": 0.07}
    for database in ("mimic4", "eicu"):
        for index, hours in enumerate((24, 48)):
            full_value = full_complete[database][str(hours)]["full_landmark"][
                "macro_auc"
            ]
            complete_value = full_complete[database][str(hours)][
                "complete_window_subset"
            ]["macro_auc"]
            x_value = index + offsets[database]
            axis.plot(
                (x_value, x_value),
                (full_value, complete_value),
                color=COLORS[database],
                linewidth=0.9,
                alpha=0.65,
            )
            axis.scatter(
                x_value,
                full_value,
                s=24,
                facecolors="white",
                edgecolors=COLORS[database],
                linewidths=1.0,
                marker="o",
                zorder=3,
            )
            axis.scatter(
                x_value,
                complete_value,
                s=24,
                color=COLORS[database],
                marker="o",
                zorder=3,
            )
    axis.plot([], [], color="#555555", marker="o", markerfacecolor="white", label="Full landmark")
    axis.plot([], [], color="#555555", marker="o", label="Complete window")
    axis.set_xticks((0, 1), ("24 h", "48 h"))
    axis.set_xlim(-0.35, 1.35)
    axis.set_ylim(0.66, 0.775)
    axis.set_title("Landmark population", loc="left", fontsize=8.5, pad=7)
    axis.set_ylabel("XGBoost macro-AUC")
    axis.grid(axis="y", color="#DDDDDD", linewidth=0.45, alpha=0.75)
    axis.set_axisbelow(True)
    axis.legend(frameon=False, loc="lower left", fontsize=6.1)
    panel_label(axis, "A")

    axis = axes[0, 1]
    for index, database in enumerate(("mimic4", "eicu")):
        auc_three = xgb[database]["definitions"]["3class"]["72"]["macro_auc"]
        auc_four = xgb[database]["definitions"]["4class"]["72"]["macro_auc"]
        axis.plot(
            (auc_three, auc_four),
            (index, index),
            color=COLORS[database],
            linewidth=0.9,
            alpha=0.65,
        )
        axis.scatter(
            auc_three,
            index,
            s=28,
            color=COLORS[database],
            marker="o",
            zorder=3,
        )
        axis.scatter(
            auc_four,
            index,
            s=26,
            facecolors="white",
            edgecolors=COLORS[database],
            marker="s",
            linewidths=1.0,
            zorder=3,
        )
        axis.text(
            max(auc_three, auc_four) + 0.003,
            index,
            f"{auc_four:.3f}",
            va="center",
            fontsize=5.8,
            color=COLORS[database],
        )
    axis.scatter([], [], color="#555555", marker="o", s=24, label="Three class")
    axis.scatter(
        [],
        [],
        facecolors="white",
        edgecolors="#555555",
        marker="s",
        s=22,
        label="Four class",
    )
    axis.set_yticks((0, 1), ("MIMIC-IV", "eICU-CRD"))
    axis.set_ylim(-0.5, 1.5)
    axis.set_xlim(0.68, 0.765)
    axis.set_title("Outcome taxonomy", loc="left", fontsize=8.5, pad=7)
    axis.set_xlabel("72 h macro-AUC")
    axis.grid(axis="x", color="#DDDDDD", linewidth=0.45, alpha=0.75)
    axis.set_axisbelow(True)
    axis.legend(frameon=False, loc="lower left", fontsize=6.1)
    panel_label(axis, "B")

    axis = axes[0, 2]
    for database in ("mimic4", "eicu"):
        values = six[database]
        axis.plot(
            (6, 48),
            (values["auc_6"], values["auc_48"]),
            color=COLORS[database],
            marker="o",
            markersize=4,
            linewidth=1.0,
            label=LABELS[database],
        )
        axis.text(
            50,
            values["auc_48"],
            f"flip {values['flip'] * 100:.2f}%",
            va="center",
            fontsize=5.9,
            color=COLORS[database],
        )
    axis.set_xticks((6, 48), ("6 h", "48 h"))
    axis.set_xlim(2, 69)
    axis.set_ylim(0.64, 0.775)
    axis.set_title("Six-to-48 h window", loc="left", fontsize=8.5, pad=7)
    axis.set_ylabel("Macro-AUC")
    axis.grid(axis="y", color="#DDDDDD", linewidth=0.45, alpha=0.75)
    axis.set_axisbelow(True)
    axis.legend(frameon=False, loc="upper left", fontsize=6.1)
    panel_label(axis, "C")

    axis = axes[1, 0]
    flip_records = (
        (
            "MIMIC-IV primary",
            shared["databases"]["mimic4"]["stability"]["flip_24_72"] * 100,
            np.asarray(
                shared["databases"]["mimic4"]["stability"]["flip_24_72_ci95"]
            )
            * 100,
            "mimic4",
            "o",
        ),
        (
            "ICU death excluded, MIMIC-IV",
            icu_death["mimic4"]["complete_three_landmark_subset"]["flip_24_72"]
            * 100,
            np.asarray(
                icu_death["mimic4"]["complete_three_landmark_subset"][
                    "flip_24_72_ci95"
                ]
            )
            * 100,
            "mimic4",
            "s",
        ),
        (
            "ICD-9, MIMIC-IV",
            icd_era["mimic4_icd9"]["complete_three_landmark_subset"][
                "flip_24_72"
            ]
            * 100,
            np.asarray(
                icd_era["mimic4_icd9"]["complete_three_landmark_subset"][
                    "flip_24_72_ci95"
                ]
            )
            * 100,
            "mimic4",
            "s",
        ),
        (
            "ICD-10, MIMIC-IV",
            icd_era["mimic4_icd10"]["complete_three_landmark_subset"][
                "flip_24_72"
            ]
            * 100,
            np.asarray(
                icd_era["mimic4_icd10"]["complete_three_landmark_subset"][
                    "flip_24_72_ci95"
                ]
            )
            * 100,
            "mimic4",
            "s",
        ),
        (
            "eICU-CRD primary",
            shared["databases"]["eicu"]["stability"]["flip_24_72"] * 100,
            np.asarray(
                shared["databases"]["eicu"]["stability"]["flip_24_72_ci95"]
            )
            * 100,
            "eicu",
            "o",
        ),
        (
            "ICU death excluded, eICU-CRD",
            icu_death["eicu"]["complete_three_landmark_subset"]["flip_24_72"]
            * 100,
            np.asarray(
                icu_death["eicu"]["complete_three_landmark_subset"][
                    "flip_24_72_ci95"
                ]
            )
            * 100,
            "eicu",
            "s",
        ),
    )
    for index, (label, estimate, interval, database, marker) in enumerate(
        reversed(flip_records)
    ):
        axis.errorbar(
            estimate,
            index,
            xerr=np.asarray([[estimate - interval[0]], [interval[1] - estimate]]),
            fmt=marker,
            markersize=4.0,
            color=COLORS[database],
            markerfacecolor=COLORS[database] if marker == "o" else "white",
            markeredgecolor=COLORS[database],
            capsize=2.0,
            linewidth=0.8,
            zorder=3,
        )
        axis.text(
            interval[1] + 0.8,
            index,
            f"{estimate:.2f}",
            va="center",
            fontsize=5.7,
            color=COLORS[database],
        )
    axis.set_yticks(
        np.arange(len(flip_records)),
        [record[0] for record in reversed(flip_records)],
    )
    axis.set_ylim(-0.55, len(flip_records) - 0.45)
    axis.set_xlim(8, 39)
    axis.set_title("Flip under cohort perturbations", loc="left", fontsize=8.5, pad=7)
    axis.set_xlabel("24-to-72 h class flip (%)")
    axis.grid(axis="x", color="#DDDDDD", linewidth=0.45, alpha=0.75)
    axis.set_axisbelow(True)
    panel_label(axis, "D")

    outer = axes[1, 1]
    outer.set_axis_off()
    panel_label(outer, "E")
    outer.text(
        0.0,
        1.00,
        "Coding-order pressure test",
        transform=outer.transAxes,
        fontsize=8.5,
        va="top",
        ha="left",
    )
    auc_axis = outer.inset_axes((0.20, 0.59, 0.76, 0.29))
    flip_axis = outer.inset_axes((0.20, 0.12, 0.76, 0.29))
    for axis, metric, scale, title in (
        (auc_axis, "auc_72", 1.0, "Delta AUC"),
        (
            flip_axis,
            "flip_24_72",
            100.0,
            "Delta flip (percentage points)",
        ),
    ):
        axis.axvline(0, color="#999999", linewidth=0.7, zorder=0)
        for y_value, database in ((1, "mimic4"), (0, "eicu")):
            paired = reversal[database]["paired_reversed_minus_standard"][metric]
            values = np.asarray(paired["values"]) * scale
            axis.scatter(
                values,
                np.full(len(values), y_value),
                s=13,
                facecolors="none",
                edgecolors=COLORS[database],
                linewidths=0.8,
                zorder=2,
            )
            axis.errorbar(
                paired["mean"] * scale,
                y_value,
                xerr=paired["sd"] * scale,
                fmt="D",
                markersize=4,
                color=COLORS[database],
                capsize=2,
                linewidth=0.8,
                zorder=3,
            )
        axis.set_yticks((0, 1), ("eICU", "M4"))
        axis.set_ylim(-0.5, 1.5)
        axis.set_title(title, loc="left", fontsize=7.0, pad=2)
        axis.grid(axis="x", color="#DDDDDD", linewidth=0.4, alpha=0.75)
        axis.set_axisbelow(True)
        axis.tick_params(axis="both", labelsize=6.0, length=2)
    auc_axis.set_xlim(-0.002, 0.018)
    flip_axis.set_xlim(-2, 5)

    axis = axes[1, 2]
    ordinal_rows = (
        ("MIMIC-IV\nnon-home", "mimic4", "non_home"),
        ("MIMIC-IV\ndeath", "mimic4", "death"),
        ("eICU-CRD\nnon-home", "eicu", "non_home"),
        ("eICU-CRD\ndeath", "eicu", "death"),
    )
    for index, (label, database, endpoint) in enumerate(ordinal_rows):
        for metric, marker in (("continuous_nri", "o"), ("idi", "s")):
            estimate = ordinal_boot[database][
                f"{metric}_{endpoint}"
            ]["estimate"]
            interval = ordinal_boot[database][
                f"{metric}_{endpoint}"
            ]["ci95"]
            axis.errorbar(
                estimate,
                index,
                xerr=np.asarray(
                    [[estimate - interval[0]], [interval[1] - estimate]]
                ),
                fmt=marker,
                markersize=3.8,
                markerfacecolor=COLORS[database] if marker == "o" else "white",
                markeredgecolor=COLORS[database],
                color=COLORS[database],
                capsize=1.8,
                linewidth=0.75,
                zorder=3,
            )
    axis.scatter([], [], color="#555555", marker="o", s=20, label="Continuous NRI")
    axis.scatter(
        [],
        [],
        facecolors="white",
        edgecolors="#555555",
        marker="s",
        s=18,
        label="IDI",
    )
    axis.set_yticks(
        np.arange(len(ordinal_rows)),
        [row[0] for row in ordinal_rows],
    )
    axis.set_ylim(-0.55, len(ordinal_rows) - 0.45)
    axis.set_xlim(-0.005, 0.61)
    axis.set_title("Exploratory reclassification", loc="left", fontsize=8.5, pad=7)
    axis.set_xlabel("Ordinal minus nominal (95% CI)")
    axis.grid(axis="x", color="#DDDDDD", linewidth=0.45, alpha=0.75)
    axis.set_axisbelow(True)
    axis.legend(frameon=False, loc="lower right", fontsize=5.8)
    panel_label(axis, "F")

    figure.subplots_adjust(
        left=0.10,
        right=0.985,
        bottom=0.08,
        top=0.95,
        wspace=0.52,
        hspace=0.48,
    )
    save_figure(figure, "figure9_sensitivity_analyses")


def figure10_hospital_validation() -> None:
    hospital = load_json(
        RESULTS / "eicu_hospital_validation" / "summary.json"
    )
    shared = load_json(RESULTS / "shared_fold" / "summary.json")
    comparison = hospital["comparison_with_patient_grouped"]
    patient_metrics = comparison["patient_grouped_metrics_recomputed"]
    hospital_metrics = comparison["hospital_grouped_metrics"]
    differences = comparison["hospital_minus_patient_grouped"]
    site_summary = hospital["hospital_level"][
        "der_simonian_laird_72h_macro_auc"
    ]
    site_definition = hospital["hospital_level"]["definition"]
    site_distributions = hospital["hospital_level"][
        "distributions_across_primary_sites"
    ]

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(7.4, 5.8),
        gridspec_kw={"height_ratios": [1.05, 0.95]},
    )

    axis = axes[0, 0]
    hours = (24, 48, 72)
    patient_values = [
        patient_metrics[str(hour)]["macro_auc"] for hour in hours
    ]
    hospital_values = [
        hospital_metrics[str(hour)]["macro_auc"] for hour in hours
    ]
    axis.plot(
        (0, 1, 2),
        patient_values,
        color="#6C7077",
        marker="o",
        markersize=4.5,
        linewidth=1.1,
        label="Patient-grouped",
    )
    axis.plot(
        (0, 1, 2),
        hospital_values,
        color=COLORS["eicu"],
        marker="s",
        markersize=4.5,
        linewidth=1.1,
        label="Hospital-held-out",
    )
    for index, (patient, heldout) in enumerate(
        zip(patient_values, hospital_values)
    ):
        axis.text(
            index - 0.07,
            patient - 0.012,
            f"{patient:.3f}",
            ha="center",
            va="top",
            fontsize=6.0,
            color="#555555",
        )
        axis.text(
            index + 0.07,
            heldout + 0.008,
            f"{heldout:.3f}",
            ha="center",
            va="bottom",
            fontsize=6.0,
            color=COLORS["eicu"],
        )
    axis.set_xticks((0, 1, 2), ("24 h", "48 h", "72 h"))
    axis.set_xlim(-0.35, 2.35)
    axis.set_ylim(0.64, 0.735)
    axis.set_ylabel("Macro-AUC")
    axis.set_title("Same stays, stricter grouping", loc="left", fontsize=8.5, pad=7)
    axis.grid(axis="y", color="#DDDDDD", linewidth=0.45, alpha=0.75)
    axis.set_axisbelow(True)
    axis.legend(frameon=False, loc="lower right", fontsize=6.2)
    panel_label(axis, "A")

    axis = axes[0, 1]
    axis.set_xlim(0, 10)
    axis.set_ylim(0, 10)
    axis.axis("off")
    cohort = hospital["cohort"]
    full_eicu = shared["databases"]["eicu"]["cohort"]["n"]
    boxes = (
        (
            7.2,
            f"Complete 72 h eICU-CRD cohort\n{full_eicu:,} stays",
        ),
        (
            4.2,
            "Patients observed in more than one hospital\n"
            f"-{cohort['patients_excluded_for_multiple_hospitals']:,} patients / "
            f"-{cohort['rows_excluded_for_multiple_hospitals']:,} stays",
        ),
        (
            1.0,
            f"Validation cohort\n{cohort['n']:,} stays | "
            f"{cohort['n_unique_patients']:,} patients | "
            f"{cohort['n_hospitals']} hospitals",
        ),
    )
    for y_value, text in boxes:
        axis.add_patch(
            FancyBboxPatch(
                (0.35, y_value),
                9.25,
                1.45,
                boxstyle="round,pad=0.03,rounding_size=0.09",
                linewidth=0.8,
                edgecolor="#4B4B4B",
                facecolor="white",
            )
        )
        axis.text(
            4.98,
            y_value + 0.73,
            text,
            ha="center",
            va="center",
            fontsize=6.7,
            linespacing=1.3,
        )
    for y_value in (7.2, 4.2):
        axis.add_patch(
            FancyArrowPatch(
                (4.98, y_value - 0.08),
                (4.98, y_value - 1.48),
                arrowstyle="-|>",
                mutation_scale=9,
                linewidth=0.8,
                color="#4B4B4B",
            )
        )
    axis.text(
        0.0,
        9.7,
        "Validation cohort construction",
        fontsize=8.5,
        fontweight="bold",
        va="top",
        ha="left",
    )
    panel_label(axis, "B")

    axis = axes[1, 0]
    contrast_records = (
        (
            "Any class flip",
            differences["flip_24_72"],
            COLORS["mimic4"],
        ),
        (
            "Non-death to\ndeath conversion",
            differences["major_upgrade_to_death"],
            "#B15C4A",
        ),
        (
            "Home to facility",
            differences["home_to_facility"],
            "#4F8588",
        ),
    )
    y_positions = np.arange(len(contrast_records))[::-1]
    for y_value, (label, record, color) in zip(
        y_positions, contrast_records
    ):
        estimate = 100 * record["estimate"]
        lower, upper = 100 * np.asarray(record["ci95"])
        axis.errorbar(
            estimate,
            y_value,
            xerr=np.asarray([[estimate - lower], [upper - estimate]]),
            fmt="o",
            markersize=4.5,
            color=color,
            markerfacecolor=color,
            capsize=2.5,
            linewidth=0.9,
            zorder=3,
        )
        axis.text(
            upper + 0.10,
            y_value,
            f"{estimate:+.2f} pp",
            va="center",
            fontsize=6.1,
            color=color,
        )
    axis.axvline(0, color="#8B8B8B", linewidth=0.75, zorder=0)
    axis.set_yticks(
        y_positions, [record[0] for record in contrast_records]
    )
    axis.set_ylim(-0.55, 2.55)
    axis.set_xlim(-0.4, 2.65)
    axis.set_xlabel("Hospital-held-out minus patient-grouped (pp, 95% CI)")
    axis.set_title("Same-row partition contrast", loc="left", fontsize=8.5, pad=7)
    axis.grid(axis="x", color="#DDDDDD", linewidth=0.45, alpha=0.75)
    axis.set_axisbelow(True)
    panel_label(axis, "C")

    axis = axes[1, 1]
    frame = pd.read_csv(
        RESULTS / "eicu_hospital_validation" / "hospital_metrics.csv"
    )
    evaluable = frame.loc[
        frame["sufficient_n_ge_50"]
        & frame["all_outcomes_observed"]
        & frame["all_outcome_counts_ge_5"]
    ]
    values = evaluable["macro_auc_72"].to_numpy()
    bins = np.linspace(0.56, 0.84, 15)
    axis.hist(
        values,
        bins=bins,
        color="#E8CDBB",
        edgecolor=COLORS["eicu"],
        linewidth=0.6,
    )
    distribution = site_distributions["macro_auc_72"]
    axis.axvline(
        distribution["p50"],
        color="#4F8588",
        linestyle="-",
        linewidth=1.0,
        label=f"Median {distribution['p50']:.3f}",
    )
    axis.axvline(
        site_summary["pooled_auc"],
        color="#6B4C9A",
        linestyle="--",
        linewidth=1.0,
        label=f"Pooled {site_summary['pooled_auc']:.3f}",
    )
    axis.set_xlim(0.56, 0.84)
    axis.set_ylim(0, 12.5)
    axis.set_xlabel("Site-level 72 h macro-AUC")
    axis.set_ylabel("Hospitals")
    axis.set_title("Between-hospital heterogeneity", loc="left", fontsize=8.5, pad=7)
    axis.text(
        0.985,
        0.93,
        (
            f"k={site_summary['k']}\n"
            f"P10-P90 {distribution['p10']:.3f}-{distribution['p90']:.3f}\n"
            f"I2={site_summary['i2_percent']:.2f}%"
        ),
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=6.2,
        linespacing=1.35,
    )
    axis.grid(axis="y", color="#DDDDDD", linewidth=0.45, alpha=0.75)
    axis.set_axisbelow(True)
    axis.legend(frameon=False, loc="upper left", fontsize=6.0)
    panel_label(axis, "D")

    figure.subplots_adjust(
        left=0.095,
        right=0.985,
        bottom=0.10,
        top=0.95,
        wspace=0.30,
        hspace=0.50,
    )
    save_figure(figure, "figure10_hospital_validation")


def figure11_clinical_transitions_safety() -> None:
    clinical = load_json(RESULTS / "clinical_flips" / "summary.json")
    mimic3 = load_json(RESULTS / "mimic3_shared" / "summary.json")
    database_order = ("mimic4", "eicu", "mimic3")
    labels = [LABELS[database] for database in database_order]

    major_values = []
    major_intervals = []
    home_values = []
    home_intervals = []
    for database in database_order:
        if database == "mimic3":
            major = mimic3["clinical_transitions"]["major_upgrade_to_death"]
            home = mimic3["clinical_transitions"]["home_to_facility"]
            major_values.append(major["estimate"])
            home_values.append(home["estimate"])
            major_intervals.append(None)
            home_intervals.append(None)
        else:
            model = clinical["databases"][database]["models"]["xgboost"]
            major = model["clinical_major_upgrades"]["major_upgrade_to_death"]
            home = model["clinical_major_upgrades"]["home_to_facility"]
            major_values.append(major["estimate"])
            home_values.append(home["estimate"])
            major_intervals.append(np.asarray(major["ci95"]))
            home_intervals.append(np.asarray(home["ci95"]))

    figure, axes = plt.subplots(
        2, 2, figsize=(7.8, 6.0), gridspec_kw={"width_ratios": [1.0, 1.18]}
    )

    axis = axes[0, 0]
    x_positions = np.arange(len(database_order))
    width = 0.32
    for offset, values, intervals, color, label, marker in (
        (
            -width / 2,
            major_values,
            major_intervals,
            "#B15C4A",
            "Non-death to\ndeath conversion",
            "o",
        ),
        (
            width / 2,
            home_values,
            home_intervals,
            "#4F8588",
            "Home to facility",
            "s",
        ),
    ):
        x_values = x_positions + offset
        axis.bar(
            x_values,
            np.asarray(values) * 100,
            width=width,
            color=color,
            alpha=0.86,
            edgecolor=color,
            linewidth=0.6,
            label=label,
        )
        for x_value, value, interval in zip(x_values, values, intervals):
            y_value = value * 100
            label_y = y_value
            if interval is not None:
                lower, upper = interval * 100
                label_y = upper
                axis.errorbar(
                    x_value,
                    y_value,
                    yerr=np.asarray(
                        [[y_value - lower], [upper - y_value]]
                    ),
                    fmt="none",
                    ecolor="#3D3D3D",
                    capsize=2,
                    linewidth=0.8,
                    zorder=3,
                )
            axis.text(
                x_value,
                label_y + 0.25,
            f"{y_value:.2f}",
                ha="center",
                va="bottom",
                fontsize=5.9,
                color="#3D3D3D",
            )
    axis.set_xticks(x_positions, labels)
    axis.set_ylim(0, 12.5)
    axis.set_ylabel("Stays, %")
    axis.set_title(
        "Major predicted-class transitions", loc="left", fontsize=8.5, pad=28
    )
    axis.grid(axis="y", color="#DDDDDD", linewidth=0.45, alpha=0.75)
    axis.set_axisbelow(True)
    axis.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=2,
        fontsize=5.8,
        columnspacing=1.0,
        handletextpad=0.4,
    )
    panel_label(axis, "A")

    axis = axes[0, 1]
    gain_values = []
    loss_values = []
    tie_values = []
    for database in database_order:
        if database == "mimic3":
            correctness = mimic3["correctness"]["among_flips"]
        else:
            correctness = clinical["databases"][database]["models"][
                "xgboost"
            ]["correctness_relative_to_final_outcome"]["flips"]
        gain_values.append(100 * correctness["gain"]["estimate"])
        loss_values.append(100 * correctness["loss"]["estimate"])
        tie_values.append(100 * correctness["tie"]["estimate"])
    y_positions = np.arange(len(database_order))[::-1]
    axis.barh(
        y_positions,
        gain_values,
        color="#4F8588",
        edgecolor="white",
        linewidth=0.6,
        label="Closer",
    )
    axis.barh(
        y_positions,
        loss_values,
        left=gain_values,
        color="#B15C4A",
        edgecolor="white",
        linewidth=0.6,
        label="Farther",
    )
    axis.barh(
        y_positions,
        tie_values,
        left=np.asarray(gain_values) + np.asarray(loss_values),
        color="#B9BDC4",
        edgecolor="white",
        linewidth=0.6,
        label="Equal distance",
    )
    for y_value, gain, loss, tie in zip(
        y_positions, gain_values, loss_values, tie_values
    ):
        if gain > 6:
            axis.text(
                gain / 2,
                y_value,
                f"{gain:.2f}",
                ha="center",
                va="center",
                fontsize=6.0,
                color="white",
            )
        if loss > 6:
            axis.text(
                gain + loss / 2,
                y_value,
                f"{loss:.2f}",
                ha="center",
                va="center",
                fontsize=6.0,
                color="white",
            )
        if tie > 5:
            axis.text(
                gain + loss + tie / 2,
                y_value,
                f"{tie:.2f}",
                ha="center",
                va="center",
                fontsize=6.0,
                color="#3D3D3D",
            )
    axis.set_yticks(y_positions, labels)
    axis.set_xlim(0, 100)
    axis.set_xlabel("Among changed predictions, %")
    axis.set_title(
        "Movement relative to observed final class",
        loc="left",
        fontsize=8.5,
        pad=7,
    )
    axis.grid(axis="x", color="#DDDDDD", linewidth=0.45, alpha=0.75)
    axis.set_axisbelow(True)
    panel_label(axis, "B")

    axis = axes[1, 0]
    improved_values = []
    mean_changes = []
    for database in database_order:
        if database == "mimic3":
            probability = mimic3["correctness"]["true_class_probability"][
                "among_flips"
            ]
            improved = probability["improved"]["estimate"]
            mean_change = probability["change_mean"]
        else:
            probability = clinical["databases"][database]["models"][
                "xgboost"
            ]["true_class_probability"]["among_flips"]
            improved = probability["improved"]["estimate"]
            mean_change = probability["pt_true_change_mean"]["estimate"]
        improved_values.append(100 * improved)
        mean_changes.append(mean_change)
    bars = axis.bar(
        np.arange(len(database_order)),
        improved_values,
        width=0.58,
        color="#6B6F9C",
        edgecolor="#6B6F9C",
        linewidth=0.6,
    )
    axis.axhline(50, color="#8B8B8B", linewidth=0.75, linestyle="--", zorder=0)
    for bar, value, mean_change in zip(bars, improved_values, mean_changes):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1.0,
            f"{value:.2f}%\nDelta p={mean_change:+.2f}",
            ha="center",
            va="bottom",
            fontsize=5.9,
            linespacing=1.25,
        )
    axis.set_xticks(np.arange(len(database_order)), labels)
    axis.set_ylim(0, 72)
    axis.set_ylabel("Higher probability, %")
    axis.set_title("Movement toward the eventual class", loc="left", fontsize=8.5, pad=7)
    axis.grid(axis="y", color="#DDDDDD", linewidth=0.45, alpha=0.75)
    axis.set_axisbelow(True)
    panel_label(axis, "C")

    axis = axes[1, 1]
    temporal_records = []
    for database, color, marker in (
        ("mimic4", COLORS["mimic4"], "o"),
        ("eicu", COLORS["eicu"], "s"),
    ):
        temporal = clinical["databases"][database]["temporal_vs_notemporal"]
        decomposition = temporal["flip_category_decomposition"]
        death_subgroup = temporal["true_death_subgroup"]
        temporal_records.append(
            (
                database,
                color,
                marker,
                {
                    "Closer, higher-coded": decomposition["correct_upgrade"][
                        "change_temporal_minus_notemporal"
                    ],
                    "Farther, higher-coded": decomposition["incorrect_upgrade"][
                        "change_temporal_minus_notemporal"
                    ],
                    "Closer, lower-coded": decomposition["correct_downgrade"][
                        "change_temporal_minus_notemporal"
                    ],
                    "Farther, lower-coded": decomposition["incorrect_downgrade"][
                        "change_temporal_minus_notemporal"
                    ],
                    "New predicted conversion": death_subgroup[
                        "new_upgrade_to_death"
                    ][
                        "change_temporal_minus_notemporal"
                    ],
                    "Death not predicted at 72 h": death_subgroup[
                        "missed_death_at72"
                    ][
                        "change_temporal_minus_notemporal"
                    ],
                },
            )
        )
    row_labels = list(temporal_records[0][3])
    y_positions = np.arange(len(row_labels))[::-1]
    for offset, (database, color, marker, records) in zip(
        (-0.16, 0.16), temporal_records
    ):
        for y_value, label in zip(y_positions, row_labels):
            record = records[label]
            estimate = 100 * record["estimate"]
            lower, upper = 100 * np.asarray(record["ci95"])
            axis.errorbar(
                estimate,
                y_value + offset,
                xerr=np.asarray(
                    [[estimate - lower], [upper - estimate]]
                ),
                fmt=marker,
                markersize=4.2,
                color=color,
                markerfacecolor=color,
                markeredgecolor=color,
                capsize=2.0,
                linewidth=0.8,
                label=LABELS[database] if label == row_labels[0] else None,
                zorder=3,
            )
    axis.axvline(0, color="#8B8B8B", linewidth=0.75, zorder=0)
    axis.set_yticks(y_positions, row_labels)
    axis.set_ylim(-0.65, len(row_labels) - 0.35)
    axis.set_xlim(-0.8, 1.45)
    axis.set_xlabel("With minus without temporal penalty (pp, 95% CI)")
    axis.set_title(
        "Temporal-penalty transition decomposition",
        loc="left",
        fontsize=8.5,
        pad=7,
    )
    axis.grid(axis="x", color="#DDDDDD", linewidth=0.45, alpha=0.75)
    axis.set_axisbelow(True)
    axis.legend(frameon=False, loc="upper right", fontsize=6.1)
    panel_label(axis, "D")

    figure.subplots_adjust(
        left=0.085,
        right=0.985,
        bottom=0.09,
        top=0.94,
        wspace=0.72,
        hspace=0.58,
    )
    save_figure(figure, "figure11_clinical_transitions_safety")


def table1_three_databases() -> None:
    frames = cohort_frames()
    rows = []
    for database in DATABASES:
        frame = frames[database].loc[frames[database]["y3"].notna()]
        for outcome, outcome_name in enumerate(
            ("Home", "Facility", "In-hospital death")
        ):
            subset = frame.loc[frame["y3"] == outcome]
            age = subset["age"]
            lactate = subset["lac_max"]
            rows.append(
                [
                    LABELS[database],
                    outcome_name,
                    f"{len(subset):,}",
                    (
                        f"{age.median():.0f} "
                        f"({age.quantile(0.25):.0f}-{age.quantile(0.75):.0f})"
                    ),
                    fixed(float(subset["male"].mean() * 100)),
                    (
                        f"{fixed(float(lactate.median()))} "
                        f"({fixed(float(lactate.quantile(0.25)))}-"
                        f"{fixed(float(lactate.quantile(0.75)))})"
                    ),
                ]
            )
    text = markdown_table(
        (
            "Database",
            "Outcome",
            "n",
            "Age, years, median (IQR)",
            "Male, %",
            "Peak lactate, mmol/L, median (IQR)",
        ),
        rows,
    )
    (TABLES / "table1_cohort.md").write_text(text, encoding="utf-8")


def table2_three_databases() -> None:
    frames = cohort_frames()
    rows = []
    for database in DATABASES:
        frame = frames[database]
        duration = duration_column(database)
        for hour in (24, 48, 72):
            subset = frame.loc[
                frame["y3"].notna() & (frame[duration] >= hour)
            ]
            rows.append(
                [
                    LABELS[database],
                    str(hour),
                    f"{len(subset):,}",
                    fixed(float((subset["y3"] == 0).mean() * 100)),
                    fixed(float((subset["y3"] == 1).mean() * 100)),
                    fixed(float((subset["y3"] == 2).mean() * 100)),
                ]
            )
    text = markdown_table(
        (
            "Database",
            "Landmark, h",
            "n",
            "Home, %",
            "Facility, %",
            "Death, %",
        ),
        rows,
    )
    (TABLES / "table2_landmarks.md").write_text(text, encoding="utf-8")


def table3_three_databases() -> None:
    shared = load_json(RESULTS / "shared_fold" / "summary.json")
    hospital = load_json(
        RESULTS / "eicu_hospital_validation" / "summary.json"
    )
    mimic3 = load_json(RESULTS / "mimic3_shared" / "summary.json")
    formal = load_json(RESULTS / "formal_summary" / "summary.json")
    rows = []
    for database in ("mimic4", "eicu"):
        values = [
            fixed(shared["databases"][database]["metrics"][str(hour)]["macro_auc"])
            for hour in (24, 48, 72)
        ]
        rows.append([LABELS[database], "XGBoost, patient-grouped", *values])
    hospital_values = [
        fixed(hospital["metrics"][str(hour)]["macro_auc"])
        for hour in (24, 48, 72)
    ]
    rows.append(["eICU-CRD", "XGBoost, hospital-held-out", *hospital_values])
    mimic3_values = [
        fixed(mimic3["discrimination"]["macro_auc"][str(hour)])
        for hour in (24, 48, 72)
    ]
    rows.append(
        ["MIMIC-III", "XGBoost, same-institution cross-era", *mimic3_values]
    )
    for database in ("mimic4", "eicu"):
        values = [
            fixed(
                formal[database]["ensemble"]["ordinal_temporal"]["metrics"][
                    "macro_auc"
                ][str(hour)]
            )
            for hour in (24, 48, 72)
        ]
        rows.append(
            [LABELS[database], "DG-GRU ordinal + temporal", *values]
        )
    text = markdown_table(
        (
            "Database",
            "Model / validation design",
            "Macro-AUC at 24 h",
            "Macro-AUC at 48 h",
            "Macro-AUC at 72 h",
        ),
        rows,
    )
    (TABLES / "table3_discrimination.md").write_text(text, encoding="utf-8")


def table4_three_databases() -> None:
    shared = load_json(RESULTS / "shared_fold" / "summary.json")["databases"]
    clinical = load_json(RESULTS / "clinical_flips" / "summary.json")["databases"]
    trajectory = load_json(
        RESULTS / "trajectory_patterns" / "summary.json"
    )["databases"]
    hospital = load_json(
        RESULTS / "eicu_hospital_validation" / "summary.json"
    )
    mimic3 = load_json(RESULTS / "mimic3_shared" / "summary.json")
    formal = load_json(RESULTS / "formal_summary" / "summary.json")
    rows = []
    for database in DATABASES:
        if database == "mimic3":
            flip = mimic3["stability"]["flip_24_72"]
            tv = mimic3["stability"]["total_variation_24_72"]["estimate"]
            confidence = mimic3["stability"]["confidence_ge_0_6"]
            coverage = confidence["coverage"]["estimate"]
            within = confidence["subsequent_flip_24_72"]["estimate"]
            major_death = mimic3["clinical_transitions"][
                "major_upgrade_to_death"
            ]["estimate"]
        else:
            record = shared[database]["stability"]
            flip = {
                "estimate": record["flip_24_72"],
                "ci95": record["flip_24_72_ci95"],
            }
            tv = record["tv_24_72"]
            confidence = record["confidence_24h"]["0.6"]
            coverage = confidence["coverage"]
            within = confidence["flip_24_72"]
            major_death = clinical[database]["models"]["xgboost"][
                "clinical_major_upgrades"
            ]["major_upgrade_to_death"]["estimate"]
        rows.append(
            [
                LABELS[database],
                "XGBoost",
                (
                    f"{fixed(flip['estimate'] * 100, 2)} "
                    f"({fixed(flip['ci95'][0] * 100, 2)}-"
                    f"{fixed(flip['ci95'][1] * 100, 2)})"
                ),
                fixed(
                    trajectory[database]["patterns"]["stable_all"][
                        "estimate"
                    ]
                    * 100,
                    2,
                ),
                fixed(
                    trajectory[database]["clinical_trajectories"][
                        "any_adjacent_change"
                    ]["estimate"]
                    * 100,
                    2,
                ),
                fixed(tv),
                fixed(coverage * 100, 2),
                fixed(within * 100, 2),
                fixed(major_death * 100, 2),
            ]
        )
    rows.append(
        [
            "eICU-CRD",
            "XGBoost, hospital-held-out",
            (
                f"{fixed(hospital['stability']['flip']['24_72']['estimate'] * 100, 2)} "
                f"({fixed(hospital['stability']['flip']['24_72']['ci95'][0] * 100, 2)}-"
                f"{fixed(hospital['stability']['flip']['24_72']['ci95'][1] * 100, 2)})"
            ),
            "",
            "",
            fixed(
                hospital["stability"]["total_variation"]["24_72"]["estimate"]
            ),
            fixed(
                hospital["stability"]["confidence_24h"]["0.6"]["coverage"]
                * 100,
                2,
            ),
            fixed(
                hospital["stability"]["confidence_24h"]["0.6"][
                    "subsequent_flip_24_72"
                ]["estimate"]
                * 100,
                2,
            ),
            fixed(
                hospital["clinical_changes"]["major_upgrade_to_death"][
                    "estimate"
                ]
                * 100,
                2,
            ),
        ]
    )
    for database in ("mimic4", "eicu"):
        record = formal[database]["ensemble"]["ordinal_temporal"]["stability"]
        rows.append(
            [
                LABELS[database],
                "DG-GRU ordinal + temporal",
                (
                    f"{fixed(record['flip_0_2'] * 100, 2)} "
                    f"({fixed(record['flip_0_2_ci95'][0] * 100, 2)}-"
                    f"{fixed(record['flip_0_2_ci95'][1] * 100, 2)})"
                ),
                "",
                "",
                fixed(record["tv_0_2"]),
                fixed(
                    record["confidence_24h"]["0.6"]["coverage"] * 100, 2
                ),
                fixed(
                    record["confidence_24h"]["0.6"]["flip"] * 100,
                    2,
                ),
                "",
            ]
        )
    text = markdown_table(
        (
            "Database",
            "Model / validation",
            "24->72 h flip, % (95% CI)",
            "Stable at all three landmarks, %",
            "Any adjacent change, %",
            "Total variation",
            "24 h confidence >=0.60, %",
            "Flip within stratum, %",
            "Predicted non-death to death conversion, %",
        ),
        rows,
    )
    (TABLES / "table4_stability.md").write_text(text, encoding="utf-8")


def table6_six_direction_transport() -> None:
    transport = load_json(
        RESULTS / "transport_extended" / "summary.json"
    )
    rows = []
    for direction in DIRECTIONS:
        record = transport["directions"][direction]
        metric = record["transport_landmarks"]["72"][
            "target_all_reaching_landmark"
        ]
        fixed_model = record["fixed_24_to_72_target_complete"]
        rows.append(
            [
                direction_label(direction),
                record["comparison_label"],
                f"{metric['n']:,}",
                fixed(metric["macro_auc"]),
                fixed(metric["accuracy"]),
                fixed(metric["log_loss"]),
                fixed(metric["brier_multiclass"]),
                fixed(metric["ece_top_label_10bin"]),
                fixed(fixed_model["flip"]["rate"] * 100, 2),
                fixed(fixed_model["total_variation_mean"]),
            ]
        )
    text = markdown_table(
        (
            "Direction",
            "Transport type",
            "Target n at 72 h",
            "Macro-AUC",
            "Accuracy",
            "Log loss",
            "Brier",
            "Top-label ECE",
            "24-to-72 h class flip, %",
            "Total variation",
        ),
        rows,
    )
    (TABLES / "table6_transport.md").write_text(text, encoding="utf-8")


def table5_sensitivity() -> None:
    sensitivity = load_json(RESULTS / "sensitivity" / "sensitivity_summary.json")
    sensitivity_v2 = load_json(
        RESULTS / "sensitivity_v2" / "sensitivity_v2_summary.json"
    )
    mimic3 = load_json(RESULTS / "mimic3_shared" / "summary.json")
    shared = load_json(RESULTS / "shared_fold" / "summary.json")
    xgb = {
        record["database"]: record
        for record in load_json(RESULTS / "landmarks" / "xgb_summary.json")
    }
    rows = []

    for record in sensitivity["six_vs_48"]:
        database = record["database"]
        subset = record["six_to_48h_complete_subset"]
        rows.append(
            [
                LABELS[database],
                "6 to 48 h landmark; available early predictors",
                f"{subset['n']:,}",
                fixed(100 * subset["flip"], 2),
                fixed(subset["tv"]),
                (
                    f"6 h {fixed(subset['auc_6'])}; "
                    f"48 h {fixed(subset['auc_48'])}"
                ),
            ]
        )

    for database in ("mimic4", "eicu"):
        definition = xgb[database]["definitions"]["4class"]["72"]
        rows.append(
            [
                LABELS[database],
                "Rehabilitation separated, four classes, 72 h XGBoost",
                f"{definition['n']:,}",
                "Not applicable",
                "Not applicable",
                fixed(definition["macro_auc"]),
            ]
        )

    for record in sensitivity["full_vs_complete"]:
        database = record["database"]
        for hour in (24, 48):
            landmark = record["landmarks"][str(hour)]
            full = landmark["full_landmark"]
            complete = landmark["complete_window_subset"]
            rows.append(
                [
                    LABELS[database],
                    f"Full {hour} h population vs complete-window subset",
                    f"{full['n']:,} vs {complete['n']:,}",
                    "Not applicable",
                    "Not applicable",
                    (
                        f"{hour} h full {fixed(full['macro_auc'])}; "
                        f"{hour} h complete {fixed(complete['macro_auc'])}"
                    ),
                ]
            )

    for scenario, label in (
        ("mimic4", "ICU-death exclusion, retrained XGBoost"),
        ("eicu", "ICU-death exclusion, retrained XGBoost"),
    ):
        subset = sensitivity_v2["icu_death_exclusion"][scenario][
            "complete_three_landmark_subset"
        ]
        rows.append(
            [
                LABELS[scenario],
                label,
                f"{subset['n']:,}",
                fixed(100 * subset["flip_24_72"], 2),
                fixed(subset["tv_24_72"]),
                fixed(subset["macro_auc"]["72"]),
            ]
        )

    for scenario, label in (
        ("mimic4_icd9", "MIMIC-IV ICD-9 sepsis coding"),
        ("mimic4_icd10", "MIMIC-IV ICD-10 sepsis coding"),
    ):
        subset = sensitivity_v2["icd_era"][scenario][
            "complete_three_landmark_subset"
        ]
        rows.append(
            [
                "MIMIC-IV",
                label,
                f"{subset['n']:,}",
                fixed(100 * subset["flip_24_72"], 2),
                fixed(subset["tv_24_72"]),
                fixed(subset["macro_auc"]["72"]),
            ]
        )

    rows.append(
        [
            "eICU-CRD",
            "Harmonised ICD-9-style diagnosis field; era strata unavailable",
            "Not applicable",
            "Not applicable",
            "Not applicable",
            "Not applicable",
        ]
    )

    for database in ("mimic4", "eicu"):
        record = sensitivity_v2["ordinal_order_reversal"][database][
            "reversed_order_restored"
        ]
        cohort_n = shared["databases"][database]["cohort"]["n"]
        rows.append(
            [
                LABELS[database],
                "Ordinal order reversed; probability axis restored",
                f"{cohort_n:,}",
                fixed(100 * record["flip_24_72"]["mean"], 2),
                fixed(record["tv_24_72"]["mean"]),
                fixed(record["auc_72"]["mean"]),
            ]
        )

    stability = mimic3["stability"]["flip_24_72"]
    rows.append(
        [
            "MIMIC-III",
            "Same-institution cross-era replication; complete cohort and shared folds",
            f"{mimic3['cohort']['n']:,}",
            (
                f"{fixed(stability['estimate'] * 100, 2)} "
                f"({fixed(stability['ci95'][0] * 100, 2)}-"
                f"{fixed(stability['ci95'][1] * 100, 2)})"
            ),
            fixed(mimic3["stability"]["total_variation_24_72"]["estimate"]),
            (
                f"24 h {fixed(mimic3['discrimination']['macro_auc']['24'])}; "
                f"48 h {fixed(mimic3['discrimination']['macro_auc']['48'])}; "
                f"72 h {fixed(mimic3['discrimination']['macro_auc']['72'])}"
            ),
        ]
    )
    text = markdown_table(
        (
            "Database",
            "Analysis",
            "n",
            "Flip, % (95% CI where available)",
            "Total variation",
            "Landmark macro-AUC (time: value)",
        ),
        rows,
    )
    (TABLES / "table5_sensitivity.md").write_text(text, encoding="utf-8")


def table7_recalibration_reassessment() -> None:
    transport = load_json(
        RESULTS / "transport_extended" / "summary.json"
    )
    risk = load_json(RESULTS / "flip_risk" / "summary.json")
    rows = []
    for direction in DIRECTIONS:
        record_72 = transport["recalibration"][direction]["72"]
        direction_text = direction_label(direction)
        for label_count in ("25", "200"):
            matched_baseline = record_72["untreated_heldout"][label_count]
            multinomial = record_72["curves"][label_count]
            temperature = record_72["curves_temperature_scaling"][label_count]
            paired = record_72[
                "temperature_scaling_minus_multinomial_logistic"
            ][label_count]["ece_top_label_10bin"]
            rows.append(
                [
                    direction_text,
                    label_count,
                    fixed(
                        matched_baseline["ece_top_label_10bin"]["mean"],
                        2,
                    ),
                    fixed(
                        multinomial["ece_top_label_10bin"]["mean"],
                        2,
                    ),
                    fixed(
                        temperature["ece_top_label_10bin"]["mean"],
                        2,
                    ),
                    (
                        f"{signed_fixed(paired['mean'], 2)} "
                        f"({signed_fixed(paired['p2_5'], 2)} to "
                        f"{signed_fixed(paired['p97_5'], 2)})"
                    ),
                    "",
                    "",
                    "",
                    "",
                ]
            )
    for database in DATABASES:
        oof = next(
            record
            for record in risk["within_database_oof"]
            if record["database"] == database
            and record["outcome"] == "any_flip"
        )
        simulation = risk["simulations"]["instability_risk_trigger"][
            "within_database"
        ][database]["risk_thresholds"]
        threshold = next(item for item in simulation if item["threshold"] == 0.1)
        rows.append(
            [
                LABELS[database],
                "Model-predicted endpoint-change trigger, 0.10",
                "",
                "",
                "",
                "",
                fixed(oof["metrics"]["auc"], 2),
                fixed(
                    threshold["reassessments_reduced_vs_all"] * 100, 2
                ),
                fixed(
                    threshold["major_death_upgrade_capture_fraction"] * 100,
                    2,
                ),
                fixed(
                    threshold["major_death_upgrade_capture_enrichment"],
                    2,
                ),
            ]
        )
    text = markdown_table(
        (
            "Direction or monitoring setting",
            "Local calibration labels",
            "Untreated ECE",
            "Multinomial ECE",
            "Temperature ECE",
            "Temperature minus multinomial ECE (paired-split 2.5th-97.5th)",
            "Trigger AUC",
            "Fewer stays selected, %",
            "Predicted conversions included, %",
            "Predicted-conversion enrichment",
        ),
        rows,
    )
    (TABLES / "table7_recalibration_reassessment.md").write_text(
        text, encoding="utf-8"
    )


def table8_shared_fold_metrics() -> None:
    shared = load_json(RESULTS / "shared_fold" / "summary.json")
    rows = []
    for database in ("mimic4", "eicu"):
        for hour in (24, 48, 72):
            metric = shared["databases"][database]["metrics"][str(hour)]
            rows.append(
                [
                    LABELS[database],
                    str(hour),
                    f"{metric['n']:,}",
                    fixed(metric["macro_auc"]),
                    fixed(metric["accuracy"]),
                    fixed(metric["log_loss"]),
                    fixed(metric["brier"]),
                    fixed(metric["ece_10bin"]),
                    fixed(metric["mean_max_probability"]),
                ]
            )
    text = markdown_table(
        (
            "Database",
            "Landmark, h",
            "n",
            "Macro-AUC",
            "Accuracy",
            "Log loss",
            "Brier score",
            "Top-label ECE",
            "Mean maximum probability",
        ),
        rows,
    )
    (TABLES / "table8_shared_fold_metrics.md").write_text(
        text, encoding="utf-8"
    )


def table9_hospital_validation() -> None:
    hospital = load_json(
        RESULTS / "eicu_hospital_validation" / "summary.json"
    )
    comparison = hospital["comparison_with_patient_grouped"]
    patient_metrics = comparison["patient_grouped_metrics_recomputed"]
    hospital_metrics = comparison["hospital_grouped_metrics"]
    differences = comparison["hospital_minus_patient_grouped"]
    site_distribution = hospital["hospital_level"][
        "distributions_across_primary_sites"
    ]["macro_auc_72"]
    site_summary = hospital["hospital_level"][
        "der_simonian_laird_72h_macro_auc"
    ]
    site_definition = hospital["hospital_level"]["definition"]

    rows = []
    for hour in (24, 48, 72):
        rows.append(
            [
                f"{hour} h macro-AUC",
                fixed(patient_metrics[str(hour)]["macro_auc"]),
                fixed(hospital_metrics[str(hour)]["macro_auc"]),
                signed_fixed(
                    differences["macro_auc"][str(hour)], 2
                ),
                "Not calculated",
            ]
        )
    rows.append(
        [
            "72 h top-label ECE",
            fixed(patient_metrics["72"]["ece_10bin"]),
            fixed(hospital_metrics["72"]["ece_10bin"]),
            signed_fixed(differences["ece_10bin"]["72"], 2),
            "Not calculated",
        ]
    )

    hospital_flip = hospital["stability"]["flip"]["24_72"]["estimate"]
    hospital_tv = hospital["stability"]["total_variation"]["24_72"][
        "estimate"
    ]
    hospital_major = hospital["clinical_changes"]["major_upgrade_to_death"][
        "estimate"
    ]
    hospital_home = hospital["clinical_changes"]["home_to_facility"][
        "estimate"
    ]
    for label, hospital_value, difference, denominator in (
        (
            "24->72 h class flip, %",
            hospital_flip,
            differences["flip_24_72"],
            1.0,
        ),
        (
            "24->72 h total variation",
            hospital_tv,
            differences["tv_24_72"],
            1.0,
        ),
        (
            "Predicted non-death to death conversion, %",
            hospital_major,
            differences["major_upgrade_to_death"],
            1.0,
        ),
        (
            "Home-to-facility transition, %",
            hospital_home,
            differences["home_to_facility"],
            1.0,
        ),
    ):
        patient_value = hospital_value - difference["estimate"]
        if label.endswith("%"):
            patient_text = fixed(100 * patient_value, 2)
            hospital_text = fixed(100 * hospital_value, 2)
            difference_only_text = signed_fixed(
                100 * difference["estimate"], 2
            )
            ci_text = (
                f"{signed_fixed(100 * difference['ci95'][0], 2)} to "
                f"{signed_fixed(100 * difference['ci95'][1], 2)}"
            )
        elif "total variation" in label:
            patient_text = fixed(patient_value)
            hospital_text = fixed(hospital_value)
            difference_only_text = signed_fixed(difference["estimate"])
            ci_text = (
                f"{signed_fixed(difference['ci95'][0])} to "
                f"{signed_fixed(difference['ci95'][1])}"
            )
        rows.append(
            [
                label,
                patient_text,
                hospital_text,
                difference_only_text,
                ci_text,
            ]
        )

    rows.append(
        [
            "Site-level 72 h macro-AUC, median (P10-P90)",
            "",
            (
                f"{fixed(site_distribution['p50'])} "
                f"({fixed(site_distribution['p10'])}-"
                f"{fixed(site_distribution['p90'])})"
            ),
            "",
            "",
        ]
    )
    rows.append(
        [
            "Descriptive random-effects summary, 72 h macro-AUC",
            "",
            (
                f"{fixed(site_summary['pooled_auc'])} "
                f"({fixed(site_summary['pooled_ci95'][0])}-"
                f"{fixed(site_summary['pooled_ci95'][1])}); "
                f"I2 {fixed(site_summary['i2_percent'], 2)}%; "
                f"k={site_summary['k']}; "
                f"n={int(site_definition['n_rows_primary']):,}; "
                f"coverage {fixed(100 * site_definition['fraction_of_cohort_primary'], 2)}%"
            ),
            "",
            "",
        ]
    )
    text = markdown_table(
        (
            "Validation metric",
            "Patient-grouped",
            "Hospital-held-out",
            "Hospital minus patient-grouped",
            "95% CI of paired difference",
        ),
        rows,
    )
    (TABLES / "table9_hospital_validation.md").write_text(
        text, encoding="utf-8"
    )


def table10_clinical_transitions() -> None:
    clinical = load_json(RESULTS / "clinical_flips" / "summary.json")
    mimic3 = load_json(RESULTS / "mimic3_shared" / "summary.json")

    def estimate(value):
        return value["estimate"] if isinstance(value, dict) else value

    rows = []
    for database in ("mimic4", "eicu"):
        models = clinical["databases"][database]["models"]
        model_specs = (
            ("xgboost", "XGBoost, shared-fold"),
            ("ordinal_notemporal", "Ordinal DG-GRU, no temporal"),
            ("ordinal_temporal", "Ordinal DG-GRU, temporal"),
        )
        for key, label in model_specs:
            model = models[key]
            flip = estimate(model["flip_24_72"])
            major = model["clinical_major_upgrades"][
                "major_upgrade_to_death"
            ]["estimate"]
            home = model["clinical_major_upgrades"]["home_to_facility"][
                "estimate"
            ]
            correctness = model["correctness_relative_to_final_outcome"][
                "flips"
            ]
            probability = model["true_class_probability"]["among_flips"]
            rows.append(
                [
                    LABELS[database],
                    label,
                    f"{correctness['gain']['denominator_n']:,}",
                    fixed(100 * flip, 2),
                    fixed(100 * major, 2),
                    fixed(100 * home, 2),
                    (
                        f"{fixed(100 * correctness['gain']['estimate'], 2)} / "
                        f"{fixed(100 * correctness['loss']['estimate'], 2)} / "
                        f"{fixed(100 * correctness['tie']['estimate'], 2)}"
                    ),
                    fixed(
                        100 * probability["improved"]["estimate"], 2
                    ),
                    signed_fixed(
                        probability["pt_true_change_mean"]["estimate"], 2
                    ),
                ]
            )

    correctness = mimic3["correctness"]["among_flips"]
    probability = mimic3["correctness"]["true_class_probability"][
        "among_flips"
    ]
    rows.append(
        [
            "MIMIC-III",
            "XGBoost, same-institution cross-era",
            f"{correctness['gain']['n']:,}",
            fixed(
                100 * mimic3["stability"]["flip_24_72"]["estimate"], 2
            ),
            fixed(
                100
                * mimic3["clinical_transitions"]["major_upgrade_to_death"][
                    "estimate"
                ],
                2,
            ),
            fixed(
                100 * mimic3["clinical_transitions"]["home_to_facility"][
                    "estimate"
                ],
                2,
            ),
            (
                f"{fixed(100 * correctness['gain']['estimate'], 2)} / "
                f"{fixed(100 * correctness['loss']['estimate'], 2)} / "
                f"{fixed(100 * correctness['tie']['estimate'], 2)}"
            ),
            fixed(100 * probability["improved"]["estimate"], 2),
            signed_fixed(probability["change_mean"], 2),
        ]
    )

    text = markdown_table(
        (
            "Database",
            "Model / validation",
            "Changed predictions, n",
            "24->72 h class change, %",
            "Non-death to death conversion, %",
            "Home to facility, %",
            (
                "Ordinal-distance closer / farther / tie "
                "among changed predictions, %"
            ),
            "Higher eventual-class probability, %",
            "Mean change in eventual-class probability",
        ),
        rows,
    )
    (TABLES / "table10_clinical_transitions.md").write_text(
        text, encoding="utf-8"
    )


def supplementary_s8_hospital_validation() -> None:
    hospital = load_json(
        RESULTS / "eicu_hospital_validation" / "summary.json"
    )
    summary = hospital["hospital_level"]["der_simonian_laird_72h_macro_auc"]
    rows = [
        [
            "All evaluable sites, random-effects summary",
            str(summary["k"]),
            f"{int(hospital['hospital_level']['definition']['n_rows_primary']):,}",
            fixed(
                100
                * hospital["hospital_level"]["definition"][
                    "fraction_of_cohort_primary"
                ],
                2,
            ),
            (
                f"{fixed(summary['pooled_auc'])} "
                f"({fixed(summary['pooled_ci95'][0])}-"
                f"{fixed(summary['pooled_ci95'][1])})"
            ),
            fixed(summary["i2_percent"], 2),
            "",
            "",
            "",
        ]
    ]
    frame = pd.read_csv(
        RESULTS / "eicu_hospital_validation" / "hospital_metrics.csv"
    )
    evaluable = frame.loc[
        frame["sufficient_n_ge_50"]
        & frame["all_outcomes_observed"]
        & frame["all_outcome_counts_ge_5"]
    ].sort_values("hospitalid")
    for _, record in evaluable.iterrows():
        rows.append(
            [
                f"Hospital {int(record['hospitalid'])}",
                "1",
                str(int(record["n"])),
                "",
                fixed(record["macro_auc_72"]),
                "",
                fixed(record["ece_10bin_72"]),
                fixed(record["flip_24_72"] * 100, 2),
                fixed(record["major_upgrade_to_death"] * 100, 2),
            ]
        )
    text = markdown_table(
        (
            "Hospital / summary",
            "Sites, k",
            "Stays, n",
            "Cohort coverage, %",
            "72 h macro-AUC",
            "I-squared, % (summary only)",
            "72 h top-label ECE",
            "24->72 h class change, %",
            "Non-death to death conversion, %",
        ),
        rows,
    )
    (TABLES / "supplementary_table_s8_hospital_validation.md").write_text(
        text, encoding="utf-8"
    )


def supplementary_s9_transport_full() -> None:
    transport = load_json(
        RESULTS / "transport_extended" / "summary.json"
    )
    rows = []
    for direction in DIRECTIONS:
        record = transport["directions"][direction]
        for hour in ("24", "48", "72"):
            metric = record["transport_landmarks"][hour][
                "target_all_reaching_landmark"
            ]
            rows.append(
                [
                    direction_label(direction),
                    record["comparison_label"],
                    hour,
                    f"{metric['n']:,}",
                    fixed(metric["macro_auc"]),
                    fixed(metric["accuracy"]),
                    fixed(metric["log_loss"]),
                    fixed(metric["brier_multiclass"]),
                    fixed(metric["ece_top_label_10bin"]),
                    fixed(metric["mean_max_probability"]),
                ]
            )
    text = markdown_table(
        (
            "Direction",
            "Transport type",
            "Landmark, h",
            "Target n",
            "Macro-AUC",
            "Accuracy",
            "Log loss",
            "Brier",
            "Top-label ECE",
            "Mean maximum probability",
        ),
        rows,
    )
    (TABLES / "supplementary_table_s9_transport_full.md").write_text(
        text, encoding="utf-8"
    )


def supplementary_s10_recalibration() -> None:
    transport = load_json(
        RESULTS / "transport_extended" / "summary.json"
    )
    rows = []
    for direction in DIRECTIONS:
        for hour in ("24", "48", "72"):
            record = transport["recalibration"][direction][hour]
            baseline = record["curves"]["0"]
            rows.append(
                [
                    direction_label(direction),
                    hour,
                    "Untreated reference",
                    baseline["requested_local_labels"],
                    f"{baseline['evaluation_n']:,}",
                    fixed(baseline["macro_auc"]["mean"]),
                    fixed(baseline["macro_auc"]["p2_5"]),
                    fixed(baseline["macro_auc"]["p97_5"]),
                    fixed(baseline["ece_top_label_10bin"]["mean"]),
                    fixed(baseline["ece_top_label_10bin"]["p2_5"]),
                    fixed(baseline["ece_top_label_10bin"]["p97_5"]),
                ]
            )
            for method, curve_key in (
                ("Multinomial logistic", "curves"),
                ("Temperature scaling", "curves_temperature_scaling"),
            ):
                curves = record[curve_key]
                for count in ("25", "50", "100", "200"):
                    curve = curves[count]
                    rows.append(
                        [
                            direction_label(direction),
                            hour,
                            method,
                            curve["requested_local_labels"],
                            f"{curve['evaluation_n']:,}",
                            fixed(curve["macro_auc"]["mean"]),
                            fixed(curve["macro_auc"]["p2_5"]),
                            fixed(curve["macro_auc"]["p97_5"]),
                            fixed(curve["ece_top_label_10bin"]["mean"]),
                            fixed(curve["ece_top_label_10bin"]["p2_5"]),
                            fixed(curve["ece_top_label_10bin"]["p97_5"]),
                        ]
                    )
    text = markdown_table(
        (
            "Direction",
            "Landmark, h",
            "Calibrator",
            "Local labels",
            "Evaluation n",
            "Macro-AUC mean",
            "AUC 2.5th repeated-split percentile",
            "AUC 97.5th repeated-split percentile",
            "ECE mean",
            "ECE 2.5th repeated-split percentile",
            "ECE 97.5th repeated-split percentile",
        ),
        rows,
    )
    (TABLES / "supplementary_table_s10_recalibration.md").write_text(
        text, encoding="utf-8"
    )


def supplementary_s11_reassessment() -> None:
    risk = load_json(RESULTS / "flip_risk" / "summary.json")
    rows = []
    for trigger_key, trigger_label in (
        ("instability_risk_trigger", "Model-predicted endpoint-change risk"),
        ("death_upgrade_risk_trigger", "Model-predicted conversion risk"),
    ):
        for database in DATABASES:
            simulation = risk["simulations"][trigger_key]["within_database"][
                database
            ]["risk_thresholds"]
            for record in simulation:
                no_upgrade_ci = record["ci95"][
                    "reassessment_without_major_death_upgrade_fraction"
                ]
                rows.append(
                    [
                        LABELS[database],
                        trigger_label,
                        fixed(record["threshold"], 2),
                        f"{record['reassessments']:,}",
                        fixed(record["reassessments_reduced_vs_all"] * 100, 2),
                        fixed(record["any_flip_capture_fraction"] * 100, 2),
                        fixed(record["major_death_upgrade_capture_fraction"] * 100),
                        fixed(
                            record[
                                "reassessment_without_major_death_upgrade_fraction"
                            ]
                            * 100,
                        ),
                        fixed(record["major_death_upgrades_per_100_reassessments"]),
                        fixed(
                            record["major_death_upgrade_capture_enrichment"],
                            2,
                        ),
                        (
                            f"{fixed(no_upgrade_ci[0] * 100, 2)}-"
                            f"{fixed(no_upgrade_ci[1] * 100, 2)}"
                        ),
                    ]
                )
    text = markdown_table(
        (
            "Database",
            "Trigger",
            "Predicted-risk threshold",
            "Selected stays",
            "Fewer stays selected, %",
            "Predicted endpoint changes included, %",
            "Predicted conversions included, %",
            "No predicted conversion among selected, %",
            "Predicted conversions per 100 selected stays",
            "Predicted-conversion enrichment",
            "No predicted conversion 95% interval, %",
        ),
        rows,
    )
    (TABLES / "supplementary_table_s11_reassessment.md").write_text(
        text, encoding="utf-8"
    )


def build() -> None:
    configure_style()
    figure1_three_databases()
    figure3_three_databases()
    figure4_three_databases()
    figure5_three_databases()
    figure7_transport_recalibration_reassessment()
    figure8_reassessment_simulation()
    figure9_sensitivity_analyses()
    figure10_hospital_validation()
    figure11_clinical_transitions_safety()
    table1_three_databases()
    table2_three_databases()
    table3_three_databases()
    table4_three_databases()
    table5_sensitivity()
    table6_six_direction_transport()
    table7_recalibration_reassessment()
    table8_shared_fold_metrics()
    table9_hospital_validation()
    table10_clinical_transitions()
    supplementary_s8_hospital_validation()
    supplementary_s9_transport_full()
    supplementary_s10_recalibration()
    supplementary_s11_reassessment()
    print(
        f"Three-database figures written to {FIGURES}; tables written to {TABLES}",
        flush=True,
    )


if __name__ == "__main__":
    build()
