"""Generate the four core figures used in the metapopulation research paper.

Run from the repository root:
    python code/publication_figures.py

Outputs are saved to figures/publication/ as PNG and PDF files.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

CODE_DIR = Path(__file__).resolve().parent
ROOT_DIR = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from movement_sweep_engine import (  # noqa: E402
    MOVEMENT_STRENGTHS,
    N_REGIONS,
    REGION_COLORS,
    run_sweep,
)

OUTPUT_DIR = ROOT_DIR / "figures" / "publication"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
})


def closest_index(values, target):
    return int(np.argmin(np.abs(np.asarray(values) - target)))


def save_figure(fig, filename):
    for extension in ("png", "pdf"):
        fig.savefig(OUTPUT_DIR / f"{filename}.{extension}", dpi=300,
                    bbox_inches="tight")


def figure_1(synchronization, peak_spreads):
    fig, ax = plt.subplots(figsize=(12, 7))

    ax.plot(MOVEMENT_STRENGTHS, synchronization["line"], color="#2E86AB",
            linewidth=3.8, marker="o", markersize=7,
            label="Line Network (Adjacent-Only Coupling)", alpha=0.85)
    ax.plot(MOVEMENT_STRENGTHS, synchronization["circle"], color="#A23B72",
            linewidth=3.5, marker="o", markersize=7, linestyle="--",
            label="Fully Connected Circle (All-Pairs Coupling)", alpha=0.85)

    ax.axhline(1.0, color="#D62828", linewidth=2, linestyle=":",
               label="Synchronization Threshold (SD = 1 day)")
    ax.fill_between(MOVEMENT_STRENGTHS, 0, 1.0, color="#06A77D",
                    alpha=0.08, label="Synchronized Regime (SD < 1 day)")

    ax.set_xlabel("Movement Strength (k)", fontweight="bold")
    ax.set_ylabel("Synchronization SD (days)", fontweight="bold")
    ax.set_title("FIGURE 1 | Peak-Time Variability Falls Faster in the Circle Network",
                 fontweight="bold", loc="left", pad=12)
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.30)
    ax.set_facecolor("#FAFAFA")
    ax.legend(frameon=True, fontsize=10, loc="upper right",
              edgecolor="black")

    k05 = closest_index(MOVEMENT_STRENGTHS, 0.05)
    k20 = closest_index(MOVEMENT_STRENGTHS, 0.20)
    line_cross = np.where(synchronization["line"] <= 1.0)[0]
    circle_cross = np.where(synchronization["circle"] <= 1.0)[0]
    line_text = (f"k = {MOVEMENT_STRENGTHS[line_cross[0]]:.2f}"
                 if len(line_cross) else "not reached")
    circle_text = (f"k = {MOVEMENT_STRENGTHS[circle_cross[0]]:.2f}"
                   if len(circle_cross) else "not reached")

    textbox = (
        "Key Quantitative Results:\n"
        f"At k = 0.05: Circle SD = {synchronization['circle'][k05]:.3f} d | "
        f"Line SD = {synchronization['line'][k05]:.3f} d\n"
        f"At k = 0.20: Circle SD = {synchronization['circle'][k20]:.3f} d | "
        f"Line SD = {synchronization['line'][k20]:.3f} d\n"
        f"Circle reaches SD <= 1 day at {circle_text}; line at {line_text}"
    )
    ax.text(0.02, 0.98, textbox, transform=ax.transAxes, fontsize=10,
            va="top", bbox=dict(boxstyle="round,pad=0.7",
            facecolor="lightyellow", alpha=0.90, edgecolor="#333"))

    fig.tight_layout()
    save_figure(fig, "figure_1_sync_variability")
    plt.close(fig)


def figure_2(peak_times):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for axis, geometry in zip(axes, ("line", "circle")):
        for region in range(1, N_REGIONS):
            if geometry == "circle" and region in (3, 4):
                continue
            difference = peak_times[geometry][:, region] - peak_times[geometry][:, 0]
            if geometry == "circle":
                label = "Region 2 ≈ Region 5" if region == 1 else "Region 3 ≈ Region 4"
            else:
                label = f"Region {region + 1}"
            axis.plot(MOVEMENT_STRENGTHS, difference,
                      color=REGION_COLORS[region], linewidth=3,
                      marker="o", markersize=6, label=label, alpha=0.85)

        axis.axhline(0.0, color="black", linewidth=1.8,
                     linestyle="--", alpha=0.8)
        axis.fill_between(MOVEMENT_STRENGTHS, -1, 1, color="#06A77D",
                          alpha=0.08, label="Synchronized Zone (±1 day)")
        axis.set_xlabel("Movement Strength (k)", fontweight="bold")
        axis.set_ylabel("Peak-Time Difference from Region 1 (days)",
                        fontweight="bold")
        if geometry == "line":
            title = "Line Network: Sequential Propagation"
        else:
            title = "Circle Network: Symmetric Pairs (R2≈R5, R3≈R4)"
        axis.set_title(title, fontweight="bold", pad=12)
        axis.grid(True, linestyle="--", linewidth=0.7, alpha=0.30)
        axis.set_facecolor("#FAFAFA")
        axis.legend(frameon=True, fontsize=10.5, loc="best",
                    edgecolor="black")

    fig.suptitle("FIGURE 2 | Regional Peak-Time Offsets Across Movement Strength",
                 fontsize=15, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_figure(fig, "figure_2_peak_time_offsets")
    plt.close(fig)


def figure_5(peak_times):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    shared_min = min(np.min(peak_times["line"]), np.min(peak_times["circle"]))
    shared_max = max(np.max(peak_times["line"]), np.max(peak_times["circle"]))

    for axis, geometry in zip(axes, ("line", "circle")):
        image = axis.imshow(
            peak_times[geometry].T,
            aspect="auto", origin="lower", interpolation="nearest",
            extent=[MOVEMENT_STRENGTHS.min(), MOVEMENT_STRENGTHS.max(),
                    0.5, N_REGIONS + 0.5],
            cmap="RdYlBu_r", vmin=shared_min, vmax=shared_max,
        )
        axis.set_xlabel("Movement Strength (k)", fontweight="bold")
        axis.set_ylabel("Region", fontweight="bold")
        axis.set_yticks(range(1, N_REGIONS + 1))
        axis.set_yticklabels([f"R{i}" for i in range(1, N_REGIONS + 1)])
        if geometry == "circle":
            axis.text(0.02, 0.98, "R2 ≈ R5\nR3 ≈ R4",
                      transform=axis.transAxes, fontsize=11,
                      va="top", fontweight="bold",
                      bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                                alpha=0.85, edgecolor="black"))
        axis.set_title(
            "Line Network:\nSequential Spatial Propagation" if geometry == "line"
            else "Circle Network:\nGlobal Homogenization",
            fontsize=14, fontweight="bold", pad=12,
        )
        axis.grid(alpha=0.15, color="white", linewidth=0.5)
        colorbar = fig.colorbar(image, ax=axis, label="Peak Time (days)", pad=0.02)
        colorbar.ax.tick_params(labelsize=10)

    fig.suptitle("FIGURE E | Heatmaps: Spatial Propagation vs Homogenization",
                 fontsize=15, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_figure(fig, "figure_e_heatmaps")
    plt.close(fig)


def figure_7(synchronization, peak_spreads):
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis("off")

    k05 = closest_index(MOVEMENT_STRENGTHS, 0.05)
    k10 = closest_index(MOVEMENT_STRENGTHS, 0.10)
    k20 = closest_index(MOVEMENT_STRENGTHS, 0.20)
    circle_cross = np.where(synchronization["circle"] <= 1.0)[0]
    line_cross = np.where(synchronization["line"] <= 1.0)[0]

    circle_transition = (f"{MOVEMENT_STRENGTHS[circle_cross[0]]:.2f}"
                         if len(circle_cross) else ">0.20")
    line_transition = (f"{MOVEMENT_STRENGTHS[line_cross[0]]:.2f}"
                       if len(line_cross) else ">0.20")

    table_data = [
        ["Metric", "Line Network", "Fully Connected Circle"],
        ["Synchronization at k=0.05", f"{synchronization['line'][k05]:.3f} days", f"{synchronization['circle'][k05]:.3f} days"],
        ["Synchronization at k=0.10", f"{synchronization['line'][k10]:.3f} days", f"{synchronization['circle'][k10]:.3f} days"],
        ["Synchronization at k=0.20", f"{synchronization['line'][k20]:.3f} days", f"{synchronization['circle'][k20]:.3f} days"],
        ["Peak Spread at k=0.05", f"{peak_spreads['line'][k05]:.1f} days", f"{peak_spreads['circle'][k05]:.1f} days"],
        ["First sampled k with SD ≤ 1", line_transition, circle_transition],
        ["Geometric Symmetry", "5 unique regions", "R2 ≈ R5, R3 ≈ R4 (3 unique)"],
        ["Observed endpoint", "SD and spread at k=0.20", "SD and spread at k=0.20"],
        ["Movement rule", "Adjacent-only", "All-pairs"],
    ]

    left = 0.08
    column_width = 0.28
    row_height = 0.095
    start_y = 0.86

    for row_index, row in enumerate(table_data):
        y = start_y - row_index * row_height
        if row_index == 0:
            background = "#2E86AB"
            foreground = "white"
            weight = "bold"
            fontsize = 11.5
        else:
            background = "#E8F4F8" if row_index % 2 == 0 else "white"
            foreground = "black"
            weight = "normal"
            fontsize = 11

        for column, text in enumerate(row):
            x = left + column * column_width
            ax.add_patch(plt.Rectangle((x, y - row_height), column_width,
                                       row_height, facecolor=background,
                                       edgecolor="#333", linewidth=1.2))
            ax.text(x + 0.015, y - row_height / 2, text,
                    fontsize=fontsize, fontweight=weight, color=foreground,
                    va="center", ha="left")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.5, 0.96,
            "FIGURE 7 | Quantitative Summary: Network Topology Comparison",
            fontsize=15, fontweight="bold", ha="center")

    fig.tight_layout()
    save_figure(fig, "figure_7_quantitative_summary")
    plt.close(fig)


def main():
    results = run_sweep()
    model_order = ("line", "circle")

    synchronization = {
        geometry: np.array([item["synchronization"] for item in results[geometry]])
        for geometry in model_order
    }
    peak_spreads = {
        geometry: np.array([item["peak_spread"] for item in results[geometry]])
        for geometry in model_order
    }
    peak_times = {
        geometry: np.array([item["peak_times"] for item in results[geometry]])
        for geometry in model_order
    }

    figure_1(synchronization, peak_spreads)
    figure_2(peak_times)
    figure_5(peak_times)
    figure_7(synchronization, peak_spreads)

    print("Generated core publication figures in:", OUTPUT_DIR)
    for path in sorted(OUTPUT_DIR.glob("figure_*")):
        print(" -", path)


if __name__ == "__main__":
    main()
