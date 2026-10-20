"""Static figures for the README. Two colors only: GLM in blue, data in orange."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FixedLocator, NullLocator  # noqa: E402

from mtpl import config  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
REF = "#9b9a95"
MODEL = "#2a78d6"   # categorical slot 1
DATA = "#eb6834"    # categorical slot 2

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK_2,
    "axes.titlecolor": INK,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.labelsize": 9,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "legend.frameon": False,
    "legend.fontsize": 8,
    "font.size": 9,
    "lines.linewidth": 2,
})

REL_TICKS = [0.5, 0.75, 1, 1.5, 2, 3, 5, 10]


def _log_rel_axis(ax, axis: str = "y", lo: float = 0.4, hi: float = 12) -> None:
    ticks = [t for t in REL_TICKS if lo <= t <= hi]
    if axis == "y":
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(FixedLocator(ticks))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.set_yticklabels([f"{t:g}" for t in ticks])
        ax.set_ylim(lo, hi)
    else:
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xticklabels([f"{t:g}" for t in ticks])
        ax.set_xlim(lo, hi)


def one_way_vs_glm(rel: pd.DataFrame, path: Path, factors=("driver_age", "bonus_malus")) -> None:
    """Relativity by level, one-way (raw data) vs GLM, with exposure underneath."""
    fig = plt.figure(figsize=(10, 5.2))
    gs = fig.add_gridspec(2, len(factors), height_ratios=[3.2, 1], hspace=0.08, wspace=0.18)
    for j, f in enumerate(factors):
        d = rel[rel.factor == f].reset_index(drop=True)
        x = np.arange(len(d))
        ax = fig.add_subplot(gs[0, j])
        axb = fig.add_subplot(gs[1, j], sharex=ax)
        ax.axhline(1, color=REF, lw=1, zorder=1)
        ax.plot(x, d.one_way_relativity, color=DATA, ls="--", marker="o", ms=5,
                label="One-way (raw frequency ÷ base)", zorder=3)
        yerr = np.vstack([d.glm_relativity - d.ci_low, d.ci_high - d.glm_relativity])
        ax.errorbar(x, d.glm_relativity, yerr=yerr, color=MODEL, marker="o", ms=5,
                    capsize=0, elinewidth=1.2, label="GLM relativity (95% CI)", zorder=4)
        _log_rel_axis(ax, "y", 0.4, 12)
        ax.set_title(config.FACTOR_LABELS[f])
        ax.tick_params(labelbottom=False)
        ax.grid(axis="x", visible=False)
        if j == 0:
            ax.set_ylabel("Relativity vs base level (log scale)")
            ax.legend(loc="upper right")
        share = d.exposure_years / d.exposure_years.sum()
        axb.bar(x, share * 100, color="#c3c2b7", width=0.7)
        axb.set_xticks(x)
        axb.set_xticklabels(d.level, rotation=35, ha="right", rotation_mode="anchor")
        axb.grid(axis="x", visible=False)
        if j == 0:
            axb.set_ylabel("% exposure")
        base_idx = int(np.flatnonzero(d.is_base)[0])
        axb.get_xticklabels()[base_idx].set_fontweight("bold")
    years = rel[rel.factor == factors[0]].exposure_years.sum()
    fig.text(0.125, -0.04, f"Base level in bold. Training data, {years / 1000:.0f}k policy-years.",
             color=INK_2, fontsize=8)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def lift_chart(lift: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    x = lift.decile.to_numpy()
    ax.plot(x, lift.predicted_freq, color=MODEL, marker="o", ms=6, label="Predicted by GLM")
    ax.plot(x, lift.actual_freq, color=DATA, ls="none", marker="D", ms=7, label="Actual (holdout)")
    ax.set_xticks(x)
    ax.set_xlabel("Decile of predicted claim rate (equal exposure in each)")
    ax.set_ylabel("Claims per policy-year")
    ax.set_title("Holdout lift: actual vs predicted claim frequency")
    ax.set_ylim(0, None)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left")
    lo, hi = lift.actual_freq.iloc[0], lift.actual_freq.iloc[-1]
    ax.annotate(f"{hi / lo:.1f}× the claim rate\nof decile 1", xy=(x[-1], hi),
                xytext=(x[-1] - 2.6, hi * 0.97), color=INK_2, fontsize=8,
                arrowprops=dict(arrowstyle="-", color=REF, lw=0.8))
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def lorenz_chart(x: np.ndarray, y: np.ndarray, gini: float, ci: tuple[float, float], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    ax.plot([0, 1], [0, 1], color=REF, lw=1, ls="--")
    ax.plot(x, y, color=MODEL)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("Share of exposure (safest first)")
    ax.set_ylabel("Share of claims")
    ax.set_title("Ordered Lorenz curve, holdout")
    ax.text(0.04, 0.9, f"Gini {gini:.3f}\n95% CI {ci[0]:.3f}–{ci[1]:.3f}", color=INK, fontsize=9)
    ax.text(0.62, 0.55, "random\nranking", color=INK_2, fontsize=8)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def forest(rel: pd.DataFrame, path: Path) -> None:
    """Every relativity with its 95% CI, one panel per factor."""
    left = ["driver_age", "bonus_malus", "vehicle_age", "vehicle_power"]
    right = ["vehicle_brand", "fuel", "density", "region"]
    n = lambda fs: [int((rel.factor == f).sum()) for f in fs]
    fig = plt.figure(figsize=(10, 12))
    outer = fig.add_gridspec(1, 2, wspace=0.55, top=0.95)
    for col, fs in enumerate([left, right]):
        inner = outer[col].subgridspec(len(fs), 1, height_ratios=[k + 1.5 for k in n(fs)], hspace=0.5)
        for i, f in enumerate(fs):
            ax = fig.add_subplot(inner[i])
            d = rel[rel.factor == f].reset_index(drop=True)
            yy = np.arange(len(d))[::-1]
            ax.axvline(1, color=REF, lw=1)
            ax.hlines(yy, d.ci_low, d.ci_high, color=MODEL, lw=1.4)
            ax.plot(d.glm_relativity, yy, "o", color=MODEL, ms=4.5)
            ax.set_yticks(yy)
            ax.set_yticklabels(d.level, fontsize=7.5)
            base_idx = int(np.flatnonzero(d.is_base)[0])
            ax.get_yticklabels()[base_idx].set_fontweight("bold")
            ax.grid(axis="y", visible=False)
            hi = 12 if f == "bonus_malus" else 2.2
            _log_rel_axis(ax, "x", 0.45, hi)
            ax.set_title(config.FACTOR_LABELS[f], fontsize=9.5)
    fig.suptitle("GLM claim-frequency relativities with 95% confidence intervals "
                 "(base level in bold, = 1.00)", x=0.07, y=0.985, ha="left", fontsize=11, color=INK)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
