"""Shared matplotlib styling and chart builders.

Static PNGs for a README/portfolio, not an interactive dashboard, so there's
no hover layer here — but the same discipline applies everywhere else:
categorical hues assigned in a fixed order (never cycled), a single sequential
hue for magnitude, status colors reserved for retained-vs-lost framing and
always paired with a label, thin marks, recessive gridlines, no dual axes, no
rainbow colormaps. Palette values are the validated default from the `dataviz`
design-system reference (references/palette.md) — swap them here if this repo
ever adopts a different brand palette.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Categorical, fixed order — never reassign by rank
CATEGORICAL = {
    "blue": "#2a78d6",
    "orange": "#eb6834",
    "aqua": "#1baf7a",
    "yellow": "#eda100",
    "magenta": "#e87ba4",
    "green": "#008300",
    "violet": "#4a3aa7",
    "red": "#e34948",
}
CATEGORICAL_ORDER = list(CATEGORICAL.values())

# Sequential blue ramp, light -> dark (step 250 .. 650; 250 is the lightest
# step that still clears 2:1 contrast, per the ordinal-ramp rule)
SEQUENTIAL_BLUE = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95"]

STATUS = {
    "good": "#0ca30c",
    "warning": "#fab219",
    "serious": "#ec835a",
    "critical": "#d03b3b",
}

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"


def set_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "sans-serif",
            "font.size": 11,
            "text.color": INK_PRIMARY,
            "axes.edgecolor": BASELINE,
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlecolor": INK_PRIMARY,
            "axes.titleweight": "bold",
            "axes.grid": True,
            "axes.grid.axis": "y",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "grid.color": GRIDLINE,
            "grid.linewidth": 0.8,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.frameon": False,
            "legend.fontsize": 10,
            "figure.dpi": 120,
            "savefig.dpi": 150,
        }
    )


def sequential_cmap(name: str = "kkbox_blue") -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(name, SEQUENTIAL_BLUE)


def _label(value) -> str:
    """Cohort month Timestamps print as '2015-06' instead of a full datetime."""
    if hasattr(value, "strftime"):
        return value.strftime("%Y-%m")
    return str(value)


def _finalize(ax, title: str, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_title(title, loc="left", pad=12)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.tick_params(length=0)


def plot_cohort_retention_lines(
    pivot: "pd.DataFrame", title: str, max_series: int = 8, save_path: str | None = None
):
    """pivot: index=cycle, columns=cohort label, values=retention_rate.
    Caps at `max_series` lines (fixed categorical order) — for more cohorts
    than that, use `plot_cohort_heatmap` instead."""
    set_style()
    cohorts = list(pivot.columns)[:max_series]
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, cohort in enumerate(cohorts):
        color = CATEGORICAL_ORDER[i % len(CATEGORICAL_ORDER)]
        series = pivot[cohort].dropna()
        ax.plot(series.index, series.values, color=color, linewidth=2, label=_label(cohort))
        ax.scatter(series.index[-1:], series.values[-1:], color=color, s=24, zorder=3)

    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(lambda y, _: f"{y:.0%}")
    _finalize(ax, title, "Subscription cycle", "Cohort still renewing")
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.0))
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig, ax


def plot_cohort_heatmap(
    pivot: "pd.DataFrame", title: str, save_path: str | None = None
):
    """pivot: index=cohort_month (chronological), columns=cycle, values=retention_rate.
    Sequential single-hue encoding for magnitude — this is the right chart for
    'are newer cohorts retaining better or worse than older ones' at a glance."""
    set_style()
    fig, ax = plt.subplots(figsize=(9, max(4, 0.28 * len(pivot))))
    im = ax.imshow(pivot.values, cmap=sequential_cmap(), vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([_label(i) for i in pivot.index])
    ax.grid(False)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.ax.yaxis.set_major_formatter(lambda y, _: f"{y:.0%}")
    cbar.outline.set_visible(False)

    ax.set_title(title, loc="left", pad=12)
    ax.set_xlabel("Subscription cycle")
    ax.set_ylabel("Registration cohort")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig, ax


def plot_revenue_outcome_bars(
    df: "pd.DataFrame", title: str, save_path: str | None = None
):
    """df: index=cohort_month, columns must include renewed / voluntary_cancel
    / lapsed_no_renewal revenue totals. Status colors, always labeled."""
    set_style()
    colors = {
        "renewed": STATUS["good"],
        "voluntary_cancel": STATUS["serious"],
        "lapsed_no_renewal": STATUS["critical"],
    }
    labels = {
        "renewed": "Renewed (retained)",
        "voluntary_cancel": "Voluntary cancel (lost)",
        "lapsed_no_renewal": "Lapsed, no renewal (lost)",
    }
    cols = [c for c in ["renewed", "voluntary_cancel", "lapsed_no_renewal"] if c in df.columns]

    fig, ax = plt.subplots(figsize=(10, 5))
    bottom = np.zeros(len(df))
    x = np.arange(len(df))
    for col in cols:
        ax.bar(
            x,
            df[col].values,
            bottom=bottom,
            color=colors[col],
            label=labels[col],
            width=0.7,
        )
        bottom += df[col].values

    ax.set_xticks(x)
    ax.set_xticklabels([_label(i) for i in df.index], rotation=45, ha="right")
    _finalize(ax, title, "Registration cohort", "Revenue (NT$)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig, ax


def plot_ordinal_bars(
    labels: list[str], values: list[float], title: str, ylabel: str, save_path: str | None = None
):
    """Discrete ordered categories (e.g. discount depth buckets) — ordinal
    ramp of the sequential blue hue, darkest = highest tier."""
    set_style()
    n = len(labels)
    step = np.linspace(0.25, 1.0, n)
    cmap = sequential_cmap()
    colors = [cmap(s) for s in step]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(labels, values, color=colors, width=0.65)
    ax.yaxis.set_major_formatter(lambda y, _: f"{y:.0%}")
    _finalize(ax, title, "", ylabel)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig, ax


def plot_segment_risk_bars(
    labels: list[str], values: list[float], title: str, xlabel: str, save_path: str | None = None
):
    """Horizontal bar, single categorical hue (one series = no legend needed),
    sorted by the caller, direct value labels at bar end."""
    set_style()
    fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(labels))))
    y = np.arange(len(labels))
    ax.barh(y, values, color=CATEGORICAL["blue"], height=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    for yi, v in zip(y, values):
        ax.text(v, yi, f"  {v:,.0f}", va="center", ha="left", color=INK_SECONDARY, fontsize=9)
    ax.grid(axis="x", color=GRIDLINE, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    _finalize(ax, title, xlabel, "")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig, ax
