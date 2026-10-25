"""Report figures. Every function saves one PNG and returns its path.

Each method keeps the same colour in every figure, assigned in a fixed
order from a colour-blind-checked categorical palette, and every figure
with more than one series has a legend so identity never relies on colour.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save files only; works without a display
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .data import gap_runs  # noqa: E402

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
METHOD_ORDER = ["ffill", "linear", "yesterday", "cubic_spline", "pchip", "harmonic", "hybrid"]
COLOURS = dict(zip(METHOD_ORDER, PALETTE))
LABELS = {"ffill": "Forward fill", "linear": "Linear", "yesterday": "Same time yesterday",
          "cubic_spline": "Cubic spline", "pchip": "PCHIP", "harmonic": "Harmonic LSQ",
          "hybrid": "Hybrid"}


def _gap_label(hours: float) -> str:
    """Readable label for a gap length in hours."""
    if hours < 1:
        return f"{hours * 60:.0f} min"
    if hours < 24:
        return f"{hours:g} h"
    return f"{hours / 24:g} d"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 200, "font.size": 10,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "axes.titlesize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "xtick.color": MUTED, "ytick.color": MUTED, "legend.frameon": False,
    "lines.linewidth": 1.6,
})


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _date_axis(ax, tz):
    loc = mdates.AutoDateLocator(tz=tz)
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc, tz=tz))


def sensor_overview(frames: dict, path: Path) -> Path:
    """Temperature records of the usable sensors, missing slots shaded.

    Args:
        frames: Dict of title to regular-step frame.
    """
    fig, axes = plt.subplots(len(frames), 1, figsize=(10, 2.6 * len(frames)), squeeze=False)
    for ax, (title, f) in zip(axes[:, 0], frames.items()):
        ax.plot(f["time"], f["temperature"], color=PALETTE[0], linewidth=0.6)
        for s, L in zip(*gap_runs(f["temperature"].isna().to_numpy())):
            if L >= 18:  # shade outages of 3 h or more (singles are invisible at this scale)
                ax.axvspan(f["time"].iloc[s], f["time"].iloc[min(s + L, len(f) - 1)],
                           color=PALETTE[7], alpha=0.25, linewidth=0)
        miss = f["temperature"].isna().mean()
        ax.set_title(f"{title}  -  {miss:.1%} of slots missing (shaded: outages >= 3 h)",
                     loc="left")
        ax.set_ylabel("Temperature (°C)")
        _date_axis(ax, f["time"].dt.tz)
    return _save(fig, path)


def gap_length_histogram(frames: dict, path: Path) -> Path:
    """Distribution of real gap lengths (count per length, log scale)."""
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for i, (title, f) in enumerate(frames.items()):
        _, lengths = gap_runs(f["temperature"].isna().to_numpy())
        vals, counts = np.unique(lengths, return_counts=True)
        ax.scatter(vals * 10, counts, s=28, color=PALETTE[i], label=title,
                   edgecolor="white", linewidth=1, zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Gap length (minutes)")
    ax.set_ylabel("Number of gaps")
    ax.set_title("Real gap lengths: overwhelmingly single missed readings", loc="left")
    ax.legend()
    return _save(fig, path)


def error_vs_gap(summary: pd.DataFrame, path: Path, title: str,
                 acceptance: float | None = None, metric: str = "MAE",
                 boundary: float | None = None) -> Path:
    """Error metric against gap length for each method, with 95% CI bands for MAE.

    Args:
        summary: Combined summary (one row per method and gap length).
        acceptance: Draw this horizontal limit (°C).
        metric: Column to plot (``MAE``, ``P95AE``, ``MaxAE`` ...).
        boundary: Gap length (hours) where the fully observed design ends;
            marked with a dotted line.
    """
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    for m in [m for m in METHOD_ORDER if m in set(summary["method"])]:
        g = summary[summary["method"] == m].sort_values("gap_hours")
        ax.plot(g["gap_hours"], g[metric], marker="o", markersize=4.5,
                color=COLOURS[m], label=LABELS[m],
                linestyle="--" if m == "hybrid" else "-")
        if metric == "MAE" and "MAE_lo" in g:
            ax.fill_between(g["gap_hours"], g["MAE_lo"], g["MAE_hi"], color=COLOURS[m],
                            alpha=0.12, linewidth=0)
    if acceptance is not None:
        ax.axhline(acceptance, color=MUTED, linestyle="--", linewidth=1)
        ax.annotate(f"limit {acceptance} °C", (summary["gap_hours"].min(), acceptance),
                    xytext=(2, 3), textcoords="offset points", color=MUTED, fontsize=8)
    if boundary is not None:
        ax.axvline(boundary, color=MUTED, linestyle=":", linewidth=1)
        ax.annotate("fully observed | partially observed", (boundary, 1), xycoords=("data", "axes fraction"),
                    xytext=(0, -10), textcoords="offset points", ha="center", color=MUTED, fontsize=8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ticks = sorted(summary["gap_hours"].unique())
    ax.set_xticks(ticks, [_gap_label(h) for h in ticks])
    ax.minorticks_off()
    ax.set_xlabel("Gap length")
    ax.set_ylabel(f"{metric} (°C, log scale)")
    ax.set_title(title, loc="left")
    ax.legend(ncol=2, fontsize=9, loc="lower right")
    return _save(fig, path)


def design_agreement(agree: pd.DataFrame, path: Path) -> Path:
    """MAE with 95% CI from the fully vs partially observed designs, per method."""
    meths = [m for m in METHOD_ORDER if m in set(agree["method"])]
    hours = sorted(agree["gap_hours"].unique())
    fig, axes = plt.subplots(1, len(hours), figsize=(4.2 * len(hours), 0.5 * len(meths) + 1.4),
                             sharey=True, squeeze=False)
    for ax, h in zip(axes[0], hours):
        g = agree[agree["gap_hours"] == h].set_index("method")
        for i, m in enumerate(meths):
            for dy, d, mk in ((-0.15, "full", "o"), (0.15, "partial", "s")):
                v, lo, hi = g.loc[m, f"MAE_{d}"], g.loc[m, f"MAE_lo_{d}"], g.loc[m, f"MAE_hi_{d}"]
                ax.errorbar(v, i + dy, xerr=[[v - lo], [hi - v]], fmt=mk, color=COLOURS[m],
                            markersize=6, capsize=3, markerfacecolor="white" if d == "partial" else COLOURS[m])
        ax.set_yticks(range(len(meths)), [LABELS[m] for m in meths])
        ax.invert_yaxis()
        ax.set_xlabel("MAE (°C)")
        ax.set_title(f"{_gap_label(h)} gaps", loc="left")
    axes[0][0].plot([], [], "o", color=MUTED, label="fully observed")
    axes[0][0].plot([], [], "s", color=MUTED, markerfacecolor="white", label="partially observed")
    axes[0][0].legend(loc="lower right", fontsize=8)
    fig.suptitle("Do the two validation designs agree? (95% CIs)", x=0.01, ha="left", fontsize=11)
    return _save(fig, path)


def gap_count_plot(table: pd.DataFrame, slopes: dict, path: Path) -> Path:
    """Runtime against number of gaps at fixed series length (log-log)."""
    fig, ax = plt.subplots(figsize=(6.6, 4))
    for m in [m for m in METHOD_ORDER if m in set(table["method"])]:
        g = table[table["method"] == m]
        ax.plot(g["gaps"], g["seconds"], marker="o", markersize=4.5, color=COLOURS[m],
                label=f"{LABELS[m]} (slope {slopes[m]:.2f})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of gaps (series length fixed at 64,000 slots)")
    ax.set_ylabel("Fill time (s)")
    ax.set_title("Harmonic cost grows with the number of gaps fitted", loc="left")
    ax.legend(fontsize=9, loc="center right")
    return _save(fig, path)


def example_gap(frame: pd.DataFrame, start: int, gap_len: int, fills: dict,
                path: Path, context: int = 288) -> Path:
    """Show one synthetic gap: real data, the hidden truth and each method's fill."""
    lo, hi = max(0, start - context), min(len(frame), start + gap_len + context)
    t = frame["time"].iloc[lo:hi]
    y = frame["temperature"].to_numpy()[lo:hi]
    in_gap = np.zeros(hi - lo, dtype=bool)
    in_gap[start - lo:start + gap_len - lo] = True
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.axvspan(t.iloc[start - lo], t.iloc[start + gap_len - lo - 1], color=GRID, alpha=0.6,
               linewidth=0, label="Hidden gap")
    ax.plot(t, np.where(in_gap, np.nan, y), color=INK, linewidth=1.2, label="Data given to methods")
    ax.plot(t, np.where(in_gap, y, np.nan), color=INK, linewidth=1.2, linestyle=":",
            label="Hidden truth")
    for m, filled in fills.items():
        ax.plot(t[in_gap], filled[lo:hi][in_gap], color=COLOURS[m], label=LABELS[m])
    ax.set_ylabel("Temperature (°C)")
    ax.set_title(f"Example {_gap_label(gap_len / 6)} gap (held-out test period)", loc="left")
    _date_axis(ax, frame["time"].dt.tz)
    ax.legend(ncol=6, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    return _save(fig, path)


def sensitivity(table: pd.DataFrame, param: str, path: Path, title: str,
                xlabel: str) -> Path:
    """MAE against one parameter, one line per gap length."""
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for i, (L, g) in enumerate(table.groupby("gap_len")):
        g = g.sort_values(param)
        ax.plot(g[param], g["MAE"], marker="o", markersize=4.5, color=PALETTE[i],
                label=f"{L / 6:g} h gaps")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("MAE (°C)")
    ax.set_title(title, loc="left")
    ax.legend()
    return _save(fig, path)


def scaling_plot(times: pd.DataFrame, exponents: pd.DataFrame, path: Path) -> Path:
    """Runtime against series length on log-log axes, with fitted exponents."""
    exp = dict(zip(exponents["method"], exponents["exponent"]))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for m in [m for m in METHOD_ORDER if m in exp]:
        g = times[times["method"] == m]
        ax.plot(g["n"], g["seconds"], marker="o", markersize=4.5, color=COLOURS[m],
                label=f"{LABELS[m]} (slope {exp[m]:.2f})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Series length n (10-minute slots)")
    ax.set_ylabel("Fill time (s)")
    ax.set_title("Runtime scaling (slope 1 = linear in n)", loc="left")
    ax.legend(fontsize=9, loc="lower right")
    return _save(fig, path)


def error_by_hour(per_gap: pd.DataFrame, frame: pd.DataFrame, gap_len: int,
                  methods: list, path: Path) -> Path:
    """MAE by the local hour a gap starts at, for one gap length."""
    g = per_gap[per_gap["gap_len"] == gap_len].copy()
    g["hour"] = frame["time"].dt.hour.to_numpy()[g["start"].to_numpy()]
    g["bin"] = (g["hour"] // 3) * 3
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for m in methods:
        s = g[g["method"] == m].groupby("bin").apply(
            lambda d: d["sum_abs"].sum() / d["n"].sum(), include_groups=False)
        ax.plot(s.index, s.values, marker="o", markersize=4.5, color=COLOURS[m], label=LABELS[m])
    ax.set_xticks(range(0, 24, 3), [f"{h:02d}:00" for h in range(0, 24, 3)])
    ax.set_xlabel("Local time the gap starts (3-hour bins)")
    ax.set_ylabel("MAE (°C)")
    ax.set_title(f"Error depends on when a {_gap_label(gap_len / 6)} gap occurs", loc="left")
    ax.legend()
    return _save(fig, path)


def reconstruction(frame: pd.DataFrame, filled: np.ndarray, start: int, gap_len: int,
                   path: Path, label: str, context: int = 432) -> Path:
    """Show a real outage filled by the selected method."""
    lo, hi = max(0, start - context), min(len(frame), start + gap_len + context)
    t = frame["time"].iloc[lo:hi]
    y = frame["temperature"].to_numpy()[lo:hi]
    f = filled[lo:hi]
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.plot(t, y, color=INK, linewidth=0.9, label="Measured")
    ax.plot(t, np.where(np.isnan(y), f, np.nan), color=PALETTE[1], linewidth=1.4,
            label=f"Reconstructed ({label})")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title(f"Real {gap_len / 144:.1f}-day outage, reconstructed", loc="left")
    _date_axis(ax, frame["time"].dt.tz)
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    return _save(fig, path)
