"""Redraw the README/docs charts in the BusPulse style from saved outputs (no raw data needed)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

IMG = config.DOCS_DIR / "img"
INK, MUTED, GRID = "#14212b", "#5b6b78", "#e6ebef"
BLUE, GREEN, AMBER, SLATE = "#1a5fd0", "#0b6e4f", "#d97706", "#9aa6b2"
plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 11, "axes.edgecolor": GRID,
                     "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
                     "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 1,
                     "xtick.major.size": 0, "ytick.major.size": 0, "figure.dpi": 100, "savefig.dpi": 200})


def frame(fig, title, sub, note):
    """Left-aligned title + subtitle on top, source note at the bottom."""
    fig.text(0.035, 0.955, title, fontsize=17, fontweight="bold", color=INK, va="top")
    fig.text(0.035, 0.875, sub, fontsize=11.5, color=MUTED, va="top")
    fig.text(0.035, 0.03, note, fontsize=9, color=SLATE, va="bottom")


def end_label(ax, x, y, text, color, dy=0, bold=False):
    ax.annotate(text, (x, y), xytext=(8, dy), textcoords="offset points", va="center", color=color,
                fontsize=11, fontweight="bold" if bold else "normal")


def hour_axis(ax):
    ax.set_xticks(range(0, 24, 3))
    ax.set_xticklabels(["12 am", "3 am", "6 am", "9 am", "12 pm", "3 pm", "6 pm", "9 pm"])
    ax.set_xlim(-0.6, 23.3)


def error_by_hour():
    df = pd.read_csv(config.MODEL_DIR / "validation_chains.csv")
    series = [("timetable", "Timetable", AMBER, 1.8), ("const_speed", "One city speed", SLATE, 1.8),
              ("gps_model", "BusPulse", BLUE, 3.0), ("gps_live", "BusPulse + live", GREEN, 2.2)]
    fig = plt.figure(figsize=(10, 5.4)); ax = fig.add_axes([0.07, 0.13, 0.76, 0.66])
    ends = {}
    for col, name, c, lw in series:
        m = df.assign(e=(df[col] - df.actual).abs() / df.actual * 100).groupby("hour").e.mean().reindex(range(24))
        ax.plot(m.index, m.values, color=c, lw=lw, solid_capstyle="round", zorder=3 if "gps" in col else 2)
        ends[name] = (23, float(m.iloc[-1]), c)
    ax.axvspan(17, 20, color="#fff4e0", zorder=0); ax.text(18.5, 2, "evening peak", ha="center", color=AMBER, fontsize=9.5)
    offs = {"BusPulse": 6, "BusPulse + live": -7}
    for name, (x, y, c) in ends.items():
        end_label(ax, x, y, name, c, offs.get(name, 0), bold=name == "BusPulse")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%")); ax.set_ylim(0, None)
    hour_axis(ax)
    frame(fig, "BusPulse stays near 15% error at every hour", "Average error on real bus trips of 5–30 stops, "
          "by departure hour. Held-out week 25–31 Oct 2026 the model never saw.",
          f"{len(df):,} real trips · source: docs/VALIDATION.md")
    fig.savefig(IMG / "validation_by_hour.png", facecolor="white"); plt.close(fig)


def predicted_vs_actual():
    df = pd.read_csv(config.MODEL_DIR / "validation_chains.csv").sample(20000, random_state=1)
    a = df.actual / 60; lim = float(np.percentile(a, 98))
    fig = plt.figure(figsize=(10, 5.4))
    for i, (col, name, c) in enumerate([("timetable", "Timetable", AMBER), ("gps_model", "BusPulse", BLUE)]):
        ax = fig.add_axes([0.07 + i * 0.47, 0.14, 0.4, 0.64]); ax.grid(axis="both")
        p = df[col] / 60
        for f in (0.8, 1.2):  # ±20% guides
            ax.plot([0, lim], [0, lim * f], color=SLATE, lw=0.9, ls=(0, (1, 2.5)), zorder=2)
        ax.plot([0, lim], [0, lim], color=INK, lw=1, ls=(0, (4, 3)), zorder=2)
        ax.hexbin(a, p, gridsize=45, extent=(0, lim, 0, lim), mincnt=1, cmap=matplotlib.colors.LinearSegmentedColormap
                  .from_list("x", ["#f3f6f9", c]), bins="log", linewidths=0, zorder=1)
        inside = ((p - a).abs() <= 0.2 * a).mean() * 100
        ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_xlabel("real trip, min"); ax.set_ylabel("predicted, min")
        ax.text(0.04, 0.95, f"{name}", transform=ax.transAxes, fontsize=13, fontweight="bold", color=c, va="top")
        ax.text(0.04, 0.86, f"{inside:.0f}% within ±20% (dotted lines)", transform=ax.transAxes, color=MUTED, va="top")
    frame(fig, "Predicted vs real: the timetable spreads, BusPulse sits on the line",
          "Each cell counts real bus trips (darker = more). Dashed line = perfect prediction.",
          "20,000 of the held-out trips, 25–31 Oct 2026 · source: docs/VALIDATION.md")
    fig.savefig(IMG / "validation_scatter.png", facecolor="white"); plt.close(fig)


def speed_by_hour():
    sp = json.loads((config.MODEL_DIR / "insights.json").read_text(encoding="utf-8"))["speed_by_hour"]
    fig = plt.figure(figsize=(10, 5.0)); ax = fig.add_axes([0.07, 0.14, 0.78, 0.63])
    for k, name, c, lw in [("sunday", "Sunday", SLATE, 1.8), ("saturday", "Saturday", AMBER, 1.8), ("weekday", "Weekday", GREEN, 3)]:
        y = np.array(sp[k]); ax.plot(range(24), y, color=c, lw=lw, solid_capstyle="round")
        end_label(ax, 23, y[-1], name, c, {"Weekday": 0, "Saturday": 15, "Sunday": -15}[name], bold=k == "weekday")
        if k == "weekday":
            ax.fill_between(range(24), y, color=GREEN, alpha=0.07)
            h = int(np.argmin(y)); ax.scatter([h], [y[h]], s=46, color=GREEN, zorder=4, edgecolor="white", linewidth=2)
            ax.annotate(f"{y[h]:.1f} km/h at 6 pm", (h, y[h]), xytext=(-12, -26), textcoords="offset points",
                        ha="right", color=INK, fontsize=11, fontweight="bold")
            ax.scatter([2], [y[2]], s=46, color=GREEN, zorder=4, edgecolor="white", linewidth=2)
            ax.annotate(f"{y[2]:.0f} km/h at 2 am", (2, y[2]), xytext=(8, 8), textcoords="offset points", color=INK, fontsize=11)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}")); ax.set_ylim(15, 47); ax.set_ylabel("km/h")
    hour_axis(ax)
    frame(fig, "Delhi's buses slow to half speed by the evening peak", "Median bus speed between stops, "
          "by hour of the day, from 31 days of bus GPS.", "Source: data/model/insights.json · docs/INSIGHTS.md")
    fig.savefig(IMG / "speed_by_hour.png", facecolor="white"); plt.close(fig)


def google():
    g = pd.read_csv(config.DOCS_DIR / "google_comparison.csv").dropna(subset=["google_min", "ours_min"])
    fig = plt.figure(figsize=(10, 5.4)); ax = fig.add_axes([0.08, 0.14, 0.5, 0.64]); ax.grid(axis="both")
    lim = float(max(g.google_min.max(), g.ours_min.max())) * 1.05
    ax.plot([0, lim], [0, lim], color=INK, lw=1, ls=(0, (4, 3)))
    ax.text(lim * 0.86, lim * 0.78, "same as Google", rotation=34, ha="center", color=MUTED, fontsize=9.5)
    for tm, name, c in [("08:30", "8:30 am", BLUE), ("18:00", "6 pm", AMBER)]:
        x = g[g.time.astype(str).str.startswith(tm[:2])]
        ax.scatter(x.google_min, x.ours_min, s=46, color=c, edgecolor="white", linewidth=1.2, label=name, zorder=3)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_xlabel("Google Maps, min"); ax.set_ylabel("BusPulse, min")
    ax.legend(frameon=False, loc="upper left")
    side = fig.add_axes([0.66, 0.2, 0.3, 0.5]); side.axis("off")
    side.text(0, 1.0, "Who was closer to the real trip?", fontsize=12.5, fontweight="bold", va="top")
    side.text(0, 0.84, "15 trips we could follow on real buses", color=MUTED, va="top")
    for i, (name, v, c) in enumerate([("BusPulse", 8.7, BLUE), ("Google Maps", 17.4, SLATE)]):
        y = 0.6 - i * 0.26
        side.text(0, y, name, fontsize=11, va="center", fontweight="bold" if i == 0 else "normal")
        side.barh(y - 0.09, v / 17.4 * 0.95, height=0.07, left=0, color=c)
        side.text(v / 17.4 * 0.95 + 0.02, y - 0.09, f"{v} min off", va="center", fontsize=10.5, color=INK)
    side.set_xlim(0, 1.35); side.set_ylim(0, 1)
    frame(fig, "Mornings we agree with Google; at 6 pm we expect the crawl",
          "Door-to-door minutes for the same 47 bus trips (Tuesday, 8:30 am and 6 pm).",
          "Source: docs/google_comparison.csv · docs/VALIDATION.md §3")
    fig.savefig(IMG / "google_scatter.png", facecolor="white"); plt.close(fig)


if __name__ == "__main__":
    IMG.mkdir(parents=True, exist_ok=True)
    error_by_hour(); predicted_vs_actual(); speed_by_hour(); google()
    print("charts →", IMG)
