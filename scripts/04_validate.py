"""Held-out validation: real bus trips on 25–31 Oct vs model trained on ≤24 Oct.

From each held-out day, chains of consecutive observed segments of the same bus (5–30 stops)
are cut out. Actual = time from leaving the first stop to reaching the last stop. Predicted by:
  gps_model      time-dependent GPS model (this project)
  gps_live       gps_model + live blend from observations in the 15 min before departure
  gps_no_hour    GPS model averaged over hours (shows the value of time-of-day)
  timetable      GTFS scheduled stop-to-stop times
  const_speed    straight-line distance / one city-wide median speed
Writes docs/validation_results.md, data/model/validation_chains.csv, docs/img/*.png (if matplotlib).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from app import gtfs as gtfs_mod  # noqa: E402
from app.live import LiveFeed  # noqa: E402
from app.traveltime import TravelTimeModel  # noqa: E402

rng = np.random.default_rng(42)
METHODS = ["gps_model", "gps_live", "gps_no_hour", "timetable", "const_speed"]


from app.calibration import bucket_of, build_chains  # noqa: E402


def md_table(df: pd.DataFrame) -> str:
    cols = [str(df.index.name or "")] + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for idx, row in df.iterrows():
        vals = [f"{int(v):,}" if c == "n" else f"{v:,.1f}" if isinstance(v, (float, np.floating)) else str(v)
                for c, v in zip(df.columns, row.tolist())]
        lines.append("| " + " | ".join([str(idx)] + vals) + " |")
    return "\n".join(lines)


def metrics(df: pd.DataFrame, col: str) -> dict:
    err = df[col] - df.actual
    ape = (err.abs() / df.actual)
    return {"MAE_min": err.abs().mean() / 60, "bias_min": err.mean() / 60, "MAPE_%": 100 * ape.mean(),
            "medAPE_%": 100 * ape.median(), "within20_%": 100 * (ape <= 0.2).mean()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, nargs="*", default=config.TEST_DAYS)
    ap.add_argument("--max-per-day", type=int, default=6000)
    a = ap.parse_args()
    g = gtfs_mod.load()
    tt = TravelTimeModel(g)
    if not tt.has_gps:
        sys.exit("model tables missing — run scripts/02_build_traveltime.py first")
    city_speed = np.nanmedian(tt.city_speed) / 3.6
    nohour = tt.RUN.mean(axis=2)            # [P, 3]
    nohour_dw = tt.DWELL.mean(axis=2)
    rows = []
    for day in a.days:
        f = config.SEG_DIR / f"seg_{config.day_name(day)}.parquet"
        if not f.exists():
            continue
        seg = pd.read_parquet(f)
        seg, wins = build_chains(seg, a.max_per_day)
        live = LiveFeed(tt, seg)
        ca, cb = seg.ca.to_numpy(), seg.cb.to_numpy()
        pis = np.array([tt.pair(x, y) for x, y in zip(ca, cb)])
        dep_a, arr_b = seg.dep_a.to_numpy(), seg.arr_b.to_numpy()
        dt = int(seg.dt.iloc[0])
        for s, e in wins:
            p = pis[s:e]
            if (p < 0).any():
                continue
            t0 = (dep_a[s] + config.IST_OFFSET_S) % 86400
            actual = arr_b[e - 1] - dep_a[s]
            r = {"day": day, "dt": dt, "hour": int(t0 // 3600), "n_stops": e - s, "actual": actual,
                 "dist_km": float(tt.dist[p].sum() / 1000)}
            r["gps_model"] = tt.chain_time(p, dt, t0)
            tt.live = live
            r["gps_live"] = tt.chain_time(p, dt, t0, now=t0)
            tt.live = None
            r["gps_no_hour"] = float(nohour[p, dt].sum() + nohour_dw[p[:-1], dt].sum())
            r["timetable"] = float(tt.sched[p].sum())
            r["const_speed"] = float(tt.dist[p].sum() / city_speed)
            r["gps_levels_pct"] = float((tt.LEVEL[p, dt, r["hour"]] >= 2).mean() * 100)
            rows.append(r)
        print(f"day {day}: {len(wins)} chains")
    if not rows:
        sys.exit("no held-out segment files found")
    df = pd.DataFrame(rows)
    df = df[df.actual > 60]
    df.to_csv(config.MODEL_DIR / "validation_chains.csv", index=False)

    overall = pd.DataFrame({m: metrics(df, m) for m in METHODS}).T.round(1)
    df["hb"] = pd.cut(df.hour, [0, 6, 10, 16, 20, 24], right=False,
                      labels=["00-06", "06-10 AM peak", "10-16", "16-20 PM peak", "20-24"])
    by_hour = df.groupby("hb", observed=True).apply(
        lambda x: pd.Series({"n": len(x), **{f"{m} MAPE": metrics(x, m)["MAPE_%"] for m in METHODS[:4]}})).round(1)
    by_dt = df.groupby("dt").apply(
        lambda x: pd.Series({"n": len(x), **{f"{m} MAPE": metrics(x, m)["MAPE_%"] for m in METHODS[:4]}})).round(1)
    by_dt.index = [config.DAYTYPES[i] for i in by_dt.index]
    df["lb"] = pd.cut(df.n_stops, [5, 10, 20, 31], right=False, labels=["5-9 stops", "10-19", "20-30"])
    by_len = df.groupby("lb", observed=True).apply(
        lambda x: pd.Series({"n": len(x), "actual_min": x.actual.median() / 60,
                             **{f"{m} MAE_min": metrics(x, m)["MAE_min"] for m in METHODS[:4]}})).round(1)

    # ablation: each model component's contribution
    abl = pd.DataFrame({
        "1. GTFS timetable": metrics(df, "timetable"),
        "2. + GPS history (all hours)": metrics(df, "gps_no_hour"),
        "3. + hour of day & day type": metrics(df, "gps_model"),
        f"4. + live blend (last {config.LIVE_WINDOW_S // 60} min)": metrics(df, "gps_live"),
    }).T.round(1)
    abl.index.name = "model"
    # calibration: do the ranges cover what they claim?
    cal_md = "calibration.json missing (run 02)"
    if tt.calib:
        q = df.hour.map(lambda h: tt.calib["by_bucket"].get(bucket_of(h), tt.calib["all"]))
        lo10 = df.gps_model * q.map(lambda x: x[0])
        q25 = df.gps_model * q.map(lambda x: x[1])
        q75 = df.gps_model * q.map(lambda x: x[3])
        q85 = df.gps_model * q.map(lambda x: x[4])
        hi90 = df.gps_model * q.map(lambda x: x[5])
        cov = pd.DataFrame({"claimed %": [50, 85, 90, 80],
                            "observed %": [100 * ((df.actual >= q25) & (df.actual <= q75)).mean(),
                                           100 * (df.actual <= q85).mean(), 100 * (df.actual <= hi90).mean(),
                                           100 * ((df.actual >= lo10) & (df.actual <= hi90)).mean()]},
                           index=["inside p25–p75", "below p85", "below p90 ('9 of 10 days')", "inside p10–p90"]).round(1)
        cov.index.name = "range"
        cal_md = md_table(cov)
    md = [f"# Validation results (auto-generated by scripts/04_validate.py)\n",
          f"Held-out days: {sorted(df.day.unique().tolist())}; trained on days {tt.meta.get('train_days')}.",
          f"{len(df):,} real bus trip chains of 5–30 consecutive stops "
          f"(median {df.actual.median()/60:.1f} min, {df.dist_km.median():.1f} km straight-line).\n",
          "## Overall\n", md_table(overall), "\n## Ablation: what each part adds\n", md_table(abl),
          "\n## Calibration of the reliability range (calibrated on training days, checked here)\n", cal_md, "\n## MAPE by time of day\n", md_table(by_hour),
          "\n## MAPE by day type\n", md_table(by_dt), "\n## MAE by chain length\n", md_table(by_len)]
    config.DOCS_DIR.mkdir(exist_ok=True)
    (config.DOCS_DIR / "validation_results.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        img = config.DOCS_DIR / "img"
        img.mkdir(exist_ok=True)
        hm = df.groupby("hour").apply(lambda x: pd.Series({m: metrics(x, m)["MAPE_%"] for m in METHODS})).reindex(range(24))
        fig, ax = plt.subplots(figsize=(8, 4))
        for m, c in zip(METHODS, ["#1f6feb", "#0a8754", "#9a6700", "#cf222e", "#6e7781"]):
            ax.plot(hm.index, hm[m], label=m, color=c, lw=2 if m.startswith("gps") else 1.4)
        ax.set_xlabel("departure hour (IST)"); ax.set_ylabel("MAPE %"); ax.set_xticks(range(0, 24, 2))
        ax.set_title("Held-out trip time error by hour"); ax.legend(frameon=False); ax.grid(alpha=.3)
        fig.tight_layout(); fig.savefig(img / "validation_by_hour.png", dpi=120); plt.close(fig)
        fig, ax = plt.subplots(figsize=(5, 5))
        sm = df.sample(min(3000, len(df)), random_state=1)
        ax.scatter(sm.actual / 60, sm.timetable / 60, s=4, alpha=.3, color="#cf222e", label="timetable")
        ax.scatter(sm.actual / 60, sm.gps_model / 60, s=4, alpha=.3, color="#1f6feb", label="GPS model")
        lim = float(np.percentile(sm.actual / 60, 99))
        ax.plot([0, lim], [0, lim], "k--", lw=1); ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.set_xlabel("actual minutes"); ax.set_ylabel("predicted minutes"); ax.legend(frameon=False)
        fig.tight_layout(); fig.savefig(img / "validation_scatter.png", dpi=120); plt.close(fig)
        print(f"charts → {img}")
    except ImportError:
        print("matplotlib not installed — skipping charts")


if __name__ == "__main__":
    main()
