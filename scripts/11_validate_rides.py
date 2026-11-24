"""Journey-planner validation on whole bus rides: planned legs vs real buses on the held-out week.

For random origin → destination trips on 25–31 Oct (model + ride calibration from 1–24 Oct), every bus
leg of the planner's best option is checked against the GPS stop visits of that day (app/ridecheck.py):
  ride   real buses of the same route that left the boarding stop within ±15 min of the planned time
         and reached the alighting stop; actual = median of their ride times.
  wait   mean wait for the next real bus of that route for a passenger arriving at a random moment
         within ±10 min — only at stops where the GPS sees ≥ 80 % of the route's buses.
  range  does the real ride fall inside the planner's "usually … / 9 of 10 days" range?
Writes docs/RIDE_VALIDATION.md and data/model/ride_validation.csv.

    python scripts/11_validate_rides.py --n 420
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from app import gtfs as gtfs_mod  # noqa: E402
from app.calibration import bucket_of  # noqa: E402
from app.planner import Planner  # noqa: E402
from app.ridecheck import BOARD_WINDOW_S, MIN_DETECTION, WAIT_WINDOW_S, collect_legs  # noqa: E402
from app.traveltime import TravelTimeModel  # noqa: E402


def score(df: pd.DataFrame, col: str) -> dict:
    err = df[col] - df.actual
    ape = err.abs() / df.actual
    return {"n": len(df), "MAE_min": err.abs().mean() / 60, "bias_min": err.mean() / 60,
            "MAPE_%": 100 * ape.mean(), "medAPE_%": 100 * ape.median(), "within20_%": 100 * (ape <= 0.2).mean()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=420, help="random trips")
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()
    t0 = time.time()
    g = gtfs_mod.load()
    tt = TravelTimeModel(g)
    try:
        from app.crowding import CrowdModel
        crowd = CrowdModel.load(g)
    except Exception:  # noqa: BLE001
        crowd = None
    pl = Planner(g, tt, crowd)
    days = [d for d in config.TEST_DAYS if (config.SEG_DIR / f"visits_{config.day_name(d)}.parquet").exists()]
    if not days:
        sys.exit("no visits_*.parquet for the held-out days — rerun scripts/01_gps_extract.py")
    df, n_plans = collect_legs(pl, g, days, a.n, seed=a.seed, live=True,
                               log=lambda m: print(f"{m} ({time.time() - t0:.0f} s)", flush=True))
    df.to_csv(config.MODEL_DIR / "ride_validation.csv", index=False)

    rides = df.dropna(subset=["actual"])
    rides = rides[rides.actual > 120].copy()
    both = rides.dropna(subset=["live"])
    cal = pl.ride_calib
    res = {"GTFS timetable": score(rides, "timetable")}
    if cal:
        res["GPS stretch model (no ride calibration)"] = score(rides, "model_raw")
    res["**BusPulse planner** (GPS model + ride calibration)"] = score(rides, "model")
    res = pd.DataFrame(res).T
    res_live = pd.DataFrame({"BusPulse planner": score(both, "model"),
                             "BusPulse planner + live blend": score(both, "live")}).T

    def table(t):
        out = ["| | legs | MAE min | bias min | MAPE % | median APE % | within 20 % |", "|---|---|---|---|---|---|---|"]
        for k, r in t.iterrows():
            out.append(f"| {k} | {int(r.n):,} | {r.MAE_min:.1f} | {r.bias_min:+.1f} | {r['MAPE_%']:.1f} | "
                       f"{r['medAPE_%']:.1f} | {r['within20_%']:.0f} |")
        return out

    lb = pd.cut(rides.actual / 60, [0, 15, 30, 45, 300], labels=["< 15 min", "15–30 min", "30–45 min", "> 45 min"])
    by_len = rides.groupby(lb, observed=True).apply(lambda x: pd.Series({
        "legs": len(x), "actual_med_min": x.actual.median() / 60,
        "timetable_MAPE": score(x, "timetable")["MAPE_%"], "model_MAPE": score(x, "model")["MAPE_%"]}))
    cover = []
    if cal:
        q = np.array([cal["by_bucket"].get(bucket_of(h), cal["all"]) for h in rides.hour])
        q = q / q[:, 2:3]
        ratio = (rides.actual / rides.model).to_numpy()
        cover = ["", "## Is the range honest?\n",
                 "The planner's ride range comes from whole-ride ratios on training days (18–24 Oct). On the held-out "
                 "week:\n", "| range | claimed % | observed % |", "|---|---|---|",
                 f"| inside 'usually' (p25–p75) | 50 | {100 * ((ratio >= q[:, 1]) & (ratio <= q[:, 3])).mean():.1f} |",
                 f"| below p85 | 85 | {100 * (ratio <= q[:, 4]).mean():.1f} |",
                 f"| below p90 ('on 9 of 10 days') | 90 | {100 * (ratio <= q[:, 5]).mean():.1f} |",
                 f"| inside p10–p90 | 80 | {100 * ((ratio >= q[:, 0]) & (ratio <= q[:, 5])).mean():.1f} |"]
    # door to door: trips where every leg has real rides and a well-detected real wait
    # real = planned walking + real waits + real rides; compared with BusPulse's and the timetable's totals
    legs_ok = df.assign(ok=df.actual.notna() & df.wait_actual_any.notna())
    g_ = legs_ok.groupby("trip").agg(n=("leg", "size"), n_legs=("n_legs", "first"), ok=("ok", "all"),
                                     walk=("walk_min", "first"), ours=("total_min", "first"),
                                     tt=("timetable_total_min", "first"),
                                     ride=("actual", "sum"), wait=("wait_actual_any", "sum"))
    d2d = g_[(g_.n == g_.n_legs) & g_.ok].copy()
    d2d["real"] = d2d.walk + (d2d.ride + d2d.wait) / 60
    d2d_lines = []
    if len(d2d) >= 10:
        def m(col):
            e = d2d[col] - d2d.real
            ape = e.abs() / d2d.real
            return e.abs().mean(), e.mean(), 100 * ape.mean(), 100 * (ape <= 0.2).mean()
        mo, mt = m("ours"), m("tt")
        d2d_lines = [
            "", "## Door to door (walk + wait + ride)\n",
            f"{len(d2d)} trips where every bus leg had real rides and a real wait at a well-detected stop that day "
            f"(real = planned walking + real waits + real rides; median {d2d.real.median():.0f} min, "
            f"{(d2d.n_legs > 1).mean() * 100:.0f} % with a change).\n",
            "| | MAE min | bias min | MAPE % | within 20 % |", "|---|---|---|---|---|",
            f"| GTFS timetable | {mt[0]:.1f} | {mt[1]:+.1f} | {mt[2]:.1f} | {mt[3]:.0f} |",
            f"| **BusPulse** | {mo[0]:.1f} | {mo[1]:+.1f} | {mo[2]:.1f} | {mo[3]:.0f} |"]
    w = df.dropna(subset=["wait_actual", "wait_model"])
    wt = w.dropna(subset=["wait_timetable"])
    lines = [
        "# Whole-ride validation (auto-generated by scripts/11_validate_rides.py)\n",
        f"{n_plans} random trips (3–18 km, 06–22 h) on the held-out days {days[0]}–{days[-1]} Oct; the model "
        f"and its ride calibration only saw 1–24 Oct. For each bus leg of the planner's best option, real buses "
        f"of the **same route** on **that day** that left the boarding stop within ±{BOARD_WINDOW_S // 60} min of "
        f"the planned time were followed to the alighting stop in the GPS stop visits. {len(rides)} of {len(df)} "
        f"legs had at least one such bus (median ride {rides.actual.median() / 60:.0f} min, "
        f"{rides.stops.median():.0f} stops; the rest: the GPS missed the boarding or alighting stop, or no bus of "
        f"that route ran in the window).\n",
        "## Ride time (boarding stop → alighting stop)\n", *table(res), "",
        "The stretch model alone under-predicts whole rides: real rides include long halts at timing points, "
        "jams slower than 3 km/h and stops the GPS missed, which the clean stretch observations leave out. "
        "The ride calibration (median actual/planned ratio per time of day, estimated on training-day rides) "
        "corrects that.\n",
        "By ride length (MAPE %):\n",
        "| actual ride | legs | median min | timetable MAPE % | BusPulse MAPE % |", "|---|---|---|---|---|",
        *[f"| {i} | {int(r.legs)} | {r.actual_med_min:.0f} | {r.timetable_MAPE:.1f} | {r.model_MAPE:.1f} |"
          for i, r in by_len.iterrows()],
        "", "### With the live blend (same legs)\n",
        "The live blend replays that day's GPS of the 15 min before departure.\n", *table(res_live),
        *cover,
        *d2d_lines,
        "", "## Waiting time at the first stop\n",
        f"Actual = mean wait until the next real bus of that route for a passenger arriving at a random moment "
        f"within ±{WAIT_WINDOW_S // 60} min of the planned arrival, at stops where the GPS sees "
        f"≥ {MIN_DETECTION:.0%} of the route's buses ({len(w)} legs).\n",
        "| | legs | actual mean min | predicted mean min | MAE min | bias min |", "|---|---|---|---|---|---|",
        f"| BusPulse (GPS headways incl. bunching, else timetable) | {len(w)} | {w.wait_actual.mean() / 60:.1f} | "
        f"{w.wait_model.mean() / 60:.1f} | {(w.wait_model - w.wait_actual).abs().mean() / 60:.1f} | "
        f"{(w.wait_model - w.wait_actual).mean() / 60:+.1f} |",
        f"| GTFS timetable (half the headway) | {len(wt)} | {wt.wait_actual.mean() / 60:.1f} | "
        f"{wt.wait_timetable.mean() / 60:.1f} | {(wt.wait_timetable - wt.wait_actual).abs().mean() / 60:.1f} | "
        f"{(wt.wait_timetable - wt.wait_actual).mean() / 60:+.1f} |",
        "", f"Per-leg data: `data/model/ride_validation.csv`. Run time {time.time() - t0:.0f} s.",
    ]
    (config.DOCS_DIR / "RIDE_VALIDATION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
