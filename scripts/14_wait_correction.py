"""Experiment: should the wait correction depend on time of day? (training days only; uses the legs
collected by scripts/13_ride_correction.py: fit = 4–17 Oct, score = 18–24 Oct)

v5 multiplies the GPS headway wait by one factor (real ÷ predicted, ratio of means) and the timetable
half-headway by another. Here: one factor per time-of-day bucket, and a blend towards the global factor.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from app.calibration import bucket_of  # noqa: E402

MAX = config.MAX_WAIT_S


def load(tag):
    d = pd.read_parquet(config.MODEL_DIR / f"exp_legs_{tag}.parquet")
    d = d.dropna(subset=["wait_actual"]).copy()
    d["b"] = d.hour.map(bucket_of)
    d["src"] = np.where(d.wait_gps_raw.notna(), "gps", "tt")
    d["raw"] = d.wait_gps_raw.fillna(d.wait_timetable)
    return d.dropna(subset=["raw"])


def factors(tr, by_bucket, k=0.0):
    g = {s: x.wait_actual.sum() / x.raw.sum() for s, x in tr.groupby("src")}
    if not by_bucket:
        return lambda s, b: g[s]
    fb = {}
    for (s, b), x in tr.groupby(["src", "b"]):
        n = len(x)
        fb[(s, b)] = (n * x.wait_actual.sum() / x.raw.sum() + k * g[s]) / (n + k)
    return lambda s, b: fb.get((s, b), g[s])


def pred(te, f):
    p = np.array([r * f(s, b) for r, s, b in zip(te.raw, te.src, te.b)])
    tt = te.wait_timetable.fillna(0).to_numpy()
    return np.where(te.src == "tt", np.maximum(tt, np.minimum(p, MAX)), np.minimum(p, MAX))


def score(te, p):
    e = p - te.wait_actual
    return {"n": len(te), "MAE_min": e.abs().mean() / 60, "bias_min": e.mean() / 60,
            "RMSE_min": np.sqrt((e ** 2).mean()) / 60}


tr, te = load("fit"), load("score")
print(f"fit waits {len(tr)}  score waits {len(te)}  (gps {int((te.src == 'gps').sum())}, timetable {int((te.src == 'tt').sum())})")
rows = {"raw (no factor)": score(te, np.minimum(te.raw, MAX)),
        "global factors (v5)": score(te, pred(te, factors(tr, False))),
        "per bucket": score(te, pred(te, factors(tr, True))),
        "per bucket, shrunk k=50": score(te, pred(te, factors(tr, True, 50)))}
print(pd.DataFrame(rows).T.round(2).to_string())
f = factors(tr, True)
print("\nper-bucket factors (gps / timetable):",
      {b: (round(f("gps", b), 2), round(f("tt", b), 2)) for b in ["am_peak", "midday", "pm_peak", "evening", "night"]})
for name, ff in [("global", factors(tr, False)), ("per bucket", factors(tr, True))]:
    e = (pred(te, ff) - te.wait_actual) / 60
    print(f"bias by bucket ({name}):", e.groupby(te.b.to_numpy()).mean().round(1).to_dict())
