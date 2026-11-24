"""Trip chains from observed segments + reliability calibration.

A *chain* = consecutive observed segments of one bus (it left stop A and next reached stop B,
then left B ...). Chains of 5–30 stops are real bus trips with known actual times.

Calibration (training days only): ratio = actual / predicted ride time. Its quantiles by time
of day turn a point estimate into a range ("usually 52–67 min; on 9 of 10 days ≤ 67 min").
The held-out validation then checks the coverage honestly.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

BUCKETS = [(0, 6, "night"), (6, 10, "am_peak"), (10, 16, "midday"), (16, 20, "pm_peak"), (20, 24, "evening")]
QS = [0.1, 0.25, 0.5, 0.75, 0.85, 0.9]


def bucket_of(hour: int) -> str:
    for a, b, n in BUCKETS:
        if a <= hour < b:
            return n
    return "night"


def build_chains(seg: pd.DataFrame, max_samples: int, rng=None):
    rng = rng or np.random.default_rng(42)
    seg = seg.sort_values(["v", "dep_a"]).reset_index(drop=True)
    v = seg.v.to_numpy()
    link = np.r_[False, (v[1:] == v[:-1]) & (seg.ca.to_numpy()[1:] == seg.cb.to_numpy()[:-1])
                 & (np.abs(seg.dep_a.to_numpy()[1:] - seg.dep_b.to_numpy()[:-1]) < 1)]
    bounds = np.r_[np.nonzero(~link)[0], len(seg)]
    windows = []
    for s, e in zip(bounds[:-1], bounds[1:]):
        k = s
        while e - k >= 5:
            L = min(int(rng.integers(5, 31)), e - k)
            windows.append((k, k + L))
            k += L
    if len(windows) > max_samples:
        idx = rng.choice(len(windows), max_samples, replace=False)
        windows = [windows[i] for i in sorted(idx)]
    return seg, windows


def chain_predictions(tt, days, max_per_day=4000):
    """Yield dicts with actual and model-predicted time for chains on the given days."""
    rows = []
    for day in days:
        f = config.SEG_DIR / f"seg_{config.day_name(day)}.parquet"
        if not f.exists():
            continue
        seg, wins = build_chains(pd.read_parquet(f), max_per_day)
        pis = np.array([tt.pair(a, b) for a, b in zip(seg.ca.to_numpy(), seg.cb.to_numpy())])
        dep_a, arr_b = seg.dep_a.to_numpy(), seg.arr_b.to_numpy()
        dt = int(seg.dt.iloc[0])
        for s, e in wins:
            p = pis[s:e]
            if (p < 0).any():
                continue
            t0 = (dep_a[s] + config.IST_OFFSET_S) % 86400
            rows.append({"day": day, "dt": dt, "hour": int(t0 // 3600), "s": s, "e": e,
                         "actual": float(arr_b[e - 1] - dep_a[s]), "pred": tt.chain_time(p, dt, t0)})
    return pd.DataFrame(rows)


def calibrate(tt, days) -> dict:
    df = chain_predictions(tt, days, 3000)
    df = df[(df.actual > 60) & (df.pred > 0)]
    df["ratio"] = df.actual / df.pred
    df["b"] = df.hour.map(bucket_of)
    out = {"days": list(days), "n": int(len(df)), "quantiles": QS, "by_bucket": {}}
    allq = df.ratio.quantile(QS).round(3).tolist() if len(df) else [0.8, 0.9, 1, 1.1, 1.2, 1.3]
    out["all"] = allq
    for _, _, name in BUCKETS:
        x = df[df.b == name].ratio
        out["by_bucket"][name] = x.quantile(QS).round(3).tolist() if len(x) >= 50 else allq
    with open(config.MODEL_DIR / "calibration.json", "w") as f:
        json.dump(out, f, indent=1)
    return out


def load_calibration() -> dict | None:
    f = config.MODEL_DIR / "calibration.json"
    return json.loads(f.read_text()) if f.exists() else None
