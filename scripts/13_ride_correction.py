"""Experiment: which whole-ride correction should the planner use? (training days only)

    python scripts/13_ride_correction.py collect --days 4-17 --n 1400 --tag fit
    python scripts/13_ride_correction.py collect --days 18-24 --n 700 --tag score
    python scripts/13_ride_correction.py compare --fit fit --score score

Real bus rides take longer than the planner's sum of stop-to-stop stretches (halts at timing points,
jams, stops the GPS barely sees). v5 corrects this with one median ratio per time of day. Here we ask
whether the ratio should also depend on how long the ride is and on the route, choosing on the inner
split (fit 4–17 Oct, score 18–24 Oct). The held-out week (25–31 Oct) is not touched by this script.
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
from app.calibration import bucket_of  # noqa: E402

LEN_EDGES = [0, 10, 20, 35, 1e9]          # minutes of raw model ride
MIN_CELL = 30
ROUTE_K = 5.0                             # shrinkage: a route needs ~5 real rides to move halfway


def days_arg(s: str) -> list[int]:
    a, _, b = s.partition("-")
    return list(range(int(a), int(b or a) + 1))


def collect(days, n, tag, seed):
    from app import gtfs as gtfs_mod
    from app.planner import Planner
    from app.ridecheck import collect_legs
    from app.traveltime import TravelTimeModel
    g = gtfs_mod.load()
    tt = TravelTimeModel(g)
    pl = Planner(g, tt)
    pl.ride_calib = None                                   # learn from the raw model, as v5 does
    days = [d for d in days if (config.SEG_DIR / f"visits_{config.day_name(d)}.parquet").exists()]
    t0 = time.time()
    df, n_plans = collect_legs(pl, g, days, n, seed=seed, live=False,
                               log=lambda m: print(f"{m} ({time.time() - t0:.0f} s)", flush=True))
    df["route_name"] = df.route.astype(str)
    out = config.MODEL_DIR / f"exp_legs_{tag}.parquet"
    df.to_parquet(out, index=False)
    print(f"{n_plans} plans, {len(df)} legs → {out}")


def prep(df):
    r = df.dropna(subset=["actual"]).copy()
    r = r[r.actual > 120]
    r["b"] = r.hour.map(bucket_of)
    r["lb"] = pd.cut(r.model_raw / 60, LEN_EDGES, labels=False, right=False)
    r["lr"] = np.log(r.actual / r.model_raw)
    return r


def fit(tr, variant):
    """Returns a dict usable by predict(); all medians in log space (median ratio)."""
    m = {"all": float(tr.lr.median()), "b": tr.groupby("b").lr.median().to_dict(), "bl": {}, "route": {}}
    if variant in ("len", "len+route", "len+route_b"):
        for (b, lb), x in tr.groupby(["b", "lb"]):
            if len(x) >= MIN_CELL:
                m["bl"][(b, int(lb))] = float(x.lr.median())
    if variant in ("len+route", "len+route_b"):
        base = base_pred(tr, m)
        res = (tr.lr - base).groupby(tr.route_name if variant == "len+route" else [tr.route_name, tr.b])
        m["route"] = {k: float(v.sum() / (len(v) + ROUTE_K)) for k, v in res}
        m["route_by_b"] = variant == "len+route_b"
    return m


def base_pred(df, m):
    out = []
    for b, lb in zip(df.b, df.lb):
        v = m["bl"].get((b, int(lb)))
        out.append(v if v is not None else m["b"].get(b, m["all"]))
    return pd.Series(out, index=df.index)


def predict(df, m):
    p = base_pred(df, m)
    if m["route"]:
        keys = list(zip(df.route_name, df.b)) if m.get("route_by_b") else list(df.route_name)
        p = p + pd.Series([m["route"].get(k, 0.0) for k in keys], index=df.index)
    return df.model_raw * np.exp(p)


def score(df, pred):
    e = pred - df.actual
    ape = e.abs() / df.actual
    return {"legs": len(df), "MAE_min": e.abs().mean() / 60, "bias_min": e.mean() / 60, "MAPE_%": 100 * ape.mean(),
            "medAPE_%": 100 * ape.median(), "within20_%": 100 * (ape <= .2).mean()}


def compare(fit_tag, score_tag):
    tr = prep(pd.read_parquet(config.MODEL_DIR / f"exp_legs_{fit_tag}.parquet"))
    te = prep(pd.read_parquet(config.MODEL_DIR / f"exp_legs_{score_tag}.parquet"))
    print(f"fit legs {len(tr)}  score legs {len(te)}  routes in fit {tr.route_name.nunique()}  "
          f"score legs on a fit route {100 * te.route_name.isin(set(tr.route_name)).mean():.0f} %")
    rows = {"raw model (no correction)": score(te, te.model_raw)}
    for v in ["bucket", "len", "len+route", "len+route_b"]:
        rows[v] = score(te, predict(te, fit(tr, v)))
    res = pd.DataFrame(rows).T
    print(res.round(2).to_string())
    print("\nbias (min) by raw-length bin on the score set:")
    for v in ["bucket", "len", "len+route"]:
        p = predict(te, fit(tr, v))
        print(f"  {v:10s}", ((p - te.actual) / 60).groupby(te.lb).mean().round(1).to_dict())
    print("\nfitted median ratio by bucket × length (fit set):")
    print(np.exp(tr.groupby(["b", "lb"]).lr.median().unstack()).round(2).to_string())
    return res


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    c = sp.add_parser("collect")
    c.add_argument("--days", required=True)
    c.add_argument("--n", type=int, default=700)
    c.add_argument("--tag", required=True)
    c.add_argument("--seed", type=int, default=21)
    k = sp.add_parser("compare")
    k.add_argument("--fit", default="fit")
    k.add_argument("--score", default="score")
    a = ap.parse_args()
    if a.cmd == "collect":
        collect(days_arg(a.days), a.n, a.tag, a.seed)
    else:
        compare(a.fit, a.score)


if __name__ == "__main__":
    main()
