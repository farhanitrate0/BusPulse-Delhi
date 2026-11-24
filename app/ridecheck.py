"""Whole-ride checks against real buses (GPS stop visits): used for ride calibration on training days
(scripts/02_build_traveltime.py) and for the held-out ride validation (scripts/11_validate_rides.py).

A planned bus leg = route r from boarding stop A to alighting stop B at time t. Real rides = buses of
route r on that day that left A within ±15 min of t and later reached B (same vehicle, no data gap,
passing most stops in between); actual ride = median of their A→B times.

Stop detection: a bus passing a stop is only seen if its GPS comes within 50 m of the stop cluster.
At some stops (stop placed off the road, two clusters close together) most passes are missed, which
makes headways look longer than they are. detection(r, c) = buses of route r seen at stop c ÷ buses
seen at the route's best-detected stops; waits are only measured / trusted where it is high.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from app.geo import haversine_m  # noqa: E402

BOARD_WINDOW_S = 900       # buses leaving the boarding stop within ±15 min of the planned time
WAIT_WINDOW_S = 600        # passenger arrival spread for the wait check
MIN_DETECTION = 0.8        # stop must see ≥ 80 % of the route's buses for an "actual wait"


class DayVisits:
    """All GPS stop visits of one day, sorted by (vehicle, time), indexed by (route, cluster)."""

    def __init__(self, day: int):
        f = config.SEG_DIR / f"visits_{config.day_name(day)}.parquet"
        d = pd.read_parquet(f).sort_values(["v", "arr"]).reset_index(drop=True)
        self.v = pd.factorize(d.v)[0]
        self.r, self.c = d.r.to_numpy(np.int64), d.c.to_numpy(np.int64)
        self.arr, self.dep = d.arr.to_numpy(), d.dep.to_numpy()
        self.gs, self.ge = d["gs"].to_numpy(), d["ge"].to_numpy()     # (d.ge would be DataFrame.ge)
        self.tod = (self.dep + config.IST_OFFSET_S) % 86400
        self.idx = pd.DataFrame({"r": self.r, "c": self.c}).groupby(["r", "c"]).indices
        nb = pd.DataFrame({"r": self.r, "c": self.c, "v": self.v}).drop_duplicates().groupby(["r", "c"]).size()
        top = nb.groupby(level=0).transform(lambda s: s.nlargest(3).median())
        self.detection = (nb / top).clip(upper=1.0).to_dict()
        self._deps: dict = {}

    def departures(self, r: int, c: int) -> np.ndarray:
        """Departure times (time of day) of route r at stop c, one per bus pass."""
        k = (r, c)
        if k not in self._deps:
            ii = self.idx.get(k, np.array([], dtype=np.int64))
            ii = ii[np.argsort(self.dep[ii])]
            t, vv = self.tod[ii], self.v[ii]
            keep = np.ones(len(ii), dtype=bool)
            last: dict = {}
            for j in range(len(ii)):                       # same bus again within 10 min = same pass
                p = last.get(vv[j])
                if p is not None and t[j] - p < 600:
                    keep[j] = False
                last[vv[j]] = t[j]
            self._deps[k] = np.sort(t[keep])
        return self._deps[k]

    def rides(self, r: int, cl_seq: list[int], t_board: float, max_s: float) -> list[float]:
        a, b = cl_seq[0], cl_seq[-1]
        mids = set(cl_seq[1:-1])
        out = []
        for i in self.idx.get((r, a), []):
            if abs(self.tod[i] - t_board) > BOARD_WINDOW_S:
                continue
            seen = 0
            for j in range(i + 1, min(i + 400, len(self.v))):
                if self.v[j] != self.v[i] or self.r[j] != r or self.arr[j] - self.dep[i] > max_s:
                    break
                if self.c[j] in mids:
                    seen += 1
                if self.c[j] == b:
                    if self.gs[j] == self.ge[i] and (not mids or seen >= 0.4 * len(mids)):
                        out.append(float(self.arr[j] - self.dep[i]))
                    break
        return out

    def actual_wait(self, r: int, c: int, t: float):
        """Mean wait for the next real bus of route r at stop c, arrival uniform in t ± 10 min."""
        if self.detection.get((r, c), 0.0) < MIN_DETECTION:
            return None
        deps = self.departures(r, c)
        if len(deps) < 2:
            return None
        arrivals = t + np.linspace(-WAIT_WINDOW_S, WAIT_WINDOW_S, 41)
        k = np.searchsorted(deps, arrivals)
        ok = k < len(deps)
        if ok.mean() < 0.9:
            return None
        w = deps[k[ok]] - arrivals[ok]
        return float(w.mean()) if w.max() < 7200 else None


def sample_trips(pl, g, rng, n: int):
    """Random origin/destination stop clusters 3–18 km apart where several routes stop."""
    busy = np.nonzero(pl.cl_npat >= 3)[0]
    out = []
    while len(out) < n:
        i, j = rng.choice(busy, 2, replace=False)
        if 3000 <= haversine_m(g["cl_lat"][i], g["cl_lon"][i], g["cl_lat"][j], g["cl_lon"][j]) <= 18000:
            out.append((f"{g['cl_lat'][i]:.5f},{g['cl_lon'][i]:.5f}", f"{g['cl_lat'][j]:.5f},{g['cl_lon'][j]:.5f}",
                        int(rng.integers(6, 22)), int(rng.choice([0, 15, 30, 45]))))
    return out


def collect_legs(pl, g, days: list[int], n: int, seed: int = 3, live: bool = True, log=print) -> tuple[pd.DataFrame, int]:
    """Plan random trips on `days` and check every bus leg of the best option against real buses."""
    from app.live import LiveFeed
    from app.planner import PlanError
    rng = np.random.default_rng(seed)
    tt = pl.tt
    rows, n_plans, trip = [], 0, 0
    per_day = int(np.ceil(n / len(days)))
    for day in days:
        dv = DayVisits(day)
        feed = None
        if live:
            seg = pd.read_parquet(config.SEG_DIR / f"seg_{config.day_name(day)}.parquet",
                                  columns=["ca", "cb", "arr_b", "run"])
            feed = LiveFeed(tt, seg, label=f"{day} Oct")
        date = f"{config.YEAR}-{config.MONTH:02d}-{day:02d}"
        for o, d, hh, mm in sample_trips(pl, g, rng, per_day):
            dep = f"{date}T{hh:02d}:{mm:02d}"
            try:
                tt.live = None
                static = pl.plan(o, d, dep, "fastest", 1)
                lv_plan = None
                if feed is not None:
                    tt.live = feed
                    lv_plan = pl.plan(o, d, dep, "fastest", 1, live=True)
            except PlanError:
                continue
            finally:
                tt.live = None
            n_plans += 1
            if not static["options"]:
                continue
            opt = static["options"][0]
            trip += 1
            live_legs = [x for x in (lv_plan["options"][0]["legs"] if lv_plan and lv_plan["options"] else [])
                         if x["type"] == "bus"]
            for li, leg in enumerate([x for x in opt["legs"] if x["type"] == "bus"]):
                P = g["patterns"][leg["pattern"]]
                r = pl.pat_rint[leg["pattern"]]
                stops = list(P["stops"])
                try:
                    bpos = stops.index(leg["board_stop"]["stop_id"])
                    apos = stops.index(leg["alight_stop"]["stop_id"], bpos + 1)
                except ValueError:
                    continue
                cl_seq = [int(c) for c in P["cl"][bpos:apos + 1]]
                hms = leg["depart"][11:19]                      # planned boarding time, IST
                tb = int(hms[:2]) * 3600 + int(hms[3:5]) * 60 + int(hms[6:8])
                act = dv.rides(r, cl_seq, tb, max_s=3 * leg["ride_s"] + 900)
                lv = live_legs[li] if li < len(live_legs) and live_legs[li]["pattern"] == leg["pattern"] else None
                wi = leg.get("wait_info") or {}
                first = li == 0
                ta = tb - leg["wait_s"]                         # planned arrival at the stop
                gw = tt.gps_wait.get((r, cl_seq[0], config.daytype_of(date), int(ta // 3600) % 24))
                rows.append({
                    "trip": trip, "leg": li, "n_legs": opt["transfers"] + 1, "walk_min": opt["walk_min"],
                    "total_min": opt["total_min"], "timetable_total_min": opt["timetable_total_min"],
                    "wait_actual_any": dv.actual_wait(r, cl_seq[0], ta),
                    "day": day, "hour": hh, "route": leg["route"], "stops": leg["stops"],
                    "actual": float(np.median(act)) if act else np.nan, "n_buses": len(act),
                    "model": leg["ride_s"], "model_raw": leg.get("ride_model_s", leg["ride_s"]),
                    "live": lv["ride_s"] if lv else np.nan, "timetable": leg["timetable_ride_s"],
                    "level": leg.get("estimate_level"),
                    "detection": dv.detection.get((r, cl_seq[0]), 0.0),
                    "wait_actual": dv.actual_wait(r, cl_seq[0], ta) if first else None,
                    "wait_model": leg["wait_s"] if first else np.nan,
                    "wait_gps_raw": gw[0] if (first and gw) else np.nan,
                    "wait_timetable": wi.get("timetable_s") if first else np.nan,
                    "wait_source": wi.get("source") if first else None,
                })
        log(f"day {day}: {len(rows)} legs so far")
    return pd.DataFrame(rows), n_plans


def calibrate_planner(pl, g, days: list[int], n: int = 700) -> dict:
    """Planner corrections from real buses on training days (never the held-out week):
    ride  actual / planned whole-ride ratio quantiles by time of day (point = median, range = quantiles)
    wait  real wait ÷ GPS-headway wait, and ÷ timetable half-headway where the GPS has no wait for the stop
    """
    from app.calibration import BUCKETS, QS, bucket_of
    saved = pl.ride_calib, pl.wait_calib
    pl.ride_calib = pl.wait_calib = None                   # calibrate the raw model, not itself
    try:
        df, _ = collect_legs(pl, g, days, n, seed=11, live=False, log=lambda *_: None)
    finally:
        pl.ride_calib, pl.wait_calib = saved
    rides = df.dropna(subset=["actual"])
    rides = rides[rides.actual > 120].copy()
    rides["ratio"] = rides.actual / rides.model
    rides["b"] = rides.hour.map(bucket_of)
    allq = rides.ratio.quantile(QS).round(3).tolist() if len(rides) else [0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
    ride = {"days": list(days), "n_legs": int(len(rides)), "quantiles": QS, "all": allq, "by_bucket": {}}
    for _, _, name in BUCKETS:
        x = rides[rides.b == name].ratio
        ride["by_bucket"][name] = x.quantile(QS).round(3).tolist() if len(x) >= 40 else allq
    w = df.dropna(subset=["wait_actual"])
    wg = w.dropna(subset=["wait_gps_raw"])
    wt = w[w.wait_gps_raw.isna()].dropna(subset=["wait_timetable"])
    # ratio of means (not median of ratios): waits are skewed and the planner needs the *expected* wait
    wait = {"days": list(days),
            "gps": round(float(wg.wait_actual.sum() / wg.wait_gps_raw.sum()), 3) if len(wg) >= 30 else 1.0,
            "timetable": round(float(wt.wait_actual.sum() / wt.wait_timetable.sum()), 3) if len(wt) >= 30 else 1.0,
            "n_gps": int(len(wg)), "n_timetable": int(len(wt))}
    return {"ride": ride, "wait": wait}
