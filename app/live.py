"""Live + historical blend.

A LiveFeed holds recent segment observations (from a real GTFS-RT connector, or — for the demo
and for validation — a *replay* of a held-out day). For a segment the bus will reach at time t,
given the current time `now`:

    live  = median run of observations on that segment in the last 15 min   (LIVE_WINDOW_S)
    w     = n / (n + K) * exp(-(t - now) / 60 min)      (n = live observations; K = 3)
    est   = w * live + (1 - w) * historical(daytype, hour)

So live data dominates for the next few stretches and fades to history further ahead; with no
live data the estimate is purely historical.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402


class LiveFeed:
    def __init__(self, tt, seg: pd.DataFrame | None, label: str = "replay", arrays: dict | None = None):
        self.tt = tt
        self.label = label
        if arrays is None:
            arrays = self.build_arrays(tt, seg)
        self.t, self.pi, self.run = arrays["t"], arrays["pi"], arrays["run"]
        self._cache: dict[int, dict[int, tuple[float, int]]] = {}

    @staticmethod
    def build_arrays(tt, seg: pd.DataFrame) -> dict:
        ncl = tt.ncl
        keys = seg.ca.to_numpy(np.int64) * ncl + seg.cb.to_numpy(np.int64)
        # vectorized pair lookup: sorted pair keys + searchsorted
        pk = np.fromiter(tt.pair_index.keys(), dtype=np.int64)
        pv = np.fromiter(tt.pair_index.values(), dtype=np.int64)
        o = np.argsort(pk)
        pk, pv = pk[o], pv[o]
        ix = np.minimum(np.searchsorted(pk, keys), len(pk) - 1)
        pi = np.where(pk[ix] == keys, pv[ix], -1)
        t = (seg.arr_b.to_numpy(np.float64) + config.IST_OFFSET_S) % 86400
        ok = pi >= 0
        o = np.argsort(t[ok], kind="stable")
        return {"t": t[ok][o], "pi": pi[ok][o].astype(np.int32), "run": seg.run.to_numpy(np.float32)[ok][o]}

    @classmethod
    def replay_day(cls, tt, day: int):
        from app.traveltime import _fresh_cache, _save_cache
        f = config.SEG_DIR / f"seg_{config.day_name(day)}.parquet"
        cache = config.SEG_DIR / f"live_{config.day_name(day)}.npz"
        sig = f"{tt.P}|{tt.ncl}"
        z = _fresh_cache(cache, sig, [f])
        if z is None:
            if not f.exists():
                return None
            z = cls.build_arrays(tt, pd.read_parquet(f, columns=["ca", "cb", "arr_b", "run"]))
            _save_cache(cache, sig, z)
        return cls(tt, None, label=f"replay of {day} Oct", arrays=z)

    def snapshot(self, now: float) -> dict[int, tuple[float, int]]:
        key = int(now // 60)
        snap = self._cache.get(key)
        if snap is not None:
            return snap
        a = np.searchsorted(self.t, now - config.LIVE_WINDOW_S)
        b = np.searchsorted(self.t, now, side="right")
        snap = {}
        if b > a:
            pi, run = self.pi[a:b], self.run[a:b]
            o = np.lexsort((run, pi))
            pi, run = pi[o], run[o]
            u, start, cnt = np.unique(pi, return_index=True, return_counts=True)
            mid = start + cnt // 2
            med = np.where(cnt % 2 == 1, run[mid], (run[mid] + run[np.maximum(mid - 1, start)]) / 2)
            snap = {int(p): (float(m), int(c)) for p, m, c in zip(u, med, cnt)}
        if len(self._cache) > 2000:
            self._cache.clear()
        self._cache[key] = snap
        return snap

    def blend(self, pi: int, base: float, t: float, now: float) -> float:
        s = self.snapshot(now).get(pi)
        if s is None:
            return base
        med, n = s
        w = n / (n + config.LIVE_K) * math.exp(-max(0.0, t - now) / config.LIVE_DECAY_S)
        return w * med + (1 - w) * base

    def coverage(self, now: float) -> int:
        return len(self.snapshot(now))


class ReplayVehicles:
    """Where every bus was at a time of day on the replay day, for the "live buses" map layer.

    Built from that day's stop-to-stop observations: between leaving stop A (dep_a) and reaching B (arr_b)
    the bus is placed along the straight line A→B by elapsed time; from arr_b to dep_b it stands at B.
    (With a real GTFS-RT feed, the connector's latest fixes are shown instead — see app/gtfsrt.py.)
    """

    def __init__(self, z: dict, route_names: list[str], cl_lat, cl_lon, label: str):
        self.v, self.r = z["v"], z["r"]
        self.ca, self.cb = z["ca"], z["cb"]
        self.dep_a, self.arr_b = z["dep_a"], z["arr_b"]
        # a bus standing at a stop for more than 10 min is parked (terminal layover): take it off the map
        self.end = np.minimum(z["dep_b"], z["arr_b"] + 600)
        self.vnames = route_names
        self.cl_lat, self.cl_lon = np.asarray(cl_lat), np.asarray(cl_lon)
        self.label = label
        self.max_span = float((self.end - self.dep_a).max()) if len(self.dep_a) else 0.0

    @classmethod
    def load(cls, g: dict, day: int):
        from app.traveltime import _fresh_cache, _save_cache
        f = config.SEG_DIR / f"seg_{config.day_name(day)}.parquet"
        cache = config.SEG_DIR / f"vehicles_{config.day_name(day)}.npz"
        sig = f"v2|{len(g['cl_lat'])}"
        z = _fresh_cache(cache, sig, [f])
        if z is None:
            if not f.exists():
                return None
            d = pd.read_parquet(f, columns=["v", "r", "ca", "cb", "dep_a", "arr_b", "dep_b"])
            d = d[d.dep_b.notna() & (d.dep_b >= d.arr_b) & (d.arr_b > d.dep_a)]
            off = config.IST_OFFSET_S
            # IST midnight of the replay day (the median row; a few rows run in from the evening before)
            day0 = float(np.floor((d.dep_a.median() + off) / 86400) * 86400 - off)
            z = {"v": pd.factorize(d.v)[0].astype(np.int32), "r": d.r.to_numpy(np.int32),
                 "ca": d.ca.to_numpy(np.int16), "cb": d.cb.to_numpy(np.int16),
                 "dep_a": (d.dep_a.to_numpy() - day0).astype(np.int32),
                 "arr_b": (d.arr_b.to_numpy() - day0).astype(np.int32),
                 "dep_b": (d.dep_b.to_numpy() - day0).astype(np.int32)}
            o = np.argsort(z["dep_a"], kind="stable")
            z = {k: v[o] for k, v in z.items()}
            _save_cache(cache, sig, z)
        names = {}
        for P in g["patterns"]:
            try:
                names.setdefault(int(float(P["route_id"])), str(P["name"]))
            except (TypeError, ValueError):
                pass
        rn = [names.get(int(x), "") for x in range(int(z["r"].max()) + 1)] if len(z["r"]) else []
        return cls(z, rn, g["cl_lat"], g["cl_lon"], label=f"replay of {day} Oct")

    def at(self, t: float, bbox=None, limit: int = 800) -> list[list]:
        """[[lat, lon, route, bearing_deg, moving, bus id], …] at t seconds after IST midnight of the replay day."""
        lo = np.searchsorted(self.dep_a, t - self.max_span)
        hi = np.searchsorted(self.dep_a, t, side="right")
        idx = np.arange(lo, hi)
        idx = idx[self.end[idx] > t]
        if not len(idx):
            return []
        # one row per bus: the latest stretch it started
        o = idx[np.lexsort((self.dep_a[idx], self.v[idx]))]
        last = np.r_[self.v[o][1:] != self.v[o][:-1], True]
        idx = o[last]
        ca, cb = self.ca[idx].astype(int), self.cb[idx].astype(int)
        f = np.clip((t - self.dep_a[idx]) / np.maximum(self.arr_b[idx] - self.dep_a[idx], 1), 0, 1)
        la = self.cl_lat[ca] + f * (self.cl_lat[cb] - self.cl_lat[ca])
        lo_ = self.cl_lon[ca] + f * (self.cl_lon[cb] - self.cl_lon[ca])
        brg = (np.degrees(np.arctan2((self.cl_lon[cb] - self.cl_lon[ca]) * np.cos(np.radians(la)),
                                     self.cl_lat[cb] - self.cl_lat[ca])) + 360) % 360
        moving = f < 1
        keep = np.ones(len(idx), bool)
        if bbox is not None:
            a0, o0, a1, o1 = bbox
            keep = (la >= a0) & (la <= a1) & (lo_ >= o0) & (lo_ <= o1)
        sel = np.nonzero(keep)[0][:limit]
        r, vid = self.r[idx], self.v[idx]
        return [[round(float(la[i]), 5), round(float(lo_[i]), 5),
                 self.vnames[r[i]] if 0 <= r[i] < len(self.vnames) else "", int(brg[i]), bool(moving[i]), int(vid[i])]
                for i in sel]
