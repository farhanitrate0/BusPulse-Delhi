"""Runtime travel-time model: per GTFS cluster pair × daytype × hour, with fallback chain.

RUN[pair, dt, h]  expected run seconds  (stop departure → next stop arrival)
DWELL[pair, dt, h] expected dwell at the *next* stop
LEVEL[pair, dt, h] 3 = GPS segment+hour (n ≥ 5), 2 = GPS segment (shrunk / hour-scaled),
                   1 = city speed at that hour × distance, 0 = timetable only
"""
from __future__ import annotations

import json
import sys
from array import array
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

LEVEL_NAMES = {3: "GPS (this stretch, this hour)", 2: "GPS (this stretch, other hours)",
               1: "city speed for this hour", 0: "timetable"}


def _fresh_cache(path: Path, sig: str, sources: list[Path]):
    """Arrays from a compiled runtime cache, if it exists, matches `sig` and no source is newer.

    Missing sources do not invalidate it (deploy bundles ship only the compiled cache).
    """
    if not path.exists():
        return None
    mt = path.stat().st_mtime
    if any(s.exists() and s.stat().st_mtime > mt for s in sources):
        return None
    try:
        with np.load(path, allow_pickle=False) as z:
            if str(z["_sig"]) != sig:
                return None
            return {k: z[k] for k in z.files if k != "_sig"}
    except Exception:  # noqa: BLE001  (corrupt / old format → rebuild)
        return None


def _save_cache(path: Path, sig: str, arrays: dict):
    try:
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, _sig=np.array(sig), **{k: np.asarray(v) for k, v in arrays.items()})
        tmp.replace(path)
    except OSError as e:                     # read-only deploy dir: just run without the cache
        print(f"[cache] could not write {path.name}: {e}")


class GpsWaits:
    """GPS-observed waits keyed (route, cluster, daytype, hour) in sorted numpy arrays.

    A Python dict of 1.6 M tuples took ~1 GB of RAM; this takes ~25 MB. Only cells with
    n >= MIN_N_WAIT headways are kept (the planner ignores the rest).
    """
    MIN_N = 6

    def __init__(self, f: Path | None, ncl: int):
        self.ncl = ncl
        self.keys = np.zeros(0, dtype=np.int64)
        self.wait = np.zeros(0, dtype=np.float32)
        self.n = np.zeros(0, dtype=np.uint16)
        self.bunched = np.zeros(0, dtype=np.float32)
        if f is None or not f.exists():
            return
        w = pd.read_parquet(f, columns=["r", "ca", "dt", "h", "n", "wait_s", "bunched_share"])
        w = w[w.n >= self.MIN_N]
        k = self._key(w.r.to_numpy(np.int64), w.ca.to_numpy(np.int64), w.dt.to_numpy(np.int64),
                      w.h.to_numpy(np.int64))
        o = np.argsort(k, kind="stable")
        self.keys = k[o]
        self.wait = w.wait_s.to_numpy(np.float32)[o]
        self.n = np.minimum(w.n.to_numpy(), 65535).astype(np.uint16)[o]
        self.bunched = w.bunched_share.to_numpy(np.float32)[o]

    @classmethod
    def from_arrays(cls, ncl, keys, wait, n, bunched):
        o = cls(None, ncl)
        o.keys, o.wait, o.n, o.bunched = keys, wait, n, bunched
        return o

    def _key(self, r, ca, dt, h):
        return ((r * self.ncl + ca) * 3 + dt) * 24 + h

    def __len__(self):
        return len(self.keys)

    def get(self, key):
        if not len(self.keys):
            return None
        r, ca, dt, h = key
        k = self._key(int(r), int(ca), int(dt), int(h))
        i = int(np.searchsorted(self.keys, k))
        if i < len(self.keys) and self.keys[i] == k:
            return float(self.wait[i]), int(self.n[i]), float(self.bunched[i])
        return None


class TravelTimeModel:
    def __init__(self, g: dict, model_dir: Path = config.MODEL_DIR, stat: str = config.STAT):
        self.g = g
        P = len(g["pairs"])
        self.P = P
        ncl = len(g["cl_lat"])
        self.ncl = ncl
        keys = g["pairs"][:, 0].astype(np.int64) * ncl + g["pairs"][:, 1]
        self.pair_index = {int(k): i for i, k in enumerate(keys)}
        dist = g["pair_dist"].astype(np.float64)
        sched = np.maximum(g["pair_sched"].astype(np.float64), 1.0)
        self.sched = sched
        self.dist = dist
        self.meta = {}
        self.has_gps = False
        mf = model_dir / "model_meta.json"
        if mf.exists():
            self.meta = json.loads(mf.read_text())
        sig = f"{stat}|{config.SHRINK_K}|{config.MIN_N_HOUR}|{P}|{ncl}|{GpsWaits.MIN_N}"
        cache = model_dir / "tt_runtime.npz"
        z = _fresh_cache(cache, sig, [model_dir / f for f in
                                      ("seg_hour.parquet", "seg_all.parquet", "city_hour.parquet",
                                       "gps_wait.parquet")])
        if z is None:
            z = self._build(model_dir, stat, dist, sched)
            _save_cache(cache, sig, z)
        self.has_gps = bool(z["has_gps"])
        if self.has_gps:
            self.city_speed, self.city_factor, self.city_dwell = z["city_speed"], z["city_factor"], z["city_dwell"]
        self.gps_wait = GpsWaits.from_arrays(ncl, z["gw_keys"], z["gw_wait"], z["gw_n"], z["gw_bunched"])
        from app.calibration import load_calibration
        self.calib = load_calibration()
        self.RUN = z["RUN"]
        self.DWELL = z["DWELL"]
        self.LEVEL = z["LEVEL"]
        self.N = z["N"]
        # compact typed arrays for fast scalar access in the planner: [dt][h] -> array over pairs
        # (array('f') indexes about as fast as a list but takes 4 bytes per value instead of 32)
        def arr(code, a):
            x = array(code)
            x.frombytes(np.ascontiguousarray(a).tobytes())
            return x
        self.run_l = [[arr("f", self.RUN[:, d, h]) for h in range(24)] for d in range(3)]
        self.dwell_l = [[arr("f", self.DWELL[:, d, h]) for h in range(24)] for d in range(3)]
        self.level_l = [[arr("b", self.LEVEL[:, d, h]) for h in range(24)] for d in range(3)]
        self.live = None   # set by app.live

    def _build(self, model_dir: Path, stat: str, dist, sched) -> dict:
        P, ncl = self.P, self.ncl
        out = {"has_gps": False}
        RUN = np.repeat(sched[:, None, None], 3, 1).repeat(24, 2)   # level 0: timetable
        DW = np.zeros((P, 3, 24))
        LV = np.zeros((P, 3, 24), dtype=np.int8)
        N = np.zeros((P, 3, 24), dtype=np.int32)
        f_hour = model_dir / "seg_hour.parquet"
        if f_hour.exists():
            out["has_gps"] = True
            col = "run_mean" if stat == "mean" else "run_med"
            dcol = "dwell_mean" if stat == "mean" else "dwell_med"
            city = pd.read_parquet(model_dir / "city_hour.parquet")
            fac = np.full((3, 24), np.nan)
            spd = np.full((3, 24), np.nan)
            cdw = np.full((3, 24), np.nan)
            fac[city.dt, city.h] = city.factor
            spd[city.dt, city.h] = city.speed_kmh
            cdw[city.dt, city.h] = city.dwell_med
            # fill hours without data (night) from nearest available hour / other daytypes
            for arr in (fac, spd, cdw):
                for dt in range(3):
                    row = arr[dt]
                    if np.isnan(row).all():
                        row[:] = np.nanmedian(arr)
                    ok = ~np.isnan(row)
                    hrs = np.arange(24)
                    row[~ok] = np.interp(hrs[~ok], np.r_[hrs[ok] - 24, hrs[ok], hrs[ok] + 24],
                                         np.r_[row[ok], row[ok], row[ok]])
            out.update(city_speed=spd, city_factor=fac, city_dwell=cdw)
            # level 1: distance / city speed (both straight-line, so consistent)
            RUN = dist[:, None, None] / (spd[None] / 3.6)
            DW = np.broadcast_to(cdw[None], (P, 3, 24)).copy()
            LV[:] = 1
            # level 2: segment all-hours × city hourly factor
            sa = pd.read_parquet(model_dir / "seg_all.parquet")
            sa["i"] = [self.pair_index.get(int(a) * ncl + int(b), -1) for a, b in zip(sa.ca, sa.cb)]
            sa = sa[sa.i >= 0]
            i = sa.i.to_numpy()
            # factor is relative to the segment's all-hours *median*
            base = sa.run_med.to_numpy()
            ratio = (sa[col] / sa.run_med).to_numpy()
            RUN[i] = base[:, None, None] * fac[None] * ratio[:, None, None]
            dwb = sa[dcol].fillna(np.nan).to_numpy()
            has_dw = ~np.isnan(dwb)
            DW[i[has_dw]] = dwb[has_dw, None, None] * (cdw[None] / np.nanmedian(cdw))
            LV[i] = 2
            # level 3: segment+hour, shrunk toward level 2 with weight n/(n+K)
            sh = pd.read_parquet(model_dir / "seg_hour.parquet")
            sh["i"] = [self.pair_index.get(int(a) * ncl + int(b), -1) for a, b in zip(sh.ca, sh.cb)]
            sh = sh[sh.i >= 0]
            i, d, h = sh.i.to_numpy(), sh.dt.to_numpy(), sh.h.to_numpy()
            n = sh.n.to_numpy().astype(float)
            w = n / (n + config.SHRINK_K)
            RUN[i, d, h] = w * sh[col].to_numpy() + (1 - w) * RUN[i, d, h]
            nd = sh.n_dwell.to_numpy().astype(float)
            wd = nd / (nd + config.SHRINK_K)
            dv = sh[dcol].to_numpy()
            okd = ~np.isnan(dv)
            DW[i[okd], d[okd], h[okd]] = wd[okd] * dv[okd] + (1 - wd[okd]) * DW[i[okd], d[okd], h[okd]]
            N[i, d, h] = n.astype(np.int32)
            LV[i, d, h] = np.where(n >= config.MIN_N_HOUR, 3, 2)
        # GPS-observed waits (route × stop cluster × daytype × hour), incl. bunching
        gw = GpsWaits(model_dir / "gps_wait.parquet", ncl)
        out.update(RUN=RUN.astype(np.float32), DWELL=np.nan_to_num(DW, nan=15.0).astype(np.float32),
                   LEVEL=LV, N=np.minimum(N, 65535).astype(np.uint16),
                   gw_keys=gw.keys, gw_wait=gw.wait, gw_n=gw.n, gw_bunched=gw.bunched)
        return out

    def pair(self, ca: int, cb: int) -> int:
        return self.pair_index.get(int(ca) * self.ncl + int(cb), -1)

    def run(self, pi: int, dt: int, t: float, now: float | None = None) -> float:
        h = int(t // 3600) % 24
        base = self.run_l[dt][h][pi]
        if self.live is not None and now is not None:
            return self.live.blend(pi, base, t, now)
        return base

    def chain_time(self, pis, dt: int, t0: float, stat_dwell: bool = True, now: float | None = None) -> float:
        """Time from departing stop 0 to arriving at last stop along consecutive pairs."""
        t = t0
        for k, pi in enumerate(pis):
            if pi < 0:
                return float("nan")
            t += self.run(pi, dt, t, now)
            if stat_dwell and k < len(pis) - 1:
                t += self.dwell_l[dt][int(t // 3600) % 24][pi]
        return t - t0

    def traffic(self, dt: int, h: int, min_n: int = 1):
        """Per-pair speed and slowdown ratio vs the segment's own fastest hour (for the map)."""
        lv = self.LEVEL[:, dt, h]
        run = self.RUN[:, dt, h].astype(float)
        free = self.RUN[:, dt, :].min(axis=1).astype(float)
        sel = np.nonzero((lv >= 2) & (self.N[:, dt, :].sum(axis=1) >= min_n))[0]
        from app.gtfs import pair_paths
        pp = pair_paths()
        g = self.g
        out = []
        for i in sel:
            a, b = g["pairs"][i]
            out.append([round(float(g["cl_lat"][a]), 5), round(float(g["cl_lon"][a]), 5),
                        round(float(g["cl_lat"][b]), 5), round(float(g["cl_lon"][b]), 5),
                        round(float(self.dist[i] / run[i] * 3.6), 1),
                        round(float(run[i] / max(free[i], 1)), 2), int(lv[i]),
                        pp.get(int(i))])          # road path along the route's shape (None → straight)
        return out
