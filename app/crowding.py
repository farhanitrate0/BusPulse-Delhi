"""Runtime crowding model: estimated passengers on board per (route, stretch, daytype, half-hour).

Built by scripts/03_tickets_crowding.py into data/model/crowd.parquet keyed by
(route_id, ca, cb, dt, slot). Every GTFS pattern of that route that runs over the same stretch
(cluster pair) gets the estimate. Levels: 0 unknown, 1 seats available, 2 standing room, 3 crowded.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

NAMES = {0: "unknown", 1: "seats available", 2: "standing room", 3: "crowded"}
CONF = {0: "none", 1: "low", 2: "medium", 3: "high"}


def level_of(load: np.ndarray) -> np.ndarray:
    seats, stand = config.BUS_SEATS, config.BUS_STANDING
    return np.where(load < 0.9 * seats, 1, np.where(load < seats + 0.5 * stand, 2, 3)).astype(np.int8)


class CrowdModel:
    def __init__(self, g: dict, df: pd.DataFrame | None, meta: dict, arrays: dict | None = None):
        self.g = g
        self.meta = meta
        P = g["patterns"]
        ns = np.array([len(p["cl"]) - 1 for p in P])
        base = np.r_[0, np.cumsum(ns)[:-1]]
        if arrays is None:
            arrays = self.build_arrays(g, df)
        L, LD, C = arrays["L"], arrays["LD"], arrays["C"]
        has = (L > 0).any(axis=(1, 2)) | (C > 0).any(axis=(1, 2))
        self.lev, self.load, self.conf = {}, {}, {}
        for p, b, n in zip(P, base, ns):
            if has[b:b + n].any():          # views shaped [dt, slot, stretch]
                self.lev[p["pid"]] = L[b:b + n].transpose(1, 2, 0)
                self.load[p["pid"]] = LD[b:b + n].transpose(1, 2, 0)
                self.conf[p["pid"]] = C[b:b + n].transpose(1, 2, 0)
        self._lists: dict = {}

    @staticmethod
    def build_arrays(g: dict, df: pd.DataFrame) -> dict:
        """Flat [stretch-of-pattern, dt, slot] arrays; every pattern of a route over the same stretch
        (cluster pair) gets the estimate."""
        P = g["patterns"]
        ns = np.array([len(p["cl"]) - 1 for p in P])
        ps = pd.DataFrame({"route_id": np.repeat([str(p["route_id"]) for p in P], ns),
                           "ca": np.concatenate([p["cl"][:-1] for p in P]).astype(np.int64),
                           "cb": np.concatenate([p["cl"][1:] for p in P]).astype(np.int64),
                           "row": np.arange(int(ns.sum()))})
        d = df.assign(route_id=df.route_id.astype(str), ca=df.ca.astype(np.int64), cb=df.cb.astype(np.int64))
        m = d.merge(ps, on=["route_id", "ca", "cb"], how="inner")
        R = int(ns.sum())
        L = np.zeros((R, 3, 48), dtype=np.int8)
        LD = np.full((R, 3, 48), np.nan, dtype=np.float16)
        C = np.zeros((R, 3, 48), dtype=np.int8)
        r, dd, s = m.row.to_numpy(), m.dt.to_numpy(np.int64), m.slot.to_numpy(np.int64)
        L[r, dd, s] = m.level.to_numpy(np.int8)
        LD[r, dd, s] = np.minimum(m.load.to_numpy(np.float32), 60000)
        C[r, dd, s] = m.conf.to_numpy(np.int8)
        return {"L": L, "LD": LD, "C": C}

    @classmethod
    def load(cls, g: dict):
        from app.traveltime import _fresh_cache, _save_cache
        f = config.MODEL_DIR / "crowd.parquet"
        cache = config.MODEL_DIR / "crowd_runtime.npz"
        mf = config.MODEL_DIR / "crowd_meta.json"
        meta = json.loads(mf.read_text()) if mf.exists() else {}
        sig = f"{len(g['patterns'])}|{sum(len(p['cl']) - 1 for p in g['patterns'])}"
        z = _fresh_cache(cache, sig, [f])
        if z is None:
            if not f.exists():
                raise FileNotFoundError(f)
            z = cls.build_arrays(g, pd.read_parquet(f))
            _save_cache(cache, sig, z)
        return cls(g, None, meta, z)

    def level(self, p: int, seg: int, dt: int, t: float) -> int:
        L = self.lev.get(p)
        if L is None:
            return 0
        return int(L[dt, int(t // 1800) % 48, seg])

    def pat_levels(self, p: int, dt: int):
        """(bytes, n_stretches): level of stretch s in half-hour slot k is b[k * n + s].

        bytes, not nested lists: 1 byte per cell, so the cache stays ~20 MB even after every
        pattern has been touched (nested lists grew to >150 MB on a long-running server).
        """
        k = (p, dt)
        if k not in self._lists:
            L = self.lev.get(p)
            self._lists[k] = None if L is None else (np.ascontiguousarray(L[dt]).tobytes(), L.shape[2])
        return self._lists[k]

    @staticmethod
    def describe(cs) -> str:
        cs = np.asarray(cs)
        known = cs[cs > 0]
        if not len(known):
            return "no ticket data for this bus"
        n3, n2 = int((known == 3).sum()), int((known == 2).sum())
        if n3:
            return f"crowded on {n3} of {len(cs)} stretches"
        if n2:
            return f"standing room on {n2} of {len(cs)} stretches"
        return "seats usually available"

    def leg_detail(self, p: int, bpos: int, apos: int, dt: int, t: float):
        LD = self.load.get(p)
        if LD is None:
            return None
        s = int(t // 1800) % 48
        seg = LD[dt, s, bpos:apos]
        conf = self.conf[p][dt, s, bpos:apos]
        if np.isnan(seg).all():
            return None
        return {"peak_load": int(np.nanmax(seg)), "seats": config.BUS_SEATS,
                "capacity": config.BUS_SEATS + config.BUS_STANDING,
                "confidence": CONF[int(conf.max())]}

    def route_profile(self, route: str, dt: int):
        q = route.strip().upper().replace(" ", "")
        pats = [p for p in self.g["patterns"] if p["pid"] in self.lev and
                (str(p["name"]).upper().replace(" ", "") == q or str(p["route_id"]).upper() == q)]
        if not pats:
            return None
        p = max(pats, key=lambda x: x["n_trips"])
        cl = p["cl"]
        LD, C = self.load[p["pid"]], self.conf[p["pid"]]
        names = [self.g["cl_name"][int(c)] for c in cl]
        hours = []
        import warnings
        warnings.filterwarnings("ignore", message="All-NaN slice")
        for h in range(24):
            ld = np.nanmax(LD[dt, 2 * h:2 * h + 2], axis=0)
            cf = C[dt, 2 * h:2 * h + 2].max(axis=0)
            lv = np.where(np.isnan(ld), 0, level_of(np.nan_to_num(ld)))
            hours.append({"hour": h, "load": [None if np.isnan(x) else round(float(x)) for x in ld],
                          "level": lv.tolist(), "confidence": [CONF[int(x)] for x in cf]})
        return {"route": p["name"], "route_id": p["route_id"], "headsign": p["headsign"],
                "daytype": config.DAYTYPES[dt], "stops": names,
                "stretches": [f"{names[k]} → {names[k+1]}" for k in range(len(cl) - 1)],
                "capacity": {"seats": config.BUS_SEATS, "standing": config.BUS_STANDING},
                "hours": hours, "assumptions": self.meta.get("assumptions")}
