"""Journey planner: frequency-based, time-dependent RAPTOR over stop clusters.

* Nodes are stop clusters (GTFS stops within 50 m, i.e. both sides of a road), so changing
  between routes at the same place costs only MIN_TRANSFER_S.
* In-vehicle time comes from the GPS travel-time model (by daytype and hour, optionally
  blended with live data); the hour is re-evaluated at every stop as the bus moves on.
* Waiting = half the headway of that route pattern at that stop around that time
  (from the GTFS timetable), capped; for rare routes, time to the next scheduled bus.
* Modes optimise a generalised cost: crowding multiplies in-vehicle time, wait/walk weighted.
* RAPTOR picks the structure of each option; `simulate` then computes every displayed time
  with the same formulas, which also gives the 24-hour profile and "quieter hour" comparison.
"""
from __future__ import annotations

import datetime as _dt
import math
import re
import sys
from bisect import bisect_left
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from app.geo import haversine_m, to_xy  # noqa: E402
from app.gtfs import pair_paths  # noqa: E402
from app.traveltime import LEVEL_NAMES  # noqa: E402

INF = float("inf")
IST = _dt.timezone(_dt.timedelta(seconds=config.IST_OFFSET_S))
CROWD_NAMES = {0: "unknown", 1: "seats available", 2: "standing room", 3: "crowded"}


class PlanError(ValueError):
    def __init__(self, msg: str, code: str = "bad_request"):
        super().__init__(msg)
        self.code = code


# what people actually type in Delhi → words that appear in the GTFS stop names
ALIASES = {
    "cp": "connaught place", "rajiv chowk": "connaught place", "ndls": "new delhi railway station",
    "new delhi station": "new delhi railway station", "old delhi station": "old delhi railway station",
    "nizamuddin station": "nizamuddin railway station", "hazrat nizamuddin": "nizamuddin railway station",
    "airport": "igi airport terminal 3", "t3": "igi airport terminal 3", "t1": "domestic airport t 1",
    "du": "delhi university north campus", "north campus": "delhi university north campus",
    "kashmere gate": "isbt kashmiri gate", "isbt": "isbt kashmiri gate",
}


def norm_name(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


class Planner:
    def __init__(self, g: dict, tt, crowd=None):
        self.g, self.tt, self.crowd = g, tt, crowd
        self.ncl = ncl = len(g["cl_lat"])
        self.cl_lat = np.asarray(g["cl_lat"])
        self.cl_lon = np.asarray(g["cl_lon"])
        cx, cy = to_xy(self.cl_lat, self.cl_lon)
        self.cx, self.cy = np.asarray(cx, dtype=float), np.asarray(cy, dtype=float)
        self.ride_calib = (tt.calib or {}).get("ride")        # set by scripts/02 (app/ridecheck.py)
        self.wait_calib = (tt.calib or {}).get("wait")
        # service area = every GTFS stop (some routes run into Haryana / UP) plus ~2 km
        self.area = (float(self.cl_lat.min()) - 0.02, float(self.cl_lat.max()) + 0.02,
                     float(self.cl_lon.min()) - 0.02, float(self.cl_lon.max()) + 0.02)
        self.tree = cKDTree(np.c_[cx, cy])
        P = g["patterns"]
        self.pat = P
        self.pat_cl = [p["cl"].tolist() for p in P]
        self.pat_pairs = [p["pairs"].tolist() for p in P]
        self.pat_off = [p["off"].tolist() for p in P]
        self.pat_starts = [p["starts"] for p in P]
        self.pat_rint = []
        for p in P:
            try:
                self.pat_rint.append(int(float(p["route_id"])))
            except (TypeError, ValueError):
                self.pat_rint.append(-1)
        # GPS waits per pattern stop [pos, dt, h] (NaN = not observed), precomputed in one vectorized lookup
        self.pat_gw: list = [None] * len(P)
        gw = tt.gps_wait
        if len(gw):
            lens = np.array([len(c) for c in self.pat_cl])
            r = np.repeat(np.array(self.pat_rint, dtype=np.int64), lens)
            ca = np.concatenate([p["cl"] for p in P]).astype(np.int64)
            base = gw._key(r, ca, 0, 0)                     # the 72 (dt, h) keys of a stop are contiguous
            lo, hi = np.searchsorted(gw.keys, base), np.searchsorted(gw.keys, base + 72)
            cnt = hi - lo
            si = np.repeat(np.arange(len(base)), cnt)
            src = np.arange(int(cnt.sum())) - np.repeat(np.cumsum(cnt) - cnt, cnt) + np.repeat(lo, cnt)
            W = np.full((len(base), 72), np.nan, dtype=np.float16)
            W[si, gw.keys[src] - base[si]] = gw.wait[src]
            W = W.reshape(-1, 3, 24)
            for pi, (a, n) in enumerate(zip(np.r_[0, np.cumsum(lens)[:-1]], lens)):
                Wp = W[a:a + n]
                if np.isnan(Wp).all():
                    continue
                # the same buses pass consecutive stops, so a stop the GPS detects poorly (or a terminal,
                # where buses wait out of range) takes the headway-based wait of the nearest well-detected
                # stop of this pattern, up to 3 stops away
                for d in (1, 2, 3):
                    gap = np.isnan(Wp)
                    if not gap.any():
                        break
                    down = np.full_like(Wp, np.nan)
                    down[:-d] = Wp[d:]
                    up = np.full_like(Wp, np.nan)
                    up[d:] = Wp[:-d]
                    Wp[gap] = np.where(np.isnan(down[gap]), up[gap], down[gap])
                self.pat_gw[pi] = Wp
        self.cl_pat: list[list[tuple[int, int]]] = [[] for _ in range(ncl)]
        for pi, cls in enumerate(self.pat_cl):
            for pos, c in enumerate(cls[:-1]):          # cannot board at the last stop
                self.cl_pat[c].append((pi, pos))
        self.cl_npat = np.array([len(x) for x in self.cl_pat])
        # walking transfers between nearby clusters
        self.transfers: list[list[tuple[int, float]]] = [[] for _ in range(ncl)]
        for a, nb in enumerate(self.tree.query_ball_point(np.c_[cx, cy], config.TRANSFER_MAX_M)):
            for b in nb:
                if b != a:
                    d = float(np.hypot(cx[a] - cx[b], cy[a] - cy[b]))
                    self.transfers[a].append((b, d * config.WALK_DETOUR / config.WALK_SPEED_MS))
        st = g["stops"]
        self.stop_ll = {s: (la, lo) for s, la, lo in zip(st.stop_id, st.lat, st.lon)}
        self.stop_name = dict(zip(st.stop_id, st.name))
        # name search index: every stop name → its cluster
        names: dict[str, set[int]] = {}
        for nm, c in zip(st.name, st.cluster):
            names.setdefault(norm_name(str(nm)), set()).add(int(c))
        self.search_names = list(names.keys())
        self.search_cls = [sorted(names[k]) for k in self.search_names]

    # ------------------------------------------------------------------ inputs
    def search_stops(self, q: str, n: int = 8) -> list[dict]:
        qn = norm_name(q)
        if not qn:
            return []
        qn = ALIASES.get(qn, qn)
        try:
            from rapidfuzz import fuzz, process
            res = process.extract(qn, self.search_names, scorer=fuzz.WRatio, limit=max(n * 8, 60))
            # WRatio alone lets a stray "1" or "sec" match anything ("<script>alert(1)" → "Dwarka Sec 1"):
            # at least one real word of the query must match a word of the name (typo or prefix ok)
            qw = [w for w in qn.split() if len(w) >= 3 and not w.isdigit()]

            def word_ok(name: str) -> bool:
                if not qw:
                    return True
                return any(b.startswith(a) or fuzz.ratio(a, b) >= 75 for a in qw for b in name.split())
            # numbers matter in Delhi addresses (Sector 21 ≠ Sector 1, Phase 2 ≠ Phase 3)
            qnum = set(re.findall(r"\d+", qn))

            def num_pen(name: str) -> float:
                return 25.0 * len(qnum - set(re.findall(r"\d+", name))) if qnum else 0.0
            scored = [(s + (6 if self.search_names[i].startswith(qn) else 0) - num_pen(self.search_names[i]), i)
                      for _, s, i in res if word_ok(self.search_names[i])]
        except ImportError:  # stdlib fallback
            import difflib
            scored = [(100 * difflib.SequenceMatcher(None, qn, nm).ratio() + (30 if qn in nm else 0), i)
                      for i, nm in enumerate(self.search_names)]
        out, seen = [], set()
        for s, i in sorted(scored, key=lambda x: -x[0]):
            for c in self.search_cls[i]:
                if c in seen:
                    continue
                seen.add(c)
                out.append({"cluster": c, "name": self.g["cl_name"][c], "matched": self.search_names[i],
                            "lat": round(float(self.cl_lat[c]), 6), "lon": round(float(self.cl_lon[c]), 6),
                            "routes": int(self.cl_npat[c]), "score": round(float(s), 1)})
        # prefer well-served stops among near-equal matches
        out.sort(key=lambda r: (-(round(r["score"] / 3)), -r["routes"]))
        return out[:n]

    def resolve(self, place: str) -> tuple[float, float, str]:
        if place is None or not str(place).strip():
            raise PlanError("origin/destination is empty")
        m = re.match(r"^\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*$", str(place))
        if m:
            lat, lon = float(m.group(1)), float(m.group(2))
            la0, la1, lo0, lo1 = self.area
            if not (la0 <= lat <= la1 and lo0 <= lon <= lo1):
                raise PlanError(f"{lat},{lon} is outside the Delhi service area", "out_of_area")
            return lat, lon, f"{lat:.5f}, {lon:.5f}"
        qn = norm_name(place)
        hits = self.search_stops(place, 1) if len(ALIASES.get(qn, qn)) >= 3 else []
        if not hits or hits[0]["score"] < 72:
            raise PlanError(f"unknown place '{place}' — use 'lat,lon' or a bus stop name", "unknown_place")
        h = hits[0]
        return h["lat"], h["lon"], h["name"]

    def nearby(self, lat: float, lon: float, radius: float) -> list[tuple[int, float, float]]:
        x, y = to_xy(lat, lon)
        idx = self.tree.query_ball_point([float(x), float(y)], radius)
        out = []
        for c in idx:
            if self.cl_npat[c] == 0 and not any(c == cl[-1] for cl in ()):  # keep alight-only stops too
                pass
            d = float(haversine_m(lat, lon, self.cl_lat[c], self.cl_lon[c]))
            out.append((c, d * config.WALK_DETOUR / config.WALK_SPEED_MS, d))
        out.sort(key=lambda r: r[2])
        return out[: config.ACCESS_MAX_STOPS]

    # ------------------------------------------------------------------ service
    def timetable_wait(self, p: int, pos: int, t: float, dt: int):
        starts = self.pat_starts[p][dt]
        if not starts:
            return None
        t0 = t - self.pat_off[p][pos]
        lo = bisect_left(starts, t0 - 1800)
        hi = bisect_left(starts, t0 + 1800)
        n = hi - lo
        if n >= 2:
            return min(1800.0 / n, config.MAX_WAIT_S)      # half of headway 3600/n
        nxt = bisect_left(starts, t0)
        if nxt < len(starts) and starts[nxt] - t0 <= 5400:
            return max(float(starts[nxt] - t0), 120.0)
        return None

    def wait(self, p: int, pos: int, t: float, dt: int, info: list | None = None):
        """Expected wait. GPS headways (with bunching) when observed, else the timetable."""
        tw = self.timetable_wait(p, pos, t, dt)
        wc = self.wait_calib
        W = self.pat_gw[p]
        if W is not None:
            h = int(t // 3600) % 24
            g0 = float(W[pos, dt, h])
            if g0 == g0:                                   # observed (not NaN), n >= 6 headways
                if wc:      # calibrated on real waits of training days (app/ridecheck.py)
                    w = g0 * wc["gps"]
                else:
                    w = g0 if tw is None else min(g0, 2.5 * tw)
                if info is not None:
                    gw = self.tt.gps_wait.get((self.pat_rint[p], self.pat_cl[p][pos], dt, h))
                    info.append({"source": "GPS", "timetable_s": None if tw is None else round(tw),
                                 "bunched_pct": round(100 * gw[2]) if gw else None})
                return min(w, config.MAX_WAIT_S)
        if tw is None:
            return None
        if info is not None:
            info.append({"source": "timetable", "timetable_s": round(tw), "bunched_pct": None})
        if wc:              # real waits run longer than half the scheduled headway (never below the schedule)
            return max(tw, min(tw * wc["timetable"], config.MAX_WAIT_S))
        return tw

    def crowd_level(self, p: int, seg: int, dt: int, t: float) -> int:
        if self.crowd is None:
            return 0
        return self.crowd.level(p, seg, dt, t)

    # ------------------------------------------------------------------ RAPTOR
    def raptor(self, acc, egr: dict, t0: float, dt: int, mode: str, banned: set, now=None):
        W = config.MODES[mode]
        cf, w_wait, w_walk, tpen = W["crowd"], W["w_wait"], W["w_walk"], W["transfer"]
        use_crowd = self.crowd is not None and mode != "fastest"
        tt = self.tt
        run_l, dwell_l = tt.run_l[dt], tt.dwell_l[dt]
        live = tt.live if now is not None else None
        # admissible lower bound of the remaining cost from each cluster (straight line at 108 km/h,
        # then the egress walk); every mode weight is >= 1, so cost >= seconds. Prunes the far side of
        # the city once a first journey is known (A*-style target pruning).
        ec = np.fromiter(egr.keys(), dtype=np.int64)
        ew = np.array([egr[int(c)][0] for c in ec], dtype=float) * w_walk
        lb = (np.hypot(self.cx[:, None] - self.cx[ec][None], self.cy[:, None] - self.cy[ec][None]) / 30.0
              + ew[None]).min(axis=1).tolist() if len(ec) else [0.0] * self.ncl
        best: dict[int, float] = {}
        T_prev = {}
        for c, ws, d in acc:
            cost = ws * w_walk
            if cost < best.get(c, INF):
                best[c] = cost
                T_prev[c] = (cost, t0 + ws, "access", (ws, d))
        rounds_R, rounds_T = [], [T_prev]
        target = INF
        cands = []
        for k in range(1, config.MAX_TRANSFERS + 2):
            first: dict[int, int] = {}
            for c in T_prev:
                for p, pos in self.cl_pat[c]:
                    if p in banned:
                        continue
                    if pos < first.get(p, 1 << 30):
                        first[p] = pos
            R = {}
            for p, pos0 in first.items():
                cls, prs = self.pat_cl[p], self.pat_pairs[p]
                boarded = False
                t = cst = 0.0
                bpos = 0
                bwait = bdep = 0.0
                cl_p = self.crowd.pat_levels(p, dt) if use_crowd else None
                if cl_p is not None:
                    cl_p, cl_n = cl_p
                for pos in range(pos0, len(cls)):
                    c = cls[pos]
                    if boarded:
                        pi = prs[pos - 1]
                        h = int(t // 3600) % 24
                        run = run_l[h][pi]
                        if live is not None:
                            run = live.blend(pi, run, t, now)
                        m = cf[cl_p[(int(t // 1800) % 48) * cl_n + pos - 1]] if cl_p is not None else cf[0]
                        t_arr = t + run
                        c_arr = cst + run * m
                        if c_arr < best.get(c, INF) and c_arr + lb[c] < target:
                            best[c] = c_arr
                            R[c] = (c_arr, t_arr, p, bpos, pos, bwait, bdep)
                            e = egr.get(c)
                            if e is not None and c_arr + e[0] * w_walk < target:
                                target = c_arr + e[0] * w_walk      # tighten pruning within the round
                        dw = dwell_l[int(t_arr // 3600) % 24][pi]
                        t = t_arr + dw
                        cst = c_arr + dw * m
                    lab = T_prev.get(c)
                    if lab is not None and pos < len(cls) - 1:
                        extra = config.MIN_TRANSFER_S if lab[2] == "bus" else 0.0
                        cb0 = lab[0] + extra * w_walk + (tpen if k > 1 else 0.0)
                        if (boarded and cb0 >= cst) or cb0 + lb[c] >= target:   # cannot win (wait >= 0)
                            continue
                        ready = lab[1] + extra
                        w = self.wait(p, pos, ready, dt)
                        if w is not None:
                            cb = cb0 + w * w_wait
                            if not boarded or cb < cst:
                                boarded, t, cst, bpos, bwait, bdep = True, ready + w, cb, pos, w, ready + w
            # walking transfers
            T = {c: (v[0], v[1], "bus", None) for c, v in R.items()}
            for c, v in R.items():
                for c2, ws in self.transfers[c]:
                    nc = v[0] + ws * w_walk
                    if nc < best.get(c2, INF) and nc + lb[c2] < target:
                        best[c2] = nc
                        T[c2] = (nc, v[1] + ws, "walk", (c, ws))
            # egress
            for c, v in R.items():
                e = egr.get(c)
                if e is not None:
                    tot = v[0] + e[0] * w_walk
                    cands.append((tot, k, c))
                    target = min(target, tot)
            rounds_R.append(R)
            rounds_T.append(T)
            T_prev = T
            if not T:
                break
        # reconstruct the best candidate for each number of rides
        out = []
        for k in sorted({k for _, k, _ in cands}):
            tot, _, e = min((x for x in cands if x[1] == k), key=lambda x: x[0])
            legs, transfers = [], []
            c, kk = e, k
            access = None
            while True:
                R = rounds_R[kk - 1]
                cost, t_arr, p, bpos, apos, w, dep = R[c]
                legs.append((p, bpos, apos))
                board_c = self.pat_cl[p][bpos]
                lab = rounds_T[kk - 1][board_c]
                if lab[2] == "access":
                    access = (board_c, lab[3][0], lab[3][1])
                    break
                if lab[2] == "walk":
                    frm, ws = lab[3]
                    transfers.append((frm, board_c, ws))
                    c = frm
                else:
                    transfers.append((board_c, board_c, 0.0))
                    c = board_c
                kk -= 1
            legs.reverse()
            transfers.reverse()
            ew, ed = egr[e][0], egr[e][1]
            out.append({"cost": tot, "access": access, "legs": legs, "transfers": transfers,
                        "egress": (e, ew, ed)})
        return out

    # ------------------------------------------------------------------ simulate
    def simulate(self, it: dict, t0: float, dt: int, now=None, detail: bool = True):
        """Compute all times for a fixed itinerary departing at t0 (seconds of the service day)."""
        tt = self.tt
        t = t0
        c0, ws, d = it["access"]
        legs_out = []
        if detail:
            legs_out.append({"type": "walk", "kind": "access", "duration_s": round(ws), "distance_m": round(d),
                             "to_stop": self._cl_info(c0)})
        t += ws
        totals = {"walk_s": ws, "wait_s": 0.0, "ride_s": 0.0}
        worst = 0
        conf_w, conf_t = 0.0, 0.0
        sched_total = ws
        for i, (p, bpos, apos) in enumerate(it["legs"]):
            if i > 0:
                frm, to, wsx = it["transfers"][i - 1]
                if frm == to:
                    t += config.MIN_TRANSFER_S
                    totals["walk_s"] += config.MIN_TRANSFER_S
                    sched_total += config.MIN_TRANSFER_S
                else:
                    t += wsx
                    totals["walk_s"] += wsx
                    sched_total += wsx
                    if detail:
                        legs_out.append({"type": "walk", "kind": "transfer", "duration_s": round(wsx),
                                         "distance_m": round(wsx * config.WALK_SPEED_MS / config.WALK_DETOUR),
                                         "from_stop": self._cl_info(frm), "to_stop": self._cl_info(to)})
            winfo: list = []
            w = self.wait(p, bpos, t, dt, winfo)
            if w is None:
                return None
            totals["wait_s"] += w
            sched_total += w
            t_board = t + w
            t = t_board
            prs = self.pat_pairs[p]
            seg_levels, crowd_segs = [], []
            for pos in range(bpos, apos):
                pi = prs[pos]
                h = int(t // 3600) % 24
                run = tt.run(pi, dt, t, now)
                lv = tt.level_l[dt][h][pi]
                cl = self.crowd_level(p, pos, dt, t)
                crowd_segs.append(cl)
                seg_levels.append(lv)
                conf_w += run * (lv / 3)
                conf_t += run
                sched_total += float(tt.sched[pi])
                t += run
                if pos < apos - 1:
                    t += tt.dwell_l[dt][int(t // 3600) % 24][pi]
            ride_model = t - t_board
            # whole-ride correction from training days (real rides take longer than the sum of their
            # clean stretches: timing-point stops, jams, poorly detected stops; see app/ridecheck.py)
            ride = ride_model * self.ride_k(t_board)
            t = t_board + ride
            totals["ride_s"] += ride
            worst = max(worst, max(crowd_segs) if crowd_segs else 0)
            if detail:
                leg = self._bus_leg(p, bpos, apos, w, t_board, t, ride, seg_levels, crowd_segs, dt)
                leg["wait_info"] = winfo[0] if winfo else None
                leg["ride_model_s"] = round(ride_model)
                legs_out.append(leg)
        ce, ew, ed = it["egress"]
        t += ew
        totals["walk_s"] += ew
        sched_total += ew
        if detail:
            legs_out.append({"type": "walk", "kind": "egress", "duration_s": round(ew), "distance_m": round(ed),
                             "from_stop": self._cl_info(ce)})
        return {"t_arrive": t, "total_s": t - t0, "legs": legs_out, "totals": totals, "worst_crowd": worst,
                "confidence": conf_w / conf_t if conf_t else 0.0, "timetable_total_s": sched_total}

    def ride_k(self, t: float) -> float:
        """Median actual/planned whole-ride ratio for this time of day (1.0 if not calibrated)."""
        rc = self.ride_calib
        if not rc:
            return 1.0
        from app.calibration import bucket_of
        return float(rc["by_bucket"].get(bucket_of(int(t // 3600) % 24), rc["all"])[2])

    def _cl_info(self, c: int) -> dict:
        return {"name": self.g["cl_name"][c], "lat": round(float(self.cl_lat[c]), 6),
                "lon": round(float(self.cl_lon[c]), 6)}

    def _bus_leg(self, p, bpos, apos, wait, t_board, t_alight, ride, seg_levels, crowd_segs, dt):
        P = self.pat[p]
        sids = P["stops"][bpos: apos + 1]
        poly = [list(map(lambda v: round(float(v), 6), self.stop_ll[s])) for s in sids]
        # road path per stretch from GTFS shapes (straight if unknown)
        pp = pair_paths()
        road = [pp.get(int(self.pat_pairs[p][j])) or [poly[j - bpos], poly[j - bpos + 1]] for j in range(bpos, apos)]
        lvc =np.bincount(np.asarray(seg_levels, dtype=int), minlength=4)
        main_level = int(np.argmax(lvc))
        crowd = None
        if self.crowd is not None:
            cs = np.asarray(crowd_segs)
            known = cs[cs > 0]
            crowd = {"level": int(known.max()) if len(known) else 0,
                     "label": CROWD_NAMES[int(known.max()) if len(known) else 0],
                     "segments": cs.tolist(),
                     "summary": self.crowd.describe(cs),
                     "detail": self.crowd.leg_detail(p, bpos, apos, dt, t_board)}
        return {"type": "bus", "route": P["name"], "route_long_name": P["long_name"], "headsign": P["headsign"],
                "pattern": int(p), "board_stop": {"name": self.stop_name[sids[0]], "stop_id": sids[0],
                                                  "lat": poly[0][0], "lon": poly[0][1]},
                "alight_stop": {"name": self.stop_name[sids[-1]], "stop_id": sids[-1],
                                "lat": poly[-1][0], "lon": poly[-1][1]},
                "stops": apos - bpos, "wait_s": round(wait), "depart_s": t_board, "arrive_s": t_alight,
                "ride_s": round(ride), "timetable_ride_s": round(float(self.pat_off[p][apos] - self.pat_off[p][bpos])),
                "estimate_source": LEVEL_NAMES[main_level], "estimate_level": main_level,
                "crowding": crowd, "polyline": poly, "road": road}

    # ------------------------------------------------------------------ public
    def plan(self, origin: str, dest: str, depart: str | None = None, mode: str = "fastest", n: int = 3,
             live: bool = False) -> dict:
        if mode not in config.MODES:
            raise PlanError(f"mode must be one of {list(config.MODES)}")
        n = max(1, min(int(n), 5))
        olat, olon, oname = self.resolve(origin)
        dlat, dlon, dname = self.resolve(dest)
        when = parse_depart(depart)
        date = when.date().isoformat()
        dt = config.daytype_of(date)
        t0 = when.hour * 3600 + when.minute * 60 + when.second
        now = t0 if (live and self.tt.live is not None) else None
        warnings = []
        direct = float(haversine_m(olat, olon, dlat, dlon))
        walk_only = {"total_s": round(direct * config.WALK_DETOUR / config.WALK_SPEED_MS), "distance_m": round(direct)}
        base = {"from": {"query": origin, "name": oname, "lat": olat, "lon": olon},
                "to": {"query": dest, "name": dname, "lat": dlat, "lon": dlon},
                "depart": when.isoformat(), "daytype": config.DAYTYPES[dt], "mode": mode,
                "live": {"used": now is not None, "source": getattr(self.tt.live, "label", None)},
                "model": {"gps": self.tt.has_gps, "train_days": self.tt.meta.get("train_days")}}
        if direct < 150:
            return {**base, "options": [], "walk_only": walk_only,
                    "warnings": ["origin and destination are the same place — just walk"]}
        radius = config.ACCESS_MAX_M
        acc = self.nearby(olat, olon, radius)
        egr_l = self.nearby(dlat, dlon, radius)
        if not acc or not egr_l:
            radius = config.ACCESS_FALLBACK_M
            acc = acc or self.nearby(olat, olon, radius)
            egr_l = egr_l or self.nearby(dlat, dlon, radius)
            if not acc or not egr_l:
                which = "origin" if not acc else "destination"
                raise PlanError(f"no bus stop within {radius:.0f} m of the {which}", "no_stop_nearby")
            warnings.append(f"nearest bus stop is more than {config.ACCESS_MAX_M:.0f} m away")
        egr = {c: (ws, d) for c, ws, d in egr_l}

        # find the first time with service (night queries roll over to the next morning)
        res, shift = None, 0
        tq, dq, dtq = t0, date, dt
        while shift <= 9 * 3600:
            tq = t0 + shift
            dq, dtq = date, dt
            if tq >= 86400 + 2 * 3600:      # past the GTFS service day → next calendar day
                tq -= 86400
                dq = (when.date() + _dt.timedelta(days=1)).isoformat()
                dtq = config.daytype_of(dq)
            res = self.raptor(acc, egr, tq, dtq, mode, set(), now if shift == 0 else None)
            if res:
                break
            shift += 1800
        if not res:
            return {**base, "options": [], "walk_only": walk_only if direct <= 5000 else None,
                    "warnings": warnings + ["no bus journey found (with up to 2 transfers)"]}
        if shift:
            now = None
            # leave as late as possible while still catching the same first bus
            it0 = min(res, key=lambda r: r["cost"])
            s_ref = self.simulate(it0, tq, dtq, None, detail=False)
            if s_ref is not None:
                best_step = 0
                for step in range(5, 181, 5):
                    s2 = self.simulate(it0, tq + step * 60, dtq, None, detail=False)
                    if s2 is None or s2["t_arrive"] > s_ref["t_arrive"] + 120:
                        break
                    best_step = step
                tq += best_step * 60
                res = self.raptor(acc, egr, tq, dtq, mode, set(), None) or res
            warnings.append(f"no bus at {fmt_hm(t0)}; the first useful bus means leaving at about {fmt_hm(tq)}")
            date, dt, t0, shift = dq, dtq, tq, 0
            base["daytype"] = config.DAYTYPES[dt]
        options, seen, banned = [], set(), set()
        for attempt in range(n + 2):
            if attempt:
                res = self.raptor(acc, egr, t0, dt, mode, banned, now)
            if not res:
                break
            new = False
            for it in sorted(res, key=lambda r: r["cost"]):
                key = tuple(self.pat[p]["name"] for p, _, _ in it["legs"])
                if key in seen:
                    continue
                seen.add(key)
                sim = self.simulate(it, t0, dt, now)
                if sim is None:
                    continue
                options.append((it, sim))
                new = True
            # ban the patterns of the best option → next run must use a different route
            best = min(res, key=lambda r: r["cost"])
            for p, b, a in best["legs"]:
                banned.add(p)
            if len(options) >= n + 2 or (not new and attempt > 1):
                break
        if not options:
            return {**base, "options": [], "walk_only": walk_only if direct <= 5000 else None,
                    "warnings": warnings + ["no bus journey found (with up to 2 transfers)"]}

        W = config.MODES[mode]
        def score(o):
            it, s = o
            tot = s["totals"]
            m = W["crowd"][s["worst_crowd"]] if self.crowd is not None else 1.0
            return (tot["ride_s"] * m + tot["wait_s"] * W["w_wait"] + tot["walk_s"] * W["w_walk"]
                    + W["transfer"] * (len(it["legs"]) - 1))
        options.sort(key=score)
        options = options[:n]
        fastest = min(s["total_s"] for _, s in options)
        out = []
        for rank, (it, s) in enumerate(options):
            prof = []
            for h in range(24):
                ss = self.simulate(it, h * 3600 + (t0 % 3600), dt, detail=False)
                prof.append(round(ss["total_s"] / 60, 1) if ss else None)
            day_hours = [(h, v) for h, v in enumerate(prof) if v is not None and 6 <= h <= 22]
            quiet = min(day_hours, key=lambda x: x[1]) if day_hours else None
            busy = max(day_hours, key=lambda x: x[1]) if day_hours else None
            o = format_option(s, date, t0)
            o["range"] = self.reliability(s, t0, date)
            o.update({
                "rank": rank + 1, "transfers": len(it["legs"]) - 1,
                "tags": [], "profile_minutes_by_hour": prof,
                "quiet_hour": {"hour": quiet[0], "total_min": quiet[1],
                               "saves_min": round(s["total_s"] / 60 - quiet[1], 1)} if quiet else None,
                "busiest_hour": {"hour": busy[0], "total_min": busy[1]} if busy else None,
            })
            if s["total_s"] == fastest:
                o["tags"].append("fastest")
            out.append(o)
        if self.crowd is not None:
            best_c = min(range(len(out)), key=lambda i: (options[i][1]["worst_crowd"] or 9, options[i][1]["total_s"]))
            known = [options[i][1]["worst_crowd"] for i in range(len(out)) if options[i][1]["worst_crowd"] > 0]
            if known and options[best_c][1]["worst_crowd"] > 0 and options[best_c][1]["worst_crowd"] < max(known):
                out[best_c]["tags"].append("most comfortable")
            mint = min(o["transfers"] for o in out)
            for o in out:
                if o["transfers"] == mint and "fewest transfers" not in sum((x["tags"] for x in out), []):
                    o["tags"].append("fewest transfers")
        # "leave later / earlier": same best itinerary at other departure times
        it0, s0 = options[0]
        shifts = []
        for dm in (-15, 15, 30):
            s2 = self.simulate(it0, t0 + dm * 60, dt, None)
            if s2 is None:
                continue
            shifts.append({"minutes": dm, "total_min": round(s2["total_s"] / 60, 1),
                           "arrive": iso_at(date, t0 + dm * 60 + s2["total_s"]),
                           "change_min": round((s2["total_s"] - s0["total_s"]) / 60, 1),
                           "crowding": CROWD_NAMES[s2["worst_crowd"]], "crowding_level": s2["worst_crowd"]})
        base["other_departures"] = shifts
        if self.crowd is not None and s0["worst_crowd"] >= 2:
            better = [x for x in shifts if x["minutes"] > 0 and 0 < x["crowding_level"] < s0["worst_crowd"]]
            if better:
                base["leave_later"] = better[0]
        return {**base, "options": out, "walk_only": walk_only if direct <= config.DIRECT_WALK_MAX_M else None,
                "warnings": warnings}

    def reliability(self, s: dict, t0: float, date: str) -> dict | None:
        """Range from calibrated actual/predicted ride-time ratios (training days only)."""
        cal = self.tt.calib
        if not cal:
            return None
        from app.calibration import bucket_of
        b = bucket_of(int(t0 // 3600) % 24)
        if self.ride_calib:          # whole real rides; the point estimate already includes their median
            q = self.ride_calib["by_bucket"].get(b, self.ride_calib["all"])     # QS = .1 .25 .5 .75 .85 .9
            q = [x / q[2] for x in q]
        else:                        # stretch chains
            q = cal["by_bucket"].get(b, cal["all"])
        tot = s["totals"]
        fixed = tot["walk_s"]
        ride, wait = tot["ride_s"], tot["wait_s"]
        lo = fixed + 0.5 * wait + ride * q[1]
        hi = fixed + 1.6 * wait + ride * q[5]
        typ_lo = fixed + 0.75 * wait + ride * q[1]
        typ_hi = fixed + 1.25 * wait + ride * q[3]
        return {"typical_min": [round(typ_lo / 60), round(typ_hi / 60)],
                "low_min": round(lo / 60), "high_min": round(hi / 60),
                "arrive_by_9_of_10": iso_at(date, t0 + hi),
                "note": "range from GPS trips on training days (actual vs predicted); wait 0.5–1.6× expected"}


def parse_depart(depart: str | None) -> _dt.datetime:
    if not depart:
        return _dt.datetime.now(IST).replace(microsecond=0, tzinfo=None)
    s = str(depart).strip().replace("Z", "+00:00")
    try:
        d = _dt.datetime.fromisoformat(s)
    except ValueError as e:
        m = re.match(r"^(\d{1,2}):(\d{2})$", s)
        if not m:
            raise PlanError(f"depart must be ISO date-time (e.g. 2026-10-27T08:30) or HH:MM, got '{depart}'") from e
        today = _dt.datetime.now(IST).date()
        d = _dt.datetime(today.year, today.month, today.day, int(m.group(1)), int(m.group(2)))
    if d.tzinfo is not None:
        d = d.astimezone(IST).replace(tzinfo=None)
    return d


def fmt_hm(t: float) -> str:
    t = int(round(t))
    return f"{(t // 3600) % 24:02d}:{(t % 3600) // 60:02d}"


def iso_at(date: str, t: float) -> str:
    base = _dt.datetime.fromisoformat(date)
    return (base + _dt.timedelta(seconds=round(t))).replace(tzinfo=IST).isoformat()


def format_option(s: dict, date: str, t0: float) -> dict:
    legs = []
    for L in s["legs"]:
        L = dict(L)
        if L["type"] == "bus":
            L["depart"] = iso_at(date, L.pop("depart_s"))
            L["arrive"] = iso_at(date, L.pop("arrive_s"))
        legs.append(L)
    tot = s["totals"]
    return {"depart": iso_at(date, t0), "arrive": iso_at(date, s["t_arrive"]),
            "total_min": round(s["total_s"] / 60, 1),
            "walk_min": round(tot["walk_s"] / 60, 1), "wait_min": round(tot["wait_s"] / 60, 1),
            "ride_min": round(tot["ride_s"] / 60, 1),
            "timetable_total_min": round(s["timetable_total_s"] / 60, 1),
            "gps_share": round(100 * s["confidence"]),
            "crowding": CROWD_NAMES[s["worst_crowd"]], "crowding_level": s["worst_crowd"],
            "legs": legs}
