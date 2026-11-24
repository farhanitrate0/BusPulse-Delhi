"""Static GTFS → stop clusters, route patterns, schedules. Cached as data/model/gtfs.pkl.

Run directly to (re)build the cache:  python -m app.gtfs
"""
from __future__ import annotations

import pickle
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from app.geo import to_xy  # noqa: E402

CACHE = config.MODEL_DIR / "gtfs.pkl"


def find_gtfs_dir(base: Path = config.GTFS_DIR) -> Path:
    """Locate stops.txt under base (unzips a GTFS zip if that is all there is)."""
    if not base.exists():
        raise FileNotFoundError(f"GTFS folder not found: {base}. Put the GTFS .txt files (or zip) there.")
    hits = sorted(base.rglob("stops.txt"))
    if not hits:
        zips = sorted(base.glob("*.zip"))
        if not zips:
            raise FileNotFoundError(f"No stops.txt or .zip in {base}")
        for z in zips:
            with zipfile.ZipFile(z) as zf:
                zf.extractall(base / z.stem)
        hits = sorted(base.rglob("stops.txt"))
    return hits[0].parent


def read_txt(d: Path, name: str, required: bool = True) -> pd.DataFrame | None:
    p = d / name
    if not p.exists():
        if required:
            raise FileNotFoundError(p)
        return None
    df = pd.read_csv(p, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    for c in df.columns:
        df[c] = df[c].str.strip()
    return df


def hms_to_s(s: pd.Series) -> np.ndarray:
    m = s.str.extract(r"^(\d+):(\d{1,2}):(\d{1,2})$").astype(float)
    return (m[0] * 3600 + m[1] * 60 + m[2]).to_numpy()


def cluster_stops(lat: np.ndarray, lon: np.ndarray, weight: np.ndarray, radius: float):
    """Greedy leader clustering: busiest stops become leaders; members within radius join."""
    x, y = to_xy(lat, lon)
    xy = np.c_[x, y]
    tree = cKDTree(xy)
    label = np.full(len(lat), -1, dtype=np.int32)
    leaders = []
    for i in np.argsort(-weight, kind="stable"):
        if label[i] >= 0:
            continue
        cid = len(leaders)
        leaders.append(i)
        for j in tree.query_ball_point(xy[i], radius):
            if label[j] < 0:
                label[j] = cid
    return label, np.array(leaders, dtype=np.int64)


def service_daytypes(d: Path) -> dict[str, list[bool]]:
    cal = read_txt(d, "calendar.txt", required=False)
    cd = read_txt(d, "calendar_dates.txt", required=False)
    out: dict[str, list[bool]] = {}
    if cal is not None and len(cal):
        wk = cal[["monday", "tuesday", "wednesday", "thursday", "friday"]].astype(int).max(axis=1) > 0
        for sid, a, b, c in zip(cal.service_id, wk, cal.saturday.astype(int) > 0, cal.sunday.astype(int) > 0):
            out[sid] = [bool(a), bool(b), bool(c)]
    if cd is not None and len(cd):
        for sid, date, et in zip(cd.service_id, cd.date, cd.exception_type):
            if et != "1" or len(date) != 8:
                continue
            dt = config.daytype_of(f"{date[:4]}-{date[4:6]}-{date[6:]}")
            out.setdefault(sid, [False, False, False])[dt] = True
    return out


def build(gtfs_dir: Path | None = None, verbose: bool = True) -> dict:
    t_start = time.time()
    d = gtfs_dir or find_gtfs_dir()
    log = print if verbose else (lambda *a, **k: None)
    log(f"[gtfs] reading {d}")
    stops = read_txt(d, "stops.txt")
    routes = read_txt(d, "routes.txt")
    trips = read_txt(d, "trips.txt")
    st = read_txt(d, "stop_times.txt")
    freq = read_txt(d, "frequencies.txt", required=False)

    stops["lat"] = pd.to_numeric(stops.stop_lat, errors="coerce")
    stops["lon"] = pd.to_numeric(stops.stop_lon, errors="coerce")
    bad = stops.lat.isna() | stops.lon.isna() | (stops.lat == 0)
    stats = {"stops_total": int(len(stops)), "stops_bad_coords": int(bad.sum())}
    stops = stops[~bad].reset_index(drop=True)
    stops["name"] = stops.get("stop_name", stops.stop_id)

    # ---- stop_times → per-trip ordered stop list
    st = st[st.stop_id.isin(set(stops.stop_id))].copy()
    st["seq"] = pd.to_numeric(st.stop_sequence, errors="coerce")
    arr = hms_to_s(st.arrival_time) if "arrival_time" in st else np.full(len(st), np.nan)
    dep = hms_to_s(st.departure_time) if "departure_time" in st else arr
    st["t"] = np.where(np.isnan(dep), arr, dep)
    st = st.sort_values(["trip_id", "seq"], kind="stable")
    st["pos"] = st.groupby("trip_id", sort=False).cumcount()
    st = st[["trip_id", "stop_id", "pos", "t"]]

    # trips served per stop → leader weight
    stop_w = st.stop_id.value_counts()
    weight = stops.stop_id.map(stop_w).fillna(0).to_numpy()
    label, leaders = cluster_stops(stops.lat.to_numpy(), stops.lon.to_numpy(), weight, config.STOP_CLUSTER_M)
    stops["cluster"] = label
    ncl = len(leaders)
    cl_lat = stops.groupby("cluster").lat.mean().reindex(range(ncl)).to_numpy()
    cl_lon = stops.groupby("cluster").lon.mean().reindex(range(ncl)).to_numpy()
    cl_name = stops.name.to_numpy()[leaders].tolist()
    cl_stops = stops.groupby("cluster").stop_id.apply(list).reindex(range(ncl)).tolist()
    stats["clusters"] = int(ncl)
    log(f"[gtfs] {len(stops)} stops → {ncl} clusters (≤{config.STOP_CLUSTER_M:.0f} m)")

    # ---- patterns
    key = st.groupby("trip_id", sort=False).stop_id.agg("|".join).rename("key")
    t0 = st[st.pos == 0].set_index("trip_id").t.rename("t0")
    trips = trips.join(key, on="trip_id").join(t0, on="trip_id")
    trips = trips[trips.key.notna()]
    trips["pkey"] = trips.route_id + "#" + trips.key
    pk = {k: i for i, k in enumerate(trips.pkey.unique())}
    trips["pid"] = trips.pkey.map(pk).astype(np.int64)

    sdt = service_daytypes(d)
    if sdt:
        flags = np.array([sdt.get(s, [False, False, False]) for s in trips.service_id], dtype=bool)
    else:
        flags = np.ones((len(trips), 3), dtype=bool)
    stats["trips_no_service_day"] = int((~flags.any(axis=1)).sum())
    for dt in range(3):
        trips[f"f{dt}"] = flags[:, dt]

    # frequencies.txt expansion (start times per trip)
    starts_by_trip: dict[str, list[float]] = {}
    if freq is not None and len(freq):
        for tid, a, b, h in zip(freq.trip_id, hms_to_s(freq.start_time), hms_to_s(freq.end_time),
                                pd.to_numeric(freq.headway_secs, errors="coerce")):
            if np.isnan(a) or np.isnan(b) or not h or np.isnan(h):
                continue
            starts_by_trip.setdefault(tid, []).extend(np.arange(a, b, h).tolist())

    # offsets: median per (pid, pos)
    st = st.join(trips.set_index("trip_id")[["pid", "t0"]], on="trip_id", how="inner")
    st["off"] = st.t - st.t0
    off_med = st.groupby(["pid", "pos"]).off.median()

    rinfo = routes.set_index("route_id")
    short = rinfo.get("route_short_name", pd.Series(dtype=str))
    long_ = rinfo.get("route_long_name", pd.Series(dtype=str))
    headsign_col = "trip_headsign" if "trip_headsign" in trips else None
    dir_col = "direction_id" if "direction_id" in trips else None

    patterns = []
    stop_idx = pd.Series(np.arange(len(stops)), index=stops.stop_id)
    for pid, g in trips.groupby("pid", sort=True):
        sids = g.key.iloc[0].split("|")
        offs = off_med.loc[pid].reindex(range(len(sids))).to_numpy(dtype=float)
        known = ~np.isnan(offs)
        if known.sum() >= 2:
            offs = np.interp(np.arange(len(sids)), np.nonzero(known)[0], offs[known])
        else:
            offs = np.arange(len(sids)) * 90.0  # unknown timetable: 90 s per stop
        cls = label[stop_idx.loc[sids].to_numpy()]
        keep = np.r_[True, cls[1:] != cls[:-1]]
        cls, offs = cls[keep], offs[keep]
        sids = [s for s, k in zip(sids, keep) if k]
        if len(cls) < 2:
            continue
        starts = []
        for dt in range(3):
            gg = g[g[f"f{dt}"]]
            s = []
            for tid, tt in zip(gg.trip_id, gg.t0):
                if tid in starts_by_trip:
                    s.extend(starts_by_trip[tid])
                elif not np.isnan(tt):
                    s.append(float(tt))
            starts.append(sorted(s))
        rid = g.route_id.iloc[0]
        sn = str(short.get(rid, "") or "")
        ln = str(long_.get(rid, "") or "")
        hs = g[headsign_col].mode().iloc[0] if headsign_col and (g[headsign_col] != "").any() else ""
        patterns.append(dict(
            pid=len(patterns), route_id=rid, name=sn or ln or rid, long_name=ln,
            headsign=hs or cl_name[cls[-1]],
            direction=(g[dir_col].mode().iloc[0] if dir_col else ""),
            stops=sids, cl=cls.astype(np.int32), off=offs.astype(np.float32),
            n_trips=int(len(g)), starts=starts,
        ))
    stats["patterns"] = len(patterns)
    stats["routes"] = int(routes.route_id.nunique())
    stats["trips"] = int(len(trips))

    # ---- adjacent cluster pairs
    pair_list: dict[tuple[int, int], int] = {}
    sched_acc: dict[int, list[float]] = {}
    for p in patterns:
        idx = []
        for j in range(len(p["cl"]) - 1):
            k = (int(p["cl"][j]), int(p["cl"][j + 1]))
            if k not in pair_list:
                pair_list[k] = len(pair_list)
            i = pair_list[k]
            idx.append(i)
            sched_acc.setdefault(i, []).extend([float(p["off"][j + 1] - p["off"][j])] * max(1, min(p["n_trips"], 50)))
        p["pairs"] = np.array(idx, dtype=np.int32)
    pairs = np.array(list(pair_list.keys()), dtype=np.int32).reshape(-1, 2)
    cx, cy = to_xy(cl_lat, cl_lon)
    pair_dist = np.hypot(cx[pairs[:, 0]] - cx[pairs[:, 1]], cy[pairs[:, 0]] - cy[pairs[:, 1]])
    pair_sched = np.array([np.median(sched_acc[i]) for i in range(len(pairs))], dtype=np.float32)
    stats["pairs"] = int(len(pairs))
    stats["build_s"] = round(time.time() - t_start, 1)
    log(f"[gtfs] {len(patterns)} patterns, {len(pairs)} adjacent cluster pairs, "
        f"{stats['trips']} trips, built in {stats['build_s']} s")

    routes_out = routes.copy()
    return dict(
        stops=stops[["stop_id", "name", "lat", "lon", "cluster"]].reset_index(drop=True),
        cl_lat=cl_lat, cl_lon=cl_lon, cl_name=cl_name, cl_stops=cl_stops,
        routes=routes_out, patterns=patterns, pairs=pairs,
        pair_dist=pair_dist.astype(np.float32), pair_sched=pair_sched, stats=stats,
        source=str(d),
    )


class PairPaths:
    """Road path per stretch from shapes.txt (scripts/17_pair_paths.py); get(i) → [[lat, lon], ...] or None."""

    def __init__(self, path=None):
        self.pos = None
        p = path or (config.MODEL_DIR / "pair_paths.npz")
        if not Path(p).exists():
            return
        z = np.load(p)
        self.pos = np.full(int(z["n_pairs"]), -1, dtype=np.int64)
        self.pos[z["idx"]] = np.arange(len(z["idx"]))
        self.off, self.lat, self.lon = z["off"], z["lat"], z["lon"]

    def get(self, i: int, nd: int = 5):
        if self.pos is None or i < 0 or i >= len(self.pos) or self.pos[i] < 0:
            return None
        k = self.pos[i]
        a, b = self.off[k], self.off[k + 1]
        return [[round(float(la), nd), round(float(lo), nd)] for la, lo in zip(self.lat[a:b], self.lon[a:b])]


_PAIR_PATHS = None


def pair_paths() -> PairPaths:
    global _PAIR_PATHS
    if _PAIR_PATHS is None:
        _PAIR_PATHS = PairPaths()
    return _PAIR_PATHS


def load(rebuild: bool = False) -> dict:
    if CACHE.exists() and not rebuild:
        with open(CACHE, "rb") as f:
            return pickle.load(f)
    g = build()
    config.ensure_dirs()
    with open(CACHE, "wb") as f:
        pickle.dump(g, f, protocol=pickle.HIGHEST_PROTOCOL)
    return g


if __name__ == "__main__":
    g = load(rebuild=True)
    print(g["stats"])
