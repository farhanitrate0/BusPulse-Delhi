"""Cut each stop-to-stop stretch's road path from GTFS shapes.txt → data/model/pair_paths.npz."""
from __future__ import annotations

import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from app import gtfs  # noqa: E402

MAX_OFF_M = 70.0     # a stop must lie within 70 m of the shape
SIMPLIFY_M = 5.0     # Douglas–Peucker tolerance


def simplify(path: list) -> list:
    if len(path) <= 2:
        return path
    la = np.array([q[0] for q in path]); lo = np.array([q[1] for q in path])
    x, y = gtfs.to_xy(la, lo)
    keep = np.zeros(len(path), bool); keep[[0, -1]] = True
    stack = [(0, len(path) - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        dx, dy = x[j] - x[i], y[j] - y[i]
        n = max(np.hypot(dx, dy), 1e-9)
        dist = np.abs(dy * (x[i + 1:j] - x[i]) - dx * (y[i + 1:j] - y[i])) / n
        k = int(np.argmax(dist))
        if dist[k] > SIMPLIFY_M:
            keep[i + 1 + k] = True
            stack += [(i, i + 1 + k), (i + 1 + k, j)]
    return [q for q, kk in zip(path, keep) if kk]


def project(P: np.ndarray, S: np.ndarray):
    """Distance and along-shape position of each point P on each segment of S."""
    A, B = S[:-1], S[1:]
    d = B - A
    L2 = np.maximum((d ** 2).sum(1), 1e-9)
    t = np.clip(((P[:, None, :] - A[None]) * d[None]).sum(2) / L2[None], 0, 1)
    Q = A[None] + t[..., None] * d[None]
    dist = np.hypot(*(P[:, None, :] - Q).transpose(2, 0, 1))
    seg_len = np.sqrt(L2)
    C = np.r_[0, np.cumsum(seg_len)]
    along = C[:-1][None] + t * seg_len[None]
    return dist, along, Q, C


def main():
    t0 = time.time()
    g = gtfs.load()
    d = gtfs.find_gtfs_dir()
    trips = gtfs.read_txt(d, "trips.txt")[["route_id", "shape_id"]]
    shapes = gtfs.read_txt(d, "shapes.txt")
    shapes["lat"] = pd.to_numeric(shapes.shape_pt_lat, errors="coerce")
    shapes["lon"] = pd.to_numeric(shapes.shape_pt_lon, errors="coerce")
    shapes["seq"] = pd.to_numeric(shapes.shape_pt_sequence, errors="coerce")
    shapes = shapes.dropna(subset=["lat", "lon", "seq"]).sort_values(["shape_id", "seq"], kind="stable")
    shp = {}
    for sid, s in shapes.groupby("shape_id", sort=False):
        la, lo = s.lat.to_numpy(), s.lon.to_numpy()
        if len(la) >= 2:
            x, y = gtfs.to_xy(la, lo)
            shp[sid] = (np.c_[x, y], la, lo)
    route_shapes = trips[trips.shape_id != ""].groupby("route_id").shape_id.agg(lambda s: list(dict.fromkeys(s)))
    print(f"[paths] {len(shp)} shapes, {len(route_shapes)} routes with shapes ({time.time() - t0:.0f} s)")

    stops = g["stops"].set_index("stop_id")
    pairs = g["pairs"]
    cx, cy = gtfs.to_xy(g["cl_lat"], g["cl_lon"])
    straight = np.hypot(cx[pairs[:, 0]] - cx[pairs[:, 1]], cy[pairs[:, 0]] - cy[pairs[:, 1]])
    best: dict[int, tuple[float, list]] = {}   # pair → (detour ratio, path); keep the most direct cut
    for p in g["patterns"]:
        sids = p["stops"]
        la = stops.lat.reindex(sids).to_numpy()
        lo = stops.lon.reindex(sids).to_numpy()
        if np.isnan(la).any():
            continue
        px, py = gtfs.to_xy(la, lo)
        P = np.c_[px, py]
        for sh in route_shapes.get(p["route_id"], []):
            if sh not in shp:
                continue
            S, sla, slo = shp[sh]
            dist, along, Q, C = project(P, S)
            prev, pos = -1.0, []
            for j in range(len(sids)):
                ok = (dist[j] < MAX_OFF_M) & (along[j] >= prev - 5)
                if not ok.any():
                    pos.append(None)
                    continue
                k = int(np.flatnonzero(ok)[np.argmin(along[j][ok])])
                pos.append((float(along[j, k]), k, Q[j, k]))
                prev = float(along[j, k])
            for j, pi in enumerate(p["pairs"]):
                a, b = pos[j], pos[j + 1]
                if a is None or b is None or b[0] <= a[0]:
                    continue
                road = b[0] - a[0]
                if road > 3 * straight[pi] + 300:
                    continue
                ratio = road / max(straight[pi], 1.0)
                if pi in best and best[pi][0] <= ratio:
                    continue
                inner = [(float(sla[i]), float(slo[i])) for i in range(a[1] + 1, b[1] + 1)]
                def ll(k, q):  # projected point → lat/lon
                    seg = S[k + 1] - S[k]
                    f = float(np.clip(np.dot(q - S[k], seg) / max(np.dot(seg, seg), 1e-9), 0, 1))
                    return (sla[k] + f * (sla[k + 1] - sla[k]), slo[k] + f * (slo[k + 1] - slo[k]))
                path = [ll(a[1], a[2])] + inner + [ll(b[1], b[2])]
                best[pi] = (ratio, path)
    idx = np.array(sorted(best), dtype=np.int32)
    off = np.zeros(len(idx) + 1, dtype=np.int64)
    lat, lon = [], []
    for n, pi in enumerate(idx):
        path = simplify(best[int(pi)][1])
        off[n + 1] = off[n] + len(path)
        lat.extend(q[0] for q in path)
        lon.extend(q[1] for q in path)
    out = config.MODEL_DIR / "pair_paths.npz"
    np.savez_compressed(out, idx=idx, off=off, lat=np.array(lat, np.float32), lon=np.array(lon, np.float32),
                        n_pairs=np.int64(len(pairs)))
    print(f"[paths] {len(idx)} of {len(pairs)} stretches cut from road shapes ({len(idx) / len(pairs):.1%}), "
          f"{len(lat)} points → {out} ({out.stat().st_size / 1e6:.1f} MB) in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
