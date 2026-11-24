"""GPS fixes → stop visits → stop-to-stop observations.

Shared by the historical pipeline (scripts/01_gps_extract.py, one day at a time) and the live GTFS-RT
connector (app/gtfsrt.py, a rolling window of recent fixes), so live and history are measured the same way.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402


def network_index(g: dict):
    """KD-tree of stop clusters + sorted adjacent-pair keys and their distances."""
    from scipy.spatial import cKDTree
    from app.geo import to_xy
    ncl = len(g["cl_lat"])
    cx, cy = to_xy(g["cl_lat"], g["cl_lon"])
    tree = cKDTree(np.c_[cx, cy])
    pkeys = g["pairs"][:, 0].astype(np.int64) * ncl + g["pairs"][:, 1]
    order = np.argsort(pkeys)
    pair_set = pkeys[order]
    pdist_sorted = g["pair_dist"][order]

    def pair_dist_lookup(keys):
        return pdist_sorted[np.searchsorted(pair_set, keys)]

    return tree, pair_set, pair_dist_lookup, ncl


def process_vehicles(v, ts, x, y, r, tree, pair_set, pair_dist_lookup, ncl, stats):
    """Arrays sorted by (v, ts). Returns DataFrame of segment observations."""
    n = len(v)
    same = np.r_[False, v[1:] == v[:-1]]
    # --- spike removal: in- and out-speed both too high (2 passes)
    keep = np.ones(n, dtype=bool)
    for _ in range(2):
        idx = np.nonzero(keep)[0]
        vv, tt, xx, yy = v[idx], ts[idx], x[idx], y[idx]
        s_ok = np.r_[False, vv[1:] == vv[:-1]]
        dtt = np.r_[1, np.diff(tt)].astype(float)
        dd = np.r_[0, np.hypot(np.diff(xx), np.diff(yy))]
        sp_in = np.where(s_ok, dd / np.maximum(dtt, 1), 0)
        sp_out = np.r_[sp_in[1:], 0]
        spike = (sp_in > config.SPIKE_SPEED_MS) & (sp_out > config.SPIKE_SPEED_MS)
        if not spike.any():
            break
        keep[idx[spike]] = False
    stats["spikes_removed"] = stats.get("spikes_removed", 0) + int((~keep).sum())
    v, ts, x, y, r = v[keep], ts[keep], x[keep], y[keep], r[keep]
    n = len(v)
    same = np.r_[False, v[1:] == v[:-1]]
    dt_prev = np.r_[0, np.diff(ts)]
    gapflag = (~same) | (dt_prev > config.DATA_GAP_S)       # gap between i-1 and i
    stats["gaps_gt_180s"] = stats.get("gaps_gt_180s", 0) + int((same & (dt_prev > config.DATA_GAP_S)).sum())
    stats["_dt_samples"] = stats.get("_dt_samples", []) + [np.median(dt_prev[same])] if same.any() else stats.get("_dt_samples", [])

    # --- densify fix pairs that are far apart so fast buses do not skip a stop
    d = np.r_[np.hypot(np.diff(x), np.diff(y)), 0]
    dtn = np.r_[np.diff(ts), 0]
    nxt_same = np.r_[same[1:], False]
    m = nxt_same & (d > config.DENSIFY_MIN_M) & (d <= config.DENSIFY_MAX_M) & (dtn > 0) & (dtn <= config.DENSIFY_MAX_DT)
    i0 = np.nonzero(m)[0]
    k = (d[i0] // config.DENSIFY_STEP_M).astype(np.int64)
    tot = int(k.sum())
    if tot:
        rep = np.repeat(i0, k)
        start = np.repeat(np.cumsum(k) - k, k)
        j = np.arange(tot) - start + 1
        frac = j / np.repeat(k + 1, k)
        nx = x[rep] + frac * (x[rep + 1] - x[rep])
        ny = y[rep] + frac * (y[rep + 1] - y[rep])
        nts = ts[rep] + frac * (ts[rep + 1] - ts[rep])
        V = np.r_[v, v[rep]]
        T = np.r_[ts.astype(float), nts]
        X = np.r_[x, nx]
        Y = np.r_[y, ny]
        R = np.r_[r, r[rep]]
        G = np.r_[gapflag, np.zeros(tot, dtype=bool)]
        o = np.lexsort((T, V))
        V, T, X, Y, R, G = V[o], T[o], X[o], Y[o], R[o], G[o]
    else:
        V, T, X, Y, R, G = v, ts.astype(float), x, y, r, gapflag
    stats["points_densified"] = stats.get("points_densified", 0) + tot
    cumgap = np.cumsum(G)

    # --- snap to nearest stop cluster within radius
    dist, cl = tree.query(np.c_[X, Y], distance_upper_bound=config.VISIT_RADIUS_M, workers=-1)
    hit = cl < ncl
    hidx = np.nonzero(hit)[0]
    stats["points_at_stop"] = stats.get("points_at_stop", 0) + int(len(hidx))
    if len(hidx) < 2:
        return None
    hv, ht, hc = V[hidx], T[hidx], cl[hidx].astype(np.int32)
    new = np.r_[True, (hv[1:] != hv[:-1]) | (hc[1:] != hc[:-1]) | (np.diff(ht) > config.VISIT_GAP_S)]
    starts = np.nonzero(new)[0]
    ends = np.r_[starts[1:] - 1, len(hidx) - 1]
    vis_v, vis_c = hv[starts], hc[starts]
    vis_arr, vis_dep = ht[starts], ht[ends]
    vis_ia, vis_id = hidx[starts], hidx[ends]
    vis_r = R[hidx[ends]]
    stats["visits"] = stats.get("visits", 0) + int(len(starts))

    # --- segment observations between consecutive visits of a vehicle
    a, b = np.arange(len(starts) - 1), np.arange(1, len(starts))
    ok = vis_v[a] == vis_v[b]
    a, b = a[ok], b[ok]
    ca, cb = vis_c[a], vis_c[b]
    keyab = ca.astype(np.int64) * ncl + cb
    adj = np.isin(keyab, pair_set)
    stats["seg_candidates"] = stats.get("seg_candidates", 0) + int(len(a))
    stats["seg_drop_not_adjacent"] = stats.get("seg_drop_not_adjacent", 0) + int((~adj).sum())
    a, b, ca, cb, keyab = a[adj], b[adj], ca[adj], cb[adj], keyab[adj]
    run = vis_arr[b] - vis_dep[a]
    dwell = vis_dep[b] - vis_arr[b]
    gap_inside = (cumgap[vis_ia[b]] - cumgap[vis_id[a]]) > 0
    dist_m = pair_dist_lookup(keyab)
    spd = dist_m / np.maximum(run, 1) * 3.6
    good = (~gap_inside) & (run > 0) & (run <= config.MAX_RUN_S) & (spd >= config.MIN_SPEED_KMH) & (spd <= config.MAX_SPEED_KMH)
    stats["seg_drop_data_gap"] = stats.get("seg_drop_data_gap", 0) + int(gap_inside.sum())
    stats["seg_drop_speed_or_time"] = stats.get("seg_drop_speed_or_time", 0) + int(((~gap_inside) & ~good).sum())
    stats["seg_kept"] = stats.get("seg_kept", 0) + int(good.sum())
    dwell = np.where(dwell > config.DWELL_CAP_S, np.nan, dwell)
    a, b = a[good], b[good]
    seg = pd.DataFrame({
        "v": vis_v[a], "r": vis_r[a], "ca": ca[good], "cb": cb[good],
        "dep_a": vis_dep[a], "arr_b": vis_arr[b], "dep_b": vis_dep[b],
        "run": run[good].astype(np.float32), "dwell": dwell[good].astype(np.float32),
    })
    # every stop visit (not only those that start a kept segment): real departures for headways/waits
    # and whole-ride validation. gs/ge = data-gap counter at arrival/departure (a gap lies between
    # visit i and a later visit j of the same bus iff gs[j] != ge[i]).
    vis = pd.DataFrame({"v": vis_v, "r": vis_r.astype(np.int32), "c": vis_c, "arr": vis_arr, "dep": vis_dep,
                        "gs": cumgap[vis_ia].astype(np.int32), "ge": cumgap[vis_id].astype(np.int32)})
    return seg, vis
