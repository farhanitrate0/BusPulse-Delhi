"""Tickets → crowding per (route, stretch, daytype, half-hour).

1. Ticket route code (e.g. 8UP) → GTFS route: exact on normalised short/long name/id; else
   the base number (UP/DOWN stripped) → candidate routes, chosen per ticket by stop-name match.
2. Start/end stop NAMES fuzzy-matched only within the candidate route's own stop list.
   Missing end stop → route's median ride length. Wrong order → the other direction.
3. Each ticket adds passenger_count to every stretch it rides, at the half-hour it is there
   (boarding time + timetable offset).
4. Per (route, stretch, daytype, half-hour): average per day ÷ buses scheduled over that stretch
   in that half-hour × 1/e-ticket share (3.5 %) × uplift for riders not in the data (passes) = load.
5. Level: seats (< 0.9·seats) / standing (< seats + ½ standing) / crowded. Confidence by ticket count.
Outputs data/model/crowd.parquet, crowd_meta.json, docs/CROWDING_CHECKS.md.
"""
from __future__ import annotations

import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from app import gtfs as gtfs_mod  # noqa: E402
from app.crowding import level_of  # noqa: E402

try:
    from rapidfuzz import fuzz, process

    def best_match(q, choices):
        r = process.extractOne(q, choices, scorer=fuzz.WRatio)
        return (r[2], r[1]) if r else (-1, 0)
except ImportError:  # pragma: no cover
    import difflib

    def best_match(q, choices):
        sc = [difflib.SequenceMatcher(None, q, c).ratio() * 100 for c in choices]
        i = int(np.argmax(sc))
        return i, sc[i]

ABBR = [(r"\bngr\b", "nagar"), (r"\bmkt\b", "market"), (r"\bstn\b", "station"), (r"\brly\b", "railway"),
        (r"\bt\b", "terminal"), (r"\bterm\b", "terminal"), (r"\bopp\b", ""), (r"\bxing\b", "crossing"),
        (r"\bext\b", "extension"), (r"\bph\b", "phase"), (r"\bsec\b", "sector"), (r"\bmetro station\b", "metro")]


def nstop(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    for a, b in ABBR:
        s = re.sub(a, b, s)
    return re.sub(r"\s+", " ", s).strip()


def nroute(s: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(s).upper())


def base_route(s: str) -> str:
    return re.sub(r"(UP|DOWN|DN|U|D)$", "", nroute(s))


def md_table(df: pd.DataFrame) -> str:
    cols = [str(df.index.name or "")] + [str(c) for c in df.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for i, r in df.iterrows():
        out.append("| " + " | ".join([str(i)] + [f"{v:,.1f}" if isinstance(v, float) else str(v) for v in r]) + " |")
    return "\n".join(out)


def main():
    t_start = time.time()
    config.ensure_dirs()
    g = gtfs_mod.load()
    P = g["patterns"]
    print(f"reading {config.TICKETS_CSV}")
    tk = pd.read_csv(config.TICKETS_CSV, dtype=str, keep_default_na=False)
    n_all = len(tk)
    tk["pax"] = pd.to_numeric(tk.passenger_count, errors="coerce").fillna(1).clip(1, 10).astype(int)
    ts = pd.to_datetime(tk.created_at.str.slice(0, 19), errors="coerce", format="%Y-%m-%d %H:%M:%S")
    tk["date"] = ts.dt.strftime("%Y-%m-%d")
    tk["sec"] = ts.dt.hour * 3600 + ts.dt.minute * 60 + ts.dt.second
    tk = tk[ts.notna()].copy()
    tk["dt"] = tk.date.map({d: config.daytype_of(d) for d in tk.date.unique()})
    tk["day"] = ts[ts.notna()].dt.day.to_numpy()

    # ---- route index from GTFS
    routes = g["routes"]
    by_name: dict[str, set] = defaultdict(set)
    by_base: dict[str, set] = defaultdict(set)
    for col in ("route_short_name", "route_long_name", "route_id"):
        if col in routes:
            for rid, v in zip(routes.route_id, routes[col]):
                if v:
                    by_name[nroute(v)].add(rid)
                    by_base[base_route(v)].add(rid)
    pats_of_route: dict[str, list[int]] = defaultdict(list)
    for p in P:
        pats_of_route[str(p["route_id"])].append(p["pid"])
    # main patterns per route: the most-trips pattern per direction/headsign, plus any big variant
    cand_pats: dict[str, list[int]] = {}
    pat_names = {p["pid"]: [nstop(g["cl_name"][int(c)]) for c in p["cl"]] for p in P}
    # use the real stop names of the pattern (more detail than cluster leader names)
    sname = dict(zip(g["stops"].stop_id, g["stops"].name))
    for p in P:
        # map cluster positions to the pattern's own stop names
        pat_names[p["pid"]] = [nstop(sname.get(s, "")) for s in p["stops"]]

    route_codes = tk.route.unique()
    code_map, code_how = {}, {}
    for code in route_codes:
        rids = by_name.get(nroute(code))
        how = "exact"
        if not rids:
            rids = by_base.get(base_route(code))
            how = "base"
        if not rids:
            code_map[code], code_how[code] = [], "none"
            continue
        pl = []
        for rid in rids:
            ps = sorted(pats_of_route.get(str(rid), []), key=lambda i: -P[i]["n_trips"])
            pl.extend(ps[:4])
        code_map[code], code_how[code] = pl, how
    tk["route_how"] = tk.route.map(code_how)
    route_match = tk.route_how.value_counts(normalize=True).mul(100).round(1).to_dict()
    print(f"route match (share of tickets): {route_match}")

    # ---- stop-name matching within candidate patterns (cached per pattern × name)
    cache: dict[tuple[int, str], tuple[int, float]] = {}

    def match(pid, name):
        k = (pid, name)
        if k not in cache:
            cache[k] = best_match(name, pat_names[pid]) if name else (-1, 0.0)
        return cache[k]

    trip_keys = tk.groupby(["route", "start_stop_name", "end_stop_name"], sort=False).size()
    print(f"{len(trip_keys):,} distinct (route, start, end) combinations")
    med_len = {}
    chosen = {}
    stats = defaultdict(int)
    for (code, s, e), cnt in trip_keys.items():
        pl = code_map.get(code) or []
        if not pl:
            stats["no_route"] += cnt
            continue
        ns, ne = nstop(s), nstop(e) if e not in ("", "NULL") else ""
        best = None
        rev = False
        for pid in pl:
            i, si = match(pid, ns)
            if si < config.FUZZY_THRESHOLD:
                continue
            j, sj = match(pid, ne) if ne else (-1, 0.0)
            if ne and sj >= config.FUZZY_THRESHOLD:
                if j > i:
                    sc = si + sj
                    if best is None or sc > best[0]:
                        best = (sc, pid, i, j, "both")
                elif j < i:
                    rev = True
            elif best is None or best[4] == "start_only" and si > best[0]:
                best = (si, pid, i, -1, "start_only")
        if best is None:
            stats["reversed" if rev else "stop_unmatched"] += cnt
            continue
        chosen[(code, s, e)] = best[1:]
        stats["matched_" + best[4]] += cnt
    # median ride length (stops) per pattern for tickets without a usable end stop
    lens = defaultdict(list)
    for (sc_pid, i, j, how) in chosen.values():
        if j > i:
            lens[sc_pid].append(j - i)
    med_len = {pid: int(np.median(v)) for pid, v in lens.items()}
    glob_med = int(np.median([x for v in lens.values() for x in v])) if lens else 6

    # ---- expand tickets → stretches
    key = list(zip(tk.route, tk.start_stop_name, tk.end_stop_name))
    sel = np.array([k in chosen for k in key])
    t2 = tk[sel].copy()
    ch = [chosen[k] for k in zip(t2.route, t2.start_stop_name, t2.end_stop_name)]
    t2["pid"] = [c[0] for c in ch]
    t2["i"] = [c[1] for c in ch]
    jj = []
    for c in ch:
        pid, i, j, how = c
        if j <= i:
            j = min(len(P[pid]["cl"]) - 1, i + med_len.get(pid, glob_med))
        jj.append(j)
    t2["j"] = jj
    t2 = t2[t2.j > t2.i]
    n_used = int(t2.pax.sum())
    L = (t2.j - t2.i).to_numpy()
    rep = np.repeat(np.arange(len(t2)), L)
    k = t2.i.to_numpy()[rep] + (np.arange(len(rep)) - np.repeat(np.cumsum(L) - L, L))
    pid = t2.pid.to_numpy()[rep]
    # flat per-pattern arrays → vectorised lookups (tens of millions of ticket-stretches)
    base = np.cumsum([0] + [len(p["cl"]) for p in P])[:-1]
    flat_off = np.concatenate([p["off"] for p in P]).astype(np.float64)
    flat_cl = np.concatenate([p["cl"] for p in P]).astype(np.int64)
    rid_arr = np.array([str(p["route_id"]) for p in P], dtype=object)
    ii = t2.i.to_numpy()[rep]
    off = flat_off[base[pid] + k] - flat_off[base[pid] + ii]
    tsec = t2.sec.to_numpy()[rep] + off
    ex = pd.DataFrame({"pid": pid, "k": k, "dt": t2.dt.to_numpy()[rep], "slot": (tsec // 1800 % 48).astype(int),
                       "pax": t2.pax.to_numpy()[rep], "day": t2.day.to_numpy()[rep]})
    ex["ca"] = flat_cl[base[pid] + k]
    ex["cb"] = flat_cl[base[pid] + k + 1]
    ex["route_id"] = rid_arr[pid]
    print(f"{len(ex):,} ticket-stretch records")
    days_per_dt = t2.groupby("dt").date.nunique().to_dict()
    print(f"days per daytype in tickets: {days_per_dt}")

    agg = ex.groupby(["route_id", "ca", "cb", "dt", "slot"]).agg(pax=("pax", "sum"), tickets=("pax", "size")).reset_index()
    agg["pax_per_day"] = agg.pax / agg.dt.map(days_per_dt)

    # ---- buses per (route, stretch, daytype, half-hour) from the timetable
    parts = []
    for p in P:
        cl = np.asarray(p["cl"], dtype=np.int64)
        offs = np.asarray(p["off"][:-1], dtype=np.float64)
        for d in range(3):
            st = np.asarray(p["starts"][d], dtype=np.float64)
            if not len(st):
                continue
            slot = ((st[:, None] + offs[None, :]) // 1800 % 48).astype(np.int16)
            parts.append(pd.DataFrame({"route_id": str(p["route_id"]),
                                       "ca": np.tile(cl[:-1], len(st)), "cb": np.tile(cl[1:], len(st)),
                                       "dt": np.int8(d), "slot": slot.ravel()}))
    bus_tbl = pd.concat(parts, ignore_index=True).groupby(["route_id", "ca", "cb", "dt", "slot"]).size() \
        .rename("buses").reset_index()
    bus_tbl["slot"] = bus_tbl.slot.astype(int)
    bus_tbl["dt"] = bus_tbl.dt.astype(int)
    agg = agg.merge(bus_tbl, on=["route_id", "ca", "cb", "dt", "slot"], how="left")
    agg["buses"] = agg.buses.fillna(0.0)
    stats["cells_without_scheduled_bus"] = int((agg.buses == 0).sum())
    # if no scheduled bus in that half-hour (timetable gaps), use the route's average for that stretch
    agg["buses"] = agg.buses.replace(0, np.nan)
    typical = agg.groupby(["route_id", "ca", "cb", "dt"]).buses.transform("median")
    agg["buses"] = agg.buses.fillna(typical).fillna(1)
    # timetable gaps: a half-hour never counts fewer buses than half the stretch's typical half-hour
    stats["cells_bus_floor"] = int((agg.buses < 0.5 * typical).sum())
    agg["buses"] = np.maximum(agg.buses, (0.5 * typical).fillna(1)).clip(lower=1)
    scale = config.UNTICKETED_UPLIFT / config.ETICKET_SHARE
    agg["load"] = agg.pax_per_day / agg.buses * scale
    # light smoothing across neighbouring half-hours
    agg = agg.sort_values(["route_id", "ca", "cb", "dt", "slot"])
    grp = agg.groupby(["route_id", "ca", "cb", "dt"]).load
    agg["load_s"] = (0.5 * agg.load + 0.25 * grp.shift(1).fillna(agg.load) + 0.25 * grp.shift(-1).fillna(agg.load))
    # capacity-constrained scaling: the e-ticket share differs by route (much higher on e-bus/app-heavy routes).
    # A bus cannot carry more than ~crush load, so if a route's busy-hour load (p95 of its weekday cells with
    # enough tickets) implies more, that route's e-ticket share must be higher: scale it down until it fits.
    crush = (config.BUS_SEATS + config.BUS_STANDING) * 1.15
    busy = agg[(agg.dt == 0) & (agg.tickets >= 10)].groupby("route_id").load_s.quantile(0.95)
    route_fac = (crush / busy).clip(upper=1.0)
    agg["load_s"] = agg.load_s * agg.route_id.map(route_fac).fillna(1.0)
    stats["cells_capped_at_crush"] = int((agg.load_s > crush).sum())
    agg["load_s"] = agg.load_s.clip(upper=crush)
    implied_share = (config.ETICKET_SHARE / route_fac).clip(upper=1.0)
    stats["routes_rescaled"] = int((route_fac < 1).sum())
    share_q = implied_share.quantile([0.25, 0.5, 0.75, 0.9]).round(3).to_dict()
    print(f"routes whose e-ticket share must exceed {config.ETICKET_SHARE:.1%}: {stats['routes_rescaled']} of "
          f"{len(route_fac)}; implied share quantiles {share_q}")
    # tickets in the 3-slot window decide confidence
    tw = agg.groupby(["route_id", "ca", "cb", "dt"]).tickets.transform(lambda x: x.rolling(3, center=True, min_periods=1).sum())
    agg["conf"] = np.where(tw >= config.CONF_HIGH_TICKETS, 3, np.where(tw >= config.CONF_MED_TICKETS, 2,
                                                                        np.where(tw >= 5, 1, 0))).astype(np.int8)
    agg["level"] = np.where(agg.conf > 0, level_of(agg.load_s.to_numpy()), 0).astype(np.int8)
    out = agg[["route_id", "ca", "cb", "dt", "slot", "load_s", "level", "conf", "tickets"]].rename(columns={"load_s": "load"})
    out = out[out.level > 0].astype({"dt": np.int8, "slot": np.int8, "level": np.int8, "conf": np.int8,
                                     "load": np.float32, "tickets": np.int32})
    out.to_parquet(config.MODEL_DIR / "crowd.parquet", index=False)

    # ---- checks against known patterns
    hourly = tk.assign(h=tk.sec // 3600).groupby(["dt", "h"]).pax.sum().unstack(0).fillna(0)
    hourly = hourly.div([days_per_dt.get(d, 1) for d in hourly.columns], axis=1).round(0)
    hourly.columns = [config.DAYTYPES[c] for c in hourly.columns]
    hourly.index.name = "hour"
    per_day = tk.groupby("date").pax.sum()
    wk = {config.DAYTYPES[d]: round(float(per_day[[x for x in per_day.index if config.daytype_of(x) == d]].mean()), 0)
          for d in range(3)}
    # week-to-week stability: route × hour boardings, week of 5–11 vs 12–18 (and 19–25)
    tk["h"] = tk.sec // 3600
    wsets = {"w1": range(5, 12), "w2": range(12, 19), "w3": range(19, 26)}
    wv = {}
    for w, days in wsets.items():
        x = tk[tk.day.isin(list(days)) & (tk.dt == 0)].groupby(["route", "h"]).pax.sum()
        wv[w] = x
    wdf = pd.DataFrame(wv).fillna(0)
    stab = {"w1_vs_w2": round(float(np.corrcoef(wdf.w1, wdf.w2)[0, 1]), 3),
            "w2_vs_w3": round(float(np.corrcoef(wdf.w2, wdf.w3)[0, 1]), 3)} if len(wdf) > 2 else {}
    names = g["cl_name"]
    peak = out[(out.dt == 0) & out.slot.isin([16, 17, 18, 19, 34, 35, 36, 37]) & (out.conf >= 2)]
    top = peak.sort_values("load", ascending=False).drop_duplicates(["route_id", "ca", "cb"]).head(15)
    rname = {str(p["route_id"]): p["name"] for p in P}
    top_tbl = pd.DataFrame({"route": [rname.get(r, r) for r in top.route_id],
                            "stretch": [f"{names[a]} → {names[b]}" for a, b in zip(top.ca, top.cb)],
                            "time": [f"{s//2:02d}:{30*(s%2):02d}" for s in top.slot],
                            "est. on board": top.load.round(0).astype(int).tolist(),
                            "tickets": top.tickets.tolist()})
    top_tbl.index = range(1, len(top_tbl) + 1)
    lv_share = out[out.level > 0].groupby("level").size()
    meta = {
        "tickets_total": int(n_all), "tickets_used_pax": n_used,
        "route_match_pct": route_match,
        "stop_match": {k: int(v) for k, v in stats.items()},
        "stop_match_pct_of_tickets": {k: round(100 * v / n_all, 1) for k, v in stats.items()
                                      if k.startswith(("matched", "no_route", "reversed", "stop_"))},
        "days_per_daytype": {config.DAYTYPES[k]: int(v) for k, v in days_per_dt.items()},
        "per_day_average_boardings": wk, "week_to_week_correlation": stab,
        "cells": int(len(out)), "cells_by_level": {NAMES[int(k)]: int(v) for k, v in lv_share.items()},
        "ac_share_pct": round(100 * float((tk.is_ac == "True").mean()), 1),
        "capacity_rescaling": {"crush_load": crush, "routes_rescaled": stats["routes_rescaled"],
                               "routes_total": int(len(route_fac)),
                               "implied_eticket_share_quantiles": {str(k): v for k, v in share_q.items()}},
        "assumptions": {"eticket_share": config.ETICKET_SHARE, "share_range": config.ETICKET_SHARE_RANGE,
                        "untracked_uplift": config.UNTICKETED_UPLIFT, "seats": config.BUS_SEATS,
                        "standing": config.BUS_STANDING,
                        "levels": "seats < 0.9·seats; standing < seats + ½·standing; else crowded"},
        "build_s": round(time.time() - t_start, 1),
    }
    with open(config.MODEL_DIR / "crowd_meta.json", "w") as f:
        json.dump(meta, f, indent=1)
    md = ["# Crowding checks (auto-generated by scripts/03_tickets_crowding.py)\n",
          "## Matching\n", f"- Route codes matched to GTFS (share of tickets): {route_match}",
          f"- Stop matching (share of tickets): {meta['stop_match_pct_of_tickets']}",
          f"- Passengers used in the load model: {n_used:,} of {tk.pax.sum():,}\n",
          "## Weekday vs weekend\n", f"Average e-ticket boardings per day: {wk}\n",
          "## Week-to-week stability (weekday route × hour boardings)\n", f"{stab}\n",
          "## Boardings per hour (average day)\n", md_table(hourly.astype(int)),
          "\n## Most crowded stretches at weekday peaks (08:00–10:00, 17:00–19:00, medium/high confidence)\n",
          md_table(top_tbl) if len(top_tbl) else "(none)",
          "\n## Capacity-constrained scaling\n",
          f"With one city-wide e-ticket share ({config.ETICKET_SHARE:.1%}) some routes would carry several hundred "
          f"people per bus, which is impossible; their riders simply use e-tickets more. {stats['routes_rescaled']} of "
          f"{len(route_fac)} routes were scaled down so that their busy-hour (p95) load fits a crush load of "
          f"{crush:.0f}. Implied e-ticket share by route (quantiles): {share_q}.",
          "\n## Assumptions\n", json.dumps(meta["assumptions"], indent=1)]
    config.DOCS_DIR.mkdir(exist_ok=True)
    (config.DOCS_DIR / "CROWDING_CHECKS.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md[:12]))
    print(f"\ncells: {len(out):,}  levels: {meta['cells_by_level']}  built in {meta['build_s']} s")


NAMES = {1: "seats", 2: "standing", 3: "crowded"}

if __name__ == "__main__":
    main()
