"""First look at the real data: one GPS day + GTFS + tickets. Paste the output into the chat.

    python scripts/00_inspect.py --day 5
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402


def gtfs_part():
    from app import gtfs as gtfs_mod
    d = gtfs_mod.find_gtfs_dir()
    print(f"\n=== GTFS in {d}")
    for f in sorted(d.glob("*.txt")):
        with open(f, encoding="utf-8-sig") as fh:
            head = [next(fh, "").strip() for _ in range(3)]
        n = sum(1 for _ in open(f, encoding="utf-8-sig")) - 1
        print(f"{f.name:22s} {n:>10,} rows | {head[0][:110]}\n{'':22s} sample: {head[1][:110]}")
    g = gtfs_mod.load(rebuild=True)
    print("stats:", g["stats"])
    r = g["routes"]
    print("routes sample:\n", r.head(8).to_string())
    return g


def gps_part(day: int, g):
    sys.path.insert(0, str(config.ROOT / "scripts"))
    gx = __import__("01_gps_extract")
    db, extracted = gx.get_day_db(day)
    print(f"\n=== GPS {db} ({db.stat().st_size/1e9:.2f} GB)")
    con = sqlite3.connect(db)
    print(con.execute("SELECT sql FROM sqlite_master").fetchall())
    print("sample rows:")
    for row in con.execute("SELECT * FROM vehicle_feed LIMIT 5"):
        print("  ", row)
    t = time.time()
    s = pd.read_sql_query("SELECT * FROM vehicle_feed LIMIT 2000000", con)
    print(f"read 2M rows in {time.time()-t:.1f}s")
    con.close()
    s["ts"] = pd.to_numeric(s.timestamp, errors="coerce")
    s["la"] = pd.to_numeric(s.lat, errors="coerce")
    s["lo"] = pd.to_numeric(s.lng, errors="coerce")
    print("vehicles:", s.vehicle_id.nunique(), "| route_ids:", s.route_id.nunique(),
          "| null route:", round(100 * s.route_id.isna().mean(), 1), "%")
    print("timestamp range (UTC):", pd.to_datetime(s.ts.min(), unit="s"), "→", pd.to_datetime(s.ts.max(), unit="s"))
    print("date/time sample:", s[["date", "time", "timestamp"]].head(3).values.tolist())
    dup = s.duplicated(["vehicle_id", "timestamp"]).mean()
    print(f"dup (vehicle,timestamp): {100*dup:.1f}%  | exact-dup rows: {100*s.duplicated().mean():.1f}%")
    inb = s.la.between(*config.BBOX[:2]) & s.lo.between(*config.BBOX[2:])
    print(f"in Delhi bbox: {100*inb.mean():.1f}%  | lat/lng parse fail: {100*(s.la.isna()|s.lo.isna()).mean():.2f}%")
    ss = s.sort_values(["vehicle_id", "ts"])
    dts = ss.groupby("vehicle_id").ts.diff().dropna()
    print("sampling interval s (quantiles 10/50/90/99):", dts.quantile([.1, .5, .9, .99]).round(1).tolist())
    print("speed col quantiles:", s.speed.quantile([.1, .5, .9, .99]).round(1).tolist())
    rids = set(pd.to_numeric(g["routes"].route_id, errors="coerce").dropna().astype(int))
    gr = s.route_id.dropna().astype(float).astype(int).unique()
    print(f"GPS route_id found in GTFS routes.txt route_id: {100*np.isin(gr, list(rids)).mean():.1f}% of {len(gr)} ids;"
          f" sample GPS ids {gr[:8].tolist()} vs GTFS {sorted(rids)[:8]}")
    print("trip_id sample:", s.trip_id.dropna().head(3).tolist())
    if extracted:
        print(f"(keeping extracted DB for 01: {db})")


def tickets_part(g):
    print(f"\n=== tickets {config.TICKETS_CSV}")
    tk = pd.read_csv(config.TICKETS_CSV, dtype=str, keep_default_na=False, nrows=300000)
    import re
    def nr(s): return re.sub(r"[^A-Z0-9]", "", str(s).upper())
    names = set()
    for c in ("route_short_name", "route_long_name", "route_id"):
        if c in g["routes"]:
            names |= {nr(x) for x in g["routes"][c]}
    codes = tk.route.value_counts()
    hit = codes[[nr(c) in names for c in codes.index]]
    print(f"ticket route codes matching a GTFS route name exactly: {len(hit)}/{len(codes)} codes, "
          f"{100*hit.sum()/codes.sum():.1f}% of tickets")
    print("unmatched examples:", [c for c in codes.index[:40] if nr(c) not in names][:15])
    sn = {str(x).lower() for x in g["stops"].name}
    st = tk.start_stop_name.str.lower()
    print(f"ticket start stop name exactly in GTFS stop names: {100*st.isin(sn).mean():.1f}%")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", type=int, default=5)
    ap.add_argument("--skip-gps", action="store_true")
    a = ap.parse_args()
    config.ensure_dirs()
    g = gtfs_part()
    tickets_part(g)
    if not a.skip_gps:
        gps_part(a.day, g)
