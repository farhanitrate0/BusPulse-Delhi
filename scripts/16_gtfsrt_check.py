"""Prove the live GTFS-RT connector works, without an API key.

    python scripts/16_gtfsrt_check.py                 # replay 17:00–18:30 of 27 Oct as a GTFS-RT feed, check
    python scripts/16_gtfsrt_check.py --serve 8790    # serve that replay over HTTP (real time) for a demo:
        BUSPULSE_GTFSRT_URL=http://localhost:8790/vehicle_positions.pb python run.py

The raw fixes of a held-out day (scripts/15_gtfsrt_window.py) are turned into GTFS-RT VehiclePositions
messages exactly as a live feed sends them: every poll, the latest position of every bus. The connector
decodes them, measures stop-to-stop runs, and the result is compared with the offline pipeline's runs for the
same day and minutes (which saw every fix, in order, after the fact).
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from app.gtfsrt import GtfsRtConnector, decode_vehicle_positions, encode_vehicle_positions  # noqa: E402

DAY = 27


def load_window():
    f = config.TMP_DIR / f"rt_window_{DAY}.parquet"
    if not f.exists():
        sys.exit(f"{f} missing — run scripts/15_gtfsrt_window.py --day {DAY} first")
    d = pd.read_parquet(f).dropna(subset=["lat", "lon", "ts"]).sort_values("ts", kind="stable").reset_index(drop=True)
    return d


class Replay:
    """Latest fix of every bus at a given time, as a GTFS-RT feed would report it."""

    def __init__(self, d):
        self.d = d
        self.ts = d.ts.to_numpy()

    def frame(self, t_prev, t):
        a, b = np.searchsorted(self.ts, t_prev, side="right"), np.searchsorted(self.ts, t, side="right")
        x = self.d.iloc[a:b].drop_duplicates("vehicle_id", keep="last")
        return encode_vehicle_positions(x.to_dict("records"), int(t))


def tod(ts):
    return (ts + config.IST_OFFSET_S) % 86400


def check(poll_s: float, minutes: int):
    from app import gtfs as gtfs_mod
    from app.live import LiveFeed
    from app.traveltime import TravelTimeModel
    d = load_window()
    rep = Replay(d)
    g = gtfs_mod.load()
    tt = TravelTimeModel(g)
    conn = GtfsRtConnector(g, tt, poll_s=poll_s)
    offline = LiveFeed.replay_day(tt, DAY)
    t_start = int(d.ts.min())
    t_end = t_start + minutes * 60
    t, prev, ms, n_bytes, checks = t_start, t_start - 1, [], [], []
    marks = {t_start + m * 60 for m in (30, 45, 60, 75, 90) if m <= minutes}
    while t <= t_end:
        msg = rep.frame(prev, t)
        n_bytes.append(len(msg))
        _, recs = decode_vehicle_positions(msg)
        t0 = time.time()
        conn.ingest(recs, now_ts=t)
        conn.process()
        ms.append(1000 * (time.time() - t0))
        if any(prev < m <= t for m in marks):
            now = tod(t)
            a, b = conn.feed.snapshot(now), offline.snapshot(now)
            common = sorted(set(a) & set(b))
            ma = np.array([a[p][0] for p in common])
            mb = np.array([b[p][0] for p in common])
            checks.append({"IST": time.strftime("%H:%M", time.gmtime(t + config.IST_OFFSET_S)),
                           "stretches live (connector)": len(a), "stretches offline": len(b), "common": len(common),
                           "median |diff| %": round(100 * float(np.median(np.abs(ma - mb) / mb)), 1) if common else None,
                           "within 10 %": round(100 * float(np.mean(np.abs(ma - mb) / mb <= .10)), 1) if common else None,
                           "corr": round(float(np.corrcoef(ma, mb)[0, 1]), 3) if len(common) > 2 else None})
        prev, t = t, t + poll_s
    res = pd.DataFrame(checks)
    print(res.to_string(index=False))
    print(f"\npolls {len(ms)}, feed size median {np.median(n_bytes) / 1024:.0f} KB, processing per poll "
          f"median {np.median(ms):.0f} ms / p95 {np.percentile(ms, 95):.0f} ms, vehicles in buffer "
          f"{conn.stats['vehicles']}, runs emitted {conn.stats['runs_emitted']:,}")
    return res, conn, ms, n_bytes


def check_bindings():
    """Cross-check our decoder/encoder with Google's official gtfs-realtime-bindings, if installed."""
    try:
        from google.transit import gtfs_realtime_pb2
    except ImportError:
        return "official bindings not installed — skipped"
    rows = [{"vehicle_id": "DL1PC1234", "route_id": "1416", "trip_id": "T9", "lat": 28.6123, "lon": 77.2311,
             "ts": 1793100600}]
    m = gtfs_realtime_pb2.FeedMessage()
    m.ParseFromString(encode_vehicle_positions(rows, 1793100600))
    e = m.entity[0].vehicle
    ok1 = (e.vehicle.id, e.trip.route_id, e.trip.trip_id, e.timestamp) == ("DL1PC1234", "1416", "T9", 1793100600)
    m2 = gtfs_realtime_pb2.FeedMessage()
    m2.header.gtfs_realtime_version = "2.0"
    m2.header.timestamp = 1793100600
    ent = m2.entity.add()
    ent.id = "x"
    ent.vehicle.vehicle.id = "DL1PC5678"
    ent.vehicle.trip.route_id = "534"
    ent.vehicle.position.latitude = 28.55
    ent.vehicle.position.longitude = 77.25
    ent.vehicle.timestamp = 1793100611
    _, recs = decode_vehicle_positions(m2.SerializeToString())
    r = recs[0]
    ok2 = (r["vehicle_id"], r["route_id"], r["ts"]) == ("DL1PC5678", "534", 1793100611) and abs(r["lat"] - 28.55) < 1e-4
    return f"official bindings: our encoder → theirs {'OK' if ok1 else 'MISMATCH'}; theirs → our decoder {'OK' if ok2 else 'MISMATCH'}"


def serve(port: int):
    d = load_window()
    rep = Replay(d)
    t_start = int(d.ts.min())
    wall0 = time.time()
    state = {"prev": t_start - 60}

    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            t = t_start + (time.time() - wall0)
            body = rep.frame(t - 60, t)        # every bus heard from in the last minute, latest fix
            self.send_response(200)
            self.send_header("Content-Type", "application/x-protobuf")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    print(f"GTFS-RT replay of {DAY} Oct from {time.strftime('%H:%M', time.gmtime(t_start + config.IST_OFFSET_S))} IST "
          f"at http://localhost:{port}/vehicle_positions.pb")
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serve", type=int)
    ap.add_argument("--poll", type=float, default=15.0)
    ap.add_argument("--minutes", type=int, default=90)
    a = ap.parse_args()
    if a.serve:
        return serve(a.serve)
    print(check_bindings())
    res, conn, ms, n_bytes = check(a.poll, a.minutes)
    lines = ["# Live GTFS-RT connector — check (auto-generated by scripts/16_gtfsrt_check.py)\n",
             f"Raw GPS of **{DAY} Oct 2026, 17:00–{17 + a.minutes // 60}:{a.minutes % 60:02d} IST** replayed as a GTFS-RT "
             f"VehiclePositions feed: every **{a.poll:.0f} s**, the latest position of every bus (what a live feed "
             "sends). The connector (`app/gtfsrt.py`) decodes it, measures stop-to-stop runs with the same code as the "
             "historical pipeline (`app/gpsproc.py`), and the live window (last 15 min) is compared with the offline "
             "pipeline's runs for the same minutes, which saw every fix after the fact.\n",
             "| time (IST) | stretches with live data (connector) | stretches (offline) | in both | median difference "
             "of the 15-min median run | within 10 % | correlation |", "|---|---|---|---|---|---|---|"]
    for r in res.itertuples(index=False):
        lines.append(f"| {r[0]} | {r[1]:,} | {r[2]:,} | {r[3]:,} | {r[4]} % | {r[5]} % | {r[6]} |")
    lines += ["", f"Feed: median {np.median(n_bytes) / 1024:.0f} KB per poll, {conn.stats['vehicles']:,} buses in the "
              f"20-min buffer. Processing per poll: median {np.median(ms):.0f} ms, p95 {np.percentile(ms, 95):.0f} ms "
              "(this PC). Runs emitted: " + f"{conn.stats['runs_emitted']:,}.", "", check_bindings(), "",
              "Switch it on: `BUSPULSE_GTFSRT_URL=<VehiclePositions URL>` (+ `BUSPULSE_GTFSRT_HEADERS=\"x-api-key: …\"`). "
              "`/health` then reports polls, buses and runs; `live=true` on `/journeys` blends the last 15 minutes "
              "with history."]
    (config.DOCS_DIR / "LIVE_CONNECTOR.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"→ {config.DOCS_DIR / 'LIVE_CONNECTOR.md'}")


if __name__ == "__main__":
    main()
