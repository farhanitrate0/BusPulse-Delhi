"""Tiny fake Delhi: GTFS + GPS day DBs (real schema, with duplicates/noise/gaps/jumps) + tickets.

    python tests/make_synthetic.py            # writes ./data_synth
    set BUSPULSE_DATA=data_synth  (PowerShell: $env:BUSPULSE_DATA="data_synth")
then run the pipeline scripts as usual. Ground truth speeds are known, so validation must
show the GPS model beating the (deliberately wrong) timetable.
"""
from __future__ import annotations

import csv
import datetime as dt
import sqlite3
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data_synth"
rng = np.random.default_rng(7)
LAT0, LON0 = 28.62, 77.20
MLAT, MLON = 110_574.0, 111_320.0 * np.cos(np.radians(28.6))
DAYS = [5, 6, 7, 8, 9, 10, 11, 25, 27]


def xy2ll(x, y):
    return LAT0 + y / MLAT, LON0 + x / MLON


# ---- network: 4 horizontal + 4 vertical lines, 7 km long, stops every 450 m ----
LINES = []
for i in range(4):
    LINES.append(("H", -3500, 3500, -2400 + i * 1600))   # (orient, from, to, fixed coordinate)
    LINES.append(("V", -3500, 3500, -2400 + i * 1600))
ROUTE_NUMS = [101, 102, 103, 104, 201, 202, 203, 204]
NAMES = ["Kashmere Gate", "Civil Lines", "Tis Hazari", "Ajmeri Gate", "Paharganj", "Karol Bagh",
         "Patel Nagar", "Rajendra Place", "Connaught Place", "Mandi House", "ITO", "Pragati Maidan",
         "Lajpat Nagar", "Moolchand", "Nehru Place", "Kalkaji", "Okhla", "Jangpura", "Bhogal",
         "Ashram", "Sarai Kale Khan", "Nizamuddin", "Khan Market", "India Gate", "Janpath",
         "Gole Market", "RK Ashram", "Jhandewalan", "Anand Parbat", "Shadipur", "Kirti Nagar",
         "Moti Nagar", "Ramesh Nagar", "Rajouri Garden", "Tagore Garden", "Subhash Nagar",
         "Tilak Nagar", "Janakpuri", "Uttam Nagar", "Dwarka Mor", "Anand Vihar", "Preet Vihar",
         "Laxmi Nagar", "Shakarpur", "Mayur Vihar", "Akshardham", "Yamuna Bank", "Indraprastha"]


def make_network():
    node_name = {}
    routes = []
    for li, (o, a, b, c) in enumerate(LINES):
        n = int((b - a) // 450) + 1
        pts = []
        for k in range(n):
            u = a + k * 450
            x, y = (u, c) if o == "H" else (c, u)
            key = (round(x / 50), round(y / 50))
            if key not in node_name:
                node_name[key] = NAMES[len(node_name) % len(NAMES)] + ("" if len(node_name) < len(NAMES) else f" {len(node_name)//len(NAMES)+1}")
            pts.append((x, y, node_name[key]))
        routes.append((ROUTE_NUMS[li], o, pts))
    return routes


def hour_speed_kmh(h, dtype):
    base = {0: 34, 1: 36, 2: 36, 3: 36, 4: 34, 5: 30, 6: 26, 7: 20, 8: 14, 9: 15, 10: 19, 11: 21,
            12: 21, 13: 21, 14: 21, 15: 20, 16: 18, 17: 14, 18: 12, 19: 14, 20: 18, 21: 22, 22: 26, 23: 30}[h]
    return base * (1.25 if dtype == 2 else 1.12 if dtype == 1 else 1.0)


def daytype(day):
    d = dt.date(2026, 10, day)
    return 2 if d.weekday() == 6 else 1 if d.weekday() == 5 else 0


def main():
    routes = make_network()
    seg_factor = {}
    gdir, dbdir, tdir = OUT / "gtfs", OUT / "gps_db", OUT / "tickets"
    for d in (gdir, dbdir, tdir):
        d.mkdir(parents=True, exist_ok=True)

    # ---------- GTFS ----------
    stops, rts, trips, sts = [], [], [], []
    route_dirs = []
    for num, o, pts in routes:
        for dname, seq in (("UP", pts), ("DOWN", pts[::-1])):
            rid = f"{num}{dname}"
            side = 12 if dname == "UP" else -12          # stops on opposite sides of the road
            sids = []
            for k, (x, y, nm) in enumerate(seq):
                sx, sy = (x, y + side) if o == "H" else (x + side, y)
                lat, lon = xy2ll(sx, sy)
                sid = f"S{num}{dname[0]}{k:02d}"
                stops.append((sid, nm, f"{lat:.6f}", f"{lon:.6f}"))
                sids.append((sid, sx, sy, nm))
            rts.append((rid, f"{num}{dname}", f"{seq[0][2]} to {seq[-1][2]}"))
            route_dirs.append((num, rid, sids))
    for num, rid, sids in route_dirs:
        for a in range(len(sids) - 1):
            seg_factor[(rid, a)] = float(rng.choice([0.7, 1.0, 1.0, 1.1, 1.3]))
        for svc, hw in (("WK", 600), ("SA", 900), ("SU", 900)):
            for k, t0 in enumerate(range(6 * 3600, 22 * 3600, hw)):
                tid = f"{rid}_{svc}_{k}"
                trips.append((rid, svc, tid, sids[-1][3]))
                for j, (sid, *_r) in enumerate(sids):
                    t = t0 + j * 75                       # timetable: fixed 75 s per stop (wrong at peaks)
                    hh = f"{t//3600:02d}:{t%3600//60:02d}:{t%60:02d}"
                    sts.append((tid, hh, hh, sid, j + 1))

    def wcsv(name, header, rows):
        with open(gdir / name, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)

    wcsv("stops.txt", ["stop_id", "stop_name", "stop_lat", "stop_lon"], stops)
    wcsv("routes.txt", ["route_id", "route_short_name", "route_long_name", "route_type"], [r + ("3",) for r in rts])
    wcsv("trips.txt", ["route_id", "service_id", "trip_id", "trip_headsign"], trips)
    wcsv("stop_times.txt", ["trip_id", "arrival_time", "departure_time", "stop_id", "stop_sequence"], sts)
    wcsv("calendar.txt", ["service_id", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
                          "sunday", "start_date", "end_date"],
         [("WK", 1, 1, 1, 1, 1, 0, 0, "20261001", "20261031"), ("SA", 0, 0, 0, 0, 0, 1, 0, "20261001", "20261031"),
          ("SU", 0, 0, 0, 0, 0, 0, 1, "20261001", "20261031")])
    print(f"GTFS: {len(stops)} stops, {len(rts)} routes, {len(trips)} trips")

    # ---------- GPS + tickets ----------
    rid_num = {rid: i + 1000 for i, (_, rid, _) in enumerate(route_dirs)}  # GPS route_id (float in DB)
    with open(gdir / "routes.txt", "w", newline="", encoding="utf-8") as f:   # GTFS route_id numeric like Delhi
        w = csv.writer(f)
        w.writerow(["route_id", "route_short_name", "route_long_name", "route_type"])
        for rid, sn, ln in rts:
            w.writerow([rid_num[rid], sn, ln, 3])
    for fn in ("trips.txt",):
        rows = list(csv.reader(open(gdir / fn, encoding="utf-8")))
        with open(gdir / fn, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(rows[0])
            for r in rows[1:]:
                w.writerow([rid_num[r[0]]] + r[1:])

    tickets = []
    for day in DAYS:
        dtp = daytype(day)
        rows = []
        midnight_utc = int(dt.datetime(2026, 10, day, tzinfo=dt.timezone.utc).timestamp()) - 19800
        hw = 600 if dtp == 0 else 900
        for num, rid, sids in route_dirs:
            for k, t0 in enumerate(range(6 * 3600, 22 * 3600, hw)):
                veh = f"DL1PC{num}{rid[-2:]}{k % 9}"
                t = t0 + rng.normal(0, 120)
                track = []
                for j in range(len(sids)):
                    _, x, y, nm = sids[j]
                    dwell = max(5.0, rng.normal(22, 8))
                    track.append((t, x, y))
                    track.append((t + dwell, x, y))
                    t += dwell
                    if j < len(sids) - 1:
                        v = hour_speed_kmh(int(t // 3600) % 24, dtp) / 3.6 / seg_factor[(rid, j)]
                        dist = np.hypot(sids[j + 1][1] - x, sids[j + 1][2] - y)
                        t += dist / v * rng.uniform(0.85, 1.15)
                tt = np.array([p[0] for p in track])
                xs = np.array([p[1] for p in track])
                ys = np.array([p[2] for p in track])
                samp = tt[0] + np.cumsum(rng.uniform(8, 15, int((tt[-1] - tt[0]) / 8)))
                samp = samp[samp < tt[-1]]
                if rng.random() < 0.1:                     # a 4-minute data gap on 10 % of trips
                    g0 = rng.uniform(samp[0], samp[-1])
                    samp = samp[(samp < g0) | (samp > g0 + 240)]
                px = np.interp(samp, tt, xs) + rng.normal(0, 8, len(samp))
                py = np.interp(samp, tt, ys) + rng.normal(0, 8, len(samp))
                jump = rng.random(len(samp)) < 0.01
                px[jump] += 900
                lat, lon = xy2ll(px, py)
                for s, a, b in zip(samp, lat, lon):
                    ts = int(midnight_utc + s)
                    ist = dt.datetime.fromtimestamp(ts, dt.timezone.utc) + dt.timedelta(seconds=19800)
                    r = (veh, f"{a:.6f}", f"{b:.6f}", 20.0, float(rid_num[rid]), f"T{rid}{k}", str(ts),
                         ist.date().isoformat(), ist.strftime("%H:%M:%S"))
                    rows.append(r)
                    if rng.random() < 0.03:
                        rows.append(r)                     # duplicate
                # tickets for this trip (stop arrival times = track[2j])
                h0 = int(t0 // 3600)
                demand = 6 if h0 in (8, 9, 17, 18, 19) else 3
                for j in range(len(sids) - 1):
                    for _ in range(rng.poisson(demand * (1.4 if 3 <= j <= 9 else 0.7))):
                        e = min(len(sids) - 1, j + 1 + int(rng.integers(1, 8)))
                        tb = track[2 * j][0]
                        ts = dt.datetime(2026, 10, day) + dt.timedelta(seconds=float(tb))
                        bname, ename = sids[j][3], sids[e][3]
                        if rng.random() < 0.2:
                            bname = bname.lower().replace("nagar", "ngr")
                        tickets.append((1, f"{rng.choice([9, 13.5, 18])}", veh if rng.random() < 0.5 else "NULL",
                                        1, ename, rid, bname, ts.strftime("%Y-%m-%d %H:%M:%S.%f") + "+05:30",
                                        "True"))
        rng.shuffle(rows)                                  # raw order is not sorted
        db = dbdir / f"bus_movements_2026_10_{day:02d}.db"
        db.unlink(missing_ok=True)
        con = sqlite3.connect(db)
        con.execute("CREATE TABLE vehicle_feed (vehicle_id TEXT, lat TEXT, lng TEXT, speed REAL, route_id REAL,"
                    " trip_id TEXT, timestamp TEXT, date DATE, time TEXT)")
        con.executemany("INSERT INTO vehicle_feed VALUES (?,?,?,?,?,?,?,?,?)", rows)
        con.commit()
        con.close()
        print(f"day {day}: {len(rows)} GPS rows")
    with open(tdir / "October-Delhi.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
        w.writerow(["passenger_count", "amount", "vehicle_number", "ticket_type_id", "end_stop_name", "route",
                    "start_stop_name", "created_at", "is_ac"])
        w.writerows(tickets)
    print(f"tickets: {len(tickets)}")


if __name__ == "__main__":
    sys.exit(main())
