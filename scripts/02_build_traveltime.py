"""Aggregate per-day segment observations (train days only) into model tables.

Outputs in data/model/:
  seg_hour.parquet   (ca, cb, dt, h) → n, run/dwell median, winsorized mean, p25/p75/p85
  seg_all.parquet    (ca, cb)        → n, run median/mean, dwell median/mean (all hours/days)
  city_hour.parquet  (dt, h)         → factor (obs run / segment median), speed_kmh, dwell
  daily_profile.csv  per-day median speed by hour (holiday check)
  model_meta.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from app import gtfs as gtfs_mod  # noqa: E402


def seg_files(days):
    fs = [config.SEG_DIR / f"seg_{config.day_name(d)}.parquet" for d in days]
    return [f for f in fs if f.exists()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, nargs="*", default=config.TRAIN_DAYS)
    a = ap.parse_args()
    config.ensure_dirs()
    g = gtfs_mod.load()
    files = seg_files(a.days)
    if not files:
        sys.exit(f"no segment files for days {a.days} in {config.SEG_DIR}")
    used = sorted(int(f.stem[-2:]) for f in files)
    print(f"training on {len(files)} days: {used}")
    con = duckdb.connect()
    con.execute(f"SET temp_directory='{(config.TMP_DIR / 'duck').as_posix()}'")
    flist = "[" + ",".join(f"'{f.as_posix()}'" for f in files) + "]"
    pairs = pd.DataFrame({"ca": g["pairs"][:, 0], "cb": g["pairs"][:, 1], "dist": g["pair_dist"]})
    con.register("pairs", pairs)
    con.execute(f"CREATE TABLE s AS SELECT ca, cb, dt, h, day, run, dwell, v, r, dep_a FROM read_parquet({flist})")
    n_obs = con.execute("SELECT count(*) FROM s").fetchone()[0]
    print(f"{n_obs:,} segment observations")

    # robust per-segment trimming: drop obs > 4x the segment's all-days median (bus out of service / detour)
    con.execute("""CREATE TABLE segmed AS SELECT ca, cb, median(run) AS m FROM s GROUP BY ca, cb""")
    out_n = con.execute("SELECT count(*) FROM s JOIN segmed USING (ca, cb) WHERE run > 4*m + 60").fetchone()[0]
    con.execute("DELETE FROM s WHERE EXISTS (SELECT 1 FROM segmed m WHERE m.ca=s.ca AND m.cb=s.cb AND s.run > 4*m.m + 60)")
    print(f"dropped {out_n:,} outliers (> 4× segment median)")

    lo, hi = config.WINSOR_LO, config.WINSOR_HI
    q = """
    WITH b AS (SELECT {keys}, quantile_cont(run, %s) AS r05, quantile_cont(run, %s) AS r95,
                      quantile_cont(dwell, %s) AS d05, quantile_cont(dwell, %s) AS d95
               FROM s GROUP BY {keys})""" % (lo, hi, lo, hi) + """
    SELECT {keys}, count(*) AS n, count(DISTINCT s.day) AS ndays,
           median(run) AS run_med, avg(least(greatest(run, r05), r95)) AS run_mean,
           quantile_cont(run, 0.25) AS run_p25, quantile_cont(run, 0.75) AS run_p75,
           quantile_cont(run, 0.85) AS run_p85,
           median(dwell) AS dwell_med, avg(least(greatest(dwell, d05), d95)) AS dwell_mean,
           count(dwell) AS n_dwell
    FROM s JOIN b USING ({keys}) GROUP BY {keys}"""
    seg_hour = con.execute(q.format(keys="ca, cb, dt, h")).fetch_df()
    seg_all = con.execute(q.format(keys="ca, cb")).fetch_df()
    # city hourly factor: observed run / segment all-hours median → typical slowdown by daytype-hour
    city = con.execute("""
        SELECT dt, h, count(*) AS n, median(run / m) AS factor,
               median(p.dist / run * 3.6) AS speed_kmh, median(dwell) AS dwell_med
        FROM s JOIN segmed USING (ca, cb) JOIN pairs p USING (ca, cb)
        GROUP BY dt, h ORDER BY dt, h""").fetch_df()
    prof = con.execute("""
        SELECT day, h, median(p.dist / run * 3.6) AS speed_kmh, count(*) AS n
        FROM s JOIN pairs p USING (ca, cb) GROUP BY day, h ORDER BY day, h""").fetch_df()

    # real waits from GPS: departures of the same route from the same stop → headways.
    # Expected wait for a random arrival = E[H²] / (2·E[H])  (equals H/2 for perfectly regular
    # buses, larger when buses bunch). One bus counted once per stop visit.
    # Departures come from every GPS stop *visit* (visits_*.parquet), restricted to stops on that route's
    # GTFS patterns; one per bus pass (same bus again within 10 min = same pass). Older runs without visit
    # files fall back to departures that start a kept stretch — those miss buses whose next visit was
    # dropped, which makes headways look longer than they are.
    vfiles = [config.SEG_DIR / f"visits_{config.day_name(d)}.parquet" for d in used]
    if all(f.exists() for f in vfiles):
        rs = []
        for p in g["patterns"]:
            try:
                rid = int(float(p["route_id"]))
            except (TypeError, ValueError):
                continue
            rs += [(rid, int(c)) for c in p["cl"][:-1]]
        con.register("route_stops", pd.DataFrame(rs, columns=["r", "ca"]).drop_duplicates())
        vlist = "[" + ",".join(f"'{f.as_posix()}'" for f in vfiles) + "]"
        con.execute(f"""CREATE TABLE deps AS
            SELECT v.r, v.c AS ca, v.dt, CAST(regexp_extract(filename, '(\\d\\d)\\.parquet$', 1) AS INTEGER) AS day,
                   CAST(floor((v.dep + {config.IST_OFFSET_S}) / 3600) AS BIGINT) % 24 AS h, v.v, v.dep AS dep_a
            FROM read_parquet({vlist}, filename=true) v
            JOIN route_stops rs ON v.r = rs.r AND v.c = rs.ca""")
        con.execute("""CREATE OR REPLACE TABLE deps AS SELECT * FROM (
            SELECT *, dep_a - lag(dep_a) OVER (PARTITION BY r, ca, day, v ORDER BY dep_a) AS same_bus_gap FROM deps)
            WHERE same_bus_gap IS NULL OR same_bus_gap >= 600""")
        src, wait_src = "deps", "GPS stop visits"
        # stop detection: buses of the route seen at this stop ÷ buses seen at the route's best-detected
        # stops (median of its top 3), summed over days. Low = the GPS usually misses this stop (stop
        # placed off the road / next to another cluster) → its headways would look too long.
        con.execute("""CREATE TABLE det AS
            WITH nb AS (SELECT r, ca, day, count(DISTINCT v) AS nb FROM deps GROUP BY r, ca, day),
                 rk AS (SELECT *, row_number() OVER (PARTITION BY r, day ORDER BY nb DESC) AS k FROM nb),
                 top AS (SELECT r, day, median(nb) AS top FROM rk WHERE k <= 3 GROUP BY r, day),
                 tot AS (SELECT r, sum(top) AS top_sum FROM top GROUP BY r)
            SELECT r, ca, least(sum(nb) / any_value(top_sum), 1.0) AS detection
            FROM nb JOIN tot USING (r) GROUP BY r, ca""")
    else:
        src, wait_src = "(SELECT DISTINCT r, ca, dt, day, h, v, dep_a FROM s WHERE r >= 0)", "kept stretches"
        con.execute("CREATE TABLE det AS SELECT DISTINCT r, ca, 1.0 AS detection FROM s")
    gw = con.execute(f"""
        WITH d AS (SELECT r, ca, dt, day, h, dep_a,
                          dep_a - lag(dep_a) OVER (PARTITION BY r, ca, day ORDER BY dep_a) AS hw
                   FROM {src}),
             w AS (SELECT r, ca, dt, h, count(*) AS n, avg(hw) AS mean_hw, avg(hw*hw) / (2*avg(hw)) AS wait_s,
                          avg(CASE WHEN hw < 120 THEN 1.0 ELSE 0.0 END) AS bunched_share
                   FROM d WHERE hw BETWEEN 15 AND 5400 GROUP BY r, ca, dt, h HAVING count(*) >= 4)
        SELECT w.*, det.detection FROM w JOIN det USING (r, ca)""").fetch_df()
    n_all = len(gw)
    gw = gw[gw.detection >= config.MIN_STOP_DETECTION].reset_index(drop=True)
    print(f"GPS waits from {wait_src}; kept {len(gw):,} of {n_all:,} cells at stops where the GPS sees "
          f"≥ {config.MIN_STOP_DETECTION:.0%} of the route's buses")
    gw.to_parquet(config.MODEL_DIR / "gps_wait.parquet", index=False)
    print(f"GPS headway cells (route × stop × daytype × hour): {len(gw):,}; "
          f"median wait {gw.wait_s.median()/60 if len(gw) else float('nan'):.1f} min, "
          f"bunched (<2 min apart) {100*gw.bunched_share.mean() if len(gw) else 0:.1f}% of arrivals")
    seg_hour.to_parquet(config.MODEL_DIR / "seg_hour.parquet", index=False)
    seg_all.to_parquet(config.MODEL_DIR / "seg_all.parquet", index=False)
    city.to_parquet(config.MODEL_DIR / "city_hour.parquet", index=False)
    wide = prof.pivot(index="day", columns="h", values="speed_kmh").round(1)
    wide.to_csv(config.MODEL_DIR / "daily_profile.csv")

    cover = len(seg_all) / max(1, len(g["pairs"]))
    meta = dict(train_days=used, n_obs=int(n_obs), outliers_dropped=int(out_n),
                segments_with_data=int(len(seg_all)), gtfs_pairs=int(len(g["pairs"])),
                pair_coverage_pct=round(100 * cover, 1),
                seg_hour_cells=int(len(seg_hour)),
                seg_hour_cells_n_ge_5=int((seg_hour.n >= config.MIN_N_HOUR).sum()))
    with open(config.MODEL_DIR / "model_meta.json", "w") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps(meta, indent=1))
    print("\ncity median bus speed (km/h) by hour, weekday / saturday / sunday:")
    piv = city.pivot(index="h", columns="dt", values="speed_kmh").round(1)
    print(piv.T.to_string())
    # holiday check: per-day 8-10h and 17-20h speed vs daytype
    peak = prof[prof.h.isin([8, 9, 17, 18, 19])].groupby("day").speed_kmh.median().round(1)
    print("\nper-day peak-hour median speed (a holiday looks like a Sunday):")
    for d, sp in peak.items():
        date = f"{config.YEAR}-{config.MONTH:02d}-{int(d):02d}"
        print(f"  {date} {pd.Timestamp(date).day_name()[:3]} dt={config.daytype_of(date)}  {sp}")

    # reliability calibration on (a sample of) training days — never the held-out week
    from app.calibration import calibrate
    from app.traveltime import TravelTimeModel
    tt = TravelTimeModel(g)
    cal = calibrate(tt, used[-7:])
    print(f"\ncalibration ratios actual/predicted (q10..q90) on {cal['n']} training chains:")
    for b, q in cal["by_bucket"].items():
        print(f"  {b:8s} {q}")
    # planner calibration: planned legs vs real buses on the last training week (never the held-out one)
    if all((config.SEG_DIR / f"visits_{config.day_name(d)}.parquet").exists() for d in used[-7:]):
        from app.planner import Planner
        from app.ridecheck import calibrate_planner
        pc = calibrate_planner(Planner(g, tt), g, used[-7:], n=700)
        cal.update(pc)
        with open(config.MODEL_DIR / "calibration.json", "w") as f:
            json.dump(cal, f, indent=1)
        rc = pc["ride"]
        print(f"\nwhole-ride ratios actual/planned (q10..q90) on {rc['n_legs']} training bus legs:")
        for b, q in rc["by_bucket"].items():
            print(f"  {b:8s} {q}")
        print(f"wait factors (real ÷ predicted): {pc['wait']}")


if __name__ == "__main__":
    main()
