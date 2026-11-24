"""Cut a time window of raw GPS fixes from one day's SQLite DB, for the GTFS-RT simulator.

    python scripts/15_gtfsrt_window.py --day 27 --start 17:00 --end 19:00

Writes data/tmp/rt_window_<day>.parquet (vehicle_id, lat, lon, ts, route_id), raw and unfiltered, exactly
as a live feed would send them, and removes the extracted DB again.
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

spec = importlib.util.spec_from_file_location("gps_extract", Path(__file__).with_name("01_gps_extract.py"))
gx = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", type=int, default=27)
    ap.add_argument("--start", default="17:00")
    ap.add_argument("--end", default="19:00")
    a = ap.parse_args()
    ist = dt.timezone(dt.timedelta(seconds=config.IST_OFFSET_S))
    def ts(hm):
        h, m = map(int, hm.split(":"))
        return int(dt.datetime(config.YEAR, config.MONTH, a.day, h, m, tzinfo=ist).timestamp())
    t0, t1 = ts(a.start), ts(a.end)
    db, extracted = gx.get_day_db(a.day)
    import duckdb
    con = duckdb.connect()
    con.execute("INSTALL sqlite; LOAD sqlite;")
    df = con.execute(f"""SELECT CAST(vehicle_id AS VARCHAR) AS vehicle_id, TRY_CAST(lat AS DOUBLE) AS lat,
            TRY_CAST(lng AS DOUBLE) AS lon, TRY_CAST(TRY_CAST("timestamp" AS DOUBLE) AS BIGINT) AS ts,
            CAST(route_id AS VARCHAR) AS route_id
        FROM sqlite_scan('{db.as_posix()}', 'vehicle_feed')
        WHERE TRY_CAST(TRY_CAST("timestamp" AS DOUBLE) AS BIGINT) BETWEEN {t0} AND {t1}
        ORDER BY ts""").fetch_df()
    con.close()
    out = config.TMP_DIR / f"rt_window_{a.day}.parquet"
    df.to_parquet(out, index=False)
    print(f"{len(df):,} fixes, {df.vehicle_id.nunique():,} vehicles, {a.start}–{a.end} IST on {a.day} Oct → {out}")
    if extracted:
        db.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
