"""GPS day DBs → stop-to-stop segment observations (one small parquet per day).

For each day (priority order from config): extract that one DB from the 7z (or use
data/gps_db/*.db if present) → clean → densify → snap to stop clusters → visits →
segment observations → write data/segments/seg_YYYY_MM_DD.parquet → delete the DB.
Resumable: days with an existing parquet are skipped.

    python scripts/01_gps_extract.py                 # all days, priority order, 2 workers
    python scripts/01_gps_extract.py --days 5 6 7 --workers 1
    python scripts/01_gps_extract.py --list          # show archive contents / solidity
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from app import gtfs as gtfs_mod  # noqa: E402
from app.geo import to_xy  # noqa: E402
from app.gpsproc import process_vehicles  # noqa: E402,F401


# ----------------------------------------------------------------------------- 7z
def seven_zip() -> str:
    for c in config.SEVEN_ZIP_CANDIDATES:
        if Path(c).exists() or shutil.which(c):
            return c
    raise FileNotFoundError("7-Zip not found. Install it (winget install 7zip.7zip).")


def archive_listing() -> tuple[list[str], str]:
    out = subprocess.run([seven_zip(), "l", "-slt", str(config.GPS_ARCHIVE)], capture_output=True, text=True).stdout
    names = [ln.split("=", 1)[1].strip() for ln in out.splitlines() if ln.startswith("Path = ") and ln.endswith(".db")]
    solid = next((ln for ln in out.splitlines() if ln.startswith("Solid")), "Solid = ?")
    return names, solid


def get_day_db(day: int) -> tuple[Path, bool]:
    """Return (path to the day's DB, whether we extracted it and must delete it)."""
    name = f"bus_movements_{config.day_name(day)}.db"
    pre = config.GPS_DB_DIR / name
    if pre.exists():
        return pre, False
    if not config.GPS_ARCHIVE.exists():
        raise FileNotFoundError(f"neither {pre} nor archive {config.GPS_ARCHIVE}")
    out_dir = config.TMP_DIR / f"day_{day:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / name
    if not target.exists():
        # 'e' ignores folder paths inside the archive; wildcard handles a sub-folder
        r = subprocess.run([seven_zip(), "e", str(config.GPS_ARCHIVE), f"-o{out_dir}", name, f"*/{name}", "-r", "-y"],
                           capture_output=True, text=True)
        if not target.exists():
            raise RuntimeError(f"7z could not extract {name}: {r.stdout[-500:]} {r.stderr[-500:]}")
    return target, True


# ----------------------------------------------------------------------------- load
def load_points(db: Path, stats: dict) -> pd.DataFrame:
    """Read the day with DuckDB's SQLite scanner (fast, out-of-core); fallback to sqlite3 chunks."""
    lat0, lat1, lon0, lon1 = config.BBOX
    try:
        import duckdb
        con = duckdb.connect()
        threads = max(2, (os.cpu_count() or 4) // int(os.environ.get("BUSPULSE_WORKERS", "1")))
        con.execute(f"SET threads={threads}; SET memory_limit='5GB'; SET temp_directory='{(config.TMP_DIR / 'duck').as_posix()}'")
        con.execute("INSTALL sqlite; LOAD sqlite;")
        con.execute(f"""CREATE TEMP TABLE raw AS SELECT vehicle_id AS v,
                TRY_CAST(lat AS DOUBLE) AS lat, TRY_CAST(lng AS DOUBLE) AS lon,
                TRY_CAST(TRY_CAST("timestamp" AS DOUBLE) AS BIGINT) AS ts,
                TRY_CAST(route_id AS BIGINT) AS r
            FROM sqlite_scan('{db.as_posix()}', 'vehicle_feed')""")
        stats["rows_total"] = con.execute("SELECT count(*) FROM raw").fetchone()[0]
        stats["rows_invalid"] = con.execute(
            "SELECT count(*) FROM raw WHERE v IS NULL OR lat IS NULL OR lon IS NULL OR ts IS NULL").fetchone()[0]
        stats["rows_out_bbox"] = con.execute(
            f"SELECT count(*) FROM raw WHERE lat IS NOT NULL AND lon IS NOT NULL AND NOT "
            f"(lat BETWEEN {lat0} AND {lat1} AND lon BETWEEN {lon0} AND {lon1})").fetchone()[0]
        # vehicle ids → integer codes inside DuckDB (keeps Python memory small on 40M-row days)
        con.execute("CREATE TEMP TABLE vk AS SELECT v, (row_number() OVER (ORDER BY v) - 1)::INTEGER AS vc "
                    "FROM (SELECT DISTINCT v FROM raw WHERE v IS NOT NULL)")
        names = [r[0] for r in con.execute("SELECT v FROM vk ORDER BY vc").fetchall()]
        df = con.execute(f"""SELECT vk.vc AS vc, ts, any_value(lat) AS lat, any_value(lon) AS lon, max(r) AS r
            FROM raw JOIN vk USING (v) WHERE ts IS NOT NULL AND lat BETWEEN {lat0} AND {lat1}
              AND lon BETWEEN {lon0} AND {lon1}
            GROUP BY vk.vc, ts ORDER BY vk.vc, ts""").fetch_df()
        con.close()
        df.attrs["names"] = names
        stats["reader"] = "duckdb"
    except Exception as e:  # noqa: BLE001
        stats["reader"] = f"sqlite3 ({type(e).__name__})"
        parts, total, invalid = [], 0, 0
        con = sqlite3.connect(db)
        q = "SELECT vehicle_id, lat, lng, timestamp, route_id FROM vehicle_feed"
        for ch in pd.read_sql_query(q, con, chunksize=2_000_000):
            total += len(ch)
            ch = pd.DataFrame({"v": ch.vehicle_id, "lat": pd.to_numeric(ch.lat, errors="coerce"),
                               "lon": pd.to_numeric(ch.lng, errors="coerce"),
                               "ts": pd.to_numeric(ch.timestamp, errors="coerce"),
                               "r": pd.to_numeric(ch.route_id, errors="coerce")})
            ok = ch.v.notna() & ch.lat.notna() & ch.lon.notna() & ch.ts.notna()
            invalid += int((~ok).sum())
            parts.append(ch[ok])
        con.close()
        df = pd.concat(parts, ignore_index=True)
        df["ts"] = df.ts.astype(np.int64)
        inb = df.lat.between(lat0, lat1) & df.lon.between(lon0, lon1)
        stats.update(rows_total=total, rows_invalid=invalid, rows_out_bbox=int((~inb).sum()))
        df = df[inb].drop_duplicates(["v", "ts"])
        codes, names = pd.factorize(df.v, sort=True)
        df = df.assign(vc=codes.astype(np.int32)).drop(columns="v").sort_values(["vc", "ts"], kind="stable")
        df.attrs["names"] = list(names)
    stats["rows_dedup"] = int(len(df))
    stats["rows_duplicates"] = int(stats["rows_total"] - stats["rows_invalid"] - stats["rows_out_bbox"] - len(df))
    df["r"] = df.r.fillna(-1).astype(np.int64)
    return df.reset_index(drop=True)


# ----------------------------------------------------------------------------- core
# process_vehicles lives in app/gpsproc.py (shared with the live GTFS-RT connector)


def process_day(day: int, g: dict | None = None) -> dict:
    t0 = time.time()
    config.ensure_dirs()
    out = config.SEG_DIR / f"seg_{config.day_name(day)}.parquet"
    vout = config.SEG_DIR / f"visits_{config.day_name(day)}.parquet"
    if out.exists() and vout.exists():
        return {"day": day, "skipped": True}
    g = g or gtfs_mod.load()
    ncl = len(g["cl_lat"])
    cx, cy = to_xy(g["cl_lat"], g["cl_lon"])
    tree = cKDTree(np.c_[cx, cy])
    pkeys = g["pairs"][:, 0].astype(np.int64) * ncl + g["pairs"][:, 1]
    order = np.argsort(pkeys)
    pair_set = pkeys[order]
    pdist_sorted = g["pair_dist"][order]

    def pair_dist_lookup(keys):
        return pdist_sorted[np.searchsorted(pair_set, keys)]

    stats: dict = {"day": day}
    db, extracted = get_day_db(day)
    stats["extract_s"] = round(time.time() - t0, 1)
    df = load_points(db, stats)
    stats["load_s"] = round(time.time() - t0, 1)
    gtfs_rids = set(pd.to_numeric(g["routes"].route_id, errors="coerce").dropna().astype(np.int64))
    ur = df.r.unique()
    stats["gps_routes"] = int(len(ur))
    stats["gps_route_ids_in_gtfs_pct"] = round(100 * float(np.isin(ur, list(gtfs_rids)).mean()), 1) if len(ur) else 0
    vnames = df.attrs["names"]
    vcode = df.vc.to_numpy(np.int32)
    stats["vehicles"] = int(len(np.unique(vcode)))
    ts = df.ts.to_numpy(np.int64)
    x, y = to_xy(df.lat.to_numpy(), df.lon.to_numpy())
    r = df.r.to_numpy(np.int64)
    # vehicles per route per IST hour (operated frequency check)
    hour = ((ts + config.IST_OFFSET_S) // 3600) % 24
    vrh = pd.DataFrame({"r": r, "h": hour, "v": vcode}).drop_duplicates().groupby(["r", "h"]).size().rename("vehicles").reset_index()
    del df

    # process in vehicle chunks to bound memory
    segs, viss = [], []
    bounds = np.searchsorted(vcode, np.linspace(0, vcode.max() + 1, 9).astype(np.int64))
    for s, e in zip(bounds[:-1], bounds[1:]):
        if e - s < 2:
            continue
        res = process_vehicles(vcode[s:e], ts[s:e], x[s:e], y[s:e], r[s:e], tree, pair_set, pair_dist_lookup, ncl, stats)
        if res is not None:
            segs.append(res[0])
            viss.append(res[1])
    seg = pd.concat(segs, ignore_index=True) if segs else pd.DataFrame(
        columns=["v", "r", "ca", "cb", "dep_a", "arr_b", "dep_b", "run", "dwell"])
    seg["v"] = np.asarray(vnames)[seg.v.to_numpy(np.int64)] if len(seg) else seg.v
    date = f"{config.YEAR}-{config.MONTH:02d}-{day:02d}"
    if viss:
        vis = pd.concat(viss, ignore_index=True)
        vis["v"] = np.asarray(vnames)[vis.v.to_numpy(np.int64)]
        vis["dt"] = np.int8(config.daytype_of(date))
        vis.to_parquet(vout.with_suffix(".tmp"), index=False, compression="zstd")
        os.replace(vout.with_suffix(".tmp"), vout)
        stats["visits_saved"] = int(len(vis))
    seg["day"] = np.int8(day)
    seg["dt"] = np.int8(config.daytype_of(date))
    seg["h"] = (((seg.dep_a.astype(np.int64) + config.IST_OFFSET_S) // 3600) % 24).astype(np.int8)
    tmp = out.with_suffix(".tmp")
    seg.to_parquet(tmp, index=False)
    vrh.to_parquet(config.SEG_DIR / f"vrh_{config.day_name(day)}.parquet", index=False)
    dts = stats.pop("_dt_samples", [])
    stats["median_sampling_interval_s"] = float(np.median(dts)) if dts else None
    stats["seg_obs"] = int(len(seg))
    stats["segments_distinct"] = int(seg[["ca", "cb"]].drop_duplicates().shape[0]) if len(seg) else 0
    stats["total_s"] = round(time.time() - t0, 1)
    with open(config.SEG_DIR / f"quality_{config.day_name(day)}.json", "w") as f:
        json.dump(stats, f, indent=1, default=float)
    os.replace(tmp, out)
    if extracted:
        db.unlink(missing_ok=True)
        shutil.rmtree(db.parent, ignore_errors=True)
    return stats


def _worker(day: int) -> dict:
    try:
        return process_day(day)
    except Exception as e:  # noqa: BLE001
        import traceback
        return {"day": day, "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-1500:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        names, solid = archive_listing()
        print(solid)
        print("\n".join(names))
        return
    config.ensure_dirs()
    gtfs_mod.load()  # build cache once before forking workers
    try:  # install the DuckDB SQLite scanner once (concurrent installs collide)
        import duckdb
        duckdb.connect().execute("INSTALL sqlite")
    except Exception as e:  # noqa: BLE001
        print(f"duckdb sqlite extension unavailable ({e}); using slower sqlite3 reader")
    days = a.days or config.DAY_PRIORITY
    os.environ["BUSPULSE_WORKERS"] = str(a.workers)
    print(f"processing {len(days)} days with {a.workers} worker(s): {days}")
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(_worker, d): d for d in days}
        for fu in as_completed(futs):
            s = fu.result()
            if s.get("skipped"):
                print(f"day {s['day']:02d}: already done")
            elif "error" in s:
                print(f"day {s['day']:02d}: ERROR {s['error']}\n{s['trace']}")
            else:
                print(f"day {s['day']:02d}: rows {s['rows_total']:,} → dedup {s['rows_dedup']:,} "
                      f"(dup {s['rows_duplicates']:,}, bbox-out {s['rows_out_bbox']:,}, spikes {s['spikes_removed']:,}) "
                      f"| visits {s['visits']:,} | segs kept {s['seg_kept']:,}/{s['seg_candidates']:,} "
                      f"| route-id match {s['gps_route_ids_in_gtfs_pct']}% | {s['total_s']} s [{s['reader']}]", flush=True)


if __name__ == "__main__":
    main()
