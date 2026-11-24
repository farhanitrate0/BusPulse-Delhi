"""Central configuration. Override the data folder with env var BUSPULSE_DATA."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = Path(os.environ.get("BUSPULSE_DATA", ROOT / "data")).resolve()

RAW_DIR = DATA / "raw"
GTFS_DIR = DATA / "gtfs"
TICKETS_CSV = DATA / "tickets" / "October-Delhi.csv"
GPS_ARCHIVE = RAW_DIR / "delhi-dtc-bus-gtfs-rt-2026-10.7z"
GPS_DB_DIR = DATA / "gps_db"          # already-extracted (or synthetic) day DBs, used if present
TMP_DIR = DATA / "tmp"
SEG_DIR = DATA / "segments"           # per-day segment observations (parquet)
MODEL_DIR = DATA / "model"            # aggregated model tables + gtfs cache
GOOGLE_CSV = DATA / "google_benchmark.csv"
# generated reports go to docs/ for the real data; any other data folder (tests, synthetic city) writes
# its reports next to its own data so a test run never overwrites the real results
DOCS_DIR = Path(os.environ.get("BUSPULSE_DOCS",
                               ROOT / "docs" if DATA == (ROOT / "data").resolve() else DATA / "docs")).resolve()

SEVEN_ZIP_CANDIDATES = [
    r"C:\Program Files\7-Zip\7z.exe",
    r"C:\Program Files (x86)\7-Zip\7z.exe",
    "7z", "7za", "7zz",
]

# ---- time ---------------------------------------------------------------
IST_OFFSET_S = 19800  # +05:30
YEAR, MONTH = 2026, 10
# Public holidays treated as Sunday. 2 Oct = Gandhi Jayanti. 20 Oct = Dussehra (verify:
# scripts/02 prints a per-day speed profile; a holiday looks like a Sunday).
HOLIDAYS = {"2026-10-02", "2026-10-20"}
DAYTYPES = ["weekday", "saturday", "sunday"]   # index 0/1/2 ("sunday" = Sunday or holiday)

TRAIN_DAYS = list(range(1, 25))       # 1–24 Oct
TEST_DAYS = list(range(25, 32))       # 25–31 Oct held out (Sun–Sat)
# Processing priority: one full week, then a few held-out days so validation is possible early.
DAY_PRIORITY = [5, 6, 7, 8, 9, 10, 11, 27, 25, 31, 12, 13, 14, 15, 16, 17, 18,
                19, 20, 21, 22, 23, 24, 1, 2, 3, 4, 26, 28, 29, 30]

# ---- GPS cleaning / matching -------------------------------------------
BBOX = (28.2, 29.0, 76.8, 77.6)       # lat_min, lat_max, lon_min, lon_max
STOP_CLUSTER_M = 50.0                 # merge GTFS stops (both sides of road) within this
VISIT_RADIUS_M = 50.0                 # a fix within this of a cluster = at the stop
VISIT_GAP_S = 120                     # max gap between fixes of one visit
DENSIFY_MIN_M = 80.0                  # densify fix pairs further apart than this ...
DENSIFY_MAX_M = 1500.0                # ... but not further than this
DENSIFY_MAX_DT = 180                  # ... and only if the time gap is at most this
DENSIFY_STEP_M = 40.0
SPIKE_SPEED_MS = 100 / 3.6            # in AND out speed above this = GPS jump
DATA_GAP_S = 180                      # raw gap that invalidates a segment observation
DWELL_CAP_S = 180                     # longer = layover; dwell not used
MIN_SPEED_KMH, MAX_SPEED_KMH = 3.0, 70.0
MAX_RUN_S = 1800

# ---- model --------------------------------------------------------------
MIN_N_HOUR = 5                        # obs needed for "segment+hour" confidence
SHRINK_K = 3.0                        # shrinkage weight toward the prior
# winsorized mean of run/dwell times: clip to these quantiles per stretch-hour, then average.
# 15–85 % chosen on the model-selection split (train 1–17, score 18–24 Oct); see docs/EXPERIMENTS.md
WINSOR_LO, WINSOR_HI = 0.15, 0.85
# GPS waits are only used at stops where the GPS sees at least this share of the route's buses
# (app/ridecheck.py explains stop detection); elsewhere the timetable headway is used
MIN_STOP_DETECTION = 0.7
STAT = os.environ.get("BUSPULSE_STAT", "mean")   # "mean" (winsorized) or "median"
REPLAY_DAY = int(os.environ.get("BUSPULSE_REPLAY_DAY", "27"))   # held-out Tue 27 Oct = demo "live" feed
# live blend, tuned on the model-selection split (docs/EXPERIMENTS.md): median of the last 15 min of
# observations on a stretch, weight n/(n+K) fading with how far ahead the stretch is (e-folding 60 min)
LIVE_WINDOW_S = 900
LIVE_K = 3.0
LIVE_DECAY_S = 3600.0

# ---- planner ------------------------------------------------------------
WALK_SPEED_MS = 1.25
WALK_DETOUR = 1.3
ACCESS_MAX_M = 800.0
ACCESS_FALLBACK_M = 1500.0
ACCESS_MAX_STOPS = 25
TRANSFER_MAX_M = 300.0
MIN_TRANSFER_S = 60                   # crossing the road at the same stop cluster
MAX_TRANSFERS = 2
MAX_WAIT_S = 1800
GEOCODE_TIMEOUT_S = float(os.environ.get("BUSPULSE_GEOCODE_TIMEOUT", "3"))  # address search; 0 effectively disables
DIRECT_WALK_MAX_M = 2000.0
MODES = {
    # wait/walk weights, transfer penalty (s), in-vehicle multipliers by crowd level
    # crowd levels: 0 unknown, 1 seats, 2 standing, 3 crowded
    "fastest":     dict(w_wait=1.0, w_walk=1.0, transfer=120, crowd=[1.0, 1.0, 1.0, 1.0]),
    "balanced":    dict(w_wait=1.25, w_walk=1.25, transfer=300, crowd=[1.07, 1.0, 1.15, 1.4]),
    "comfortable": dict(w_wait=1.5, w_walk=1.5, transfer=600, crowd=[1.15, 1.0, 1.3, 1.8]),
}

# ---- crowding -----------------------------------------------------------
ETICKET_SHARE = 0.035                 # e-tickets ≈ 3–4 % of all tickets (P3 brief)
ETICKET_SHARE_RANGE = (0.03, 0.04)
UNTICKETED_UPLIFT = 1.10              # pass holders / free travel not in data
BUS_SEATS = 35
BUS_STANDING = 35
FUZZY_THRESHOLD = 78
CONF_HIGH_TICKETS = 200
CONF_MED_TICKETS = 50


def ensure_dirs() -> None:
    for d in (TMP_DIR, SEG_DIR, MODEL_DIR, DOCS_DIR / "img"):
        d.mkdir(parents=True, exist_ok=True)


try:                                   # report folder must exist for every script that writes one
    (DOCS_DIR / "img").mkdir(parents=True, exist_ok=True)
except OSError:
    pass


def day_name(day: int) -> str:
    return f"{YEAR:04d}_{MONTH:02d}_{day:02d}"


def daytype_of(date_str: str) -> int:
    """date_str 'YYYY-MM-DD' -> 0 weekday, 1 saturday, 2 sunday/holiday."""
    import datetime as _dt
    if date_str in HOLIDAYS:
        return 2
    wd = _dt.date.fromisoformat(date_str).weekday()
    return 2 if wd == 6 else (1 if wd == 5 else 0)
