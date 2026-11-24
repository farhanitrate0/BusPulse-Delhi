"""Copy the small model tables the app needs into ./bundle (committed to git) for online deploys.

The raw data (124 GB GPS, tickets, GTFS) never goes into git; the app only needs ~tens of MB:
GTFS cache, travel-time tables, GPS waits, calibration, crowding, insights and one replay day.
    python scripts/09_bundle.py           then deploy with BUSPULSE_DATA=bundle (render.yaml does this)
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

# compiled runtime caches (numpy, compressed) instead of the parquet tables: the server then loads
# without pandas/parquet peaks and fits a 512 MB instance
FILES = ["gtfs.pkl", "pair_paths.npz","tt_runtime.npz", "crowd_runtime.npz", "calibration.json", "model_meta.json",
         "crowd_meta.json", "insights.json"]


def compile_caches():
    """Load every runtime model once so its .npz cache in data/model is fresh."""
    from app import gtfs as gtfs_mod
    from app.crowding import CrowdModel
    from app.live import LiveFeed
    from app.traveltime import TravelTimeModel
    g = gtfs_mod.load()
    tt = TravelTimeModel(g)
    CrowdModel.load(g)
    LiveFeed.replay_day(tt, config.REPLAY_DAY)
    from app.live import ReplayVehicles
    ReplayVehicles.load(g, config.REPLAY_DAY)


def main():
    out = config.ROOT / "bundle"
    if out.exists():
        shutil.rmtree(out)
    (out / "model").mkdir(parents=True, exist_ok=True)
    (out / "segments").mkdir(parents=True, exist_ok=True)
    compile_caches()
    total = 0
    for f in FILES:
        src = config.MODEL_DIR / f
        if src.exists():
            shutil.copy2(src, out / "model" / f)
            total += src.stat().st_size
            print(f"  model/{f:22s} {src.stat().st_size / 1e6:7.1f} MB")
        else:
            print(f"  (missing) {f}")
    rep = config.SEG_DIR / f"live_{config.day_name(config.REPLAY_DAY)}.npz"
    if rep.exists():
        shutil.copy2(rep, out / "segments" / rep.name)
        total += rep.stat().st_size
        print(f"  segments/{rep.name} (live replay day)")
    veh = config.SEG_DIR / f"vehicles_{config.day_name(config.REPLAY_DAY)}.npz"
    if veh.exists():
        shutil.copy2(veh, out / "segments" / veh.name)
        total += veh.stat().st_size
        print(f"  segments/{veh.name} (live buses map layer)")
    print(f"bundle: {total / 1e6:.1f} MB → {out}")


if __name__ == "__main__":
    main()
