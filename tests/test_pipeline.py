"""End-to-end tests on the synthetic city (ground truth known). Run: python -m pytest -q

The fixture builds synthetic data (GTFS + GPS DBs with duplicates/jumps/gaps + tickets) in a temp
folder and runs the whole pipeline: extract → model → crowding → planner/API handlers.
"""
from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def synth(tmp_path_factory):
    d = ROOT / "data_synth"
    env = {**os.environ, "BUSPULSE_DATA": str(d)}
    if not (d / "gps_db").exists():
        subprocess.run([sys.executable, str(ROOT / "tests" / "make_synthetic.py")], check=True, env=env)
    for cmd in (["scripts/01_gps_extract.py", "--workers", "1", "--days", "5", "6", "7", "8", "9", "10", "11", "25", "27"],
                ["scripts/02_build_traveltime.py"], ["scripts/03_tickets_crowding.py"], ["scripts/07_insights.py"]):
        subprocess.run([sys.executable, *cmd], check=True, env=env, cwd=ROOT)
    os.environ["BUSPULSE_DATA"] = str(d)
    import config
    importlib.reload(config)
    return d


def _handlers():
    from app import handlers
    handlers._geocode = lambda q, n: []      # offline + deterministic: the synthetic city is not on OpenStreetMap
    return handlers


def test_segments_cleaned(synth):
    import json
    q = json.loads((synth / "segments" / "quality_2026_10_05.json").read_text())
    assert q["rows_duplicates"] > 0 and q["spikes_removed"] > 0       # synthetic noise is detected
    assert q["seg_kept"] / q["seg_candidates"] > 0.6


def test_model_learns_peak_slowdown(synth):
    from app import gtfs
    from app.traveltime import TravelTimeModel
    tt = TravelTimeModel(gtfs.load())
    assert tt.has_gps
    # synthetic truth: 14 km/h at 08h vs 21 km/h at 13h on weekdays → peak runs ~1.5× longer
    ratio = tt.RUN[:, 0, 8].mean() / tt.RUN[:, 0, 13].mean()
    assert 1.3 < ratio < 1.7


def test_journey_differs_by_hour(synth):
    H = _handlers()
    c1, a = H.journeys("Kashmere Gate", "Nehru Place", "2026-10-27T08:30")
    c2, b = H.journeys("Kashmere Gate", "Nehru Place", "2026-10-27T13:00")
    assert c1 == c2 == 200 and a["options"] and b["options"]
    assert a["options"][0]["total_min"] > b["options"][0]["total_min"]
    o = a["options"][0]
    kinds = [l["type"] for l in o["legs"]]
    assert kinds[0] == "walk" and kinds[-1] == "walk" and "bus" in kinds
    assert o["range"]["low_min"] <= o["total_min"] <= o["range"]["high_min"]
    assert len(o["profile_minutes_by_hour"]) == 24 and o["quiet_hour"] is not None


@pytest.mark.parametrize("frm,to,code", [("999,999", "Nehru Place", 422), ("zzqqxx", "Nehru Place", 422),
                                         ("Nehru Place", "Nehru Place", 200)])
def test_edge_cases(synth, frm, to, code):
    c, body = _handlers().journeys(frm, to, "2026-10-27T08:30")
    assert c == code


def test_night_rolls_to_morning(synth):
    c, body = _handlers().journeys("Kashmere Gate", "Nehru Place", "2026-10-27T23:50")
    assert c == 200 and body["options"] and body["warnings"]


def test_modes_and_crowding(synth):
    H = _handlers()
    for m in ("fastest", "balanced", "comfortable"):
        c, body = H.journeys("Kashmere Gate", "Nehru Place", "2026-10-27T18:00", m)
        assert c == 200 and body["options"]
    c, r = H.crowding_route("101UP", "weekday")
    assert c == 200 and len(r["hours"]) == 24


def test_matrix_and_traffic(synth):
    H = _handlers()
    c, m = H.matrix("Kashmere Gate|Karol Bagh", "Nehru Place")
    assert c == 200 and len(m["rows"]) == 2
    c, t = H.traffic("weekday", 18)
    assert c == 200 and t["segments"]
