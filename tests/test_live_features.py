"""Unit tests for the live features: GTFS-RT decoding, the live blend store, and the live-buses positions."""
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from app.gtfsrt import StreamFeed, _enc_field, decode_vehicle_positions, encode_vehicle_positions  # noqa: E402
from app.live import ReplayVehicles  # noqa: E402


def test_gtfsrt_round_trip():
    rows = [{"vehicle_id": "DL1PC1234", "route_id": "534", "trip_id": "T1", "lat": 28.6123, "lon": 77.2311, "ts": 1793100600},
            {"vehicle_id": "DL51EV0001", "route_id": "2111", "lat": 28.55, "lon": 77.25, "ts": 1793100611}]
    hts, recs = decode_vehicle_positions(encode_vehicle_positions(rows, 1793100620))
    assert hts == 1793100620 and len(recs) == 2
    assert recs[0]["vehicle_id"] == "DL1PC1234" and recs[0]["route_id"] == "534" and recs[0]["trip_id"] == "T1"
    assert abs(recs[0]["lat"] - 28.6123) < 1e-4 and abs(recs[1]["lon"] - 77.25) < 1e-4
    assert recs[1]["ts"] == 1793100611


def test_gtfsrt_decoder_skips_unknown_fields_and_uses_header_time():
    # an entity with an extra unknown field (99) and no own timestamp → header time
    pos = _enc_field(1, 5, struct.pack("<f", 28.6)) + _enc_field(2, 5, struct.pack("<f", 77.2))
    vp = _enc_field(2, 2, pos) + _enc_field(99, 0, 7)
    ent = _enc_field(1, 2, b"bus-9") + _enc_field(4, 2, vp)
    hdr = _enc_field(1, 2, b"2.0") + _enc_field(3, 0, 1793100000)
    _, recs = decode_vehicle_positions(_enc_field(1, 2, hdr) + _enc_field(2, 2, ent))
    assert recs[0]["vehicle_id"] == "bus-9" and recs[0]["ts"] == 1793100000


def test_stream_feed_blend_weights_and_window():
    f = StreamFeed()
    now = 18 * 3600.0
    f.add(np.array([now - 60, now - 120, now - 2000]), np.array([5, 5, 5]), np.array([100.0, 120.0, 999.0]))
    snap = f.snapshot(now)
    assert snap[5] == (110.0, 2)                    # the 2000-s-old run is outside the 15-min window
    base = 200.0
    w = 2 / (2 + config.LIVE_K)
    assert f.blend(5, base, now, now) == pytest.approx(w * 110 + (1 - w) * base)
    assert f.blend(5, base, now + 3 * 3600, now) == pytest.approx(base, rel=0.05)   # fades further ahead
    assert f.blend(6, base, now, now) == base                                        # no live data → history


def test_replay_vehicles_interpolates_and_drops_parked_buses():
    z = {"v": np.array([0, 1], np.int32), "r": np.array([0, 0], np.int32),
         "ca": np.array([0, 1], np.int16), "cb": np.array([1, 0], np.int16),
         "dep_a": np.array([1000, 1000], np.int32), "arr_b": np.array([1100, 1050], np.int32),
         "dep_b": np.array([1120, 5000], np.int32)}
    rv = ReplayVehicles(z, ["534UP"], [28.60, 28.61], [77.20, 77.20], "test")
    mid = rv.at(1050)
    bus0 = next(v for v in mid if v[5] == 0)
    assert bus0[0] == pytest.approx(28.605, abs=1e-4) and bus0[4] is True and bus0[2] == "534UP"
    assert 0 <= bus0[3] < 10 or bus0[3] > 350                       # heading north
    later = {v[5] for v in rv.at(1700)}
    assert 1 not in later                                           # stood > 10 min at the stop: parked
    assert rv.at(1050, bbox=(0, 0, 1, 1)) == []
