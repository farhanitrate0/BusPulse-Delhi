"""Live GTFS-Realtime connector: vehicle positions → stop-to-stop times → the live blend.

    BUSPULSE_GTFSRT_URL=https://…/VehiclePositions.pb            (required to switch it on)
    BUSPULSE_GTFSRT_HEADERS="x-api-key: …"                        (optional, ';'-separated)
    BUSPULSE_GTFSRT_POLL_S=15                                     (optional)

Every poll fetches the feed, keeps the last 20 minutes of fixes per bus, and runs them through the
**same** code as the historical pipeline (app/gpsproc.py: drop GPS jumps, mark data gaps, densify,
snap to stop clusters, visits, stop-to-stop runs). New stop-to-stop runs go into a StreamFeed, which the
planner blends with history exactly like the replay (app/live.py). Late fixes are sorted in, repeated
fixes are dropped, and gaps longer than 3 minutes never produce a run.

The protobuf decoder covers what is needed here (FeedMessage → FeedEntity → VehiclePosition: trip.route_id,
trip.trip_id, position.latitude/longitude, timestamp, vehicle.id), so no extra dependency is required.
"""
from __future__ import annotations

import math
import os
import struct
import sys
import threading
import time
import urllib.request
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

BUFFER_S = 1200          # fixes kept per bus (a stop-to-stop run is ≤ 30 min, the live window 15 min)
KEEP_OBS_S = 7200        # observations kept in the StreamFeed


# ----------------------------------------------------------------------------- protobuf (minimal)
def _varint(b: bytes, i: int) -> tuple[int, int]:
    x = s = 0
    while True:
        c = b[i]
        i += 1
        x |= (c & 0x7F) << s
        if c < 0x80:
            return x, i
        s += 7


def _fields(b: bytes):
    i, n = 0, len(b)
    while i < n:
        key, i = _varint(b, i)
        f, wt = key >> 3, key & 7
        if wt == 0:
            v, i = _varint(b, i)
        elif wt == 1:
            v, i = b[i:i + 8], i + 8
        elif wt == 2:
            ln, i = _varint(b, i)
            v, i = b[i:i + ln], i + ln
        elif wt == 5:
            v, i = b[i:i + 4], i + 4
        else:
            raise ValueError(f"unsupported wire type {wt}")
        yield f, wt, v


def decode_vehicle_positions(data: bytes) -> tuple[int | None, list[dict]]:
    """FeedMessage bytes → (header timestamp, [{vehicle_id, route_id, trip_id, lat, lon, ts}])."""
    header_ts, out = None, []
    for f, _, v in _fields(data):
        if f == 1:                                        # FeedHeader
            for hf, _, hv in _fields(v):
                if hf == 3:
                    header_ts = hv
        elif f == 2:                                      # FeedEntity
            ent_id, vp = None, None
            for ef, _, ev in _fields(v):
                if ef == 1:
                    ent_id = ev.decode("utf-8", "replace")
                elif ef == 4:
                    vp = ev
            if vp is None:
                continue
            rec = {"vehicle_id": ent_id, "route_id": None, "trip_id": None, "lat": None, "lon": None, "ts": None}
            for vf, wt, vv in _fields(vp):
                if vf == 1:                               # TripDescriptor
                    for tf, _, tv in _fields(vv):
                        if tf == 1:
                            rec["trip_id"] = tv.decode("utf-8", "replace")
                        elif tf == 5:
                            rec["route_id"] = tv.decode("utf-8", "replace")
                elif vf == 2:                             # Position
                    for pf, pwt, pv in _fields(vv):
                        if pf in (1, 2):
                            val = struct.unpack("<f", pv)[0] if pwt == 5 else struct.unpack("<d", pv)[0]
                            rec["lat" if pf == 1 else "lon"] = val
                elif vf == 5 and wt == 0:                 # timestamp
                    rec["ts"] = vv
                elif vf == 8:                             # VehicleDescriptor
                    for df, _, dv in _fields(vv):
                        if df == 1:
                            rec["vehicle_id"] = dv.decode("utf-8", "replace")
            if rec["ts"] is None:
                rec["ts"] = header_ts
            out.append(rec)
    return header_ts, out


def _enc_varint(x: int) -> bytes:
    o = bytearray()
    while True:
        b = x & 0x7F
        x >>= 7
        if x:
            o.append(b | 0x80)
        else:
            o.append(b)
            return bytes(o)


def _enc_field(f: int, wt: int, payload) -> bytes:
    key = _enc_varint((f << 3) | wt)
    if wt == 0:
        return key + _enc_varint(int(payload))
    if wt == 2:
        return key + _enc_varint(len(payload)) + payload
    if wt == 5:
        return key + payload
    raise ValueError(wt)


def encode_vehicle_positions(rows, header_ts: int) -> bytes:
    """[{vehicle_id, route_id, trip_id?, lat, lon, ts}] → FeedMessage bytes (used by the simulator/tests)."""
    hdr = _enc_field(1, 2, b"2.0") + _enc_field(2, 0, 0) + _enc_field(3, 0, header_ts)
    out = [_enc_field(1, 2, hdr)]
    for r in rows:
        vid = str(r["vehicle_id"]).encode()
        trip = b""
        if r.get("trip_id"):
            trip += _enc_field(1, 2, str(r["trip_id"]).encode())
        if r.get("route_id") is not None:
            trip += _enc_field(5, 2, str(r["route_id"]).encode())
        pos = _enc_field(1, 5, struct.pack("<f", r["lat"])) + _enc_field(2, 5, struct.pack("<f", r["lon"]))
        vp = (_enc_field(1, 2, trip) + _enc_field(2, 2, pos) + _enc_field(5, 0, int(r["ts"]))
              + _enc_field(8, 2, _enc_field(1, 2, vid)))
        out.append(_enc_field(2, 2, _enc_field(1, 2, vid) + _enc_field(4, 2, vp)))
    return b"".join(out)


# ----------------------------------------------------------------------------- live observations
class StreamFeed:
    """Same interface as app.live.LiveFeed (snapshot / blend / coverage), filled while running."""

    def __init__(self, label: str = "GTFS-RT live"):
        self.label = label
        self._obs: deque = deque()            # (time of day IST [s], pair index, run [s]) in arrival order
        self._lock = threading.Lock()
        self._version = 0
        self._newest: float | None = None
        self._cache: dict = {}

    def add(self, t_tod: np.ndarray, pi: np.ndarray, run: np.ndarray):
        with self._lock:
            for t, p, r in zip(t_tod.tolist(), pi.tolist(), run.tolist()):
                self._obs.append((t, p, r))
                # newest time of day seen so far (a batch is ordered by bus, not by time; wraps at midnight)
                if self._newest is None or (t - self._newest) % 86400 < 43200:
                    self._newest = t
            if self._newest is not None:
                keep = [o for o in self._obs if (self._newest - o[0]) % 86400 <= KEEP_OBS_S]
                if len(keep) != len(self._obs):
                    self._obs = deque(keep)
            self._version += 1

    def __len__(self):
        return len(self._obs)

    def snapshot(self, now: float) -> dict[int, tuple[float, int]]:
        key = (int(now // 60), self._version)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        with self._lock:
            rows = [(p, r) for t, p, r in self._obs if 0 <= (now - t) % 86400 <= config.LIVE_WINDOW_S]
        by: dict[int, list] = {}
        for p, r in rows:
            by.setdefault(p, []).append(r)
        snap = {p: (float(np.median(v)), len(v)) for p, v in by.items()}
        self._cache = {key: snap}
        return snap

    def blend(self, pi: int, base: float, t: float, now: float) -> float:
        s = self.snapshot(now).get(pi)
        if s is None:
            return base
        med, n = s
        w = n / (n + config.LIVE_K) * math.exp(-max(0.0, t - now) / config.LIVE_DECAY_S)
        return w * med + (1 - w) * base

    def coverage(self, now: float) -> int:
        return len(self.snapshot(now))


# ----------------------------------------------------------------------------- connector
class GtfsRtConnector:
    def __init__(self, g: dict, tt, url: str | None = None, headers: dict | None = None,
                 poll_s: float = 15.0, feed: StreamFeed | None = None):
        from app.gpsproc import network_index
        self.url, self.headers, self.poll_s = url, headers or {}, poll_s
        self.tt = tt
        self.feed = feed or StreamFeed()
        self.tree, self.pair_set, self.pair_dist, self.ncl = network_index(g)
        self._buf: dict[str, dict[int, tuple[float, float, int]]] = {}   # vehicle → {ts: (x, y, route)}
        self._emitted: dict[str, float] = {}                              # vehicle → newest arr_b sent
        self._vcode: dict[str, int] = {}
        self.latest: dict[str, tuple] = {}                                # vehicle → (lat, lon, ts, route_id)
        self._thread = None
        self._stop = threading.Event()
        self.stats = {"polls": 0, "errors": 0, "last_error": None, "fixes_in": 0, "fixes_new": 0,
                      "runs_emitted": 0, "vehicles": 0, "last_poll_utc": None, "last_feed_ts": None,
                      "process_ms": None}

    # -- ingest
    def ingest(self, recs: list[dict], now_ts: float | None = None):
        from app.geo import to_xy
        ok = [r for r in recs if r.get("vehicle_id") and r.get("lat") is not None and r.get("lon") is not None
              and r.get("ts")]
        lat0, lat1, lon0, lon1 = config.BBOX
        ok = [r for r in ok if lat0 <= r["lat"] <= lat1 and lon0 <= r["lon"] <= lon1]
        self.stats["fixes_in"] += len(recs)
        if not ok:
            return 0
        xs, ys = to_xy(np.array([r["lat"] for r in ok]), np.array([r["lon"] for r in ok]))
        new = 0
        for r in ok:                                       # newest known position per bus, for the map
            prev = self.latest.get(r["vehicle_id"])
            if prev is None or int(r["ts"]) > prev[2]:
                brg = prev[4] if prev is not None and len(prev) > 4 else -1
                if prev is not None:                       # heading from the last move of ≥ 15 m
                    dy = (float(r["lat"]) - prev[0]) * 111_320
                    dx = (float(r["lon"]) - prev[1]) * 111_320 * math.cos(math.radians(prev[0]))
                    if dx * dx + dy * dy >= 225:
                        brg = int((math.degrees(math.atan2(dx, dy)) + 360) % 360)
                self.latest[r["vehicle_id"]] = (float(r["lat"]), float(r["lon"]), int(r["ts"]), r.get("route_id"), brg)
        for r, x, y in zip(ok, xs, ys):
            b = self._buf.setdefault(r["vehicle_id"], {})
            ts = int(r["ts"])
            if ts in b:
                continue                                   # the same fix again (feeds repeat until a bus moves on)
            try:
                route = int(float(r["route_id"])) if r.get("route_id") not in (None, "") else -1
            except ValueError:
                route = -1
            b[ts] = (float(x), float(y), route)
            new += 1
        horizon = (now_ts or max(int(r["ts"]) for r in ok)) - BUFFER_S
        for v in list(self._buf):
            b = self._buf[v]
            for ts in [t for t in b if t < horizon]:
                del b[ts]
            if not b:
                del self._buf[v]
                self._emitted.pop(v, None)
        for v in [v for v, p in self.latest.items() if p[2] < horizon]:
            del self.latest[v]                             # not heard from for 20 min: off the map
        self.stats["fixes_new"] += new
        self.stats["vehicles"] = len(self._buf)
        self.stats["last_feed_ts"] = max(int(r["ts"]) for r in ok)
        return new

    # -- buffer → stop-to-stop runs → feed
    def process(self) -> int:
        from app.gpsproc import process_vehicles
        t0 = time.time()
        V, T, X, Y, R, names = [], [], [], [], [], []
        for v, b in self._buf.items():
            if len(b) < 2:
                continue
            code = self._vcode.setdefault(v, len(self._vcode))
            for ts in sorted(b):                            # late fixes are sorted into place here
                x, y, r = b[ts]
                V.append(code), T.append(ts), X.append(x), Y.append(y), R.append(r)
        if len(V) < 2:
            return 0
        res = process_vehicles(np.array(V, np.int64), np.array(T, np.int64), np.array(X), np.array(Y),
                               np.array(R, np.int64), self.tree, self.pair_set, self.pair_dist, self.ncl, {})
        self.stats["process_ms"] = round(1000 * (time.time() - t0))
        if res is None:
            return 0
        seg = res[0]
        if not len(seg):
            return 0
        inv = {c: v for v, c in self._vcode.items()}
        vname = seg.v.map(inv)
        last = vname.map(lambda v: self._emitted.get(v, -1.0))
        seg = seg[seg.arr_b.to_numpy() > last.to_numpy()]
        if not len(seg):
            return 0
        keys = seg.ca.to_numpy(np.int64) * self.ncl + seg.cb.to_numpy(np.int64)
        pi = np.array([self.tt.pair_index.get(int(k), -1) for k in keys])
        good = pi >= 0
        t_tod = (seg.arr_b.to_numpy(np.float64) + config.IST_OFFSET_S) % 86400
        self.feed.add(t_tod[good], pi[good], seg.run.to_numpy(np.float32)[good])
        for v, a in zip(seg.v.map(inv), seg.arr_b):
            if a > self._emitted.get(v, -1.0):
                self._emitted[v] = float(a)
        self.stats["runs_emitted"] += int(good.sum())
        return int(good.sum())

    def vehicles(self, bbox=None, limit: int = 800, max_age_s: int = 300) -> list[list]:
        """Latest position of every bus heard from in the last 5 min: [[lat, lon, route, bearing, moving]]."""
        if not self.latest:
            return []
        newest = max(p[2] for p in self.latest.values())
        names = getattr(self, "route_names", {})
        out = []
        for vid, (lat, lon, ts, rid, brg) in self.latest.items():
            if newest - ts > max_age_s:
                continue
            if bbox is not None and not (bbox[0] <= lat <= bbox[2] and bbox[1] <= lon <= bbox[3]):
                continue
            try:
                nm = names.get(int(float(rid)), str(rid or ""))
            except (TypeError, ValueError):
                nm = str(rid or "")
            out.append([round(lat, 5), round(lon, 5), nm, brg, True, vid])
            if len(out) >= limit:
                break
        return out

    # -- network loop
    def poll_once(self) -> int:
        req = urllib.request.Request(self.url, headers={"User-Agent": "BusPulse-Delhi/1.0", **self.headers})
        with urllib.request.urlopen(req, timeout=20) as r:
            data = r.read()
        _, recs = decode_vehicle_positions(data)
        self.ingest(recs)            # buffer horizon follows the feed's own clock, not this server's
        n = self.process()
        self.stats["polls"] += 1
        self.stats["last_poll_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        return n

    def _loop(self):
        backoff = self.poll_s
        while not self._stop.is_set():
            try:
                self.poll_once()
                backoff = self.poll_s
            except Exception as e:  # noqa: BLE001  (network errors must never take the API down)
                self.stats["errors"] += 1
                self.stats["last_error"] = f"{type(e).__name__}: {str(e)[:200]}"
                backoff = min(backoff * 2, 300)
            self._stop.wait(backoff)

    def start(self):
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, name="gtfsrt", daemon=True)
            self._thread.start()
        return self

    def stop(self):
        self._stop.set()


def from_env(g: dict, tt) -> GtfsRtConnector | None:
    url = os.environ.get("BUSPULSE_GTFSRT_URL", "").strip()
    if not url:
        return None
    headers = {}
    raw = os.environ.get("BUSPULSE_GTFSRT_HEADERS", "")
    hf = config.DATA / "gtfsrt.env"            # local convenience: key kept in the gitignored data folder
    if not raw and hf.exists():
        raw = hf.read_text(encoding="utf-8").strip()
    for part in raw.split(";"):
        if ":" in part:
            k, v = part.split(":", 1)
            headers[k.strip()] = v.strip()
    poll = float(os.environ.get("BUSPULSE_GTFSRT_POLL_S", "15"))
    return GtfsRtConnector(g, tt, url, headers, poll)
