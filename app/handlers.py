"""Framework-independent request handlers shared by the FastAPI app and the stdlib server."""
from __future__ import annotations

import json
import sys
import threading
import time
from collections import OrderedDict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from app import gtfs as gtfs_mod  # noqa: E402
from app.planner import Planner, PlanError, norm_name, parse_depart  # noqa: E402
import re  # noqa: E402

try:
    from rapidfuzz import fuzz  # noqa: E402
except ImportError:  # same optional-dependency fallback as planner.search_stops
    import difflib

    class fuzz:  # noqa: N801
        @staticmethod
        def token_set_ratio(a: str, b: str) -> float:
            return 100 * difflib.SequenceMatcher(None, " ".join(sorted(a.split())), " ".join(sorted(b.split()))).ratio()

_LATLON = re.compile(r"^\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*$")
from app.traveltime import TravelTimeModel  # noqa: E402

_state: dict = {}
_lock = threading.Lock()
STARTED = time.time()

# one-tap example trips shown in the web app (fixed date = the replayed held-out Tuesday, so the
# warm-up below and a click produce the same request and the answer comes from the cache)
EXAMPLES = [
    {"from": "ISBT Kashmiri Gate", "to": "Nehru Place", "at": "2026-10-27T08:30",
     "en": "Kashmere Gate → Nehru Place · 8:30 am", "hi": "कश्मीरी गेट → नेहरू प्लेस · सुबह 8:30"},
    {"from": "Anand Vihar ISBT", "to": "Dhaula Kuan", "at": "2026-10-27T18:00",
     "en": "Anand Vihar → Dhaula Kuan · 6 pm", "hi": "आनंद विहार → धौला कुआँ · शाम 6"},
    {"from": "Dwarka Sec 21", "to": "Connaught Place", "at": "2026-10-27T09:00",
     "en": "Dwarka Sec 21 → Connaught Place · 9 am", "hi": "द्वारका सेक्टर 21 → कनॉट प्लेस · सुबह 9"},
]

# small LRU of finished answers: on a slow shared CPU a repeated question costs nothing
_cache: "OrderedDict[tuple, dict]" = OrderedDict()
_cache_lock = threading.Lock()
CACHE_MAX = 512


def state() -> dict:
    if _state:
        return _state
    with _lock:
        if _state:
            return _state
        t = time.time()
        g = gtfs_mod.load()
        tt = TravelTimeModel(g)
        crowd = None
        try:
            from app.crowding import CrowdModel
            crowd = CrowdModel.load(g)
        except Exception as e:  # noqa: BLE001
            print(f"[state] crowding not loaded: {e}")
        live_day, rt = None, None
        try:                                   # a real GTFS-RT feed if configured, else the replay demo
            from app.gtfsrt import from_env
            rt = from_env(g, tt)
            if rt is not None:
                tt.live = rt.feed
                rt.start()
                print(f"[state] GTFS-RT connector polling {rt.url} every {rt.poll_s:.0f} s")
        except Exception as e:  # noqa: BLE001
            rt = None
            print(f"[state] GTFS-RT connector not started: {e}")
        vehicles = None
        if rt is None:
            try:
                from app.live import LiveFeed
                for d in (config.REPLAY_DAY, *config.TEST_DAYS):
                    feed = LiveFeed.replay_day(tt, d)
                    if feed is not None:
                        tt.live, live_day = feed, d
                        break
            except Exception as e:  # noqa: BLE001
                print(f"[state] live replay not loaded: {e}")
            try:
                from app.live import ReplayVehicles
                vehicles = ReplayVehicles.load(g, live_day or config.REPLAY_DAY)
            except Exception as e:  # noqa: BLE001
                print(f"[state] replay vehicles not loaded: {e}")
        else:
            names = {}
            for P in g["patterns"]:
                try:
                    names.setdefault(int(float(P["route_id"])), str(P["name"]))
                except (TypeError, ValueError):
                    pass
            rt.route_names = names
        insights = None
        fi = config.MODEL_DIR / "insights.json"
        if fi.exists():
            insights = json.loads(fi.read_text(encoding="utf-8"))
        _state.update(g=g, tt=tt, planner=Planner(g, tt, crowd), crowd=crowd, live_day=live_day, rt=rt, vehicles=vehicles,
                      insights=insights, load_s=round(time.time() - t, 1))
        print(f"[state] ready in {_state['load_s']} s (gps model: {tt.has_gps}, crowding: {crowd is not None}, "
              f"live replay day: {live_day})")
    return _state


def _err(code: int, msg: str, kind: str = "bad_request"):
    return code, {"error": kind, "message": msg}


def journeys(frm: str, to: str, depart: str | None = None, mode: str = "fastest", n: int = 3, live: bool = False):
    t = time.time()
    try:
        n = int(n)
    except (TypeError, ValueError):
        return _err(400, "n must be an integer")
    # "now" queries are cached per minute; explicit times exactly (a time without a date means today)
    ist_day = time.strftime("%Y-%m-%d", time.gmtime(t + config.IST_OFFSET_S))
    key = (frm, to, depart or f"now@{int(t // 60)}", ist_day, mode, n, bool(live))
    with _cache_lock:
        hit = _cache.get(key)
        if hit is not None:
            _cache.move_to_end(key)
    if hit is not None:
        return 200, {**hit, "compute_ms": 0, "cached": True}
    try:
        pl = state()["planner"]
        names = {}
        for side in ("from", "to"):          # an address that is not (clearly) a stop name → OpenStreetMap
            q = frm if side == "from" else to
            if _LATLON.match(str(q)):
                continue
            stop = pl.search_stops(q, 1)
            if stop and stop[0]["score"] >= 88:
                continue
            hit = _geocode(q, 1) if len(str(q).strip()) >= 3 else []
            if not hit or (stop and stop[0]["score"] >= 72
                           and fuzz.token_set_ratio(norm_name(q), norm_name(hit[0]["name"])) < 80):
                continue                     # keep the stop match; plan() raises unknown_place if there is none
            if hit:
                names[side] = hit[0]["name"]
                if side == "from":
                    frm = f"{hit[0]['lat']},{hit[0]['lon']}"
                else:
                    to = f"{hit[0]['lat']},{hit[0]['lon']}"
        res = pl.plan(frm, to, depart, mode, int(n), bool(live))
        for side, nm in names.items():
            res[side]["name"] = nm
    except PlanError as e:
        return _err(422 if e.code != "bad_request" else 400, str(e), e.code)
    except (TypeError, ValueError) as e:
        return _err(400, str(e))
    res["compute_ms"] = round(1000 * (time.time() - t))
    with _cache_lock:
        _cache[key] = res
        while len(_cache) > CACHE_MAX:
            _cache.popitem(last=False)
    return 200, res


def examples():
    return 200, {"examples": EXAMPLES}


def warm_up():
    """Load the model and answer the example trips once (all modes), in the background at start-up."""
    def run():
        try:
            state()
            for ex in EXAMPLES:
                for mode in ("fastest", "balanced", "comfortable"):
                    journeys(ex["from"], ex["to"], ex["at"], mode, 3, False)
            print(f"[warm-up] {len(EXAMPLES)} example trips ready")
        except Exception as e:  # noqa: BLE001
            print(f"[warm-up] failed: {e}")
    threading.Thread(target=run, daemon=True).start()


def matrix(origins: str, destinations: str, depart: str | None = None, mode: str = "fastest"):
    os_ = [o for o in origins.split("|") if o.strip()]
    ds = [d for d in destinations.split("|") if d.strip()]
    if not os_ or not ds:
        return _err(400, "origins and destinations are '|'-separated lists of 'lat,lon' or stop names")
    if len(os_) * len(ds) > 100:
        return _err(400, "at most 100 origin–destination cells per request")
    pl = state()["planner"]
    rows = []
    for o in os_:
        row = []
        for d in ds:
            try:
                r = pl.plan(o, d, depart, mode, 1)
                op = r["options"][0] if r["options"] else None
                row.append({"total_min": op["total_min"] if op else None,
                            "range_min": [op["range"]["low_min"], op["range"]["high_min"]] if op and op.get("range") else None,
                            "transfers": op["transfers"] if op else None,
                            "status": "OK" if op else "NO_ROUTE"})
            except PlanError as e:
                row.append({"total_min": None, "status": e.code.upper()})
        rows.append(row)
    return 200, {"origins": os_, "destinations": ds, "depart": depart, "mode": mode, "rows": rows}


def stops_search(q: str, n: int = 8):
    if not q or len(q.strip()) < 2:
        return 200, {"results": []}
    return 200, {"results": state()["planner"].search_stops(q, int(n))}


PHOTON_URL = "https://photon.komoot.io/api/"
_geo_cache: "OrderedDict[str, list]" = OrderedDict()


def _geocode(q: str, n: int) -> list[dict]:
    """Addresses / landmarks from OpenStreetMap (Photon, free, no key), limited to the service area.
    Any failure (offline, slow, rate-limited) just returns [] so stop search keeps working."""
    import urllib.parse
    import urllib.request
    key = q.strip().lower()
    with _cache_lock:
        if key in _geo_cache:
            _geo_cache.move_to_end(key)
            return _geo_cache[key][:n]
    pl = state()["planner"]
    la0, la1, lo0, lo1 = pl.area
    url = PHOTON_URL + "?" + urllib.parse.urlencode(
        {"q": q, "limit": max(n * 2, 8), "lat": 28.61, "lon": 77.21, "bbox": f"{lo0},{la0},{lo1},{la1}"})
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "BusPulse-Delhi/1.0 (transit demo)"})
        with urllib.request.urlopen(req, timeout=config.GEOCODE_TIMEOUT_S) as r:
            feats = json.loads(r.read()).get("features", [])
    except Exception:  # noqa: BLE001
        return []
    out, seen = [], set()
    for f in feats:
        lon, lat = f.get("geometry", {}).get("coordinates", [None, None])[:2]
        if lat is None or not (la0 <= lat <= la1 and lo0 <= lon <= lo1):
            continue
        p = f.get("properties", {})
        name = p.get("name") or " ".join(x for x in (p.get("housenumber"), p.get("street")) if x)
        if not name:
            continue
        area = ", ".join(dict.fromkeys(x for x in (p.get("street") if p.get("name") else None, p.get("district"),
                                                    p.get("locality"), p.get("city")) if x and x != name))
        k = (name.lower(), round(lat, 2), round(lon, 2))   # same name within ~1 km = one entry
        if k in seen:
            continue
        seen.add(k)
        out.append({"kind": "place", "name": name, "area": area, "lat": round(lat, 6), "lon": round(lon, 6)})
    with _cache_lock:
        _geo_cache[key] = out
        while len(_geo_cache) > CACHE_MAX:
            _geo_cache.popitem(last=False)
    return out[:n]


def places_search(q: str, n: int = 8):
    """Bus stops (typo tolerant) plus addresses/landmarks: strong stop matches first, then places."""
    if not q or len(q.strip()) < 2:
        return 200, {"results": []}
    n = int(n)
    stops = [dict(s, kind="stop") for s in state()["planner"].search_stops(q, n)]
    strong = [s for s in stops if s.get("score", 0) >= 88]
    places = _geocode(q, n) if len(q.strip()) >= 3 else []
    # weak stop matches are noise next to real addresses; keep them only when the address lookup found nothing
    rest = [] if places else [s for s in stops if s not in strong]
    res = (strong[:4] + places[: max(3, n - len(strong[:4]) - 2)] + rest)[:n]
    return 200, {"results": res}


def traffic(daytype: str = "weekday", hour: int = 18):
    if daytype not in config.DAYTYPES:
        return _err(400, f"daytype must be one of {config.DAYTYPES}")
    hour = int(hour)
    if not 0 <= hour <= 23:
        return _err(400, "hour must be 0–23")
    tt = state()["tt"]
    if not tt.has_gps:
        return 200, {"segments": [], "note": "GPS model not built"}
    segs = tt.traffic(config.DAYTYPES.index(daytype), hour)
    return 200, {"daytype": daytype, "hour": hour,
                 "columns": ["lat1", "lon1", "lat2", "lon2", "speed_kmh", "slowdown_vs_best_hour", "level", "path"],
                 "segments": segs}


def crowding_route(route: str, daytype: str = "weekday"):
    c = state()["crowd"]
    if c is None:
        return _err(404, "crowding model not built (run scripts/03_tickets_crowding.py)", "not_built")
    if daytype not in config.DAYTYPES:
        return _err(400, f"daytype must be one of {config.DAYTYPES}")
    res = c.route_profile(route, config.DAYTYPES.index(daytype))
    if res is None:
        return _err(404, f"no crowding data for route '{route}'", "unknown_route")
    return 200, res


def insights():
    ins = state()["insights"]
    if ins is None:
        return _err(404, "insights not built (run scripts/07_insights.py)", "not_built")
    return 200, ins


def live_vehicles(at: str | None = None, bbox: str | None = None, limit: int = 800):
    """Buses on the map: the GTFS-RT feed's latest positions, or the replay day at this time of day."""
    s = state()
    try:
        limit = max(1, min(int(limit), 3000))
        bb = None
        if bbox:
            bb = [float(x) for x in bbox.split(",")]
            if len(bb) != 4:
                raise ValueError
    except (TypeError, ValueError):
        return _err(400, "bbox must be 'lat0,lon0,lat1,lon1'; limit an integer")
    if s.get("rt") is not None:
        v = s["rt"].vehicles(bb, limit)
        return 200, {"mode": "gtfs-rt", "source": "GTFS-RT feed", "vehicles": v,
                     "columns": ["lat", "lon", "route", "bearing", "moving", "id"]}
    rv = s.get("vehicles")
    if rv is None:
        return 200, {"mode": "off", "vehicles": []}
    if at:
        m = __import__("re").match(r"^(\d{1,2}):(\d{2})(?::(\d{2}))?$", str(at).strip())
        if not m or int(m.group(1)) > 23 or int(m.group(2)) > 59 or int(m.group(3) or 0) > 59:
            return _err(400, "at must be HH:MM")
        sec = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3) or 0)
    else:
        now = time.time() + config.IST_OFFSET_S
        sec = int(now % 86400)
    return 200, {"mode": "replay", "source": rv.label, "time": f"{sec // 3600:02d}:{sec % 3600 // 60:02d}:{sec % 60:02d}",
                 "vehicles": rv.at(sec, bb, limit), "columns": ["lat", "lon", "route", "bearing", "moving", "id"]}


def health():
    s = state()
    g, tt = s["g"], s["tt"]
    rt = s.get("rt")
    live = ({"mode": "gtfs-rt", "observations": len(rt.feed), **{k: v for k, v in rt.stats.items()}}
            if rt is not None else {"mode": "replay" if s["live_day"] else "off", "replay_day": s["live_day"]})
    return 200, {"status": "ok", "uptime_s": round(time.time() - STARTED), "load_s": s["load_s"],
                 "gtfs": g["stats"], "gps_model": tt.meta, "crowding": s["crowd"] is not None,
                 "live_replay_day": s["live_day"], "live": live, "stat": config.STAT}
