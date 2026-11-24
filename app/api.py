"""FastAPI app. Run: python run.py  (or: uvicorn app.api:app --port 8000). Docs at /docs."""
from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from app import handlers as H  # noqa: E402

WEB = config.ROOT / "web"

app = FastAPI(
    title="BusPulse Delhi API",
    version="1.0",
    description=(
        "Bus journey planner for Delhi whose travel times come from **real bus GPS** "
        "(31 days, DTC fleet), not just the timetable. Times depend on day type and hour, "
        "can blend in live data, come with a reliability range, and carry a crowding level "
        "estimated from e-tickets.\n\n"
        "Example: `/journeys?from=Kashmere Gate ISBT&to=Nehru Place&depart=2026-10-27T08:30`"),
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"], allow_headers=["*"])
app.add_middleware(GZipMiddleware, minimum_size=2000)   # /traffic with road paths compresses ~5x


def _resp(res):
    code, body = res
    return JSONResponse(body, status_code=code)


@app.on_event("startup")
def _warm():
    H.state()
    H.warm_up()          # example trips answered in the background → instant first clicks


@app.get("/examples", tags=["planner"], summary="Example trips shown in the web app")
def examples():
    return _resp(H.examples())


@app.get("/journeys", tags=["planner"], summary="Bus journey options between two places")
def journeys(
    from_: str = Query(..., alias="from", description="'lat,lon' or a bus stop name", examples=["Kashmere Gate ISBT"]),
    to: str = Query(..., description="'lat,lon' or a bus stop name", examples=["Nehru Place"]),
    depart: str | None = Query(None, description="ISO date-time in IST, e.g. 2026-10-27T08:30 (default: now)"),
    mode: str = Query("fastest", description="fastest | balanced | comfortable (crowding-aware)"),
    n: int = Query(3, ge=1, le=5, description="number of options"),
    live: bool = Query(False, description="blend live observations (demo: replay of a held-out day)"),
):
    """Each option lists: walk to the first stop, boarding stop, route, wait, ride, alighting stop,
    transfers, walk to the destination, total time, a reliability range, the same trip at a quieter
    hour, a 24-hour profile, the share of the estimate backed by GPS, and crowding per leg."""
    return _resp(H.journeys(from_, to, depart, mode, n, live))


@app.get("/matrix", tags=["planner"], summary="Travel-time matrix (like a distance matrix API)")
def matrix(
    origins: str = Query(..., description="'|'-separated 'lat,lon' or stop names"),
    destinations: str = Query(..., description="'|'-separated 'lat,lon' or stop names"),
    depart: str | None = Query(None), mode: str = Query("fastest"),
):
    return _resp(H.matrix(origins, destinations, depart, mode))


@app.get("/stops/search", tags=["stops"], summary="Stop name autocomplete (typo tolerant)")
def stops_search(q: str = Query(..., min_length=1), n: int = Query(8, ge=1, le=20)):
    return _resp(H.stops_search(q, n))


@app.get("/places/search", tags=["stops"],
         summary="Search box: bus stops plus addresses/landmarks (OpenStreetMap via Photon)")
def places_search(q: str = Query(..., min_length=1), n: int = Query(8, ge=1, le=20)):
    return _resp(H.places_search(q, n))


@app.get("/live/vehicles", tags=["live"],
         summary="Buses on the map now (GTFS-RT feed, or the replay day at this time of day)")
def live_vehicles(at: str | None = Query(None, description="HH:MM[:SS] on the replay day; default = now (IST)"),
                  bbox: str | None = Query(None, description="lat0,lon0,lat1,lon1"),
                  limit: int = Query(800, ge=1, le=3000)):
    return _resp(H.live_vehicles(at, bbox, limit))


@app.get("/traffic", tags=["traffic model"], summary="Bus speed on every stretch at a day type and hour")
def traffic(daytype: str = Query("weekday", description="weekday | saturday | sunday"),
            hour: int = Query(18, ge=0, le=23)):
    return _resp(H.traffic(daytype, hour))


@app.get("/insights", tags=["traffic model"], summary="Slowest stretches, peak vs night, timetable vs reality")
def insights():
    return _resp(H.insights())


@app.get("/crowding/route", tags=["crowding"], summary="Crowding along one route by time of day")
def crowding_route(route: str = Query(..., examples=["8UP"]), daytype: str = Query("weekday")):
    return _resp(H.crowding_route(route, daytype))


@app.get("/health", tags=["system"], summary="Server and model status")
def health():
    return _resp(H.health())


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(WEB / "index.html")


if (WEB / "static").exists():
    app.mount("/static", StaticFiles(directory=WEB / "static"), name="static")
