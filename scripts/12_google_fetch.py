"""Fill the Google Maps benchmark automatically with the Google Routes API (transit, bus only).

    python scripts/12_google_fetch.py                    # all rows of the template
    python scripts/12_google_fetch.py --times 08:30,18:00

The API key is read from env GOOGLE_MAPS_API_KEY or from data/google_key.txt (data/ is gitignored, so
the key never reaches the repo). Each template row is asked for the next Tuesday at its departure time
(IST), bus only, and the top route's total duration + bus numbers go into data/google_benchmark.csv,
which scripts/05_google_compare.py then compares. Raw responses are kept in data/model/google_raw.json
so every number can be audited.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

URL = "https://routes.googleapis.com/directions/v2:computeRoutes"
FIELDS = ",".join([
    "routes.duration",
    "routes.legs.steps.travelMode",
    "routes.legs.steps.transitDetails.transitLine.nameShort",
    "routes.legs.steps.transitDetails.transitLine.name",
    "routes.legs.steps.transitDetails.transitLine.vehicle.type",
    "routes.legs.steps.transitDetails.stopDetails.departureTime",
])
IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


def api_key() -> str:
    k = os.environ.get("GOOGLE_MAPS_API_KEY", "").strip()
    f = config.DATA / "google_key.txt"
    if not k and f.exists():
        k = f.read_text(encoding="utf-8").strip()
    if not k:
        sys.exit(f"No API key: put it in {f} (one line) or set GOOGLE_MAPS_API_KEY.")
    return k


def next_date(weekday: str) -> dt.date:
    today = dt.datetime.now(IST).date()
    want = WEEKDAYS.index(weekday.strip().lower())
    return today + dt.timedelta(days=(want - today.weekday()) % 7 or 7)  # always a future day


def ask(key: str, r, when: dt.datetime) -> dict:
    body = {
        "origin": {"location": {"latLng": {"latitude": float(r.origin_lat), "longitude": float(r.origin_lon)}}},
        "destination": {"location": {"latLng": {"latitude": float(r.dest_lat), "longitude": float(r.dest_lon)}}},
        "travelMode": "TRANSIT",
        "departureTime": when.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "transitPreferences": {"allowedTravelModes": ["BUS"]},
        "languageCode": "en",
    }
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), method="POST", headers={
        "Content-Type": "application/json", "X-Goog-Api-Key": key, "X-Goog-FieldMask": FIELDS})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        msg = e.read().decode(errors="replace")
        if e.code in (400, 401, 403):
            sys.exit(f"Google refused the request ({e.code}):\n{msg}")
        return {"error": f"{e.code}: {msg[:300]}"}


def summarise(resp: dict) -> tuple[float | None, str, str]:
    routes = resp.get("routes") or []
    if not routes:
        return None, "", "no route"
    top = routes[0]
    minutes = round(int(str(top["duration"]).rstrip("s")) / 60, 1)
    lines, modes, first_dep = [], set(), ""
    for leg in top.get("legs", []):
        for s in leg.get("steps", []):
            td = s.get("transitDetails")
            if not td:
                continue
            ln = td.get("transitLine", {})
            lines.append(ln.get("nameShort") or ln.get("name") or "?")
            modes.add(ln.get("vehicle", {}).get("type", "?"))
            first_dep = first_dep or td.get("stopDetails", {}).get("departureTime", "")
    note = "" if modes <= {"BUS"} else "non-bus: " + "/".join(sorted(modes))
    if not lines:
        note = "walk only"
    return minutes, " + ".join(lines), note


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", default=str(config.DATA / "google_benchmark_template.csv"))
    ap.add_argument("--times", default="", help="comma-separated departure times to fetch (default: all)")
    a = ap.parse_args()
    key = api_key()
    df = pd.read_csv(a.template, dtype={"depart_time": str})
    if a.times:
        df = df[df.depart_time.isin(a.times.split(","))].reset_index(drop=True)
    raw, out = [], []
    for i, r in enumerate(df.itertuples()):
        day = next_date(str(r.weekday))
        hh, mm = map(int, str(r.depart_time)[:5].split(":"))
        when = dt.datetime(day.year, day.month, day.day, hh, mm, tzinfo=IST)
        resp = ask(key, r, when)
        mins, routes, note = summarise(resp) if "error" not in resp else (None, "", resp["error"])
        raw.append({"origin": r.origin_name, "dest": r.dest_name, "depart": when.isoformat(), "response": resp})
        out.append({**r._asdict(), "google_minutes": mins, "google_routes": routes, "google_note": note,
                    "google_query_date": day.isoformat()})
        print(f"{i + 1:3d}/{len(df)} {r.origin_name} -> {r.dest_name} {r.depart_time}: "
              f"{mins if mins is not None else '-'} min {routes} {note}")
        time.sleep(0.15)
    res = pd.DataFrame(out).drop(columns=["Index"])
    res.to_csv(config.GOOGLE_CSV, index=False)
    (config.MODEL_DIR / "google_raw.json").write_text(json.dumps(raw, indent=1), encoding="utf-8")
    got = res.google_minutes.notna()
    print(f"\n{int(got.sum())}/{len(res)} rows with a Google bus route -> {config.GOOGLE_CSV}")
    print(f"raw responses -> {config.MODEL_DIR / 'google_raw.json'}")


if __name__ == "__main__":
    main()
