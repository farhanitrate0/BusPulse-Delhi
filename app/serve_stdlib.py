"""Zero-dependency fallback server (stdlib http.server) with the same endpoints as the FastAPI app.

    python -m app.serve_stdlib --port 8000
"""
from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from app import handlers as H  # noqa: E402


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):  # noqa: N802
        u = urlparse(self.path)
        q = {k: v[-1] for k, v in parse_qs(u.query).items()}
        try:
            if u.path in ("/", "/index.html"):
                return self._send(200, (config.ROOT / "web" / "index.html").read_bytes(), "text/html; charset=utf-8")
            if u.path == "/journeys":
                if "from" not in q or "to" not in q:
                    return self._send(400, {"error": "bad_request", "message": "from and to are required"})
                return self._send(*H.journeys(q["from"], q["to"], q.get("depart"), q.get("mode", "fastest"),
                                              int(q.get("n", 3)), q.get("live", "false").lower() in ("1", "true")))
            if u.path == "/matrix":
                return self._send(*H.matrix(q.get("origins", ""), q.get("destinations", ""), q.get("depart"),
                                            q.get("mode", "fastest")))
            if u.path == "/stops/search":
                return self._send(*H.stops_search(q.get("q", ""), int(q.get("n", 8))))
            if u.path == "/live/vehicles":
                return self._send(*H.live_vehicles(q.get("at"), q.get("bbox"), q.get("limit", 800)))
            if u.path == "/places/search":
                return self._send(*H.places_search(q.get("q", ""), int(q.get("n", 8))))
            if u.path == "/traffic":
                return self._send(*H.traffic(q.get("daytype", "weekday"), int(q.get("hour", 18))))
            if u.path == "/insights":
                return self._send(*H.insights())
            if u.path == "/crowding/route":
                return self._send(*H.crowding_route(q.get("route", ""), q.get("daytype", "weekday")))
            if u.path == "/health":
                return self._send(*H.health())
            if u.path == "/examples":
                return self._send(*H.examples())
            return self._send(404, {"error": "not_found"})
        except ValueError as e:
            return self._send(400, {"error": "bad_request", "message": str(e)})
        except Exception as e:  # noqa: BLE001
            return self._send(500, {"error": "internal", "message": f"{type(e).__name__}: {e}"})

    def log_message(self, fmt, *args):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    H.state()
    H.warm_up()
    print(f"BusPulse (stdlib server) on http://{a.host}:{a.port}")
    ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
