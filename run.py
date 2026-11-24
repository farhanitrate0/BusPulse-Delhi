"""One-command start: python run.py [--port 8000] [--host 0.0.0.0]
Uses FastAPI/uvicorn when installed, otherwise the zero-dependency stdlib server.
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)))
    ap.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    a = ap.parse_args()
    try:
        import uvicorn  # noqa: F401
        import fastapi  # noqa: F401
    except ImportError:
        sys.argv = [sys.argv[0], "--port", str(a.port), "--host", a.host]
        from app.serve_stdlib import main
        main()
    else:
        print(f"BusPulse Delhi → http://{'localhost' if a.host in ('127.0.0.1', '0.0.0.0') else a.host}:{a.port}"
              f"  (API docs at /docs)")
        uvicorn.run("app.api:app", host=a.host, port=a.port, log_level="warning")
