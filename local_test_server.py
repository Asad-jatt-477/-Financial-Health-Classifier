"""
LOCAL TESTING ONLY -- not part of the Vercel deployment.
Serves the FastAPI backend (api/) and the static frontend (public/) together
on one port, exactly mirroring how Vercel will route them in production, but
without needing a Vercel account or project link.

Run:  python local_test_server.py
Then open: http://localhost:8000
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "api"))

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import uvicorn

from index import app as api_app  # noqa: E402  (api/index.py)

app = FastAPI()
# api_app's own routes are already full paths (/api/config, /api/predict, ...),
# so they're added directly rather than mounted under an extra prefix.
app.router.routes.extend(api_app.router.routes)
# Anything that isn't an /api/* route falls through to the static frontend.
app.mount("/", StaticFiles(directory="public", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)