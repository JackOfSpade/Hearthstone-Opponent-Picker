"""Local web dashboard for hop.

Deliberately built on the Python standard library (``http.server``) so it needs
no FastAPI/uvicorn install - the dashboard is always available. It serves a
single self-contained page that polls a JSON status endpoint and posts control
actions (start / stop / acknowledge alarm / set criteria), plus a throttled
live-view of the phone screen.
"""

from .server import DashboardServer, serve

__all__ = ["DashboardServer", "serve"]
