"""
Serves the viewer: ui/index.html, plus /summary.json built fresh from runs/ on each request (every run so far, and the
analysis of the finished ones). The analysis is recomputed only when the set of finished runs changes.

    .venv/bin/python ui/serve.py        # then open http://localhost:8000
"""

import argparse
import json
import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

UI_DIR = Path(__file__).parent
sys.path.insert(0, str(UI_DIR.parent))

from analyze import analysis, finished, load_all  # noqa: E402
from train import GROUPS  # noqa: E402

_cache: dict[str, dict] = {}


def _no_nan(value):
    """JSON has no NaN; a measure with nothing to average (e.g. a run that never pressed) becomes null."""
    if isinstance(value, float) and value != value:
        return None
    if isinstance(value, dict):
        return {k: _no_nan(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_no_nan(v) for v in value]
    return value


def summary() -> dict:
    runs = load_all()
    done = finished(runs)
    key = json.dumps({group: [run["seed"] for run in group_runs] for group, group_runs in done.items()})
    if key not in _cache:
        _cache.clear()
        _cache[key] = analysis(done)
    return {"groups": GROUPS, "runs": runs, "analysis": _cache[key]}


class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path.split("?")[0] != "/summary.json":
            return super().do_GET()
        body = json.dumps(_no_nan(summary()), allow_nan=False).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # keep the terminal quiet
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8000)
    port = parser.parse_args().port
    print(f"Viewer at http://localhost:{port}")
    ThreadingHTTPServer(("", port), partial(Handler, directory=str(UI_DIR))).serve_forever()
