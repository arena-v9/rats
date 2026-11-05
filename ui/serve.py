"""
Serves the viewer: ui/index.html, plus /summary.json built fresh from runs/ on each request (every run so far, and the
analysis of the finished ones). The analysis is recomputed only when the set of finished runs changes. For the replay
(ui/replay.html): /episodes.json?group=G&seed=S lists a run's recorded episodes, /episode.json?group=G&seed=S&i=N
gives one episode step by step.

    .venv/bin/python ui/serve.py        # then open http://localhost:8000
"""

import argparse
import json
import sys
from functools import partial
from urllib.parse import parse_qs, urlsplit
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

UI_DIR = Path(__file__).parent
sys.path.insert(0, str(UI_DIR.parent))

from analyze import analysis, finished, load_all  # noqa: E402
from recording import FILE_NAME, episode, episodes, read  # noqa: E402
from train import GROUPS, RUNS_DIR  # noqa: E402

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


_recordings: dict[Path, tuple] = {}  # path -> (file size, records, episode list), reread when the file grows


def recording(query: dict) -> tuple:
    """The recording of the run named in the query, or a KeyError/ValueError for a bad or missing one."""
    group, seed = query["group"][0], int(query["seed"][0])
    if group not in GROUPS:
        raise KeyError(group)
    path = RUNS_DIR / group / f"seed_{seed}" / FILE_NAME
    size = path.stat().st_size
    if path not in _recordings or _recordings[path][0] != size:
        steps = read(path)
        phase1_steps = json.loads((path.parent / "metrics.json").read_text())["phase1_steps"]  # depends on the schedule
        _recordings[path] = (size, steps, episodes(steps, phase1_steps), phase1_steps)
    return _recordings[path]


def episode_list(query: dict) -> dict:
    _, steps, listing, phase1_steps = recording(query)
    return {"phase1_steps": phase1_steps, "steps": len(steps), **listing}


def one_episode(query: dict) -> dict:
    _, steps, listing, _ = recording(query)
    i = int(query["i"][0])
    if not 0 <= i < listing["count"]:
        raise KeyError(i)
    start = listing["start"][i]
    return {"i": i, "start": start, **episode(steps, start, listing["length"][i])}


ROUTES = {"/summary.json": lambda query: _no_nan(summary()), "/episodes.json": episode_list,
          "/episode.json": one_episode}


class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        url = urlsplit(self.path)
        if url.path not in ROUTES:
            return super().do_GET()
        try:
            content = ROUTES[url.path](parse_qs(url.query))
        except (KeyError, ValueError, FileNotFoundError):
            self.send_error(404, "No such run, recording or episode")
            return
        body = json.dumps(content, allow_nan=False).encode()
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
