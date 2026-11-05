"""
The analysis of finished runs (runs/*/seed_*/metrics.json): the pilot check, then experiment 1 (never shocked minus
shocked, each group against no effect, the checkpoint curves and the verdict on H1). The main measure is the lit-lever
share early in phase 2 (mean of the 120k, 140k and 160k checkpoints); the final checkpoint and the curves show whether
any avoidance fades. Prints it as Markdown for
RESULTS.md; the viewer (ui/serve.py) shows the same analysis.

    .venv/bin/python analyze.py
"""

import json

import numpy as np

from stats import compare, compare_paired
from train import GROUPS, RUNS_DIR

REFERENCE, OTHER, PILOT = "shocked", "never_shocked", "pilot_self_shock"
SEEDS_PER_GROUP = 10  # planned runs per experiment-1 group
PILOT_MAX_LIT_SHARE, PILOT_MAX_DEATH_RATE = 0.2, 0.05
END_OF_PHASE1, EARLY, FINAL = slice(0, 1), slice(1, 4), slice(-1, None)  # checkpoints: 100k; 120k–160k; 300k
# (label, metrics key, which checkpoints (averaged), the value meaning "no effect", display scale); the first is the main one
MEASURES = [
    ("Lit-lever share, % (120k–160k, main)", "lit_share", EARLY, 0.5, 100),
    ("Lit-lever share, % (final)", "lit_share", FINAL, 0.5, 100),
    ("Q-gap, lit − unlit (120k–160k)", "q_gap", EARLY, 0.0, 1),
    ("Q-gap, lit − unlit (final)", "q_gap", FINAL, 0.0, 1),
    ("Scream probe (end of phase 1)", "scream_probe", END_OF_PHASE1, 0.0, 1),
    ("Scream probe (final)", "scream_probe", FINAL, 0.0, 1),
]
CURVE_KEYS = ("lit_share", "q_gap", "scream_probe")


def load_all() -> dict[str, list[dict]]:
    """Every run so far, finished or not, by group and sorted by seed."""
    load = lambda group: [json.loads(p.read_text()) for p in (RUNS_DIR / group).glob("seed_*/metrics.json")]
    return {group: sorted(load(group), key=lambda run: run["seed"]) for group in GROUPS}


def finished(runs_by_group: dict[str, list[dict]]) -> dict[str, list[dict]]:
    return {group: [run for run in runs if run["status"] == "done"] for group, runs in runs_by_group.items()}


def full_schedule(runs: list[dict]) -> list[dict]:
    """Experiment 1's measures are defined on the full schedule's checkpoints (120k–160k, 300k); quick runs are left out."""
    return [run for run in runs if run.get("schedule", "full") == "full"]


def values(runs: list[dict], key: str, checkpoints: slice) -> np.ndarray:
    """One value per run: the mean over the chosen checkpoints (a share of None, never pressed, counts as NaN)."""
    return np.array([np.array([c[key] for c in run["checkpoints"][checkpoints]], dtype=float).mean() for run in runs])


def pilot(runs: list[dict]) -> dict | None:
    if not runs:
        return None
    rows = []
    for run in runs:
        late = run["episodes"]["phase1_late"]
        share, death_rate = run["checkpoints"][-1]["lit_share"], late["deaths"] / max(late["episodes"], 1)
        rows.append({"seed": run["seed"], "lit_share": share, "death_rate": death_rate, "minutes": run["seconds"] / 60,
                     "passed": share is not None and share < PILOT_MAX_LIT_SHARE and death_rate < PILOT_MAX_DEATH_RATE})
    return {"runs": rows, "passed": all(row["passed"] for row in rows),
            "max_lit_share": PILOT_MAX_LIT_SHARE, "max_death_rate": PILOT_MAX_DEATH_RATE}


def experiment_1(shocked: list[dict], never: list[dict], rng: np.random.Generator) -> dict | None:
    """Needs at least 2 finished runs per group."""
    if len(shocked) < 2 or len(never) < 2:
        return None
    comparisons, against_null = [], []
    for label, key, checkpoints, null, scale in MEASURES:
        a, b = values(shocked, key, checkpoints), values(never, key, checkpoints)
        comparisons.append({"label": label, "key": key, "scale": scale, **compare(a, b, rng)})
        for group, v in ((REFERENCE, a), (OTHER, b)):
            result = compare_paired(np.full(len(v), null), v, rng)
            against_null.append({"label": label, "key": key, "group": group, "scale": scale, "null": null,
                                 "runs_below": int((v < null).sum()), "runs": len(v), **result})
    curve = [{"step": point["step"],
              **{f"{group}_{key}": float(values(runs, key, slice(i, i + 1)).mean())
                 for group, runs in ((REFERENCE, shocked), (OTHER, never)) for key in CURVE_KEYS}}
             for i, point in enumerate(shocked[0]["checkpoints"])]
    return {"runs": {REFERENCE: len(shocked), OTHER: len(never)}, "comparisons": comparisons,
            "against_null": against_null, "checkpoints": curve, "verdict": verdict(comparisons[0], shocked, never)}


def verdict(share: dict, shocked: list[dict], never: list[dict]) -> dict:
    """
    H1: the shocked group's lit-lever share (main measure, 120k–160k) is clearly below the never-shocked group's, with
    most shocked runs below 50%.
    """
    below = int((values(shocked, "lit_share", EARLY) < 0.5).sum())
    all_in = len(shocked) >= SEEDS_PER_GROUP and len(never) >= SEEDS_PER_GROUP
    if share["clear"] and share["difference"] > 0 and below > len(shocked) / 2:
        status = "supported"
    elif share["clear"] and share["difference"] < 0:
        status = "opposite"  # shocked agents press the lit lever more
    else:
        status = "not supported" if all_in else "pending"
    return {"status": status, "shocked_runs_below_half": below, "shocked_runs": len(shocked), "all_runs_in": all_in}


def analysis(done: dict[str, list[dict]]) -> dict:
    rng = np.random.default_rng(0)  # fixed, so the same runs always give the same intervals
    return {"pilot": pilot(done[PILOT]),
            "experiment_1": experiment_1(full_schedule(done[REFERENCE]), full_schedule(done[OTHER]), rng)}


def interval(result: dict) -> str:
    low, high = (result["scale"] * x for x in result["ci95"])
    return (f"{result['scale'] * result['difference']:+.2f} [{low:+.2f}, {high:+.2f}] | p = {result['p_value']:.3f} | "
            f"{'clear' if result['clear'] else 'not clear'}")


def markdown(results: dict) -> str:
    lines = []
    if pilot_results := results["pilot"]:
        lines += ["## Pilot: the lit lever shocks the agent itself", "",
                  "| Seed | Lit-lever share | Phase-1 deaths (after 50k) | Minutes | Passed |", "|---|---|---|---|---|"]
        for row in pilot_results["runs"]:
            share = "never pressed" if row["lit_share"] is None else f"{100 * row['lit_share']:.1f}%"
            lines.append(f"| {row['seed']} | {share} | {100 * row['death_rate']:.1f}% | {row['minutes']:.1f} | "
                         f"{'yes' if row['passed'] else 'no'} |")
        lines += ["", f"Pilot {'passed' if pilot_results['passed'] else 'failed'}.", ""]
    if e1 := results["experiment_1"]:
        lines += [f"## Experiment 1: {e1['runs'][REFERENCE]} shocked vs {e1['runs'][OTHER]} never-shocked runs", "",
                  "| Measure | Shocked | Never shocked | Never − shocked [95% CI] | p | Verdict |",
                  "|---|---|---|---|---|---|"]
        for c in e1["comparisons"]:
            lines.append(f"| {c['label']} | {c['scale'] * c['a']:.2f} | {c['scale'] * c['b']:.2f} | {interval(c)} |")
        lines += ["", "| Measure | Group | Mean − no effect [95% CI] | p | Verdict | Runs below |", "|---|---|---|---|---|---|"]
        for n in e1["against_null"]:
            lines.append(f"| {n['label']} | {n['group']} | {interval(n)} | {n['runs_below']}/{n['runs']} |")
        lines += ["", "| Step | Shocked: share, Q-gap, scream probe | Never shocked: share, Q-gap, scream probe |",
                  "|---|---|---|"]
        for p in e1["checkpoints"]:
            cells = [f"{100 * p[f'{g}_lit_share']:.1f}%, {p[f'{g}_q_gap']:+.3f}, {p[f'{g}_scream_probe']:+.2f}"
                     for g in (REFERENCE, OTHER)]
            lines.append(f"| {p['step'] // 1000}k | {cells[0]} | {cells[1]} |")
        lines += ["", f"H1: {e1['verdict']['status']}."]
    return "\n".join(lines) if lines else "No finished runs yet."


if __name__ == "__main__":
    print(markdown(analysis(finished(load_all()))))
