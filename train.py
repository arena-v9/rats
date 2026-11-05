"""
Trains one run per (group, seed): phase 1 (pre-exposure) then phase 2 (the lever task), evaluating at the end of phase
1 and regularly through phase 2. Results go to runs/<group>/seed_<seed>/: metrics.json (rewritten at every log point,
so the viewer, ui/serve.py, can follow a run while it trains) and recording.bin (every step, for the replay).

Two schedules: "full" (100k + 200k steps) and "quick" (40k + 80k), each with its own main window for the analysis.
With --experiment, runs go to runs/<experiment>/buf<size>k/<group>/seed_<seed>/, one collection per replay buffer size
(--buffers), so a buffer ablation is one command. Each run records its schedule, buffer size and code commit.

    .venv/bin/python train.py --groups shocked never_shocked --seeds {1..10}
    .venv/bin/python train.py --groups pilot_self_shock --seeds 1 2 3 4 5 --schedule quick
    .venv/bin/python train.py --experiment exp1 --buffers 20000 50000 100000 --schedule quick \
        --groups shocked never_shocked pilot_self_shock --seeds {1..10} --pilot-seeds 1 2 3
"""

import argparse
import copy
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch as t

from agent import BATCH_SIZE, BUFFER_SIZE, LEARNING_RATE, MAX_GRAD_NORM, QNetwork, ReplayBuffer, train_step
from env import EPISODE_STEPS, NUM_ACTIONS, OBSERVATION, REST, TIME_COMPRESSION, LeverTask, per_step, steps
from evaluate import evaluate
from exact import exact_trial_q
from recording import FILE_NAME, Recorder

# Each group changes one thing. The pilot checks the agent can learn to avoid a lit lever that shocks itself.
# Random shocks: 0.004 per uncompressed step, i.e. 0.1 per trial interval
GROUPS = {
    "shocked": {"phase1_shock_prob": per_step(0.004), "phase2_target": "other"},
    "never_shocked": {"phase1_shock_prob": 0.0, "phase2_target": "other"},
    "pilot_self_shock": {"phase1_shock_prob": per_step(0.004), "phase2_target": "self"},
}


def _schedule(phase1, phase2, warmup, epsilon_end, checkpoint_every, log_every, main_window) -> dict:
    """A schedule written in uncompressed steps (as in version 3.3), in steps now."""
    return {"phase1_steps": steps(phase1), "phase2_steps": steps(phase2), "warmup_steps": steps(warmup),
            "epsilon_end_step": steps(epsilon_end), "checkpoint_every": steps(checkpoint_every),
            "log_every": steps(log_every), "main_window": tuple(steps(x) for x in main_window)}


SCHEDULES = {
    "full": _schedule(100_000, 200_000, 20_000, 50_000, 20_000, 5_000, (120_000, 160_000)),
    "quick": _schedule(40_000, 80_000, 10_000, 25_000, 5_000, 1_000, (45_000, 60_000)),
}
# warmup: random actions fill the buffer before training starts; checkpoint_every: phase-2 steps between greedy
# evaluations; log_every: steps between training-curve points; main_window: the checkpoints the main measure averages
# (full: 120k, 140k, 160k uncompressed; quick: 45k to 60k, the first 20k uncompressed steps of phase 2)
EPSILON_START, EPSILON_END = 1.0, 0.02
# Gradient updates per environment step: 1, so 10 updates per trial (version 3.3: 25). Version 4.0 compressed further
# (5 steps per trial): with 1 update per step its values were less accurate; 5 per step matched version 3.3 but cost
# as much. The option stays for ablations.
UPDATES_PER_STEP = 1
TARGET_SYNC_STEPS = steps(200)  # the target network syncs every 8 trials (200 uncompressed steps) of updates
MAX_PARALLEL = min(10, os.cpu_count() or 1)  # one core per run
RUNS_DIR = Path(__file__).parent / "runs"

LEVERS_ACTIVE, LIT_B = OBSERVATION.index("levers_active"), OBSERVATION.index("lit_B")  # lit_B is 1 when B is lit
SCREAM = OBSERVATION.index("scream")
# Kinds of transition, for the TD errors: what the update teaches about each
TD_KINDS = ("lit_press", "unlit_press", "trial_rest", "scream", "death", "other")


def epsilon(step: int, schedule: dict) -> float:
    progress = step / schedule["epsilon_end_step"]
    return max(EPSILON_END, EPSILON_START + (EPSILON_END - EPSILON_START) * progress)


def commit() -> str:
    """The code version: the commit hash, with "+changes" if tracked files differ from it."""
    git = lambda *args: subprocess.run(["git", *args], cwd=Path(__file__).parent, capture_output=True, text=True).stdout
    head = git("rev-parse", "--short", "HEAD").strip() or "unknown"
    return head + ("+changes" if git("status", "--porcelain", "--untracked-files=no").strip() else "")


def collection_for(experiment: str | None, buffer_size: int) -> str:
    return f"{experiment}/buf{buffer_size // 1000}k" if experiment else ""


def train(group: str, seed: int, schedule_name: str = "full", buffer_size: int = BUFFER_SIZE,
          collection: str = "", updates_per_step: int = UPDATES_PER_STEP) -> None:
    settings, schedule = GROUPS[group], SCHEDULES[schedule_name]
    phase1_steps, warmup = schedule["phase1_steps"], schedule["warmup_steps"]
    run_dir = RUNS_DIR / collection / group / f"seed_{seed}"
    code_version = commit()
    target_sync_updates = TARGET_SYNC_STEPS * updates_per_step  # the same number of trials between syncs
    run_dir.mkdir(parents=True, exist_ok=True)
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    t.manual_seed(seed)
    env_rng, agent_rng = np.random.default_rng(seed), np.random.default_rng([seed, 1])

    q_net = QNetwork().to(device)
    target_net = copy.deepcopy(q_net)
    optimizer = t.optim.Adam(q_net.parameters(), lr=LEARNING_RATE)
    buffer = ReplayBuffer(agent_rng, capacity=buffer_size)
    total_steps = phase1_steps + schedule["phase2_steps"]
    episodes = {"phase1_late": {"episodes": 0, "deaths": 0}, "phase2": {"episodes": 0, "deaths": 0}}
    checkpoints, training, updates, start = [], [], 0, time.time()
    window = new_window()  # what happened since the last training-curve point
    recorder = Recorder(run_dir / FILE_NAME)
    episode_return = 0.0

    def checkpoint(step: int) -> None:
        checkpoints.append({"step": step, **evaluate(q_net, settings["phase2_target"], device)})

    def save(step: int, status: str) -> None:
        recorder.flush()
        metrics = {"group": group, "seed": seed, "settings": settings, "schedule": schedule_name,
                   "buffer_size": buffer_size, "collection": collection, "commit": code_version, "status": status,
                   "step": step, "total_steps": total_steps, "phase1_steps": phase1_steps,
                   "main_window": schedule["main_window"], "time_compression": TIME_COMPRESSION,
                   "episode_steps": EPISODE_STEPS, "updates_per_step": updates_per_step,
                   "target_sync_updates": target_sync_updates,
                   "seconds": round(time.time() - start), "episodes": episodes, "training": training,
                   "checkpoints": checkpoints}
        temporary = run_dir / "metrics.json.tmp"
        temporary.write_text(json.dumps(metrics, indent=2))
        temporary.replace(run_dir / "metrics.json")  # atomic, so the viewer never reads half a file

    env = LeverTask(env_rng, settings["phase1_shock_prob"], lit_target=None)
    obs = env.reset()
    for step in range(total_steps):
        if step == phase1_steps:
            checkpoint(step)
            t.save(q_net.state_dict(), run_dir / "end_of_phase1.pt")
            env = LeverTask(env_rng, 0.0, lit_target=settings["phase2_target"])
            obs = env.reset()

        # Same draws from agent_rng as without the recording: the Q-values are computed every step but use no randomness
        explored = step < warmup or agent_rng.random() < epsilon(step, schedule)
        with t.no_grad():
            q = q_net(t.as_tensor(obs[None], device=device))[0]
        action = int(agent_rng.integers(NUM_ACTIONS)) if explored else int(q.argmax())
        before = ((env.agent.health, env.agent.energy), (env.other.health, env.other.energy))
        if env.levers_active:
            count_trial(window, action, env.lit_lever, explored)
        next_obs, reward, terminated, truncated = env.step(action)
        window["agent_shocks"] += env.events["agent_shock"] > 0
        window["other_shocks"] += env.events["other_shock"]
        recorder.add(env, action, explored, q.cpu().numpy(), before, reward, terminated, truncated)
        buffer.add(obs, action, reward, next_obs, terminated)
        obs, episode_return = next_obs, episode_return + reward
        if terminated or truncated:
            phase = "phase2" if step >= phase1_steps else "phase1_late" if step >= schedule["epsilon_end_step"] else None
            if phase:
                episodes[phase]["episodes"] += 1
                episodes[phase]["deaths"] += terminated
            window["returns"].append(episode_return)
            window["deaths"] += terminated
            obs, episode_return = env.reset(), 0.0

        if step >= warmup:
            for _ in range(updates_per_step):
                batch = buffer.sample(BATCH_SIZE, device)
                count_update(window, train_step(q_net, target_net, optimizer, batch), batch)
                updates += 1
                if updates % target_sync_updates == 0:
                    window["target_drift"].append(target_drift(q_net, target_net, device))
                    target_net.load_state_dict(q_net.state_dict())
        if step >= phase1_steps and (step + 1 - phase1_steps) % schedule["checkpoint_every"] == 0:
            checkpoint(step + 1)
        if (step + 1) % schedule["log_every"] == 0:
            lit_target = settings["phase2_target"] if step >= phase1_steps else None
            training.append({**summarize_window(step + 1, epsilon(step, schedule), window),
                             **diagnostics(q_net, target_net, buffer, device, lit_target, step + 1 - phase1_steps)})
            window = new_window()
            save(step + 1, "running")

    t.save(q_net.state_dict(), run_dir / "final.pt")
    save(total_steps, "done")


def new_window() -> dict:
    return {"returns": [], "deaths": 0, "losses": [], "grad_norms": [], "clipped": 0, "q": [], "target": [],
            "trials": 0, "presses": 0, "lit_presses": 0, "lit_presses_greedy": 0, "lit_presses_explored": 0,
            "greedy_trials": 0, "agent_shocks": 0, "other_shocks": 0,
            "target_drift": [], "td": {kind: [0, 0.0, 0.0] for kind in TD_KINDS}}  # td: [count, sum, sum of squares]


def count_trial(window: dict, action: int, lit_lever: int, explored: bool) -> None:
    window["trials"] += 1
    window["greedy_trials"] += not explored
    if action != REST:
        window["presses"] += 1
        if action == lit_lever:
            window["lit_presses"] += 1
            window["lit_presses_explored" if explored else "lit_presses_greedy"] += 1


def count_update(window: dict, stats: dict, batch: tuple[t.Tensor, ...]) -> None:
    """The update's loss, gradient and TD errors (target − prediction: positive means the network is too low), by kind."""
    window["losses"].append(stats["loss"])
    window["grad_norms"].append(stats["grad_norm"])
    window["clipped"] += stats["grad_norm"] > MAX_GRAD_NORM
    window["q"].append(float(stats["q"].mean()))
    window["target"].append(float(stats["target"].mean()))
    obs, action, _, next_obs, terminated = (x.cpu().numpy() for x in batch)
    trial = obs[:, LEVERS_ACTIVE] == 1
    lit = trial & (action == obs[:, LIT_B])
    unlit = trial & (action != REST) & ~lit
    rest = trial & (action == REST)
    scream = (next_obs[:, SCREAM] == 1) & ~trial
    death = terminated == 1
    kinds = {"lit_press": lit, "unlit_press": unlit, "trial_rest": rest, "scream": scream & ~death, "death": death}
    kinds["other"] = ~(lit | unlit | rest | scream | death)
    error = stats["target"] - stats["q"]
    for kind, mask in kinds.items():
        tally = window["td"][kind]
        tally[0] += int(mask.sum())
        tally[1] += float(error[mask].sum())
        tally[2] += float((error[mask] ** 2).sum())


def summarize_window(step: int, current_epsilon: float, window: dict) -> dict:
    """One training-curve point. Rates are over the episodes and trials that ended in the window (exploration included)."""
    mean = lambda xs: float(np.mean(xs)) if xs else None
    ratio = lambda a, b: a / b if b else None
    td = {}
    for kind, (count, total, squares) in window["td"].items():
        td[f"td_bias_{kind}"] = total / count if count else None  # mean target − prediction
        td[f"td_rmse_{kind}"] = float(np.sqrt(squares / count)) if count else None
        td[f"td_count_{kind}"] = count
    return {
        "step": step,
        "epsilon": current_epsilon,
        "episode_return": mean(window["returns"]),
        "death_rate": ratio(window["deaths"], len(window["returns"])),
        "deaths": window["deaths"],
        "loss": mean(window["losses"]),
        "grad_norm": mean(window["grad_norms"]),
        "grad_norm_max": max(window["grad_norms"]) if window["grad_norms"] else None,
        "clip_share": ratio(window["clipped"], len(window["grad_norms"])),  # updates whose gradient was clipped
        "batch_q": mean(window["q"]),
        "batch_target": mean(window["target"]),
        "target_drift": mean(window["target_drift"]),  # mean |online − target| on the probe grid just before each sync
        "lit_share": ratio(window["lit_presses"], window["presses"]),
        "trials": window["trials"],
        "presses": window["presses"],
        "lit_presses_greedy": window["lit_presses_greedy"],
        "lit_presses_explored": window["lit_presses_explored"],
        "greedy_trials": window["greedy_trials"],
        "agent_shocks": window["agent_shocks"],
        "other_shocks": window["other_shocks"],
        **td,
    }


@t.no_grad()
def target_drift(q_net, target_net, device: t.device) -> float:
    """How far the online network has moved from the target network since the last sync, on the probe grid."""
    probe = t.as_tensor(PROBE, device=device)
    return float((q_net(probe) - target_net(probe)).abs().mean())


# A fixed grid of trial states for probing the network: health × energy × which lever is lit, nobody screaming
PROBE_HEALTH, PROBE_ENERGY = np.linspace(0.4, 1.0, 7), np.linspace(0.1, 1.0, 10)
PROBE = np.array([[h, e, 0.0, 1.0, lit == 0, lit == 1] for lit in (0, 1) for h in PROBE_HEALTH for e in PROBE_ENERGY],
                 dtype=np.float32)
PROBE_LIT = PROBE[:, LIT_B].astype(int)  # 0 = A lit, 1 = B lit
PROBE_LOW = PROBE[:, 0] <= 0.6  # health at or below 0.6: a shock can kill


@t.no_grad()
def diagnostics(q_net, target_net, buffer: ReplayBuffer, device: t.device, lit_target: str | None,
                phase2_steps_done: int) -> dict:
    """
    The network and the buffer at a log point. Uses no randomness. In phase 2, the probe grid is also compared with
    the exact Q-values (exact.py): `lit_target` is the phase-2 target, or None in phase 1 (values not exact there).
    """
    obs, action, reward = buffer.obs[: buffer.size], buffer.action[: buffer.size], buffer.reward[: buffer.size]
    trial = obs[:, LEVERS_ACTIVE] == 1
    lit = trial & (action == obs[:, LIT_B])
    unlit = trial & (action != REST) & ~lit
    rows = np.arange(len(PROBE))
    probe = t.as_tensor(PROBE, device=device)
    q = q_net(probe).cpu().numpy()
    q_lit, q_unlit = q[rows, PROBE_LIT], q[rows, 1 - PROBE_LIT]
    gap = q_lit - q_unlit
    result = {
        "buffer_size": buffer.size,
        "buffer_phase1_share": max(0, buffer.size - max(0, phase2_steps_done)) / buffer.size if buffer.size else None,
        "buffer_lit_presses": int(lit.sum()),
        "buffer_unlit_presses": int(unlit.sum()),
        "buffer_lit_press_reward": float(reward[lit].mean()) if lit.any() else None,
        "buffer_deaths": int(buffer.terminated[: buffer.size].sum()),
        "probe_q_lit": float(q_lit.mean()),
        "probe_q_unlit": float(q_unlit.mean()),
        "probe_q_rest": float(q[:, REST].mean()),
        "probe_gap": float(gap.mean()),
        "probe_lit_preferred": float((gap > 0).mean()),  # share of probe states where Q(lit) > Q(unlit)
        "probe_lit_preferred_A": float((gap[PROBE_LIT == 0] > 0).mean()),  # A lit
        "probe_lit_preferred_B": float((gap[PROBE_LIT == 1] > 0).mean()),
        "probe_lit_preferred_low_health": float((gap[PROBE_LOW] > 0).mean()),
        "probe_lit_preferred_high_health": float((gap[~PROBE_LOW] > 0).mean()),
        **{f"weight_norm_{i}": float(layer.weight.norm()) for i, layer in enumerate(q_net.layers) if hasattr(layer, "weight")},
    }
    if lit_target is not None:  # phase 2: compare with the exact values
        exact = np.array([exact_trial_q(h, e, lit_target) for h, e in PROBE[:, :2]])  # columns: lit, unlit, rest
        exact_gap = exact[:, 0] - exact[:, 1]
        for name, mask in (("", slice(None)), ("_low_health", PROBE_LOW), ("_high_health", ~PROBE_LOW)):
            result[f"probe_exact_gap{name}"] = float(exact_gap[mask].mean())
            result[f"probe_gap_error{name}"] = float((gap - exact_gap)[mask].mean())  # learned − exact
            result[f"probe_q_unlit_error{name}"] = float((q_unlit - exact[:, 1])[mask].mean())
            result[f"probe_q_lit_error{name}"] = float((q_lit - exact[:, 0])[mask].mean())
            # Where the exact gap is negative (pressing lit is worse), does the network agree?
            worse = (exact_gap < -1e-9) & (np.ones(len(PROBE), bool) if isinstance(mask, slice) else mask)
            result[f"probe_gap_sign_correct{name}"] = float((gap[worse] < 0).mean()) if worse.any() else None
    return result


def launch(jobs: list[tuple[str, int, int, str]], schedule_name: str, updates_per_step: int) -> None:
    """
    Runs each job (group, seed, buffer size, collection) in its own process, at most MAX_PARALLEL at once; each logs
    to its run directory.
    """
    processes = []
    for group, seed, buffer_size, collection in jobs:
        while sum(p.poll() is None for p in processes) >= MAX_PARALLEL:
            time.sleep(5)
        log = RUNS_DIR / collection / group / f"seed_{seed}" / "train.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, __file__, "--groups", group, "--seeds", str(seed), "--schedule", schedule_name,
                   "--buffers", str(buffer_size), "--collection", collection,
                   "--updates-per-step", str(updates_per_step)]
        processes.append(subprocess.Popen(command, stdout=log.open("w"), stderr=subprocess.STDOUT))
    failed = [job for job, p in zip(jobs, processes) if p.wait() != 0]
    print(f"{len(jobs) - len(failed)}/{len(jobs)} runs finished" + (f"; failed (see train.log): {failed}" if failed else ""))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--groups", nargs="+", choices=GROUPS, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--schedule", choices=SCHEDULES, default="full")
    parser.add_argument("--updates-per-step", type=int, default=UPDATES_PER_STEP)
    parser.add_argument("--buffers", nargs="+", type=int, default=[BUFFER_SIZE], help="replay buffer sizes")
    parser.add_argument("--experiment", help="name; runs go to runs/<experiment>/buf<size>k/")
    parser.add_argument("--pilot-seeds", nargs="+", type=int, help="seeds for pilot_self_shock (default: --seeds)")
    parser.add_argument("--collection", help=argparse.SUPPRESS)  # set by launch() for a single job
    args = parser.parse_args()
    if len(args.buffers) > 1 and not args.experiment:
        parser.error("several buffer sizes need --experiment, so their runs go to separate folders")
    # Buffer by buffer, then seed by seed, alternating groups, so every batch holds the groups equally
    jobs = [(group, seed, buffer_size, collection_for(args.experiment, buffer_size) if args.collection is None
             else args.collection)
            for buffer_size in args.buffers for seed in sorted(set(args.seeds) | set(args.pilot_seeds or []))
            for group in args.groups
            if seed in (args.pilot_seeds or args.seeds if group == "pilot_self_shock" else args.seeds)]
    if len(jobs) == 1:
        t.set_num_threads(1)
        train(jobs[0][0], jobs[0][1], args.schedule, jobs[0][2], jobs[0][3], args.updates_per_step)
    else:
        launch(jobs, args.schedule, args.updates_per_step)


if __name__ == "__main__":
    main()
