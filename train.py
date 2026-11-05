"""
Trains one run per (group, seed): phase 1 (pre-exposure) then phase 2 (the lever task), evaluating at the end of phase
1 and every 20k steps of phase 2. Results go to runs/<group>/seed_<seed>/; metrics.json is rewritten every 5k steps so
the viewer (ui/serve.py) can follow a run while it trains.

    .venv/bin/python train.py --groups shocked never_shocked --seeds {1..10}   # at most 10 runs at once
"""

import argparse
import copy
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch as t

from agent import BATCH_SIZE, LEARNING_RATE, QNetwork, ReplayBuffer, train_step
from env import NUM_ACTIONS, REST, LeverTask
from evaluate import evaluate

# Each group changes one thing. The pilot checks the agent can learn to avoid a lit lever that shocks itself.
GROUPS = {
    "shocked": {"phase1_shock_prob": 0.004, "phase2_target": "other"},
    "never_shocked": {"phase1_shock_prob": 0.0, "phase2_target": "other"},
    "pilot_self_shock": {"phase1_shock_prob": 0.004, "phase2_target": "self"},
}
PHASE1_STEPS = 100_000
PHASE2_STEPS = 200_000
WARMUP_STEPS = 20_000  # random actions fill the buffer before training starts
EPSILON_START, EPSILON_END, EPSILON_END_STEP = 1.0, 0.02, 50_000
TARGET_UPDATE_EVERY = 200  # gradient steps
CHECKPOINT_EVERY = 20_000  # phase-2 steps
LOG_EVERY = 5_000  # steps between training-curve points
MAX_PARALLEL = 10
RUNS_DIR = Path(__file__).parent / "runs"


def epsilon(step: int) -> float:
    return max(EPSILON_END, EPSILON_START + (EPSILON_END - EPSILON_START) * step / EPSILON_END_STEP)


def train(group: str, seed: int) -> None:
    settings = GROUPS[group]
    run_dir = RUNS_DIR / group / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    t.manual_seed(seed)
    env_rng, agent_rng = np.random.default_rng(seed), np.random.default_rng([seed, 1])

    q_net = QNetwork().to(device)
    target_net = copy.deepcopy(q_net)
    optimizer = t.optim.Adam(q_net.parameters(), lr=LEARNING_RATE)
    buffer = ReplayBuffer(agent_rng)
    total_steps = PHASE1_STEPS + PHASE2_STEPS
    episodes = {"phase1_late": {"episodes": 0, "deaths": 0}, "phase2": {"episodes": 0, "deaths": 0}}
    checkpoints, training, updates, start = [], [], 0, time.time()
    window = new_window()  # what happened since the last training-curve point
    episode_return = 0.0

    def checkpoint(step: int) -> None:
        checkpoints.append({"step": step, **evaluate(q_net, settings["phase2_target"], device)})

    def save(step: int, status: str) -> None:
        metrics = {"group": group, "seed": seed, "settings": settings, "status": status, "step": step,
                   "total_steps": total_steps, "phase1_steps": PHASE1_STEPS, "seconds": round(time.time() - start),
                   "episodes": episodes, "training": training, "checkpoints": checkpoints}
        temporary = run_dir / "metrics.json.tmp"
        temporary.write_text(json.dumps(metrics, indent=2))
        temporary.replace(run_dir / "metrics.json")  # atomic, so the viewer never reads half a file

    env = LeverTask(env_rng, settings["phase1_shock_prob"], lit_target=None)
    obs = env.reset()
    for step in range(total_steps):
        if step == PHASE1_STEPS:
            checkpoint(step)
            t.save(q_net.state_dict(), run_dir / "end_of_phase1.pt")
            env = LeverTask(env_rng, 0.0, lit_target=settings["phase2_target"])
            obs = env.reset()

        if step < WARMUP_STEPS or agent_rng.random() < epsilon(step):
            action = int(agent_rng.integers(NUM_ACTIONS))
        else:
            with t.no_grad():
                action = int(q_net(t.as_tensor(obs[None], device=device)).argmax())
        if env.levers_active and action != REST:  # a press on a trial step
            window["presses"] += 1
            window["lit_presses"] += action == env.lit_lever
        next_obs, reward, terminated, truncated = env.step(action)
        buffer.add(obs, action, reward, next_obs, terminated)
        obs, episode_return = next_obs, episode_return + reward
        if terminated or truncated:
            phase = "phase2" if step >= PHASE1_STEPS else "phase1_late" if step >= EPSILON_END_STEP else None
            if phase:
                episodes[phase]["episodes"] += 1
                episodes[phase]["deaths"] += terminated
            window["returns"].append(episode_return)
            window["deaths"] += terminated
            obs, episode_return = env.reset(), 0.0

        if step >= WARMUP_STEPS:
            window["losses"].append(train_step(q_net, target_net, optimizer, buffer.sample(BATCH_SIZE, device)))
            updates += 1
            if updates % TARGET_UPDATE_EVERY == 0:
                target_net.load_state_dict(q_net.state_dict())
        if step >= PHASE1_STEPS and (step + 1 - PHASE1_STEPS) % CHECKPOINT_EVERY == 0:
            checkpoint(step + 1)
        if (step + 1) % LOG_EVERY == 0:
            training.append(summarize_window(step + 1, epsilon(step), window))
            window = new_window()
            save(step + 1, "running")

    t.save(q_net.state_dict(), run_dir / "final.pt")
    save(total_steps, "done")


def new_window() -> dict:
    return {"returns": [], "deaths": 0, "losses": [], "presses": 0, "lit_presses": 0}


def summarize_window(step: int, current_epsilon: float, window: dict) -> dict:
    """One training-curve point. Rates are over the episodes and trials that ended in the window (exploration included)."""
    mean = lambda xs: float(np.mean(xs)) if xs else None
    return {
        "step": step,
        "epsilon": current_epsilon,
        "episode_return": mean(window["returns"]),
        "death_rate": window["deaths"] / len(window["returns"]) if window["returns"] else None,
        "loss": mean(window["losses"]),
        "lit_share": window["lit_presses"] / window["presses"] if window["presses"] else None,
    }


def launch(jobs: list[tuple[str, int]]) -> None:
    """Runs each job in its own process, at most MAX_PARALLEL at once; each logs to its run directory."""
    processes = []
    for group, seed in jobs:
        while sum(p.poll() is None for p in processes) >= MAX_PARALLEL:
            time.sleep(5)
        log = RUNS_DIR / group / f"seed_{seed}" / "train.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, __file__, "--groups", group, "--seeds", str(seed)]
        processes.append(subprocess.Popen(command, stdout=log.open("w"), stderr=subprocess.STDOUT))
    failed = [job for job, p in zip(jobs, processes) if p.wait() != 0]
    print(f"{len(jobs) - len(failed)}/{len(jobs)} runs finished" + (f"; failed (see train.log): {failed}" if failed else ""))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--groups", nargs="+", choices=GROUPS, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    args = parser.parse_args()
    # Seed by seed, alternating groups, so every batch holds both groups equally
    jobs = [(group, seed) for seed in args.seeds for group in args.groups]
    if len(jobs) == 1:
        t.set_num_threads(1)
        train(*jobs[0])
    else:
        launch(jobs)


if __name__ == "__main__":
    main()
