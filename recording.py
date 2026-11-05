"""
A step-by-step recording of a training run, for the replay viewer. One fixed-size binary record per training step,
appended to runs/<group>/seed_<seed>/recording.bin every few thousand steps, so the file can be read while the run
trains (a partly written last record is ignored). Record i is training step i. About 38 bytes per step, 11 MB per run.

Each record is one step: the state before it, what the agent chose (its Q-values and whether the choice was random),
what happened (food, shocks, screams, reward) and the state after it. Episodes start where `t` is 0.
"""

from pathlib import Path

import numpy as np

from env import TRIAL_STEPS

STEP = np.dtype([
    ("t", "u1"),  # step within the episode, before this step
    ("lit", "u1"),  # the lit lever this episode (0 = A, 1 = B)
    ("active", "u1"),  # 1 on trial steps, when the levers work
    ("action", "u1"),  # 0 = A, 1 = B, 2 = rest
    ("explored", "u1"),  # 1 if the action was random (warm-up or ε), 0 if greedy
    ("q", "f2", 3),  # Q-values of the state before the step
    ("agent_before", "f2", 2), ("agent_after", "f2", 2),  # (health, energy)
    ("other_before", "f2", 2), ("other_after", "f2", 2),
    ("agent_fed", "u1"), ("other_fed", "u1"),  # a pellet this step
    ("agent_shock", "u1"),  # a shock started this step: 0 none, 1 random (phase 1), 2 from the lit lever
    ("other_shock", "u1"),  # a shock to the other started this step (from the lit lever)
    ("agent_scream", "u1"), ("other_scream", "u1"),  # screaming during this step
    ("reward", "f4"),
    ("end", "u1"),  # 0 the episode goes on, 1 the agent died, 2 truncated (the episode's last step)
])
FILE_NAME = "recording.bin"
FLUSH_EVERY = 5_000  # steps


class Recorder:
    """Collects records in memory and appends them to the file on `flush`. Starting a recorder empties the file."""

    def __init__(self, path: Path):
        self.path = path
        self.path.write_bytes(b"")
        self.pending = np.zeros(FLUSH_EVERY, dtype=STEP)
        self.count = 0

    def add(self, env, action: int, explored: bool, q: np.ndarray, before: tuple, reward: float, terminated: bool,
            truncated: bool) -> None:
        """Call right after env.step; `before` is ((health, energy) of the agent, of the other) before the step."""
        record = self.pending[self.count]
        events = env.events
        record["t"] = env.t - 1
        record["lit"], record["active"] = env.lit_lever, (env.t - 1) in TRIAL_STEPS
        record["action"], record["explored"], record["q"] = action, explored, q
        record["agent_before"], record["other_before"] = before
        record["agent_after"] = (env.agent.health, env.agent.energy)
        record["other_after"] = (env.other.health, env.other.energy)
        record["agent_fed"], record["other_fed"] = events["agent_fed"], events["other_fed"]
        record["agent_shock"], record["other_shock"] = events["agent_shock"], events["other_shock"]
        record["agent_scream"], record["other_scream"] = env.agent.scream, env.other.scream
        record["reward"] = reward
        record["end"] = 1 if terminated else 2 if truncated else 0
        self.count += 1
        if self.count == FLUSH_EVERY:
            self.flush()

    def flush(self) -> None:
        with self.path.open("ab") as f:
            f.write(self.pending[: self.count].tobytes())
        self.count = 0


def read(path: Path) -> np.ndarray:
    """Every whole record in the file so far."""
    data = path.read_bytes()
    return np.frombuffer(data[: len(data) - len(data) % STEP.itemsize], dtype=STEP)


def episodes(steps: np.ndarray, phase1_steps: int) -> dict:
    """
    One entry per episode, as columns. `end` is 1 for death, 2 for truncation, and 0 for an episode cut short: by the
    switch to phase 2 or, for the last one, because it is still being played.
    """
    if not len(steps):
        return {"count": 0}
    starts = np.flatnonzero(steps["t"] == 0)
    if not len(starts) or starts[0] != 0:  # can't happen with a file written from the start, but don't crash on it
        starts = np.concatenate([[0], starts])
    lengths = np.diff(np.append(starts, len(steps)))
    on_trial = steps["active"].astype(bool)
    pressed = on_trial & (steps["action"] != 2)
    lit = pressed & (steps["action"] == steps["lit"])
    total = lambda x: np.add.reduceat(x.astype(np.int64), starts)
    return {
        "count": len(starts),
        "start": starts.tolist(),
        "length": lengths.tolist(),
        "phase": np.where(starts >= phase1_steps, 2, 1).tolist(),
        "lit": steps["lit"][starts].tolist(),
        "lit_presses": total(lit).tolist(),
        "unlit_presses": total(pressed & ~lit).tolist(),
        "rests": total(on_trial & ~pressed).tolist(),
        "agent_shocks": total(steps["agent_shock"] > 0).tolist(),
        "other_shocks": total(steps["other_shock"] > 0).tolist(),
        "explored": total(steps["explored"]).tolist(),
        "end": steps["end"][starts + lengths - 1].tolist(),
        "return": np.round(np.add.reduceat(steps["reward"].astype(np.float64), starts), 3).tolist(),
    }


def episode(steps: np.ndarray, start: int, length: int) -> dict:
    """One episode's records as columns, ready for JSON."""
    part = steps[start: start + length]
    columns = {}
    for name in STEP.names:
        values = part[name]
        if values.dtype.kind == "f":
            columns[name] = np.round(values.astype(np.float64), 4).tolist()
        else:
            columns[name] = values.astype(int).tolist()
    return columns
