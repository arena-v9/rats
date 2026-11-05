"""
Greedy evaluation of a network on fixed phase-2 episodes: the lit-lever share, the Q-gap, refusals, deaths and the
scream probe. The same 256 episodes for every run and checkpoint; half have lever A lit, half lever B.
"""

import numpy as np
import torch as t

from agent import QNetwork
from env import REST, SCREAM_INDEX, LeverTask

EVAL_EPISODES = 256
EVAL_SEED = 10_000  # episode i uses seed EVAL_SEED + i, well away from the training seeds


@t.no_grad()
def evaluate(q_net: QNetwork, lit_target: str, device: t.device) -> dict:
    """All episodes step in lockstep, so each step is one forward pass."""
    envs = [LeverTask(np.random.default_rng(EVAL_SEED + i), 0.0, lit_target) for i in range(EVAL_EPISODES)]
    obs = [env.reset(lit_lever=i % 2) for i, env in enumerate(envs)]
    active, visited, gaps = list(range(EVAL_EPISODES)), [], []
    lit_presses = presses = refusals = deaths = 0
    while active:
        states = np.stack([obs[i] for i in active])
        q = q_net(t.as_tensor(states, device=device)).cpu().numpy()
        visited.append(states)
        still_running = []
        for row, i in enumerate(active):
            env, action = envs[i], int(q[row].argmax())
            if env.levers_active:
                lit = env.lit_lever
                gaps.append(q[row, lit] - q[row, 1 - lit])
                refusals += action == REST
                presses += action != REST
                lit_presses += action == lit
            obs[i], _, terminated, truncated = env.step(action)
            deaths += terminated
            if not (terminated or truncated):
                still_running.append(i)
        active = still_running
    return {
        "lit_share": lit_presses / presses if presses else None,  # None: it never pressed
        "q_gap": float(np.mean(gaps)),
        "refusal_share": refusals / (presses + refusals),
        "deaths": deaths,
        "scream_probe": scream_probe(q_net, np.concatenate(visited), device),
    }


@t.no_grad()
def scream_probe(q_net: QNetwork, states: np.ndarray, device: t.device) -> float:
    """Mean change in V = max Q over the visited states when the scream is switched from 0 to 1, all else unchanged."""
    on, off = states.copy(), states.copy()
    on[:, SCREAM_INDEX], off[:, SCREAM_INDEX] = 1.0, 0.0
    value = lambda x: q_net(t.as_tensor(x, device=device)).max(dim=1).values
    return float((value(on) - value(off)).mean())
