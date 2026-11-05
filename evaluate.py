"""
Greedy evaluation of a network on fixed phase-2 episodes: the lit-lever share, the Q-gap, refusals, deaths and the
scream probe. The same 256 episodes for every run and checkpoint; half have lever A lit, half lever B.
"""

import numpy as np
import torch as t

from agent import QNetwork
from env import REST, SCREAM_INDEX, TRIAL_STEPS, LeverTask
from exact import exact_trial_q

EVAL_EPISODES = 256
EVAL_SEED = 10_000  # episode i uses seed EVAL_SEED + i, well away from the training seeds


@t.no_grad()
def evaluate(q_net: QNetwork, lit_target: str, device: t.device) -> dict:
    """All episodes step in lockstep, so each step is one forward pass."""
    envs = [LeverTask(np.random.default_rng(EVAL_SEED + i), 0.0, lit_target) for i in range(EVAL_EPISODES)]
    obs = [env.reset(lit_lever=i % 2) for i, env in enumerate(envs)]
    active, visited, gaps = list(range(EVAL_EPISODES)), [], []
    lit_presses = presses = refusals = deaths = 0
    # Diagnostics: presses broken down by which lever is lit and by trial, to see where any lit pressing comes from
    by_lever = {"A": [0, 0], "B": [0, 0]}  # [lit presses, presses]
    by_trial = {str(step): [0, 0] for step in TRIAL_STEPS}
    trial_states = []  # (health, energy) on each trial step, for the comparison with the exact values
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
                trial_states.append((env.agent.health, env.agent.energy))
                refusals += action == REST
                presses += action != REST
                lit_presses += action == lit
                for tally in (by_lever["AB"[lit]], by_trial[str(env.t)]):
                    tally[0] += action == lit
                    tally[1] += action != REST
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
        "lit_preferred_share": float(np.mean(np.array(gaps) > 0)),  # trial states where Q(lit) > Q(unlit)
        "lit_share_by_lever": {k: share(*v) for k, v in by_lever.items()},  # key: which lever was lit
        "lit_share_by_trial": {k: share(*v) for k, v in by_trial.items()},
        **exact_comparison(np.array(gaps), trial_states, lit_target),
    }


def exact_comparison(gaps: np.ndarray, trial_states: list, lit_target: str) -> dict:
    """The learned Q-gap on the visited trial steps against the exact one (exact.py), and where it has the wrong sign."""
    exact = np.array([exact_trial_q(h, e, lit_target) for h, e in trial_states])
    exact_gap = exact[:, 0] - exact[:, 1]
    worse = exact_gap < -1e-9  # pressing the lit lever is truly worse
    return {
        "exact_q_gap": float(exact_gap.mean()),
        "q_gap_error": float((gaps - exact_gap).mean()),  # learned − exact
        "gap_sign_correct": float((gaps[worse] < 0).mean()) if worse.any() else None,
        "trial_health_mean": float(np.mean([h for h, _ in trial_states])),
        "trial_health_min": float(np.min([h for h, _ in trial_states])),
    }


def share(lit: int, presses: int) -> float | None:
    return lit / presses if presses else None


@t.no_grad()
def scream_probe(q_net: QNetwork, states: np.ndarray, device: t.device) -> float:
    """Mean change in V = max Q over the visited states when the scream is switched from 0 to 1, all else unchanged."""
    on, off = states.copy(), states.copy()
    on[:, SCREAM_INDEX], off[:, SCREAM_INDEX] = 1.0, 0.0
    value = lambda x: q_net(t.as_tensor(x, device=device)).max(dim=1).values
    return float((value(on) - value(off)).mean())
