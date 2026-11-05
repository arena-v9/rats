"""
Exact Q-values for the agent in phase 2, where the task is deterministic (no random shocks), so they can be computed by
simulating the agent's body. Used to check what the network learns against the truth, during training and at
checkpoints.

From a trial state (health, energy) the agent takes a first action, then follows the best policy: press the unlit
lever on every trial (one every TRIAL_INTERVAL steps; a pellet costs nothing, the unlit lever never shocks) and rest
otherwise. The body runs on past the truncation at the end of an episode, as the TD targets do (truncation
bootstraps). Rewards and discount are the training ones. Phase 1 has random shocks, so its values are not exact (not
computed here).
"""

import math
from functools import lru_cache

from agent import GAMMA
from env import BETA, EPISODE_STEPS, PELLET, TRIAL_INTERVAL, TRIAL_STEPS, Body

assert all(b - a == TRIAL_INTERVAL for a, b in zip(TRIAL_STEPS, TRIAL_STEPS[1:]))
assert TRIAL_STEPS[0] + EPISODE_STEPS - TRIAL_STEPS[-1] == TRIAL_INTERVAL  # the spacing holds across episodes too
HORIZON = math.ceil(math.log(1e-6) / math.log(GAMMA))  # beyond this, discounting leaves less than a millionth
FIRST_ACTIONS = ("lit", "unlit", "rest")


@lru_cache(maxsize=200_000)
def exact_q(health: float, energy: float, first: str, lit_shocks_agent: bool) -> float:
    """
    Q of a trial state for the first action "lit", "unlit" or "rest". `lit_shocks_agent`: the pilot (the lit lever
    shocks the agent); otherwise pressing it costs the agent nothing. Round the state before calling, for the cache.
    """
    assert first in FIRST_ACTIONS
    body = Body(energy=energy)
    body.health = health
    total, discount = 0.0, 1.0
    for k in range(HORIZON):
        if k == 0:
            food, shocked = (0.0 if first == "rest" else PELLET), (first == "lit" and lit_shocks_agent)
        else:
            food, shocked = (PELLET if k % TRIAL_INTERVAL == 0 else 0.0), False
        before = body.drive
        body.step(food, shocked)
        total += discount * BETA * (before - body.drive)
        if not body.alive:
            break
        discount *= GAMMA
    return total


def exact_trial_q(health: float, energy: float, lit_target: str | None) -> tuple[float, float, float]:
    """(Q(lit), Q(unlit), Q(rest)) at a trial state, rounded to 0.01 for the cache."""
    h, e, shocks = round(float(health), 2), round(float(energy), 2), lit_target == "self"
    return tuple(exact_q(h, e, first, shocks) for first in FIRST_ACTIONS)
