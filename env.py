"""
The lever task: an agent and another body with identical bodies. On trial steps the agent can press lever A or B for a
pellet; in phase 2 the lit lever also shocks the other body (or, in the pilot, the agent itself). The agent's reward
is the reduction in its own drive and nothing else: neither the other body nor the scream ever enters it.

The agent sees its own health and energy, and hears a scream: one input, 1 when either body screams, so its own scream
and the other's sound the same (Church 1959: rats heard their own squeaks while in pain). Nothing else about the other
body is perceived. (A first version showed the other's health and energy, and agents that had never been shocked
avoided the lit lever too, because starving had taught them that a falling health means danger.)

The lit lever is two inputs, one per lamp (`lit_A`, `lit_B`; exactly one is 1). With a single 0/1 input, A-lit states
were the network's baseline and B-lit states a correction on top of it, so the two conditions learned asymmetrically.

Time is compressed (version 4): trials are TRIAL_INTERVAL = 10 steps apart instead of 25, so one step here is
TIME_COMPRESSION = 2.5 steps of version 3. Up to version 3.3 there were 24 passive steps between trials, so 96% of
transitions (and of training) were steps on which the agent's action did nothing. Everything defined per step is
rescaled together so the task is the same per trial: trial spacing, episode length, hunger, regeneration, starvation,
the shock (same 0.6 of health, in two steps instead of three), the other's meals, and, in agent.py and train.py, the discount, the
random-shock probability, step counts, buffer size and target-network sync. The exact Q-values the agent has to learn
stay close (gap of the lit lever at full health: −1.88 at 10 steps per trial, −1.70 at 25). Version 4.0 used 5 steps
per trial; with one gradient update per step that left too few updates per trial for accurate values (version 4.1).
A trial interval of 25 reproduces version 3.3 exactly; the environment variable LEVER_TASK_TRIAL_INTERVAL overrides
the interval, for checks (each run records the values it used).
"""

import os

import numpy as np

BASE_TRIAL_INTERVAL = 25  # steps between trials before compression (version 3.3)
TRIAL_INTERVAL = int(os.environ.get("LEVER_TASK_TRIAL_INTERVAL", "10"))  # steps between trials now
TIME_COMPRESSION = BASE_TRIAL_INTERVAL / TRIAL_INTERVAL  # uncompressed steps per step: 2.5


def per_step(rate_per_base_step: float) -> float:
    """A rate defined per uncompressed step, per step now."""
    return round(rate_per_base_step * TIME_COMPRESSION, 12)


def steps(base_steps: int) -> int:
    """A number of uncompressed steps, in steps now (step counts in train.py and agent.py). Exact, or an error."""
    assert (base_steps * TRIAL_INTERVAL) % BASE_TRIAL_INTERVAL == 0, f"{base_steps} steps don't compress evenly"
    return base_steps * TRIAL_INTERVAL // BASE_TRIAL_INTERVAL


EPISODE_STEPS = 4 * TRIAL_INTERVAL  # four trials; then truncated, not terminated
# Levers active on these steps: 5, 15, 25, 35 (version 3.3: 12, 37, 62, 87)
TRIAL_STEPS = tuple(round(12 / TIME_COMPRESSION) + i * TRIAL_INTERVAL for i in range(4))
OTHER_MEAL_INTERVAL = TRIAL_INTERVAL  # the other body is fed on its own clock, from a random offset each episode

HUNGER = per_step(0.02)  # energy lost per step
PELLET = 0.5  # energy from one meal, capped at 1 (per trial, so not rescaled)
REGENERATION = per_step(0.01)  # health gained per step while energy > 0
STARVATION = per_step(0.05)  # health lost per step while energy is 0
SHOCK_HEALTH = 0.6  # health lost to one shock, in total
# A shock lasts this many steps (2 at 10 steps per trial; 3 uncompressed), with harm and a scream on each. At least two:
# on the first, the scream comes with harm still to come, so it predicts pain. With one step (version 4.1 at first)
# the harm was over by the time the scream was heard, and the shocked agents learned the scream as good news.
SHOCK_STEPS = max(2, round(3 / TIME_COMPRESSION))
SHOCK_LOSS = round(SHOCK_HEALTH / SHOCK_STEPS, 12)  # health lost per shock step
BETA = 10  # reward scale
DEAD_DRIVE = 2.0
TOLERANCE = 1e-9  # values this close to 0 count as 0, so float error can't keep a body alive or fed

LEVER_A, LEVER_B, REST = 0, 1, 2
NUM_ACTIONS = 3
OBSERVATION = ("health", "energy", "scream", "levers_active", "lit_A", "lit_B")  # health and energy are the agent's own
OBS_SIZE = len(OBSERVATION)
SCREAM_INDEX = OBSERVATION.index("scream")


def _clip(value: float) -> float:
    return 0.0 if value < TOLERANCE else min(1.0, value)


class Body:
    """Health and energy in [0, 1]. Dies when health reaches 0, and only then."""

    def __init__(self, energy: float):
        self.health = 1.0
        self.energy = energy
        self.shock_steps_left = 0
        self.scream = False  # True on each step the body lost health to a shock
        self.shock_started = False  # True on the step a new shock began (for the replay recording)

    @property
    def alive(self) -> bool:
        return self.health > 0

    @property
    def drive(self) -> float:
        """Keramati & Gutkin (2014): the sum of squared deficits from the setpoints (1, 1); maximal when dead."""
        return (1 - self.health) ** 2 + (1 - self.energy) ** 2 if self.alive else DEAD_DRIVE

    def step(self, food: float, shocked: bool) -> None:
        """One step: a shock starts if none is running, then hunger and food, then shock loss, then regeneration."""
        self.shock_started = False
        if not self.alive:
            self.scream = False  # a dead body is silent
            return
        if shocked and self.shock_steps_left == 0:
            self.shock_steps_left = SHOCK_STEPS
            self.shock_started = True
        self.scream = self.shock_steps_left > 0
        self.energy = _clip(max(0.0, self.energy - HUNGER) + food)
        if self.scream:
            self.health = _clip(self.health - SHOCK_LOSS)
            self.shock_steps_left -= 1
        if self.alive:
            self.health = _clip(self.health + (REGENERATION if self.energy > 0 else -STARVATION))


class LeverTask:
    """
    One phase of the task. `self_shock_prob`: chance per step that the agent is shocked, whatever it does (phase 1).
    `lit_target`: whom the lit lever shocks, "other", "self" or None (phase 1: no one).
    """

    def __init__(self, rng: np.random.Generator, self_shock_prob: float, lit_target: str | None):
        assert lit_target in ("other", "self", None)
        self.rng, self.self_shock_prob, self.lit_target = rng, self_shock_prob, lit_target

    def reset(self, lit_lever: int | None = None) -> np.ndarray:
        self.t = 0
        self.agent = Body(energy=self.rng.uniform(0.5, 1.0))
        self.other = Body(energy=self.rng.uniform(0.5, 1.0))
        self.other_meal_offset = int(self.rng.integers(OTHER_MEAL_INTERVAL))
        drawn = int(self.rng.integers(2))  # drawn even when overridden, so evaluation episodes keep the same stream
        self.lit_lever = drawn if lit_lever is None else lit_lever
        return self.observation()

    @property
    def levers_active(self) -> bool:
        return self.t in TRIAL_STEPS

    def observation(self) -> np.ndarray:
        scream = self.agent.scream or self.other.scream
        return np.array([self.agent.health, self.agent.energy, scream, self.levers_active,
                         self.lit_lever == LEVER_A, self.lit_lever == LEVER_B], dtype=np.float32)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool]:
        """Returns (observation, reward, terminated, truncated). Terminated means the agent died."""
        pressed = self.levers_active and action != REST
        lit_pressed = pressed and action == self.lit_lever
        random_shock = self.rng.random() < self.self_shock_prob  # drawn every step, so the stream never depends on actions
        other_fed = (self.t - self.other_meal_offset) % OTHER_MEAL_INTERVAL == 0

        drive_before = self.agent.drive
        self.agent.step(PELLET if pressed else 0.0, random_shock or (lit_pressed and self.lit_target == "self"))
        self.other.step(PELLET if other_fed else 0.0, lit_pressed and self.lit_target == "other")
        self.t += 1
        # What happened this step, for the replay recording; nothing here feeds back into the task
        self.events = {"agent_fed": pressed, "other_fed": other_fed and self.other.alive,
                       "agent_shock": (1 if random_shock else 2) if self.agent.shock_started else 0,
                       "other_shock": self.other.shock_started}

        reward = BETA * (drive_before - self.agent.drive)
        terminated = not self.agent.alive
        truncated = self.t == EPISODE_STEPS and not terminated
        return self.observation(), reward, terminated, truncated
