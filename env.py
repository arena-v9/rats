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
"""

import numpy as np

EPISODE_STEPS = 100  # then truncated, not terminated
TRIAL_STEPS = (12, 37, 62, 87)  # steps on which the levers are active
OTHER_MEAL_INTERVAL = 25  # the other body is fed on its own clock, from a random offset each episode

HUNGER = 0.02  # energy lost per step
PELLET = 0.5  # energy from one meal, capped at 1
REGENERATION = 0.01  # health gained per step while energy > 0
STARVATION = 0.05  # health lost per step while energy is 0
SHOCK_STEPS = 3  # a shock lasts this many steps; the body screams throughout
SHOCK_LOSS = 0.2  # health lost per shock step (0.6 per shock)
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
