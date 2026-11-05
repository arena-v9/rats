"""
The Q-network, the replay buffer with its store of deaths, and the Double DQN update.
"""

import numpy as np
import torch as t
from torch import nn

from env import NUM_ACTIONS, OBS_SIZE

GAMMA = 0.99
LEARNING_RATE = 5e-4
BATCH_SIZE = 128
BUFFER_SIZE = 100_000
DEATH_STORE_SIZE = 5_000
DEATH_SHARE = 0.05  # share of each batch taken from the store of deaths, so the agent doesn't forget them
MAX_GRAD_NORM = 10.0


class QNetwork(nn.Module):
    """A plain MLP from the observation to one Q-value per action."""

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(OBS_SIZE, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, NUM_ACTIONS)
        )

    def forward(self, obs: t.Tensor) -> t.Tensor:
        return self.layers(obs)


class _Store:
    """A circular store of transitions, kept in numpy and sampled with replacement."""

    def __init__(self, capacity: int):
        self.capacity, self.size, self.next = capacity, 0, 0
        self.obs = np.zeros((capacity, OBS_SIZE), dtype=np.float32)
        self.next_obs = np.zeros((capacity, OBS_SIZE), dtype=np.float32)
        self.action = np.zeros(capacity, dtype=np.int64)
        self.reward = np.zeros(capacity, dtype=np.float32)
        self.terminated = np.zeros(capacity, dtype=np.float32)

    def add(self, obs, action, reward, next_obs, terminated) -> None:
        i = self.next
        self.obs[i], self.action[i], self.reward[i], self.next_obs[i], self.terminated[i] = (
            obs, action, reward, next_obs, terminated
        )
        self.next = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def indices(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.integers(self.size, size=n)


class ReplayBuffer:
    """
    Every transition goes in the main store; those ending in death are also copied to a small store of deaths, which
    supplies `DEATH_SHARE` of each batch once it holds any. Truncation is not death, so it is never stored as terminal.
    """

    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self.main, self.deaths = _Store(BUFFER_SIZE), _Store(DEATH_STORE_SIZE)

    def add(self, obs, action, reward, next_obs, terminated: bool) -> None:
        self.main.add(obs, action, reward, next_obs, terminated)
        if terminated:
            self.deaths.add(obs, action, reward, next_obs, terminated)

    def sample(self, n: int, device: t.device) -> tuple[t.Tensor, ...]:
        num_deaths = round(n * DEATH_SHARE) if self.deaths.size else 0
        parts = [(self.main, self.main.indices(n - num_deaths, self.rng))]
        if num_deaths:
            parts.append((self.deaths, self.deaths.indices(num_deaths, self.rng)))
        fields = ("obs", "action", "reward", "next_obs", "terminated")
        return tuple(
            t.as_tensor(np.concatenate([getattr(store, f)[i] for store, i in parts])).to(device) for f in fields
        )


def dqn_loss(q_net: QNetwork, target_net: QNetwork, batch: tuple[t.Tensor, ...]) -> t.Tensor:
    """Double DQN: the online network picks the next action, the target network values it. Huber loss."""
    obs, action, reward, next_obs, terminated = batch
    q = q_net(obs).gather(1, action[:, None]).squeeze(1)
    with t.no_grad():
        next_action = q_net(next_obs).argmax(dim=1, keepdim=True)
        next_q = target_net(next_obs).gather(1, next_action).squeeze(1)
        target = reward + GAMMA * (1 - terminated) * next_q
    return nn.functional.smooth_l1_loss(q, target)


def train_step(q_net: QNetwork, target_net: QNetwork, optimizer: t.optim.Optimizer, batch) -> float:
    loss = dqn_loss(q_net, target_net, batch)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(q_net.parameters(), MAX_GRAD_NORM)
    optimizer.step()
    return loss.item()
