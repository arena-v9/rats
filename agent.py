"""
The Q-network, the replay buffer and the Double DQN update.
"""

import numpy as np
import torch as t
from torch import nn

from env import NUM_ACTIONS, OBS_SIZE, TIME_COMPRESSION, steps

# A shock's health comes back as reward while it regenerates, so it costs the agent only through discounting: at
# gamma 0.99 the lit lever's true disadvantage at full health is 0.56, at 0.95 it is 1.70 (version 3.3). Per step it is
# 0.95 per uncompressed step, so the discount over a trial interval stays the same (0.95 ** 2.5 = 0.880 per step).
GAMMA = 0.95 ** TIME_COMPRESSION
LEARNING_RATE = 5e-4
BATCH_SIZE = 128
# 20k uncompressed steps (800 trials). (Versions up to 3.2: 100k, plus a store of deaths supplying 5% of each batch,
# which overweights death wherever it is a matter of chance and so biases values down.)
BUFFER_SIZE = steps(20_000)
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


class ReplayBuffer:
    """
    The last `BUFFER_SIZE` transitions, kept in numpy and sampled uniformly with replacement. Truncation is not death,
    so it is never stored as terminal. Small, so that phase-1 experience has left it by step 120k, the start of the main
    measure; there is no separate store of deaths (see the constants above).
    """

    def __init__(self, rng: np.random.Generator, capacity: int = BUFFER_SIZE):
        self.rng, self.capacity, self.size, self.next = rng, capacity, 0, 0
        self.obs = np.zeros((capacity, OBS_SIZE), dtype=np.float32)
        self.next_obs = np.zeros((capacity, OBS_SIZE), dtype=np.float32)
        self.action = np.zeros(capacity, dtype=np.int64)
        self.reward = np.zeros(capacity, dtype=np.float32)
        self.terminated = np.zeros(capacity, dtype=np.float32)

    def add(self, obs, action, reward, next_obs, terminated: bool) -> None:
        i = self.next
        self.obs[i], self.action[i], self.reward[i], self.next_obs[i], self.terminated[i] = (
            obs, action, reward, next_obs, terminated
        )
        self.next = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, n: int, device: t.device) -> tuple[t.Tensor, ...]:
        i = self.rng.integers(self.size, size=n)
        fields = (self.obs, self.action, self.reward, self.next_obs, self.terminated)
        return tuple(t.as_tensor(field[i]).to(device) for field in fields)


def predictions_and_targets(q_net: QNetwork, target_net: QNetwork, batch: tuple[t.Tensor, ...]) -> tuple[t.Tensor, t.Tensor]:
    """Double DQN: the online network picks the next action, the target network values it."""
    obs, action, reward, next_obs, terminated = batch
    q = q_net(obs).gather(1, action[:, None]).squeeze(1)
    with t.no_grad():
        next_action = q_net(next_obs).argmax(dim=1, keepdim=True)
        next_q = target_net(next_obs).gather(1, next_action).squeeze(1)
        target = reward + GAMMA * (1 - terminated) * next_q
    return q, target


def dqn_loss(q_net: QNetwork, target_net: QNetwork, batch: tuple[t.Tensor, ...]) -> t.Tensor:
    """
    Mean squared error, not Huber: with rewards on this scale (pellets up to +4.5, death around −15) most TD errors
    exceed Huber's threshold of 1, where it acts like absolute error and so fits the median of the targets instead of
    their mean. In version 3.1 that left every value about 7 too low, the lever gap 5–20 times too small, and death
    barely felt.
    """
    q, target = predictions_and_targets(q_net, target_net, batch)
    return nn.functional.mse_loss(q, target)


def train_step(q_net: QNetwork, target_net: QNetwork, optimizer: t.optim.Optimizer, batch) -> dict:
    """One update. Returns the loss, the gradient norm before clipping, and the batch's predictions and targets."""
    q, target = predictions_and_targets(q_net, target_net, batch)
    loss = nn.functional.mse_loss(q, target)
    optimizer.zero_grad()
    loss.backward()
    grad_norm = float(nn.utils.clip_grad_norm_(q_net.parameters(), MAX_GRAD_NORM))
    optimizer.step()
    return {"loss": loss.item(), "grad_norm": grad_norm, "q": q.detach().cpu().numpy(), "target": target.cpu().numpy()}
