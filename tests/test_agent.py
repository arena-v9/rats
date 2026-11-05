import numpy as np
import torch as t

from agent import QNetwork, ReplayBuffer, dqn_loss
from env import OBS_SIZE


def test_buffer_keeps_only_the_latest_transitions():
    buffer = ReplayBuffer(np.random.default_rng(0), capacity=10)
    for i in range(25):
        obs = np.full(OBS_SIZE, i, dtype=np.float32)
        buffer.add(obs, 0, float(i), obs, terminated=False)
    assert buffer.size == 10
    _, _, reward, _, _ = buffer.sample(500, t.device("cpu"))
    assert set(reward.tolist()) == set(float(i) for i in range(15, 25))  # 0–14 were overwritten


def test_loss_is_the_mean_squared_td_error_even_for_large_errors():
    """Large errors must count in full (Huber would cap them): the loss is the plain mean of squared TD errors."""
    t.manual_seed(0)
    q_net, target_net = QNetwork(), QNetwork()
    obs = t.rand(8, OBS_SIZE)
    next_obs = t.rand(8, OBS_SIZE)
    action = t.randint(0, 3, (8,))
    reward = t.full((8,), -15.0)  # death-sized rewards, far beyond Huber's threshold
    terminated = t.ones(8)
    loss = dqn_loss(q_net, target_net, (obs, action, reward, next_obs, terminated))
    q = q_net(obs).gather(1, action[:, None]).squeeze(1)
    assert t.isclose(loss, ((q - reward) ** 2).mean())
