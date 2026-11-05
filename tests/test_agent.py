import numpy as np
import torch as t

from agent import ReplayBuffer
from env import OBS_SIZE


def test_batches_take_five_percent_from_the_death_store():
    buffer = ReplayBuffer(np.random.default_rng(0))
    obs = np.zeros(OBS_SIZE, dtype=np.float32)
    for i in range(1000):
        buffer.add(obs, 0, 0.0, obs, terminated=i % 100 == 0)  # 1% of transitions are deaths
    _, _, _, _, terminated = buffer.sample(200, t.device("cpu"))
    assert terminated.sum() >= 10  # 5% of 200 from the death store, plus any drawn from the main store
