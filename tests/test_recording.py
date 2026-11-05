import numpy as np

from env import LEVER_A, LEVER_B, REST, TRIAL_STEPS, LeverTask
from recording import STEP, Recorder, episode, episodes, read


def play(path, actions, lit_target="self", self_shock_prob=0.0, seed=0):
    """Plays the actions (restarting after each episode) into a recording, the way train.py does."""
    env = LeverTask(np.random.default_rng(seed), self_shock_prob, lit_target)
    env.reset(lit_lever=LEVER_A)
    recorder = Recorder(path)
    for action in actions:
        before = ((env.agent.health, env.agent.energy), (env.other.health, env.other.energy))
        _, reward, terminated, truncated = env.step(action)
        recorder.add(env, action, False, np.zeros(3), before, reward, terminated, truncated)
        if terminated or truncated:
            env.reset(lit_lever=LEVER_A)
    recorder.flush()
    return read(path)


def test_records_each_step_with_what_it_caused(tmp_path):
    actions = [REST] * TRIAL_STEPS[0] + [LEVER_A] + [REST] * 3
    steps = play(tmp_path / "r.bin", actions)
    assert len(steps) == len(actions)
    press = steps[TRIAL_STEPS[0]]
    assert press["t"] == TRIAL_STEPS[0] and press["active"] == 1 and press["action"] == LEVER_A and press["lit"] == 0
    assert press["agent_fed"] == 1 and press["agent_shock"] == 2 and press["agent_scream"] == 1
    assert press["agent_after"][0] < press["agent_before"][0]  # the shock cost health
    assert steps["agent_shock"].sum() == 2  # one shock started, nothing else
    assert steps["agent_scream"][TRIAL_STEPS[0]:].tolist() == [1, 1, 1, 0]


def test_pressing_the_unlit_lever_feeds_without_shock(tmp_path):
    steps = play(tmp_path / "r.bin", [REST] * TRIAL_STEPS[0] + [LEVER_B])
    assert steps[-1]["agent_fed"] == 1 and steps[-1]["agent_shock"] == 0 and steps[-1]["agent_scream"] == 0


def test_random_shocks_are_marked_as_random(tmp_path):
    steps = play(tmp_path / "r.bin", [REST] * 100, lit_target=None, self_shock_prob=0.2)
    assert set(steps["agent_shock"].tolist()) <= {0, 1} and (steps["agent_shock"] == 1).any()


def test_episodes_split_at_resets_and_count_presses(tmp_path):
    first = [REST] * TRIAL_STEPS[0] + [LEVER_A] + [REST] * (TRIAL_STEPS[1] - TRIAL_STEPS[0] - 1) + [LEVER_B]
    actions = first + [REST] * (100 - len(first)) + [REST] * 10  # one full episode, then 10 steps of the next
    steps = play(tmp_path / "r.bin", actions)
    listing = episodes(steps, phase1_steps=50)
    assert listing["count"] == 2 and listing["start"] == [0, 100] and listing["length"] == [100, 10]
    assert listing["lit_presses"] == [1, 0] and listing["unlit_presses"] == [1, 0]
    assert listing["rests"] == [2, 0]  # trials 3 and 4 of the first episode; the second ends before its first trial
    assert listing["end"] == [2, 0] and listing["phase"] == [1, 2]
    one = episode(steps, 100, 10)
    assert one["t"] == list(range(10)) and len(one["q"]) == 10


def test_a_partly_written_record_is_ignored(tmp_path):
    path = tmp_path / "r.bin"
    play(path, [REST] * 5)
    with path.open("ab") as f:
        f.write(b"\x00" * (STEP.itemsize // 2))
    assert len(read(path)) == 5

