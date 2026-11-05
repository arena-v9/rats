import numpy as np
import pytest

from env import (EPISODE_STEPS, HUNGER, LEVER_A, LEVER_B, OBSERVATION, PELLET, REGENERATION, REST, SCREAM_INDEX,
                 SHOCK_HEALTH, SHOCK_LOSS, SHOCK_STEPS, STARVATION, TIME_COMPRESSION, TRIAL_INTERVAL, TRIAL_STEPS, Body,
                 LeverTask)

# The rules are written in terms of the constants, so these tests hold at any time compression (run them with
# LEVER_TASK_TRIAL_INTERVAL=25 too: that is version 3.3's task)

LEVERS_ACTIVE_INDEX = OBSERVATION.index("levers_active")
AGENT_INDICES = [OBSERVATION.index("health"), OBSERVATION.index("energy")]


def make_task(seed=0, self_shock_prob=0.0, lit_target="other", lit_lever=LEVER_B):
    task = LeverTask(np.random.default_rng(seed), self_shock_prob, lit_target)
    task.reset(lit_lever=lit_lever)
    return task


def run_to(task, step, action=REST):
    while task.t < step:
        task.step(action)


# Body rules


def test_compression_scales_every_rate_and_keeps_the_task_per_trial():
    assert TRIAL_INTERVAL * TIME_COMPRESSION == 25 and EPISODE_STEPS == 4 * TRIAL_INTERVAL
    assert HUNGER * TRIAL_INTERVAL == pytest.approx(0.5) == PELLET  # a pellet per trial balances the hunger
    assert (REGENERATION, STARVATION) == pytest.approx((0.01 * TIME_COMPRESSION, 0.05 * TIME_COMPRESSION))
    assert SHOCK_STEPS * SHOCK_LOSS == pytest.approx(SHOCK_HEALTH) == pytest.approx(0.6)
    assert all(b - a == TRIAL_INTERVAL for a, b in zip(TRIAL_STEPS, TRIAL_STEPS[1:]))


def test_hunger_and_eating_capped_at_one():
    body = Body(energy=0.48)
    body.step(food=PELLET, shocked=False)
    assert body.energy == pytest.approx(0.48 - HUNGER + PELLET)
    body.step(food=PELLET, shocked=False)
    assert body.energy == 1.0
    body.step(food=0.0, shocked=False)
    assert body.energy == pytest.approx(1 - HUNGER)


def test_regenerates_only_while_fed_and_starves_at_zero_energy():
    body = Body(energy=0.5)
    body.health = 0.5
    body.step(food=0.0, shocked=False)
    assert body.health == pytest.approx(0.5 + REGENERATION)
    body.energy = HUNGER / 2
    body.step(food=0.0, shocked=False)
    assert body.energy == 0.0 and body.health == pytest.approx(0.5 + REGENERATION - STARVATION)


def test_shock_lasts_its_steps_with_scream_then_regenerates():
    body = Body(energy=1.0)
    healths, screams = [], []
    for step in range(SHOCK_STEPS + 2):
        body.step(food=PELLET, shocked=step == 0)
        healths.append(body.health)
        screams.append(body.scream)
    assert screams == [True] * SHOCK_STEPS + [False, False]
    expected, health = [], 1.0
    for step in range(SHOCK_STEPS + 2):
        health = min(1.0, health - (SHOCK_LOSS if step < SHOCK_STEPS else 0) + REGENERATION)
        expected.append(health)
    assert healths == pytest.approx(expected)


def test_new_shock_cannot_start_during_one():
    body = Body(energy=1.0)
    for step in range(SHOCK_STEPS):
        body.step(food=PELLET, shocked=True)  # asking again while a shock runs starts nothing
    assert body.health == pytest.approx(1 - SHOCK_HEALTH + SHOCK_STEPS * REGENERATION)
    body.step(food=PELLET, shocked=False)
    assert not body.scream


def test_dies_only_when_health_reaches_zero_and_stays_dead_and_silent():
    survivor = Body(energy=1.0)
    survivor.health = SHOCK_LOSS + 0.01  # one shock step leaves it just alive
    survivor.step(food=PELLET, shocked=True)
    assert survivor.alive
    body = Body(energy=1.0)
    body.health = 0.5 * SHOCK_LOSS
    body.step(food=PELLET, shocked=True)
    assert not body.alive and body.drive == 2.0 and body.scream  # it screams on the step it dies
    body.step(food=PELLET, shocked=False)
    assert body.health == 0.0 and not body.scream


def test_drive_ignores_the_scream():
    quiet, screaming = Body(energy=0.7), Body(energy=0.7)
    quiet.health = screaming.health = 0.8
    screaming.scream = True
    assert quiet.drive == screaming.drive == pytest.approx(0.04 + 0.09)


# Reward


def test_death_reward_is_drive_before_minus_two_and_terminates():
    task = make_task()
    task.agent.health, task.agent.energy = 0.04, 0.0  # one step of starvation kills
    before = task.agent.drive
    _, reward, terminated, _ = task.step(REST)
    assert terminated and reward == pytest.approx(10 * (before - 2.0))


def test_eating_reward():
    task = make_task()
    run_to(task, TRIAL_STEPS[0])
    task.agent.energy = 0.48
    _, reward, terminated, _ = task.step(LEVER_A)
    assert not terminated
    assert reward == pytest.approx(10 * ((1 - 0.48) ** 2 - (1 - (0.48 - HUNGER + PELLET)) ** 2))


def test_episode_truncates_at_the_step_limit():
    task = make_task()
    for step in range(EPISODE_STEPS):
        _, _, terminated, truncated = task.step(LEVER_A)
        assert not terminated
        assert truncated == (step == EPISODE_STEPS - 1)


# The levers and the shock


def test_levers_feed_only_on_trial_steps():
    task = make_task()
    run_to(task, TRIAL_STEPS[0] - 1)
    obs = task.observation()
    assert obs[LEVERS_ACTIVE_INDEX] == 0.0
    energy = task.agent.energy
    task.step(LEVER_A)
    assert task.agent.energy == pytest.approx(energy - HUNGER)
    assert task.observation()[LEVERS_ACTIVE_INDEX] == 1.0
    task.step(LEVER_A)
    assert task.agent.energy == pytest.approx(min(1.0, energy - 2 * HUNGER + PELLET))


@pytest.mark.parametrize("lit_lever", [LEVER_A, LEVER_B])
def test_lit_lever_shocks_only_the_other_body(lit_lever):
    task = make_task(lit_lever=lit_lever)
    run_to(task, TRIAL_STEPS[0])
    task.step(lit_lever)
    assert task.other.scream and task.other.health == pytest.approx(1 - SHOCK_LOSS + REGENERATION)
    assert not task.agent.scream and task.agent.health == 1.0


def test_unlit_lever_shocks_no_one():
    task = make_task(lit_lever=LEVER_B)
    run_to(task, TRIAL_STEPS[0])
    task.step(LEVER_A)
    assert not task.other.scream and not task.agent.scream


def test_lit_lever_targets_the_agent_in_the_pilot():
    task = make_task(lit_target="self", lit_lever=LEVER_A)
    run_to(task, TRIAL_STEPS[0])
    task.step(LEVER_A)
    assert task.agent.scream and not task.other.scream


def test_phase1_lit_lever_does_nothing_and_random_shocks_hit_only_the_agent():
    task = make_task(seed=3, self_shock_prob=0.2, lit_target=None)
    agent_screams = other_screams = 0
    for _ in range(EPISODE_STEPS):
        _, _, terminated, _ = task.step(task.lit_lever)
        agent_screams += task.agent.scream
        other_screams += task.other.scream
        if terminated:
            break
    assert agent_screams > 0 and other_screams == 0


def test_agent_unaffected_by_what_happens_to_the_other():
    """Same seed and actions: the agent's rewards and its body match whether or not the lit lever shocks the other."""
    actions = [LEVER_B] * EPISODE_STEPS  # the lit lever, pressed on every trial
    outcomes = {}
    for target in ("other", None):
        task = make_task(seed=7, self_shock_prob=0.05, lit_target=target)
        steps = [task.step(int(a)) for a in actions]
        outcomes[target] = steps
    other_differs = False
    for (obs_a, r_a, term_a, _), (obs_b, r_b, term_b, _) in zip(outcomes["other"], outcomes[None]):
        assert r_a == r_b and term_a == term_b
        assert np.array_equal(obs_a[AGENT_INDICES], obs_b[AGENT_INDICES])
        other_differs |= obs_a[SCREAM_INDEX] != obs_b[SCREAM_INDEX]
    assert other_differs  # the actions did make the other scream, so the test is not vacuous


def test_either_body_screaming_sets_the_one_scream_input():
    for target in ("other", "self"):
        task = make_task(lit_target=target, lit_lever=LEVER_A)
        run_to(task, TRIAL_STEPS[0])
        obs, _, _, _ = task.step(LEVER_A)
        assert obs[SCREAM_INDEX] == 1.0


@pytest.mark.parametrize("lit_lever", [LEVER_A, LEVER_B])
def test_lit_lever_is_one_input_per_lamp(lit_lever):
    task = make_task(lit_lever=lit_lever)
    for _ in range(EPISODE_STEPS - 1):  # the lamp stays on all episode, trial or not
        obs, _, _, _ = task.step(REST)
        assert obs[OBSERVATION.index("lit_A")] == (lit_lever == LEVER_A)
        assert obs[OBSERVATION.index("lit_B")] == (lit_lever == LEVER_B)


def test_other_body_is_perceived_only_through_its_scream():
    """Changing the other's health and energy changes nothing the agent sees."""
    task = make_task()
    run_to(task, TRIAL_STEPS[0])
    before = task.observation()
    task.other.health, task.other.energy = 0.2, 0.0
    assert np.array_equal(task.observation(), before)
