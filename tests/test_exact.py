import pytest

from exact import exact_q, exact_trial_q
from train import SCHEDULES, epsilon


def test_in_the_pilot_the_lit_lever_is_worse_and_more_so_at_low_health():
    gaps = [exact_q(h, 0.5, "lit", True) - exact_q(h, 0.5, "unlit", True) for h in (1.0, 0.8, 0.7)]
    assert gaps[0] > gaps[1] > gaps[2] < 0


def test_time_compression_keeps_the_lit_levers_disadvantage():
    """
    Version 3.3 (25 steps per trial): -1.70 at full health (independent simulation); compression must keep it close.
    10 steps per trial, two-step shock: -1.57 (the setting in use). 5 steps per trial fails here (-1.23: a two-step
    shock is 40% of the interval), so 5 is not a supported setting; 25 is kept for checks against version 3.3.
    """
    gap = exact_q(1.0, 0.5, "lit", True) - exact_q(1.0, 0.5, "unlit", True)
    assert gap == pytest.approx(-1.70, abs=0.25)


def test_a_shock_at_low_health_kills():
    # health 0.5: three shock steps of 0.2 take it to 0 before regeneration can help
    assert exact_q(0.5, 0.5, "lit", True) < exact_q(0.5, 0.5, "unlit", True) - 10


def test_when_the_lit_lever_shocks_the_other_it_costs_the_agent_nothing():
    lit, unlit, rest = exact_trial_q(1.0, 0.5, "other")
    assert lit == unlit and rest < unlit  # resting forgoes the pellet


@pytest.mark.parametrize("name", list(SCHEDULES))
def test_epsilon_falls_to_its_floor_by_the_schedule_end_step(name):
    schedule = SCHEDULES[name]
    assert epsilon(0, schedule) == 1.0
    assert epsilon(schedule["epsilon_end_step"], schedule) == pytest.approx(0.02)
    assert schedule["warmup_steps"] < schedule["epsilon_end_step"] < schedule["phase1_steps"]
