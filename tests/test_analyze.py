import sys

import numpy as np
import pytest

import train
from analyze import END_OF_PHASE1, FINAL, MAIN, values


def fake_run(schedule: str, shares: dict[int, float]) -> dict:
    run = {"schedule": schedule, "phase1_steps": train.SCHEDULES[schedule]["phase1_steps"],
           "checkpoints": [{"step": step, "lit_share": share} for step, share in sorted(shares.items())]}
    if schedule == "quick":
        run["main_window"] = list(train.SCHEDULES["quick"]["main_window"])
    return run


def test_quick_main_window_is_the_first_20k_uncompressed_steps_of_phase_2():
    quick = train.SCHEDULES["quick"]
    p1, every = quick["phase1_steps"], quick["checkpoint_every"]
    assert quick["main_window"] == (p1 + every, p1 + 4 * every)  # four checkpoints
    shares = {p1: 0.9, p1 + every: 0.1, p1 + 2 * every: 0.2, p1 + 3 * every: 0.3, p1 + 4 * every: 0.4,
              p1 + 5 * every: 0.9, p1 + quick["phase2_steps"]: 0.5}
    run = fake_run("quick", shares)
    assert values([run], "lit_share", MAIN)[0] == pytest.approx(np.mean([0.1, 0.2, 0.3, 0.4]))
    assert values([run], "lit_share", END_OF_PHASE1)[0] == 0.9
    assert values([run], "lit_share", FINAL)[0] == 0.5


def test_runs_without_a_schedule_use_the_full_main_window():
    full = train.SCHEDULES["full"]
    p1, every = full["phase1_steps"], full["checkpoint_every"]
    run = fake_run("full", {p1 + i * every: share for i, share in enumerate([0.9, 0.1, 0.2, 0.3, 0.9])})
    del run["schedule"]
    assert np.isclose(values([run], "lit_share", MAIN)[0], 0.2)


def test_buffer_ablation_jobs(monkeypatch):
    captured = {}
    monkeypatch.setattr(train, "launch", lambda jobs, schedule, updates_per_step: captured.update(jobs=jobs))
    monkeypatch.setattr(sys, "argv", ["train.py", "--experiment", "exp", "--buffers", "20000", "100000",
                                      "--groups", "shocked", "pilot_self_shock", "--seeds", "1", "2", "3",
                                      "--pilot-seeds", "1"])
    train.main()
    jobs = captured["jobs"]
    assert ("shocked", 3, 100_000, "exp/buf100k") in jobs and ("pilot_self_shock", 1, 20_000, "exp/buf20k") in jobs
    assert not any(group == "pilot_self_shock" and seed > 1 for group, seed, _, _ in jobs)
    assert len(jobs) == 2 * (3 + 1)
