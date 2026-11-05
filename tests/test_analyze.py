import sys

import numpy as np

import train
from analyze import END_OF_PHASE1, FINAL, MAIN, values


def fake_run(schedule: str, shares: dict[int, float]) -> dict:
    run = {"schedule": schedule, "phase1_steps": train.SCHEDULES[schedule]["phase1_steps"],
           "checkpoints": [{"step": step, "lit_share": share} for step, share in sorted(shares.items())]}
    if schedule == "quick":
        run["main_window"] = list(train.SCHEDULES["quick"]["main_window"])
    return run


def test_quick_main_window_is_the_first_20k_steps_of_phase_2():
    run = fake_run("quick", {40_000: 0.9, 45_000: 0.1, 50_000: 0.2, 55_000: 0.3, 60_000: 0.4, 65_000: 0.9, 120_000: 0.5})
    assert values([run], "lit_share", MAIN)[0] == np.mean([0.1, 0.2, 0.3, 0.4])
    assert values([run], "lit_share", END_OF_PHASE1)[0] == 0.9
    assert values([run], "lit_share", FINAL)[0] == 0.5


def test_runs_without_a_schedule_use_the_full_main_window():
    run = fake_run("full", {100_000: 0.9, 120_000: 0.1, 140_000: 0.2, 160_000: 0.3, 180_000: 0.9})
    del run["schedule"]
    assert np.isclose(values([run], "lit_share", MAIN)[0], 0.2)


def test_buffer_ablation_jobs(monkeypatch):
    captured = {}
    monkeypatch.setattr(train, "launch", lambda jobs, schedule: captured.update(jobs=jobs))
    monkeypatch.setattr(sys, "argv", ["train.py", "--experiment", "exp", "--buffers", "20000", "100000",
                                      "--groups", "shocked", "pilot_self_shock", "--seeds", "1", "2", "3",
                                      "--pilot-seeds", "1"])
    train.main()
    jobs = captured["jobs"]
    assert ("shocked", 3, 100_000, "exp/buf100k") in jobs and ("pilot_self_shock", 1, 20_000, "exp/buf20k") in jobs
    assert not any(group == "pilot_self_shock" and seed > 1 for group, seed, _, _ in jobs)
    assert len(jobs) == 2 * (3 + 1)
