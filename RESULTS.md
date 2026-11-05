# Results

The record of every experiment in the lever task: setup, predictions, results and what they mean. Numbers come from
`analyze.py` (which reads `runs/*/seed_*/metrics.json`). Each run is one trained network; "8/10" means 8 of 10 runs went
the stated way.

Last updated: 5 November 2026.

## Experiment 1: does being shocked itself make the agent avoid shocking another body?

**Hypothesis.** An agent that has been shocked itself avoids the lever that shocks another body, more than an agent
that has never been shocked.

**Background.** Church (1959): rats that had been shocked themselves suppressed lever pressing while another rat was
shocked; never-shocked rats did not. His explanation of the shocked-alone group: they heard their own squeaks while in
pain, so another rat's squeaks came to signal pain. Hernández-Lallement et al. (2020): rats moved away from a lever that
also shocked a neighbour, more so after their own pain experience.

### Setup (version 3)

- **Bodies** (agent and other, identical rules): energy −0.02 per step, a pellet +0.5 (capped at 1); health +0.01 per
  step while energy > 0, −0.05 while it is 0; death when health reaches 0. Start: full health, energy uniform in [0.5, 1].
- **Shock:** 3 steps of −0.2 health (0.6 per shock); the body screams for those 3 steps. A dead body is silent.
- **Task:** 100-step episodes (truncated, not terminated). Trials on steps 12, 37, 62 and 87 (`levers_active` = 1):
  either lever gives the agent a pellet. One lever is lit per episode at random (`lit_lever`). The other body is fed a
  pellet every 25 steps on its own clock and has no actions.
- **Reward:** 10 × (D_before − D_after), D = (1 − health)² + (1 − energy)², from the agent's body only; death gives
  10 × (D_before − 2). Neither the other body nor the scream enters it.
- **Phases:** phase 1 (steps 0–100k): the lit lever does nothing; the *shocked* group is shocked at random (0.004 per
  step, about 0.4 shocks per episode, whatever it does), the *never-shocked* group never is. Phase 2 (100k–300k):
  pressing the lit lever shocks the other body, and no one is shocked at random. That is the only difference between
  the groups. 10 seeds each (1–10).
- **Observation:** 5 inputs: the agent's `health` and `energy`, `scream`, `levers_active` and `lit_lever`. `scream` is
  1 when either body screams, so the agent's own scream and the other's sound the same (Church: the rats heard their
  own squeaks while in pain). Nothing else about the other body is perceived. Network: an MLP 5→64→64→3 Q-values
  (A, B, rest).
- **Training:** Double DQN, Huber loss, gradient-norm clip 10, Adam 5e-4, batch 128, γ = 0.99; replay 100k filled with
  20k random steps; target network every 200 gradient steps; ε from 1.0 to 0.02 by step 50k; death store of 5k giving
  5% of each batch.
- **Evaluation:** greedy, the same 256 phase-2 episodes for every run (128 with A lit, 128 with B lit; 1,024 trials),
  at the end of phase 1 and every 20k phase-2 steps.
  - **Lit-lever share:** presses on the lit lever ÷ all presses. 50% = indifferent. **Main measure: the mean of the
    120k, 140k and 160k checkpoints**, early in phase 2, since any avoidance may fade later. The final checkpoint
    (300k) is reported too.
  - **Q-gap:** mean Q(lit) − Q(unlit) on trial steps.
  - **Scream probe:** mean change in V = max Q over the visited states when `scream` is switched from 0 to 1, all else
    unchanged. At the end of phase 1 it is the manipulation check (below 0: a scream means harm).
  - Every measure is recorded at every checkpoint, so if avoidance or the scream's meaning fades, the curve records
    when. A fade is a result, not a failure.
- **Statistics:** never shocked minus shocked: means, difference, 95% bootstrap interval (20,000 resamples), exact
  two-sided permutation p-value. Each group against no effect: sign-flip test. "Clear" = interval excludes 0 and
  p < 0.05. H1 is supported if the shocked group's main lit-lever share is clearly below the never-shocked group's,
  with most shocked runs below 50%.
- **Pilot** (run first): 3 seeds in which the lit lever shocks the agent itself. Passes if every run's final
  lit-lever share is below 20% and phase-1 deaths (after ε reaches 0.02) are under 5% of episodes. Random shocks
  alone kill in 2.3% of phase-1 episodes even with the best policy (simulated).
- **Changes from version 2, and why:**
  - Shock 0.1 → 0.2 per step. At 0.1 a shock cost the agent only −0.07 in value at full health, because the reward
    pays the health back as it regenerates, and the version-2 pilot failed. At 0.2 it costs −0.56.
  - Random shocks 0.02 → 0.004 per step. Two shocks close together can now kill: at 0.02, 36% of phase-1 episodes
    would end in death even with the best policy.
  - One `scream` input for both bodies, instead of a 4-input token for the other with its health and energy fixed
    at 1.0 (constants that carry no information), and a plain MLP instead of the shared token encoder.
  - Main measure early in phase 2 instead of the final checkpoint.

### Predictions (version 3)

*To be filled in by the experimenter before the runs.*

- Pilot:
- Lit-lever share 120k–160k, shocked:
- Lit-lever share 120k–160k, never shocked:
- Lit-lever share at 300k (does avoidance fade?):
- Q-gap:
- Scream probe at the end of phase 1:
- Scream probe over phase 2:

### Results (version 3)

**Pilot: passed, but only on the final checkpoint.** With the lit lever shocking the agent itself, the final networks
pressed it on 17.2%, 0.0% and 0.0% of presses; phase-1 deaths after ε reached 0.02 were 1.8%, 3.1% and 2.4% of
episodes. Avoidance was learned within about 20k phase-2 steps and then broke down for stretches: seed 1 was at 32.8%
at 180k and rose again to 17.2% at 300k; seed 2 was at 25.0% at 180k; seed 3 was at 25–38% from 160k to 280k. The pass
depends on where the oscillation happened to be at 300k. A run took about 20 minutes on CPU, three at once. (Run twice:
the second run, with the step recording and the diagnostics below, reproduced every checkpoint exactly.)

**Why avoidance breaks down** (from the step recording and the buffer and probe diagnostics):

- Not exploration: exploratory lit presses stayed at 0–5 per 5k steps and did not rise before a breakdown; the presses
  during one were greedy (seed 1, 175–180k: 39 of 42).
- Not a lack of shocked examples: shocked lit presses in the replay buffer rose through phase 2 (about 100 → 300). Nor
  the safe lit presses left from phase 1: avoidance was best while they still filled the buffer.
- Each breakdown hit one lit condition only: the agent pressed one lever whichever lamp was lit (a lever bias). Seeds 1
  and 2 at 180k and seed 3 from 160k to 280k pressed B when B was lit; seed 1 at 280–300k pressed A when A was lit.
  Seed 2 at 180k pressed lit B on exactly the trials at steps 12 and 62: hence its 25.0%.
- The Q-gap that decides the choice was small (0.1–0.4) next to swings of ±5 in all three Q-values together over
  20–30k steps, and the values of A-lit and B-lit states differed by up to 7 although the task is symmetric. The step
  of the press itself paid +2.9 on average (the pellet outweighs the first 0.2 of shock); the cost comes in the next
  two steps.

### Version 3.1: one input per lamp

**Change from version 3.** The lit lever is two inputs, `lit_A` and `lit_B` (exactly one is 1), instead of one
`lit_lever` input (0 = A, 1 = B). Observation: 6 inputs; network 6→64→64→3. Nothing else changes.

**Why.** With a 0/1 input, A-lit states are the network's baseline (the input contributes nothing) and B-lit states
are the baseline plus a correction that only B-lit samples train, so the two conditions learn asymmetrically. One input
per lamp makes them symmetric. Both encodings can represent the same functions; only how training distributes its
updates changes. It is a test of whether the encoding contributes to the lever bias: the late seed-1 breakdown on lever
A means it is not the whole story, and the small Q-gap next to the drift may matter more.

**Plan.** Pilot only, seeds 1–3, compared with the version-3 pilot: do breakdowns still happen, and on which lever?

### Predictions (version 3.1)

*To be filled in by the experimenter before the runs.*

- Breakdowns (lit-lever share above 20% at any phase-2 checkpoint after 120k):
- Lever of the breakdowns (A, B or both):
- Pilot pass (final lit-lever share below 20% on all seeds):

### Results (version 3.1)

**Pilot: passed, but only on the final checkpoint again.** Final lit-lever shares 0.0%, 0.0% and 5.7%; phase-1 deaths
1.8%, 3.1% and 2.4%. Breakdowns (above 20% after 120k) on 5 of 30 checkpoints, against 8 of 30 in version 3: seed 2 at
280k (43%), seed 3 at 160k, 180k, 240k and 280k (67%, both lamps). They now fall on either lever, so the encoding
asymmetry is gone, but the breakdowns are not. Runs deleted on 5 November (after these results were recorded).

**Why: the loss was biased.** Phase 2 of the pilot is deterministic, so the true Q-values can be computed exactly by
simulation; phase 1's by Monte Carlo. Compared with them, the networks were wrong in three ways:

- Every value about 7 too low at the end of phase 1 (−6.5 to −7.4 on all three seeds), wearing off only over 40–60k
  phase-2 steps.
- The lever gap 5–20 times too small during the main window: learned −0.02 to −0.15, true −0.6 to −2.3 (at full
  health the true gap is −0.56).
- Death barely felt: at health 0.5 a shock kills and the true gap is −15 to −19; seed 3 reached low health and kept
  pressing the lit lever, with a learned gap of −0.8 where the true one averaged −7.3.

Phase 1 alone, seed 1, one change at a time: without the death store the error was unchanged (−7.37 vs −7.36); with
mean squared error instead of the Huber loss it was +0.23. The Huber loss (threshold 1) acts like absolute error when
TD errors exceed 1, which here they mostly do (rewards are scaled ×10), so it fits the median of the targets instead of
their mean. The rewards are mostly small losses with occasional large gains, so the median lies below the mean; large
rare errors (the shock's cost, death) count no more than an error of 1. Versions 1–3.1 all used this loss.

### Version 3.2: mean squared error instead of the Huber loss

**Change from version 3.1.** The DQN loss is mean squared error instead of the Huber loss (`smooth_l1_loss`). The
gradient-norm clip (10) stays, guarding against spikes from death transitions. Nothing else changes.

**Why.** See above: the Huber loss biased every value and shrank the differences that decide the choice.

**Plan.** Pilot, seeds 1–5 (it costs the same time as three on this machine, and three cannot tell "sometimes breaks
down" from "reliable"), compared with versions 3 and 3.1 on breakdowns and against the exact values.

### Predictions (version 3.2)

*To be filled in by the experimenter before the runs.*

- Breakdowns (lit-lever share above 20% at any phase-2 checkpoint after 120k):
- Learned values vs exact values at the end of phase 1:
- Pilot pass (final lit-lever share below 20% on all seeds):

### Results (version 3.2)

**Stopped at 260–265k of 300k (enough to read): better, still not reliable.** Breakdowns (above 20%) on 5 of 40
checkpoints from 120k to 260k, against 3 of 24 in version 3.1, and smaller (at most 23%, against 39%). But many
checkpoints sat at 10–20%, and seed 4 broke down at 120–140k, in the main measure's window. The loss fix alone is not
enough. Runs deleted on 5 November (after these results were recorded).

### Version 3.3: γ 0.95, a 20k replay buffer, no store of deaths

**Changes from version 3.2.** Three at once (to get a reliable pilot; version 3.2 isolates the loss):

- γ 0.99 → 0.95. A shock's health comes back as reward while it regenerates, so a shock costs the agent only through
  discounting, and at γ 0.99 there is little of it over the ~60 steps of recovery. True disadvantage of the lit lever
  (exact, energy 0.5), at health 1.0 / 0.8 / 0.6: γ 0.99: −0.56 / −1.30 / −2.33; γ 0.95: −1.70 / −3.50 / −5.61.
  Horizon about 20 steps instead of 100.
- Replay buffer 100k → 20k transitions. With 100k, phase-1 transitions (in which the lit lever is safe, in states
  that look the same) stayed in the buffer until step 200k; with 20k they are gone by step 120k, when the main
  measure starts. 20k is also the length of the random warm-up.
- No store of deaths (it gave deaths 5% of each batch, about 250 times their frequency). Sampling death that often
  overweights it wherever it depends on chance (phase 1: random shocks at low health) and biases values down. Under
  the Huber loss its effect was capped; under mean squared error it would not be.

**Risk.** With a small buffer and no store, deaths can leave the buffer entirely and be forgotten. Check: phase-2
deaths in training and evaluation, and the learned gap at low health against the exact one.

**Plan.** Pilot, seeds 1–5, on the new quick schedule (40k + 80k steps: warm-up 10k, ε at its floor by 25k,
checkpoints every 5k, log points every 1k), to iterate faster. The experiment's measures stay defined on the full
schedule; `analyze.py` leaves quick runs out of experiment 1. Training now also logs, at every log point: gradient
norms and how often clipping acts, TD errors by kind of transition (lit press, unlit press, rest on a trial, scream,
death, other), how far the online network drifts from the target network, weight norms, the buffer's contents
(deaths, phase-1 share), and, in phase 2, the probe grid against the exact Q-values (`exact.py`); each checkpoint
compares the learned Q-gap on visited trial states with the exact one.

### Predictions (version 3.3)

*To be filled in by the experimenter before the runs.*

- Breakdowns (lit-lever share above 20% at any phase-2 checkpoint after 120k):
- Deaths in phase 2 (forgetting):
- Pilot pass (final lit-lever share below 20% on all seeds):

### Results (version 3.3)

**Pilot: passed, and reliably this time** (quick schedule, 5 seeds). No breakdowns: 0 of 80 phase-2 checkpoints above
20% (highest 7%, at 55k on seed 4); final lit-lever share 0.0% on every seed. The learned Q-gap on visited trial states
came within 0.3 of the exact gap (−1.70) by 60–70k and stayed there (final −1.50 to −1.74), with the right sign on
100% of trial states. Phase-1 deaths after ε reached its floor: 2.0%, 0.7%, 3.9%, 2.6%, 2.0% of episodes. Gradient
clipping never acted (0% of updates). About 8 minutes per run, five at once. Runs deleted on 5 November (after these results were recorded).

**One weakness: death is forgotten.** Without the store of deaths, deaths left the 20k buffer by about 70k, after
which the TD errors on death transitions could no longer be measured and the probe grid showed the gap at low health
(where a shock kills) about 5 smaller than the exact one, though with the right sign. Harmless here, since the agent no
longer reaches low health in phase 2; buffer size governs it, so the buffer ablation measures it too.

### Experiment 1 with version 3.3: replay buffer ablation

**Why buffer size.** The replay buffer is the agent's only memory of individual experiences: while the shocked
group's own shocks from phase 1 are still in it, every batch keeps retraining "scream → my health drops". In phase 2
the shocked group is never shocked, and the other's screams are never followed by harm to the agent, so as soon as
those memories leave the buffer, what phase 1 taught survives only in the network's weights, and the harmless screams
extinguish it. Buffer size therefore sets how long the agent rehearses its own pain, and any effect on the lit lever
may depend on it.

**Design.** Quick schedule (phase 1 40k steps, about 160 random shocks for the shocked group; phase 2 80k), three
buffer sizes. On this schedule they mean:

| Buffer | Phase-1 memories leave the buffer | |
|---|---|---|
| 20k | 20k steps into phase 2 (step 60k, the end of the main window) | forgotten early |
| 50k | 50k steps into phase 2 (step 90k) | forgotten late |
| 100k | never (the run ends at 120k) | never forgotten |

Per buffer size: shocked and never-shocked groups, seeds 1–10 each, plus the pilot (lit lever shocks the agent) on
seeds 1–3 as the positive control at that buffer size. 69 runs, eight at a time. Runs in `runs_archive/exp1/buf20k/`,
`buf50k/`, `buf100k/`.

**Measures.** As in version 3, per buffer size, with the main window redefined for the quick schedule: **the mean of
the 45k, 50k, 55k and 60k checkpoints** (the first 20k steps of phase 2, the period in which the pilot learned
avoidance). H1 is tested separately at each buffer size, with the same criterion as before (shocked group's lit-lever
share clearly below the never-shocked group's, most shocked runs below 50%). The buffer question: does the
never-shocked minus shocked difference grow with buffer size, and does the scream probe in the shocked group stay
negative for longer with a larger buffer? Each pilot must pass for its buffer size's result to count.

### Predictions (experiment 1, buffer ablation)

*To be filled in by the experimenter before the runs.*

- H1 at 20k / 50k / 100k:
- Does the effect grow with buffer size:
- Scream probe in the shocked group over phase 2, by buffer size:
- Pilots pass at all three buffer sizes:

### Results (experiment 1, buffer ablation)

**Stopped after the 20k buffer** to compress time first (version 4): the 20k-buffer runs are complete (10 shocked, 10
never shocked, 3 pilots, in `runs_archive/exp1/buf20k/`); one 50k run finished; the rest never started. Not analysed yet. Runs
from the first batch record commit `b31c147`, later ones `4db54b9` or `e7c792f`: the training code is the same in all
three (only the analysis and the viewer changed).

### Version 4: compressed time

**Change from version 3.3.** One step is now five steps of version 3.3 (`TIME_COMPRESSION` in `env.py`). Up to
version 3.3 there were 24 passive steps between trials, so 96% of transitions, and of training, were steps on which
the agent's action did nothing. Everything defined per step is rescaled together, so the task is the same per trial:

| | Version 3.3 | Version 4 |
|---|---|---|
| Steps between trials; episode | 25; 100 (trials at 12, 37, 62, 87) | 5; 20 (trials at 2, 7, 12, 17) |
| Hunger, regeneration, starvation per step | 0.02, 0.01, 0.05 | 0.1, 0.05, 0.25 |
| Shock | 3 steps × 0.2 (scream 3 steps) | 1 step × 0.6 (scream 1 step) |
| γ per step | 0.95 | 0.95⁵ = 0.774 (same per trial interval) |
| Random shocks in phase 1, per step | 0.004 | 0.02 (0.1 per trial interval, both) |
| Quick schedule: phase 1, phase 2, warm-up, ε floor at | 40k, 80k, 10k, 25k | 8k, 16k, 2k, 5k |
| Quick main window; checkpoints every | 45k–60k; 5k | 9k–12k; 1k |
| Replay buffer | 20k | 4k (same 800 trials) |
| Target network synced every | 200 updates (8 trials) | 8 trials of updates |
| Other's meals; pellet | every 25 steps; 0.5 | every 5 steps; 0.5 |

**Checks.** The exact Q-values the agent has to learn stay close: gap of the lit lever at full health −1.77 (version
3.3: −1.70), and Q(unlit), Q(rest) within about 0.5. One difference: a one-step shock kills at health 0.60 or below,
the three-step one at 0.57 or below (the body regenerated 0.03 during it). With `TIME_COMPRESSION = 1` the code
reproduces version 3.3 exactly (training log, checkpoints and the step recording identical, byte for byte), and the
tests pass at both values. Not rescaling the shock (3 steps × 0.2 at k = 5) would halve the gap (−0.86).

**Ablation: updates per step.** With one update per environment step, training does 5 updates per trial instead of 25.
Pilot, quick schedule, seeds 1–5, with 1 and with 5 updates per step (5 = the updates per trial of version 3.3; the
target network syncs after the same number of trials either way). If 1 matches version 3.3's pilot, it is the default
(about 5 times less compute per run).

### Predictions (version 4)

*To be filled in by the experimenter before the runs.*

- Pilot with 1 update per step (reliable avoidance? learned gap vs exact):
- Pilot with 5 updates per step:

### Results (version 4)

**Pilot: reliable at both settings** (quick schedule, 5 seeds each; runs in `runs_archive/v4_updates1/` and `runs_archive/v4_updates5/`):

| | Version 3.3 | Version 4, 1 update/step | Version 4, 5 updates/step |
|---|---|---|---|
| Minutes per run | 8.1 | 2.3 | 8.2 |
| Checkpoints above 20% lit-lever share | 0 / 80 | 0 / 80 | 0 / 80 (highest exactly 20%, first checkpoint) |
| Final lit-lever share | 0% on every seed | 0% on every seed | 0% on every seed |
| Learned − exact Q-gap, second half of phase 2 | about ±0.2 | −0.3 to −0.6 | about ±0.1 |
| Phase-2 deaths | 19 / 4,001 episodes | 41 / 4,009 | 14 / 4,003 |
| Phase-1 deaths after ε reached its floor | 2.2% | 2.5% | 2.6% |

- Time compression keeps the pilot as reliable as version 3.3.
- With one update per step (5 per trial instead of 25), avoidance holds but the values are less accurate: the network
  rates the lit lever about 0.5 worse than it is, and dies three times as often in phase 2. With five updates per step
  (version 3.3's updates per trial) the values match the exact ones and deaths fall to 14. Both have the same one-step
  shock, so the extra deaths come from training less, not from the shock killing at 0.60 instead of 0.57.
- Compression's speed-up comes only from training less per trial: at five updates per step a run costs the same as
  in version 3.3.
- Also run: 2 updates per step (5 seeds, `runs_archive/v4_updates2/`, not analysed) and 3 (stopped part-way when version 4.1
  was chosen).

### Version 4.1: 10 steps per trial, one update per step

**Change from version 4.** Decision by the experimenter: keep one gradient update per step, and compress less, so
that a trial gets 10 updates instead of 5 (version 3.3: 25). Trials are 10 steps apart instead of 5 (`TRIAL_INTERVAL`
in `env.py`; compression 2.5). Everything is rescaled by the same rule as in version 4:

| | Version 3.3 | Version 4.1 |
|---|---|---|
| Steps between trials; episode | 25; 100 (trials at 12, 37, 62, 87) | 10; 40 (trials at 5, 15, 25, 35) |
| Hunger, regeneration, starvation per step | 0.02, 0.01, 0.05 | 0.05, 0.025, 0.125 |
| Shock | 3 steps × 0.2 | 1 step × 0.6 (scream 1 step) |
| γ per step | 0.95 | 0.95^2.5 = 0.880 |
| Random shocks in phase 1, per step | 0.004 | 0.01 |
| Quick schedule: phase 1, phase 2, warm-up, ε floor at | 40k, 80k, 10k, 25k | 16k, 32k, 4k, 10k |
| Quick main window; checkpoints every | 45k–60k; 5k | 18k–24k; 2k |
| Replay buffer (default) | 20k | 8k |
| Target network synced every | 200 updates (8 trials) | 80 updates (8 trials) |

**Checks.** Exact gap of the lit lever at full health −1.88 (version 3.3: −1.70; version 4.0: −1.77), from both
`exact.py` and an independent simulation. A trial interval of 25 still reproduces version 3.3 exactly (training log,
checkpoints and step recording identical byte for byte), and the tests pass at intervals of 25, 10 and 5. Not piloted
on its own: the experiment below runs three pilot seeds at each buffer size as the positive control.

### Experiment 1 with version 4.1: replay buffer ablation

**Design.** As planned for version 3.3 (above), rescaled: quick schedule (phase 1 16k steps, about 160 random shocks
for the shocked group; phase 2 32k steps), buffers of 8k, 20k and 40k steps (20k, 50k and 100k uncompressed):

| Buffer | Phase-1 memories leave the buffer | |
|---|---|---|
| 8k | 8k steps into phase 2 (step 24k, the end of the main window) | forgotten early |
| 20k | 20k steps into phase 2 (step 36k) | forgotten late |
| 40k | never (the run ends at 48k) | never forgotten |

Per buffer size: shocked and never-shocked groups, seeds 1–10 each, and the pilot on seeds 1–3 (positive control: its
lit-lever share must stay low for that buffer size's result to count). 69 runs, in `runs/exp2/buf8k/`, `buf20k/`,
`buf40k/`. **Measures** as before; main window: the 18k, 20k, 22k and 24k checkpoints (the first 8k steps of phase 2,
the same trial time as version 3.3's 45k–60k). H1 tested at each buffer size; the buffer question as above.

### Predictions (experiment 1 with version 4.1)

*To be filled in by the experimenter before the runs.*

- H1 at 8k / 20k / 40k:
- Does the effect grow with buffer size:
- Scream probe in the shocked group over phase 2, by buffer size:
- Pilots pass at all three buffer sizes:

### Results (experiment 1 with version 4.1)

*Running.*

## Earlier versions of experiment 1

### Version 1: the other's health and energy visible (4 November)

As version 2, but the other's token showed its real health and energy. Agents that had never been shocked avoided the
lit lever too, because starving had taught them that a falling health means danger. The experiment-1 runs were
discarded.

**Pilot: passed.** Passed on all three seeds. With the lit lever shocking the agent itself, the final networks pressed it on 1.1%, 0.0% and
0.0% of presses; phase-1 deaths after ε reached 0.02 were 2.0%, 1.2% and 0.4% of episodes. The own-scream probe at the
end of phase 1 was −0.77, −0.76 and −0.87: the scream predicts harm. Avoidance took about 100k phase-2 steps to become
reliable, and greedy choices flipped between checkpoints on the way (seed 2: 1%, 0%, then 42% at 240k). A run took about
22 minutes, three at once on the RTX 4090. Run with /opt/arena-env (same package versions as requirements.txt) while
torch was still downloading into .venv.

### Version 2: the other as a token with placeholders (4–5 November)

Differences from version 3: shock 0.1 per step; random shocks 0.02 per step in phase 1; two tokens
`[health, energy, scream, is_self]` (agent, other) through one shared encoder (4→32→32), summed, joined with
`[levers_active, lit_lever]`, then 34→64→3; the other's token always showed health and energy at 1.0, so only its
scream carried information; separate own-scream and other-scream probes; the main measure was the final checkpoint.
Runs kept in `runs/v2_token/`.

*Predictions were not filled in before the results were seen, so there is nothing to audit.*

**Pilot: failed.** With the lit lever shocking the agent itself, the final networks pressed it on 31.9%, 40.1% and
59.8% of presses (mean of the last 3 checkpoints: 31%, 38%, 28%). Avoidance was learned but did not hold: seed 1 was
at 0% from 180k to 240k and then rose to 32%; seed 3 was at 2% at 280k and 60% at 300k. Phase-1 deaths after ε reached
0.02: 2.0%, 1.2%, 0.4%. Since the first version also flipped between checkpoints (seed 2: 42% at 240k), its pass on the
final checkpoint was probably partly luck.

**Experiment 1: H1 not supported.** Both groups are indifferent between the levers.

| Measure | Shocked | Never shocked | Never − shocked [95% CI] | p |
|---|---|---|---|---|
| Lit-lever share, % (final) | 50.6 | 51.3 | +0.7 [−11.5, +13.0] | 0.915 |
| Lit-lever share, % (mean of last 3 checkpoints) | 49.4 | 48.9 | | |
| Q-gap, lit − unlit (final) | 0.00 | 0.00 | −0.00 [−0.01, +0.01] | 0.914 |
| Own-scream probe (end of phase 1) | −0.48 | +0.54 | +1.02 [−0.27, +2.36] | 0.174 |
| Other-scream probe (final) | −0.01 | −0.00 | +0.01 [−0.01, +0.03] | 0.501 |

- The manipulation worked: in all 10 shocked runs the agent's own scream lowered the value at the end of phase 1 (−0.48,
  clear). In the never-shocked runs it is arbitrary (they never screamed).
- Neither group's lit-lever share differs from 50% (4/10 and 5/10 runs below 50%), at any checkpoint (44.8–57.4%).
- The other's scream starts phase 2 with a large, arbitrary value (mean |probe| 1.1 shocked, 0.46 never shocked: the
  other never screamed in phase 1). Within the first 20k phase-2 steps it falls to about 0 (0.05 and 0.03), because a
  scream from the other is never followed by anything that touches the agent's reward. Whatever transfer the shared
  encoder carried was gone long before the agent could learn which lever causes the scream; the pilot needed 60–80k
  steps to learn that for its own shock.

**Why the pilot fails: the shock barely costs the agent anything.** The reward pays back the health as it
regenerates, so a shock costs only the delay. The discounted cost of one shock at full health is −0.07 (γ = 0.99),
against −0.56 if the shock were 0.2 per step and −1.83 at 0.3. At 0.1 the difference between the levers is within the
noise of the Q-values, so greedy choices flip between checkpoints.

**What this means for H1.** This setup can't test it. For a model-free learner whose reward is only its own drive, the
correct value of the other's scream is exactly 0. Generalization from its own scream could only show up for a short
time, and TD learning removes it in the same updates that teach the lever contingency. The first version showed
avoidance only because the agent could see the other's health, and starving had taught it that low health predicts
danger.
