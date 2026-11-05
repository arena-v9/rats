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

*Not run yet.*

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
