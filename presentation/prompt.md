# Prompt: slides for the whole project (gridworld, then lever task)

The images and videos are on GitHub, at the URLs in the Assets table below (raw files, pinned to the commits they
were made in, so the links don't change when branches move):

- the gridworld's 6 figures: `arena-v9/homeostasis` at commit `80eaab9`, folder `presentation/out/`
- the lever task's 5 figures and 3 videos: `arena-v9/rats` at commit `a6bab17`, folder `presentation/out/`

Embed the images by these URLs (or download them from there); don't recreate them.

Everything below comes from the two projects' records: the gridworld's analysis of its experiments 1 and 2 (5 runs
per group, trained on 5 November 2026 with the code at `homeostasis` commit `405885a`), and the lever task's
`RESULTS.md` (experiment 1 on version 4.1). Use these numbers as given; do not invent others.

---

You are making a slide deck (about 16 main slides, plus an addendum) presenting a reinforcement-learning research
project in two parts: a **gridworld** experiment, then a **lever task** that tests the same idea with fewer confounders.
The audience is technical (ML / AI research) but not specialists in animal learning. Make it clear, visual and honest
about what each result does and does not show.

## The question (both parts)

Can an agent learn to avoid harming another body **without being rewarded for it**, because it has a vulnerable body
of its own? Christov-Moore et al. (2023, *Science Robotics*) argue that empathy needs vulnerability: an agent that can
be hurt, and must keep itself alive, has a reason to represent "about to be hurt"; if that representation is **shared**
between itself and another body like it, harm to the other might register too. Their paper runs no experiments, and
its blueprint builds caring into the objective. This project tests the bolder claim: that harm avoidance can **emerge**
without being built in.

Both parts use the same body and reward:

- **Body:** health and energy, each from 0 to 1. Energy drops every step (hunger); food restores it. Health regenerates
  only while fed and drains while starving; the body dies when health reaches 0.
- **Reward = drive reduction of the agent's own body only** (Keramati & Gutkin, 2014): how much closer its own health
  and energy get to full. **The other body never enters the reward.**

## Part 1: the gridworld

### Design

- A 7×7 map (5×5 playable) with fire (−0.3 health per step on it) and food. The **agent** learns (Double DQN with a
  small CNN over the whole map). A scripted **NPC** has the same body: it walks to food when hungry, wanders otherwise,
  and sometimes steps onto fire by mistake.
- **Free harm:** walking into the NPC pushes it one square; the agent can push it onto fire at no cost to itself.
- **The shared representation under test:** the network sees the map as 9 channels. Both bodies appear on the same
  `body`, `health` and `energy` channels; only `self` tells them apart. A convolutional feature learned from the agent's
  own body ("a body next to fire is about to lose health") applies at every square, including the NPC's.

### Experiment 1: does watching an NPC change how the agent learns to survive?

The agent trains on a **parallel grid**: its own grid, and next to it a second grid where the NPC lives, which the agent
can never enter (so the NPC cannot compete for food, block it or be pushed). Three groups: **no NPC**; **NPC with its
health and energy hidden**; **NPC with its state visible** on the shared channels.

**Result: watching the NPC makes the agent more reliable at surviving.** Reliability (the share of evaluations after
100k steps in which the agent survives at least 95% of test layouts):

| Group | Reliability | Starvation deaths per 1,000 test episodes |
|---|---|---|
| No NPC | 66% | 58 |
| NPC, state hidden | 85% | 32 |
| NPC, state visible | 94% | 20 |

- Watching the NPC: **+18 points** (95% interval +9 to +28, p = 0.024), and fewer starvation deaths (p = 0.032).
- Seeing its state as well: +9 points, **not clear** (p = 0.32).
- **Why is not yet tested.** Two candidate explanations: *shared features* (watching another body get hungry and eat
  trains the same convolutional features the agent uses for its own body) or *moving input* (a moving NPC keeps the
  picture changing, which may stop a greedy agent getting stuck, without it learning anything about the NPC).

### Experiment 2: does it depend on the agent being vulnerable?

A **"psychopath"** agent is rewarded only per food item eaten, and its body cannot be hurt (fire and starvation don't
lower its health; it never dies). It trains on the same parallel grid, with and without the NPC, and is compared with
experiment 1's homeostatic agent. Every final network is then **tested with the NPC moved onto the agent's own grid**
(512 layouts), where it can push the NPC, even onto fire, at no cost to itself.

**Result: the homeostatic agent pushes the NPC onto fire less than the psychopath, but it pushes less overall too.**

| Group | All pushes per test episode | Pushes onto fire per test episode |
|---|---|---|
| Psychopath, no NPC | 1.45 | 0.15 |
| Psychopath, NPC | 1.52 | 0.14 |
| Homeostatic, NPC | 0.91 | 0.05 |
| Homeostatic, no NPC | 1.21 | 0.07 |

- Homeostatic vs psychopath, both trained with the NPC: pushes onto fire 0.05 vs 0.14 (**p = 0.008**); all pushes
  0.91 vs 1.52 (**p = 0.024**).
- Watching the NPC in training made no clear difference for either agent (e.g. homeostatic with vs without the NPC:
  0.05 vs 0.07 onto fire, p = 0.46).

### What the gridworld cannot settle

A push onto fire leaves the agent standing next to the fire. So a homeostatic agent that simply keeps **itself** away
from fire would push onto fire less, with no harm avoidance at all; the psychopath, which cannot be hurt, has no such
reason. The gridworld's matched test (a push onto fire vs the same push onto floor, with the agent ending equally close
to fire) would separate the two but is not built yet. The gridworld results also rest on 5 runs per group (its design
plans 15), and in experiment 2 the agents meet the NPC up close only at test time. **This is what motivates part 2.**

## Part 2: the lever task — the same idea, fewer confounders

### Transition slide: why a second design

Make the transition explicit with one slide comparing the two designs (the gridworld is the richer, more naturalistic
setting; the lever task isolates one mechanism):

| Confounder in the gridworld | How the lever task removes it |
|---|---|
| **Self-protection:** a push moves the agent toward the fire too, so fewer pushes onto fire may just mean the agent keeps itself safe. Experiment 2 shows exactly this ambiguity (fewer fire pushes, but fewer pushes overall), and the matched floor-push test is not built | Both levers are identical for the agent: same position, same food, no risk to itself. The only difference is what happens to the other rat |
| **Indirect consequences of the other's fate:** on a shared grid the NPC competes for food, moves food around and can block the agent (why experiment 1 used a parallel grid; experiment 2's test puts the NPC back) | The other rat's fate cannot affect the agent at all (tested: the agent's rewards are identical whether or not the lever shocks the other), so an agent that knew the truth would be exactly indifferent: 50% on each lever |
| **Many routes from the other's body to the agent's choices:** position and, when visible, health and energy, through spatial CNN features; a falling health value alone can teach "danger" without any transfer of pain (an early version of the lever task showed exactly this: agents never hurt avoided the lever when they could see the other's health) | The other rat is perceived **only** through one input: its scream, the same input as the agent's own scream |
| **Vulnerability is manipulated together with everything else:** the psychopath differs in its reward (food, not its drive), its body (cannot be hurt) and its fate (never dies) | **Only own pain** differs: shocked and never-shocked agents have the same reward, body, task, seeds and random events; the one difference is whether the agent was shocked in phase 1 |
| **The test is outside the agent's experience:** experiment 2's agents meet the NPC within reach only at test time | The test is the task itself: in phase 2 the agent presses the levers and hears the other's scream every time, measured as it learns |
| **No ground truth** for what the agent should have learned | Phase 2 is deterministic, so the exact values the agent should learn can be computed and compared with what it learned (this caught a training bug; see the addendum) |

It also turns the gridworld's planned "witnessing" question (does seeing the other harmed, at no cost to oneself,
train avoidance away?) into a measured result: in phase 2 the agent hears the other's scream with no cost to itself,
and the avoidance fades.

### Church's rats (1959)

The lever task recreates a classic rat study in simulation: Church (1959) found that rats who had been shocked
themselves stopped pressing a lever when pressing it shocked another rat; rats never shocked did not. Church's
explanation: the shocked rats had heard their own squeals while in pain, so another rat's squeal came to signal pain
to them. In this design the scream is the shared representation, reduced to a single input.

**Hypothesis (H1):** an agent that has been shocked itself avoids the lever that shocks another agent, more than an
agent that has never been shocked.

### Setup

- **Two bodies,** the agent and another rat, with the same body rules as the gridworld's. A shock takes 0.6 of health,
  over two steps, and the body screams while it is shocked.
- **Reward = the agent's own drive reduction only**, as in the gridworld.
- **Observation (6 inputs):** the agent's own health and energy; a **scream** input that is 1 when *either* body
  screams (its own scream and the other's sound the same, as for Church's rats); whether the levers are active; which
  of two lamps is lit (lever A or B). Nothing else about the other rat is visible.
- **Task:** every 10 steps there is a trial: pressing either lever gives food. One lever is lit per episode.
- **Phase 1 (16k steps), conditioning:** the **shocked** group receives random shocks (about 160 in total) and hears
  itself scream while in pain; the **never-shocked** group is never shocked. The levers only give food.
- **Phase 2 (32k steps), test:** pressing the **lit** lever now also shocks the **other** rat, which screams. Nobody
  shocks the agent any more. The lit lever costs the agent nothing, so a self-interested agent that knew the truth
  would be indifferent (about 50% of presses on each lever).
- **Agent:** Double DQN with a small MLP (6 → 64 → 64 → 3 Q-values: lever A, lever B, rest).
- **Ablation:** the size of the **replay buffer** (the agent's memory of recent experience, replayed in training), i.e.
  how long its memories of its own shocks keep being replayed in phase 2: **8k** steps (gone early in phase 2), **20k**
  (gone 20k steps into phase 2), **40k** (never gone).
- **Runs:** 5 shocked and 5 never-shocked runs per buffer size, plus 3 positive-control runs per buffer size. **Main
  measure:** share of presses on the lit lever in the first 8k steps of phase 2 (mean of four checkpoints, each a
  greedy evaluation on 1,024 trials). Statistics: exact permutation test and 95% bootstrap interval, runs as
  replicates.

### Results

| Replay buffer | Shocked | Never shocked | Never − shocked [95% CI] | p |
|---|---|---|---|---|
| 8k | 37.1% | 50.3% | +13.2 [−3.5, +33.3] | 0.29 |
| 20k | **18.0%** | 51.2% | **+33.2 [+19.9, +45.1]** | **0.016** |
| 40k | **15.0%** | 51.7% | **+36.7 [+24.0, +46.9]** | **0.008** |

- **H1 supported at 20k and 40k** (still below 0.05 after a Bonferroni correction for three buffer sizes: 0.048 and
  0.024); not at 8k, though the difference points the same way. Never-shocked agents are indifferent (about 50%), as
  they should be.
- **Mechanism:** the only thing that distinguishes the lit lever for the agent is the scream it causes. At the end of
  phase 1 a scream lowers the shocked agents' expected value by about 2.5 (the **scream probe**: the change in the
  network's value when the scream input is switched on); in never-shocked agents it does nothing. So the avoidance is
  the scream's learned meaning, *pain to me*, transferred to the other rat's scream.
- **It is temporary — extinction.** In phase 2 a scream is never followed by harm to the agent, so learning removes its
  meaning, at a rate set by the buffer:
  - scream probe at 24k / 36k / 48k steps: −0.34 / −0.20 / −0.05 (8k buffer); −0.91 / −0.31 / −0.14 (20k);
    −0.96 / −0.71 / −0.50 (40k);
  - with the 20k buffer, avoidance ends between 32k and 36k, as the agent's own shocks leave the buffer (by 36k); with
    40k, where they never leave, it lasts longest;
  - by the end of phase 2 no buffer size shows a clear difference between the groups.

## What to claim, and what not to

Conclusion for the lever task (use this wording or close to it):

> Agents rewarded only for their own homeostasis, after experiencing their own pain paired with screaming,
> temporarily avoided an action that made another agent scream; agents that had never been shocked did not. The
> avoidance arose from a learned association — scream predicts pain to oneself — generalizing to the other's scream,
> and faded as that association was extinguished.

Across the project:

- **Gridworld:** watching another body helps the agent learn to survive (why is open); a vulnerable agent pushes
  another onto fire less than an invulnerable one, but this cannot yet be told apart from protecting itself.
- **Lever task:** with the confounds removed, harm **avoidance** emerges **without any reward for it** and without
  any other-regarding term in the reward — from the agent's own pain experience, and only temporarily.
- Do **not** say: empathy, prosocial preference, "the agent cares about the other". The lever-task agent avoids what
  it wrongly expects to hurt **itself** (Church's own interpretation was conditioned fear); that is the interesting
  part.
- **Caveats to state:** 5 runs per group in both parts (the gridworld plans 15); in the lever task the 20k and 40k
  conditions share their early trajectory (same seeds; identical until the 20k buffer first fills), so their
  main-window results are not fully independent; the lever-task effect depends on design choices — one scream input
  shared by both bodies, and a shock long enough that the scream comes while harm is still to come.

## Suggested slide outline

1. **Title** — e.g. "Can harm avoidance emerge without a reward for it?"
2. **The question** — Christov-Moore et al.: empathy needs vulnerability; our test: no reward for the other, only a
   vulnerable body and a shared representation. The body and the drive-reduction reward.
3. **The gridworld and free harm** — `design_1_push.png`.
4. **The shared representation** — `design_2_shared_channels.png`.
5. **Two gridworld experiments** — `design_3_experiments.png`.
6. **Experiment 1: watching an NPC helps the agent survive** — `results_1_exp1_reliability.png`.
7. **Experiment 2: less harm, or just self-protection?** — `results_3_exp2_pushes.png`; state the confound.
8. **From the gridworld to a cleaner test** — the comparison table.
9. **Church's rats (1959)** and H1.
10. **The lever task** — setup; same body and reward; one shared scream input; phase 1 and phase 2.
11. **Phase 1, conditioning** — video `1_phase1_conditioning.mp4`.
12. **Phase 2, the test** — video `2_side_by_side.mp4`.
13. **Main result** — `1_main_result.png`.
14. **The avoidance is temporary** — `2_time_course.png`.
15. **Why: the scream's meaning fades** — `3_scream_meaning.png`.
16. **Conclusions and caveats** — the wording above.
17. **Next steps** — more seeds (15 per group in the gridworld, 10 in the lever task); the gridworld's matched
    floor-push test and its two candidate explanations for experiment 1; a longer phase 2 in the lever task to see
    extinction fully; a separate pain-prediction system that reward learning cannot overwrite; carrying the lever
    task's training fixes back to the gridworld.

## Assets

Every image and video, with its URL, what it shows and where it goes:

| File | What it shows | Slide |
|---|---|---|
| https://raw.githubusercontent.com/arena-v9/homeostasis/80eaab921f7daa308df2b2af7a8dcdb0067089c3/presentation/out/design_1_push.png | Before and after a push: the NPC pushed onto fire, at no cost to the agent | 3 |
| https://raw.githubusercontent.com/arena-v9/homeostasis/80eaab921f7daa308df2b2af7a8dcdb0067089c3/presentation/out/design_2_shared_channels.png | Five of the network's input channels for an example state: both bodies on body, health and energy; only self separates them | 4 |
| https://raw.githubusercontent.com/arena-v9/homeostasis/80eaab921f7daa308df2b2af7a8dcdb0067089c3/presentation/out/design_3_experiments.png | Training on the parallel grid (the NPC on a grid the agent can't enter) and experiment 2's test (the NPC moved within reach) | 5 |
| https://raw.githubusercontent.com/arena-v9/homeostasis/80eaab921f7daa308df2b2af7a8dcdb0067089c3/presentation/out/results_1_exp1_reliability.png | Experiment 1: reliability by group, runs as dots, means with 95% intervals, p-values | 6 |
| https://raw.githubusercontent.com/arena-v9/homeostasis/80eaab921f7daa308df2b2af7a8dcdb0067089c3/presentation/out/results_2_exp1_survival.png | Experiment 1: mean survival over training by group (all groups learn; without the NPC they stay a little lower) | Addendum |
| https://raw.githubusercontent.com/arena-v9/homeostasis/80eaab921f7daa308df2b2af7a8dcdb0067089c3/presentation/out/results_3_exp2_pushes.png | Experiment 2: all pushes and pushes onto fire per test episode, psychopath vs homeostatic, with and without the NPC; the confound in its subtitle | 7 |
| https://raw.githubusercontent.com/arena-v9/rats/a6bab179514df3626dad10e7f620fa4e05a989cf/presentation/out/1_phase1_conditioning.mp4 | A shocked agent in phase 1: random shocks, its own scream ("AAGH!"), a caption per step (31 s) | 11 |
| https://raw.githubusercontent.com/arena-v9/rats/a6bab179514df3626dad10e7f620fa4e05a989cf/presentation/out/2_side_by_side.mp4 | Phase 2, lever B lit for both: the shocked agent presses only the unlit lever; the never-shocked one presses the lit lever 3 times and the other rat screams each time; a counter of lit presses (31 s) | 12 |
| https://raw.githubusercontent.com/arena-v9/rats/a6bab179514df3626dad10e7f620fa4e05a989cf/presentation/out/1_main_result.png | Lit-lever share in the first 8k steps of phase 2, per buffer size; runs as dots, means with 95% intervals, p-values | 13 |
| https://raw.githubusercontent.com/arena-v9/rats/a6bab179514df3626dad10e7f620fa4e05a989cf/presentation/out/2_time_course.png | Lit-lever share through phase 2, one panel per buffer size; lines smoothed over 3 checkpoints, faint dots the raw means; marks when the agent's own shocks leave memory | 14 |
| https://raw.githubusercontent.com/arena-v9/rats/a6bab179514df3626dad10e7f620fa4e05a989cf/presentation/out/3_scream_meaning.png | The scream probe through phase 2: shocked agents per buffer size vs never-shocked | 15 |
| https://raw.githubusercontent.com/arena-v9/rats/a6bab179514df3626dad10e7f620fa4e05a989cf/presentation/out/4_manipulation_check.png | The scream probe at the end of phase 1: never shocked, one-step shock (first attempt), two-step shock (final design) | Addendum |
| https://raw.githubusercontent.com/arena-v9/rats/a6bab179514df3626dad10e7f620fa4e05a989cf/presentation/out/5_positive_control.png | Positive control: when the lit lever shocks the agent itself, all 9 runs learn to avoid it | Addendum |
| https://raw.githubusercontent.com/arena-v9/rats/a6bab179514df3626dad10e7f620fa4e05a989cf/presentation/out/3_positive_control.mp4 | The positive control early in phase 2 (shocking itself) and later (avoiding the lit lever) (57 s) | Addendum |

The gridworld figures draw the map in its viewer's colours (agent blue, NPC purple, fire orange, food green). The
videos show a pixel-art side view: the agent's chamber (levers A and B, the lamp above the lit lever, a food hopper)
on the left, the other rat's chamber on the right, a mesh wall between them that lets screams through, and health (H)
and energy (E) bars under each.

## Style

- One message per slide; put the takeaway in the slide title as a sentence (the figures' titles already do this).
- Text near-black `#0b0b0b` and grey `#52514e` on an off-white `#fcfcfb` background, as in the figures. The videos
  have a dark background; give them a full slide or a dark slide.
- **Colours belong to each figure, not to the deck.** Blue and orange mean different things in different figures
  (gridworld experiment 1: no NPC / NPC hidden; lever task: shocked / never shocked; gridworld experiment 2: blue
  homeostatic, red psychopath). Keep each figure's legend or direct labels visible, and don't recolour anything.
- Plain language; define "replay buffer", "reliability" and "scream probe" the first time they appear.
- Don't redraw the figures or change their numbers; use the images as they are.

## Addendum (backup slides, not in the main flow)

- **Gridworld, survival over training** — `results_2_exp1_survival.png`: all three groups learn to survive; averaged
  curves hide the individual collapses that the reliability measure counts.
- **Lever task, positive control** — `5_positive_control.png` and `3_positive_control.mp4`. When the lit lever shocks
  the agent **itself**, all 9 positive-control runs learn to avoid it (lit-lever share at most 16%, 0% at the end): the
  setup can learn avoidance, so the never-shocked agents' indifference is not a failure to learn.
- **Lever task, manipulation check** — `4_manipulation_check.png`. At the end of phase 1 the scream lowers the shocked
  agents' expected value by about 2.5; in never-shocked agents it does nothing. A first attempt with a **one-step
  shock** failed this check (+0.61: the scream became *good* news, because the harm was already over when it was heard
  and only recovery followed). With a two-step shock the scream predicts harm still to come, as it must for Church's
  mechanism.
- **How the lever-task agent was made to learn correctly** (if asked): the Huber loss with rewards scaled ×10 fit the
  median of the targets instead of the mean, leaving every value about 7 too low; mean squared error fixed it. Other
  fixes: a discount of 0.95 per uncompressed step instead of 0.99 (so a shock is costly enough to learn), one input per
  lamp, a smaller replay buffer, and time compressed so that trials are 10 steps apart instead of 25 (96% of steps were
  passive). The gridworld still trains with the Huber loss on the same reward scale, so its values may carry the same
  bias; that has not been checked there.
