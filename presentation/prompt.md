# Prompt: slides for the lever-task experiment

Use this prompt with the figures and videos in `presentation/out/` attached. Everything below is from the experiment's
records (`RESULTS.md`, experiment 1 on version 4.1); use these numbers as given and do not invent others.

---

You are making the second part of a slide deck (about 8–10 main slides, plus an addendum) about a
reinforcement-learning research project. The audience is technical (ML / AI research) but not specialists in animal
learning. Make it clear, visual and honest about what the result does and does not show.

## Where this fits: after the gridworld experiment

The deck **opens with the gridworld experiment** ("Emergent Harm Avoidance from Shared Vulnerability: A Gridworld
Test", repository `homeostasis`, described here as of commit `405885a`), presented before these slides. By the time
your slides start, the audience has seen:

- the question: can an agent learn to avoid harming another body **without being rewarded for it**, because it has a
  vulnerable body of its own (Christov-Moore et al., 2023, argue that empathy needs vulnerability);
- the homeostatic body (health and energy) and the reward, **drive reduction** of the agent's own body only
  (Keramati & Gutkin, 2014);
- the mechanism under test, a **shared representation**: a scripted NPC with the same body appears on the same input
  channels as the agent's own (body, health, energy), so a CNN feature learned from the agent's own harm may also
  respond at the NPC's square;
- the gridworld's design: a 7×7 map with fire and food, where walking into the NPC pushes it (onto fire, at no cost to
  the agent); **experiment 1**, whether watching the NPC (on a separate "parallel grid" the agent cannot enter, with
  its health and energy hidden or visible) changes how the agent learns to survive; and **experiment 2**, a
  **"psychopath"** agent (rewarded only per food item eaten; fire and starvation cannot hurt it) against the
  homeostatic agent, tested on pushes onto fire with the NPC moved onto the agent's own grid.

**The gridworld's results** (5 runs per group, trained on 5 November with the code at `405885a`; its design plans 15
per group; figures in `homeostasis/presentation/out/`, listed under Assets). These are presented in the gridworld part;
refer back to them, do not re-present them:

- **Experiment 1: watching the NPC makes the agent more reliable at surviving.** Reliability (share of evaluations
  after 100k steps with at least 95% survival): no NPC 66%, NPC with its state hidden 85%, NPC with its state visible
  94%. Watching the NPC: +18 points (95% interval +9 to +28, p = 0.024), and fewer starvation deaths (58 vs 32 per
  1,000 test episodes, p = 0.032). Seeing its state as well: +9 points, not clear (p = 0.32). Why is not yet tested
  (candidates: shared features, or a moving input that stops the agent getting stuck).
- **Experiment 2: the homeostatic agent pushes the NPC onto fire less than the psychopath, but pushes less overall
  too.** Pushes onto fire per test episode: psychopath with the NPC 0.14, homeostatic with the NPC 0.05 (p = 0.008).
  All pushes: 1.52 vs 0.91 (p = 0.024). Watching the NPC in training made no clear difference for either agent
  (homeostatic with vs without the NPC: 0.05 vs 0.07 onto fire, p = 0.46).
- **Why the gridworld cannot settle it:** a push onto fire leaves the agent next to fire, so a homeostatic agent that
  keeps away from fire for its own sake would push onto fire less, with no harm avoidance at all; the matched
  floor-push test that would separate the two is not built. This is the bridge to the lever task: a design where the
  harmful and the harmless action are identical for the agent.

Do not repeat the gridworld's explanations at length: refer back to them ("the same body and reward as in the
gridworld", "as in the gridworld's experiment 2").

**This experiment (the lever task) is the same idea in a stripped-down design with fewer confounders.** It is the part
with results. Make the transition explicit with one slide comparing the two designs (the gridworld is the richer,
more naturalistic setting; the lever task isolates one mechanism):

| Confounder in the gridworld | How the lever task removes it |
|---|---|
| **Self-protection:** a push moves the agent too, toward the fire behind the NPC, so fewer pushes onto fire may just mean the agent keeps itself away from fire. Experiment 2 shows exactly this ambiguity: the homeostatic agent pushes onto fire less (0.05 vs 0.14 per episode) but also pushes less overall (0.91 vs 1.52), and the matched floor-push test is not built | Both levers are identical for the agent: same position, same food, no risk to itself. The only difference is what happens to the other rat |
| **Indirect consequences of the other's fate:** on a shared grid the NPC competes for food, moves food around and can block or be pushed out of the way (why the gridworld's experiment 1 moved it to a parallel grid; its push test puts it back on the agent's grid) | The other rat's fate cannot affect the agent at all (tested: the agent's rewards are identical whether or not the lever shocks the other), so an agent that knew the truth would be exactly indifferent: 50% on each lever |
| **Many routes from the other's body to the agent's choices:** the NPC is seen through its position and, when visible, its health and energy, through spatial CNN features; a falling health value alone can teach "danger" without any transfer of pain (an earlier version of the lever task showed exactly this: agents that had never been hurt avoided the lever when they could see the other's health) | The other rat is perceived **only** through one input: its scream, the same input as the agent's own scream. Nothing else about it is visible |
| **Vulnerability is manipulated together with everything else:** the psychopath agent differs from the homeostatic one in its reward (food eaten, not its drive), its body (it cannot be hurt) and its fate (it never dies), so a difference between them can come from any of these | **Only own pain** is manipulated: the **shocked** and **never-shocked** groups have the same reward, body and task, the same seeds and the same random events; the one difference is whether the agent was shocked in phase 1 |
| **The test is outside the agent's experience:** in experiment 2 the agents trained with the NPC on a separate grid and meet it on their own grid only at test time, so their behaviour there may reflect novelty rather than harm | The test is the task itself: in phase 2 the agent presses the levers and hears the other's scream every time, and its choices are measured as it learns |
| **No ground truth:** whether the agent learned the right values is hard to check | Phase 2 is deterministic, so the exact values the agent should learn can be computed and compared with what it learned (this caught a training bug, see the addendum) |

It also turns the gridworld's "witnessing" question (does seeing the other harmed, at no cost to oneself, train
avoidance away? its stored vs dropped NPC-harm condition) into a measured result: in phase 2 the agent hears the
other's scream with no cost to itself, and the avoidance fades, at a rate set by how long its own pain stays in its
replay memory.

## The question

The lever task recreates a classic rat study in simulation: Church (1959) found that rats who had been shocked
themselves stopped pressing a lever when pressing it shocked another rat; rats never shocked did not. Church's
explanation: the shocked rats had heard their own squeals while in pain, so another rat's squeal came to signal pain
to them. In this design the scream is the shared representation, reduced to a single input.

**Hypothesis (H1):** an agent that has been shocked itself avoids the lever that shocks another agent, more than an
agent that has never been shocked.

## The setup

- **Two bodies:** the agent and another rat, each with health and energy, with the same body rules as the
  gridworld's (hunger, food, regeneration only while fed, starvation, death at zero health). A shock takes 0.6 of
  health (over two steps).
- **Reward = the agent's own drive reduction only**, as in the gridworld (how much closer its own health and energy get to full). The
  other rat and the scream never enter the reward. This is the crux of the experiment.
- **Observation (6 inputs):** the agent's own health and energy; a **scream** input that is 1 when *either* body
  screams (the agent's own scream and the other's sound the same, as for Church's rats); whether the levers are active;
  which of two lamps is lit (lever A or B). Nothing else about the other rat is visible.
- **Task:** every 10 steps there is a trial: pressing either lever gives food. One lever is lit per episode.
- **Phase 1 (16k steps), conditioning:** the **shocked** group receives random shocks (about 160 in total) and hears
  itself scream while in pain; the **never-shocked** group is never shocked. The levers only give food.
- **Phase 2 (32k steps), test:** pressing the **lit** lever now also shocks the **other** rat, which screams. Nobody
  shocks the agent any more. The lit lever costs the agent nothing, so a purely self-interested agent that knew the
  truth would be indifferent (about 50% of presses on each lever).
- **Agent:** Double DQN, a small MLP (6 → 64 → 64 → 3 Q-values: lever A, lever B, rest), one gradient update per
  step.
- **Ablation:** the size of the replay buffer, i.e. how long the agent's memories of its own shocks are replayed in
  phase 2: **8k** steps (gone early in phase 2), **20k** (gone 20k steps into phase 2), **40k** (never gone).
- **Runs:** 5 shocked and 5 never-shocked runs per buffer size (seeds 1–5), plus 3 positive-control runs per buffer
  size. **Main measure:** share of presses on the lit lever in the first 8k steps of phase 2 (mean of four
  checkpoints, each a greedy evaluation on 1,024 trials). Statistics: exact permutation test and 95% bootstrap
  interval, runs as replicates.

## The results

| Replay buffer | Shocked | Never shocked | Never − shocked [95% CI] | p |
|---|---|---|---|---|
| 8k | 37.1% | 50.3% | +13.2 [−3.5, +33.3] | 0.29 |
| 20k | **18.0%** | 51.2% | **+33.2 [+19.9, +45.1]** | **0.016** |
| 40k | **15.0%** | 51.7% | **+36.7 [+24.0, +46.9]** | **0.008** |

- **H1 supported at 20k and 40k** (still below 0.05 after a Bonferroni correction for three buffer sizes: 0.048 and
  0.024); not at 8k, though the difference points the same way. Never-shocked agents are indifferent (about 50%),
  as they should be.
- **Mechanism:** the only thing that distinguishes the lit lever for the agent is the scream it causes. At the end
  of phase 1 a scream lowers the shocked agents' expected value by about 2.5 (the "scream probe"); in never-shocked
  agents it does nothing. So the avoidance is the scream's learned meaning, *pain to me*, transferred to the other
  rat's scream.
- **It is temporary — extinction.** In phase 2 a scream is never followed by harm to the agent, so learning removes
  its meaning, at a rate set by the buffer:
  - scream probe at 24k / 36k / 48k: −0.34 / −0.20 / −0.05 (8k buffer); −0.91 / −0.31 / −0.14 (20k);
    −0.96 / −0.71 / −0.50 (40k);
  - with the 20k buffer, avoidance ends between 32k and 36k, as the agent's own shocks leave the buffer (by 36k);
    with 40k, where they never leave, it lasts longest;
  - by the end of phase 2 no buffer size shows a clear difference between the groups.

## What to claim, and what not to

Use this wording (or close to it) for the conclusion:

> Agents rewarded only for their own homeostasis, after experiencing their own pain paired with screaming,
> temporarily avoided an action that made another agent scream; agents that had never been shocked did not. The
> avoidance arose from a learned association — scream predicts pain to oneself — generalizing to the other's scream,
> and faded as that association was extinguished.

- Say: harm **avoidance** emerged **without any reward for it** and without any other-regarding term in the reward.
- Do **not** say: empathy, prosocial preference, "the agent cares about the other". The agent avoids what it wrongly
  expects to hurt **itself**; that is the interesting part (Church's own interpretation was conditioned fear).
- **Caveats to state on the results or conclusion slide:** 5 runs per group; the 20k and 40k conditions share their
  early trajectory (same seeds; identical until the 20k buffer first fills), so their main-window results are not
  fully independent; the effect depends on design choices — a shared scream input for both bodies, and a shock long
  enough that the scream precedes part of the harm.

## Suggested slide outline

1. **From the gridworld to a cleaner test** — the comparison table above: the same idea, fewer confounders.
2. **Church's rats (1959)** — the classic result this design recreates, and why it isolates own pain: shocked vs
   never-shocked, the other perceived only through its scream.
3. **The setup** — same body and reward as the gridworld; one shared scream input; phase 1 (conditioning) and
   phase 2 (test). A simple diagram or the scene from the videos.
4. **Phase 1, conditioning** — video `1_phase1_conditioning.mp4`.
5. **Phase 2, the test** — video `2_side_by_side.mp4`: a shocked and a never-shocked agent on the same task.
6. **Main result** — `1_main_result.png` and the results table (or its key numbers).
7. **Over time** — `2_time_course.png`: the avoidance is temporary, and lasts as long as the agent's own pain stays in
   memory.
8. **Why: the scream's meaning** — `3_scream_meaning.png`: a scream means pain, until experience teaches otherwise.
9. **Conclusion and caveats** — the wording above; what it does and does not show.
10. **Next steps** — more seeds (10 per group), a longer phase 2 to see extinction fully, a model with a separate
    pain-prediction system that reward learning cannot overwrite, and back to the gridworld with what was learned
    here (the training fixes, and a direct manipulation of the agent's own pain).

## Assets (in `presentation/out/`)

| File | What it shows | Slide |
|---|---|---|
| `1_main_result.png` | Lit-lever share in the first 8k steps of phase 2, per buffer size; dots are runs, bars group means with 95% intervals, p-values on top | Main result |
| `2_time_course.png` | Lit-lever share through phase 2, one panel per buffer size; lines smoothed over 3 checkpoints, faint dots the raw means; marks when the agent's own shocks leave memory | Over time |
| `3_scream_meaning.png` | Scream probe through phase 2: shocked agents per buffer size vs never-shocked | Why |
| `1_phase1_conditioning.mp4` | A shocked agent in phase 1: random shocks, its own scream ("AAGH!"), captions per step (31 s) | Phase 1 |
| `2_side_by_side.mp4` | Phase 2, lever B lit for both: the shocked agent presses only the unlit lever; the never-shocked one presses the lit lever 3 times and the other rat screams each time; counter of lit presses (31 s) | Phase 2 |

Figures for the gridworld part (in `homeostasis/presentation/out/`, same palette and style; for the slides before
yours, and for the transition slide if a picture of the confound helps):

| File | What it shows |
|---|---|
| `design_1_push.png` | Before and after a push: the NPC pushed onto fire, at no cost to the agent |
| `design_2_shared_channels.png` | The example state's input channels: both bodies on body, health and energy; only self separates them |
| `design_3_experiments.png` | Training on the parallel grid (the NPC on a grid the agent can't enter) and experiment 2's test (the NPC moved within reach) |
| `results_1_exp1_reliability.png` | Experiment 1: reliability by group, runs as dots, p-values |
| `results_2_exp1_survival.png` | Experiment 1: mean survival over training by group |
| `results_3_exp2_pushes.png` | Experiment 2: all pushes and pushes onto fire per test episode, psychopath vs homeostatic, with and without the NPC |

The videos show a pixel-art side view: the agent's chamber (levers A and B, the lamp above the lit lever, a food
hopper) on the left, the other rat's chamber on the right, a mesh wall between them that lets screams through, and
health (H) and energy (E) bars under each.

## Style

- Keep the visual style consistent with the gridworld slides that come before.
- One message per slide; put the takeaway in the slide title as a sentence (the figures' titles already do this).
- Colours, to match the figures: **shocked = blue `#2a78d6`**, **never shocked = orange `#eb6834`**; text near-black
  `#0b0b0b` and grey `#52514e` on an off-white `#fcfcfb` background. The videos have a dark background; give them a
  full slide or a dark slide.
- Plain language; define "replay buffer" and "scream probe" the first time they appear.
- Don't redraw the figures or change their numbers; use the images as they are.

## Addendum (backup slides, not in the main flow)

- **Positive control** — `5_positive_control.png` and `3_positive_control.mp4` (57 s). When the lit lever shocks the
  agent **itself**, all 9 positive-control runs learn to avoid it (lit-lever share at most 16%, 0% at the end): the
  setup can learn avoidance, so the never-shocked agents' indifference is not a failure to learn.
- **Manipulation check** — `4_manipulation_check.png`. At the end of phase 1 the scream lowers the shocked agents'
  expected value by about 2.5; in never-shocked agents it does nothing. A first attempt with a **one-step shock**
  failed this check (+0.61: the scream became *good* news, because the harm was already over when it was heard and
  only recovery followed). With a two-step shock the scream predicts harm still to come, as it must for Church's
  mechanism.
- **How the agent was made to learn correctly** (if asked): the Huber loss with rewards scaled ×10 fit the median of
  the targets instead of the mean, leaving every value about 7 too low; mean squared error fixed it. Other fixes: a
  discount of 0.95 per uncompressed step instead of 0.99 (so a shock is costly enough to learn), one input per lamp, a smaller
  replay buffer, and time compressed so that trials are 10 steps apart instead of 25 (96% of steps were passive).
