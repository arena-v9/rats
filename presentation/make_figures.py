"""
Slide figures for experiment 1 (version 4.1), from runs/exp1/buf*/ (and the archived one-step-shock attempt). PNGs
in presentation/out/, 16:9, light surface. Colors: the data-viz reference palette (shocked = categorical slot 1,
never shocked = slot 2; buffer sizes = an ordinal blue ramp), validated for colour-vision deficiency.

    .venv/bin/python presentation/make_figures.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RUNS, ARCHIVE, OUT = ROOT / "runs" / "exp1", ROOT / "runs_archive", Path(__file__).resolve().parent / "out"
BUFFERS = {"buf8k": "8k", "buf20k": "20k", "buf40k": "40k"}
# Reference palette (light): categorical slots, ordinal blue ramp, chrome and ink
SHOCKED, NEVER = "#2a78d6", "#eb6834"
RAMP = {"buf8k": "#86b6ef", "buf20k": "#2a78d6", "buf40k": "#104281"}
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
LABELS = {"shocked": "Shocked in phase 1", "never_shocked": "Never shocked"}
COLORS = {"shocked": SHOCKED, "never_shocked": NEVER}
P_VALUES = {"buf8k": 0.29, "buf20k": 0.016, "buf40k": 0.008}  # permutation tests, from analyze.py (RESULTS.md)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "text.color": INK, "axes.labelcolor": INK2,
    "axes.edgecolor": AXIS, "xtick.color": MUTED, "ytick.color": MUTED, "axes.facecolor": SURFACE,
    "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
    "legend.frameon": False, "lines.linewidth": 2, "lines.solid_capstyle": "round",
})


def load(folder: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(folder.glob("seed_*/metrics.json"))]


def series(runs: list[dict], key: str) -> tuple[np.ndarray, np.ndarray]:
    steps = np.array([c["step"] for c in runs[0]["checkpoints"]]) / 1000
    values = np.array([[np.nan if c[key] is None else c[key] for c in r["checkpoints"]] for r in runs], dtype=float)
    return steps, values


def main_window(run: dict) -> float:
    first, last = run["main_window"]
    return np.mean([c["lit_share"] for c in run["checkpoints"] if first <= c["step"] <= last])


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    means = rng.choice(values, (20_000, len(values))).mean(axis=1)
    return tuple(np.percentile(means, [2.5, 97.5]))


def title(fig, text: str, sub: str) -> None:
    fig.text(0.04, 0.95, text, fontsize=15, fontweight="bold", color=INK, va="top")
    fig.text(0.04, 0.885, sub, fontsize=10.5, color=INK2, va="top")


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=200)
    plt.close(fig)
    print("wrote", OUT / name)


def main_result() -> None:
    """Lit-lever share in the main window, per buffer and group: runs as dots, group mean with 95% interval."""
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(10, 5.6))
    fig.subplots_adjust(left=0.1, right=0.97, top=0.74, bottom=0.11)
    for i, buffer in enumerate(BUFFERS):
        for j, group in enumerate(("shocked", "never_shocked")):
            v = 100 * np.array([main_window(r) for r in load(RUNS / buffer / group)])
            x = i + (j - 0.5) * 0.36
            jitter = np.linspace(-0.05, 0.05, len(v))
            ax.scatter(x + jitter, v, s=28, color=COLORS[group], alpha=0.45, linewidths=0, zorder=3)
            low, high = bootstrap_ci(v, rng)
            ax.plot([x, x], [low, high], color=COLORS[group], lw=2, zorder=4)
            ax.plot([x - 0.09, x + 0.09], [v.mean()] * 2, color=COLORS[group], lw=3, zorder=5)
            ax.text(x + 0.11, v.mean(), f"{v.mean():.0f}%", color=INK2, fontsize=10, va="center")
        p = P_VALUES[buffer]
        ax.text(i, 96, f"p = {p:g}", ha="center", fontsize=10.5, color=INK if p < 0.05 else MUTED,
                fontweight="bold" if p < 0.05 else "normal")
    ax.axhline(50, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=2)
    ax.text(-0.57, 51.5, "indifferent", color=MUTED, fontsize=9.5, ha="left")
    ax.set_xticks(range(3), [f"Replay buffer {b}" for b in BUFFERS.values()])
    ax.tick_params(axis="x", colors=INK2, labelsize=11, length=0)
    ax.set_xlim(-0.6, 2.65)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_ylabel("Presses on the lever that shocks the other")
    ax.grid(axis="x", visible=False)
    handles = [plt.Line2D([], [], color=COLORS[g], lw=3, label=LABELS[g]) for g in COLORS]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0, 1.12), ncol=2, fontsize=10.5)
    title(fig, "Agents that were shocked themselves avoid shocking the other",
          "Early in phase 2 (first 8k steps). Reward never includes the other.\n"
          "Dots: runs (5 per group). Bars: group mean with 95% bootstrap interval.")
    save(fig, "1_main_result.png")


def time_course() -> None:
    """Lit-lever share through phase 2, one panel per buffer size."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 5), sharey=True)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.74, bottom=0.14, wspace=0.08)
    forget = {"buf8k": 24, "buf20k": 36, "buf40k": None}  # step (k) by which phase-1 memories have left the buffer
    for ax, buffer in zip(axes, BUFFERS):
        first, last = load(RUNS / buffer / "shocked")[0]["main_window"]
        ax.axvspan(first / 1000, last / 1000, color=GRID, alpha=0.6, lw=0, zorder=1)
        for group in ("never_shocked", "shocked"):
            steps, v = series(load(RUNS / buffer / group), "lit_share")
            keep = steps >= 16
            ax.plot(steps[keep], 100 * np.nanmean(v, axis=0)[keep], color=COLORS[group], zorder=3)
        ax.axhline(50, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=2)
        if forget[buffer]:
            ax.axvline(forget[buffer], color=INK2, lw=1, zorder=2)
            ax.text(forget[buffer] + 0.5, 4, "own shocks\nleave memory", fontsize=8.5, color=INK2, va="bottom")
        else:
            ax.text(47.5, 4, "own shocks stay\nin memory", fontsize=8.5, color=INK2, va="bottom", ha="right")
        ax.set_title(f"Replay buffer {BUFFERS[buffer]}", fontsize=11.5, color=INK, loc="left")
        ax.set_xlim(16, 48)
        ax.set_ylim(0, 85)
        ax.set_xticks([16, 24, 32, 40, 48], ["16k", "24k", "32k", "40k", ""] if buffer != "buf40k"
                      else ["16k", "24k", "32k", "40k", "48k"])
        ax.set_xlabel("Training step (phase 2)")
    axes[0].set_yticks([0, 25, 50, 75], ["0%", "25%", "50%", "75%"])
    axes[0].set_ylabel("Presses on the lever that shocks the other")
    axes[0].text(18.2, 80, "main\nwindow", fontsize=8.5, color=MUTED, va="top")
    handles = [plt.Line2D([], [], color=COLORS[g], lw=2, label=LABELS[g]) for g in COLORS]
    axes[0].legend(handles=handles, loc="upper left", bbox_to_anchor=(0, 1.2), ncol=2, fontsize=10.5)
    title(fig, "The avoidance is temporary, and lasts as long as the agent's own pain stays in memory",
          "Mean of 5 runs per group, at every 2k-step checkpoint. Dashed: indifferent (50%).")
    save(fig, "2_time_course.png")


def scream_meaning() -> None:
    """Scream probe (change in value when a scream is heard) through phase 2, by buffer size."""
    fig, ax = plt.subplots(figsize=(10, 5.6))
    fig.subplots_adjust(left=0.1, right=0.78, top=0.8, bottom=0.13)
    never, ends = [], {}
    for buffer in BUFFERS:
        steps, v = series(load(RUNS / buffer / "shocked"), "scream_probe")
        mean = v.mean(axis=0)
        ax.plot(steps, mean, color=RAMP[buffer], zorder=3)
        ends[f"shocked, buffer {BUFFERS[buffer]}"] = (mean[-1], INK2)
        never.append(series(load(RUNS / buffer / "never_shocked"), "scream_probe")[1])
    never_mean = np.concatenate(never).mean(axis=0)
    ax.plot(steps, never_mean, color=MUTED, lw=2, ls=(0, (4, 3)), zorder=2)
    ends["never shocked"] = (never_mean[-1], MUTED)
    # End labels, spread vertically so none overlap (at least 0.13 apart), with a leader to each line's end
    placed = []
    for label, (y, color) in sorted(ends.items(), key=lambda item: -item[1][0]):
        y_text = min(y, placed[-1] - 0.13) if placed else y
        placed.append(y_text)
        ax.annotate(label, xy=(48, y), xytext=(49.2, y_text), color=color, fontsize=10, va="center",
                    annotation_clip=False, arrowprops={"arrowstyle": "-", "color": AXIS, "lw": 0.8})
    ax.axhline(0, color=AXIS, lw=1)
    ax.axvline(16, color=INK2, lw=1)
    ax.text(16.4, -2.75, "phase 2 starts: no more\nshocks to the agent", fontsize=9, color=INK2, va="bottom")
    ax.set_xlim(15, 48)
    ax.set_xticks([16, 24, 32, 40, 48], ["16k", "24k", "32k", "40k", "48k"])
    ax.set_xlabel("Training step (phase 2)")
    ax.set_ylabel("Change in expected value when a scream is heard")
    ax.set_ylim(-2.8, 0.3)
    title(fig, "A scream means pain, until experience teaches otherwise",
          "Scream probe: how much hearing a scream lowers the agent's expected value. Mean of 5 runs. "
          "Lower = the scream predicts harm.")
    save(fig, "3_scream_meaning.png")


def manipulation_check() -> None:
    """Scream probe at the end of phase 1: never shocked, shocked with a one-step shock, shocked with two steps."""
    rng = np.random.default_rng(1)
    end = lambda runs: np.array([r["checkpoints"][0]["scream_probe"] for r in runs])  # first checkpoint = end of phase 1
    groups = [
        ("Never shocked", end(load(RUNS / "buf8k" / "never_shocked") + load(RUNS / "buf20k" / "never_shocked")), NEVER),
        ("Shocked, one-step shock\n(first attempt)", end(load(ARCHIVE / "exp1_one_step_shock" / "buf8k" / "shocked")), MUTED),
        ("Shocked, two-step shock\n(final design)", end(load(RUNS / "buf8k" / "shocked") + load(RUNS / "buf20k" / "shocked")), SHOCKED),
    ]
    fig, ax = plt.subplots(figsize=(10, 5.6))
    fig.subplots_adjust(left=0.1, right=0.97, top=0.8, bottom=0.17)
    for i, (label, v, color) in enumerate(groups):
        ax.bar(i, v.mean(), width=0.5, color=color, alpha=0.9, zorder=3)
        ax.scatter(i + np.linspace(-0.12, 0.12, len(v)), v, s=22, color=INK, alpha=0.5, linewidths=0, zorder=4)
        low, high = bootstrap_ci(v, rng)
        ax.plot([i, i], [low, high], color=INK, lw=1.5, zorder=5)
        ax.text(i + 0.29, v.mean(), f"{v.mean():+.2f}", ha="left", va="center", fontsize=11.5, color=INK,
                fontweight="bold")
    ax.axhline(0, color=AXIS, lw=1.2)
    ax.set_xticks(range(3), [g[0] for g in groups])
    ax.tick_params(axis="x", colors=INK2, labelsize=10.5, length=0)
    ax.grid(axis="x", visible=False)
    ax.set_ylim(-3.2, 1.5)
    ax.set_ylabel("Change in expected value\nwhen a scream is heard")
    ax.text(0.5, 1.3, "↑ scream is good news", color=MUTED, fontsize=9.5, ha="center")
    ax.text(0.5, -3.05, "↓ scream predicts pain", color=MUTED, fontsize=9.5, ha="center")
    title(fig, "The scream only means pain if harm is still to come when it is heard",
          "End of phase 1. With a one-step shock the harm is over before the scream is heard; with two steps, more "
          "harm follows. Dots: runs.")
    save(fig, "4_manipulation_check.png")


def positive_control() -> None:
    """Pilot: the lit lever shocks the agent itself."""
    fig, ax = plt.subplots(figsize=(10, 5.6))
    fig.subplots_adjust(left=0.1, right=0.86, top=0.8, bottom=0.13)
    for buffer in BUFFERS:
        steps, v = series(load(RUNS / buffer / "pilot_self_shock"), "lit_share")
        keep = steps >= 16
        ax.plot(steps[keep], 100 * v.mean(axis=0)[keep], color=RAMP[buffer], zorder=3)
    ax.text(48.6, 3, "all 9 runs\nend at 0%", color=INK2, fontsize=10, va="center")
    ax.axhline(50, color=MUTED, lw=1, ls=(0, (4, 3)))
    ax.set_xlim(16, 48)
    ax.set_ylim(0, 85)
    ax.set_xticks([16, 24, 32, 40, 48], ["16k", "24k", "32k", "40k", "48k"])
    ax.set_yticks([0, 25, 50, 75], ["0%", "25%", "50%", "75%"])
    ax.set_xlabel("Training step (phase 2)")
    ax.set_ylabel("Presses on the lever that shocks\nthe agent itself")
    handles = [plt.Line2D([], [], color=RAMP[b], lw=2, label=f"buffer {BUFFERS[b]}") for b in BUFFERS]
    ax.legend(handles=handles, loc="upper right", fontsize=10)
    title(fig, "Positive control: when the lever hurts the agent itself, it learns to avoid it",
          "Pilot runs (3 per buffer size), mean lit-lever share through phase 2.")
    save(fig, "5_positive_control.png")


def before_forgetting() -> None:
    """Each run's lit-lever share at the end of phase 1 (16k) and 2k steps into phase 2 (18k); 20k/40k buffers."""
    rng = np.random.default_rng(2)
    at = lambda runs, step: 100 * np.array([next(c["lit_share"] for c in r["checkpoints"] if c["step"] == step) for r in runs])
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.6), sharey=True)
    fig.subplots_adjust(left=0.1, right=0.97, top=0.74, bottom=0.12, wspace=0.08)
    for ax, (step, name) in zip(axes, ((16000, "End of phase 1\n(lit lever has done nothing yet)"),
                                       (18000, "2k steps into phase 2\n(lit lever makes the other scream)"))):
        for j, group in enumerate(("shocked", "never_shocked")):
            v = at(load(RUNS / "buf40k" / group), step)
            ax.scatter(j + np.linspace(-0.08, 0.08, len(v)), v, s=40, color=COLORS[group], alpha=0.55, linewidths=0, zorder=3)
            low, high = bootstrap_ci(v, rng)
            ax.plot([j, j], [low, high], color=COLORS[group], lw=2, zorder=4)
            ax.plot([j - 0.14, j + 0.14], [v.mean()] * 2, color=COLORS[group], lw=3, zorder=5)
            ax.text(j + 0.18, v.mean(), f"{v.mean():.0f}%", color=INK2, fontsize=10.5, va="center")
        ax.axhline(50, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=2)
        ax.set_title(name, fontsize=11, color=INK, loc="left")
        ax.set_xticks([0, 1], [LABELS["shocked"], LABELS["never_shocked"]])
        ax.tick_params(axis="x", colors=INK2, labelsize=10.5, length=0)
        ax.set_xlim(-0.5, 1.6)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylim(0, 100)
    axes[0].set_yticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    axes[0].set_ylabel("Presses on the lever that shocks the other")
    axes[1].text(1.55, 95, "p = 0.056", ha="right", fontsize=10, color=MUTED)
    title(fig, "The avoidance appears as soon as the lit lever makes the other scream",
          "Replay buffer 20k and 40k (identical runs this early). Dots: runs (5 per group). Bars: mean with 95% "
          "bootstrap interval.\nExploratory: chosen after seeing the results.")
    save(fig, "6_before_forgetting.png")


if __name__ == "__main__":
    main_result()
    time_course()
    scream_meaning()
    manipulation_check()
    positive_control()
    before_forgetting()
