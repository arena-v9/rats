"""
Slide videos of the pixel scene (presentation/scene.py), from the step recordings of experiment 1. MP4s, 1920 × 1080,
in presentation/out/: just the scene, a title, and one plain caption per step saying what happens.

    1. phase1_conditioning.mp4  a shocked agent in phase 1: random shocks, its own scream
    2. side_by_side.mp4         early phase 2: a shocked and a never-shocked agent, same lever lit
    3. positive_control.mp4     the pilot: the lit lever shocks the agent itself, early on and after learning

Every press on a trial also gives food; the captions name only what differs between the levers.

    .venv/bin/python presentation/make_videos.py
"""

import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from recording import read  # noqa: E402
from scene import draw  # noqa: E402

RUNS, OUT = ROOT / "runs" / "exp1", Path(__file__).resolve().parent / "out"
WIDTH, HEIGHT, FPS = 1920, 1080, 8
FRAMES_PER_STEP, FRAMES_PER_EVENT = 2, 6  # 4 steps a second; trials and screams held 0.75 s
BACKGROUND, INK, INK2, MUTED = (13, 13, 16), (240, 239, 236), (195, 194, 183), (137, 135, 129)
SHOCKED, NEVER = (57, 135, 229), (217, 89, 38)  # the figures' series colours, dark-surface steps
FONTS = "/usr/share/fonts/truetype/dejavu/"
font = lambda size, bold=False: ImageFont.truetype(FONTS + ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"), size)


def episode(collection: str, group: str, seed: int, start: int):
    run = RUNS / collection / group / f"seed_{seed}"
    steps = read(run / "recording.bin")
    length = json.loads((run / "metrics.json").read_text())["episode_steps"]
    return steps[start: start + length]


def caption(step, phase: int, lit_target: str) -> str:
    """What happens on this step, in plain words (a press and a random shock on the same step are both named)."""
    parts = []
    if step["active"]:
        action, lit = int(step["action"]), int(step["lit"])
        lever = "AB"[action] if action != 2 else None
        if action == 2:
            parts.append("Trial: rests (no food)")
        elif phase == 1:
            parts.append(f"Trial: presses lever {lever} → food")
        elif action == lit:
            parts.append(f"Presses the LIT lever {lever} → shocks itself" if lit_target == "self"
                         else f"Presses the LIT lever {lever} → the other rat is shocked")
        else:
            parts.append(f"Presses the unlit lever {lever} → nobody hurt")
    if step["agent_shock"] == 1:
        parts.append("a random shock hits it (not the lever): it screams" if parts else
                     "A random shock hits it: it screams in pain")
    elif step["agent_scream"] and not (step["active"] and step["action"] == step["lit"] and lit_target == "self"):
        parts.append("still being shocked: it screams" if parts else "Still being shocked: it screams")
    elif step["other_scream"] and not step["active"]:
        parts.append("The other rat screams; the agent hears it")
    return "; ".join(parts)


def is_event(step) -> bool:
    return bool(step["active"] or step["agent_scream"] or step["other_scream"])


def scene_image(step, flick: int, scale: int) -> Image.Image:
    pixels = draw(step, flick)
    return Image.fromarray(np.repeat(np.repeat(pixels, scale, axis=0), scale, axis=1))


def header(d: ImageDraw.ImageDraw, title: str, subtitle: str) -> None:
    d.text((64, 44), title, font=font(46, True), fill=INK)
    d.text((64, 108), subtitle, font=font(26), fill=INK2)


def single(path: Path, segments: list[dict]) -> None:
    """One scene, full width. Segments: {title, subtitle, steps, phase, lit_target}."""
    scale = 11
    sw, sh = 128 * scale, 72 * scale
    x0, y0 = (WIDTH - sw) // 2, 162
    with imageio.get_writer(path, fps=FPS, codec="libx264", quality=8, pixelformat="yuv420p", macro_block_size=2) as out:
        frame_number = 0
        for seg in segments:
            n = len(seg["steps"])
            for k, step in enumerate(seg["steps"]):
                text = caption(step, seg["phase"], seg.get("lit_target"))
                for _ in range(FRAMES_PER_EVENT if is_event(step) else FRAMES_PER_STEP):
                    frame = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
                    d = ImageDraw.Draw(frame)
                    header(d, seg["title"], seg["subtitle"])
                    d.text((WIDTH - 64, 52), f"step {k + 1} of {n}", font=font(26), fill=MUTED, anchor="ra")
                    frame.paste(scene_image(step, frame_number, scale), (x0, y0))
                    d.text((WIDTH // 2, y0 + sh + 44), text, font=font(36, True), fill=INK, anchor="ma")
                    out.append_data(np.asarray(frame))
                    frame_number += 1
            for _ in range(FPS):  # hold the last frame a second
                out.append_data(np.asarray(frame))
    print("wrote", path)


def side_by_side(path: Path, title: str, subtitle: str, left: dict, right: dict) -> None:
    """Two agents on the same task, step by step. Panels: {label, color, steps, phase, lit_target}."""
    scale = 7
    sw, sh = 128 * scale, 72 * scale
    xs, y0 = (48, WIDTH - 48 - sw), 262
    n = max(len(left["steps"]), len(right["steps"]))
    pad = lambda steps: [steps[min(k, len(steps) - 1)] for k in range(n)]
    left_steps, right_steps = pad(left["steps"]), pad(right["steps"])
    with imageio.get_writer(path, fps=FPS, codec="libx264", quality=8, pixelformat="yuv420p", macro_block_size=2) as out:
        frame_number = 0
        for k in range(n):
            steps = (left_steps[k], right_steps[k])
            frames = FRAMES_PER_EVENT if any(is_event(s) for s in steps) else FRAMES_PER_STEP
            for _ in range(frames):
                frame = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
                d = ImageDraw.Draw(frame)
                header(d, title, subtitle)
                d.text((WIDTH - 64, 52), f"step {k + 1} of {n}", font=font(26), fill=MUTED, anchor="ra")
                for x, panel, step in zip(xs, (left, right), steps):
                    d.rectangle([x, y0 - 62, x + 14, y0 - 48], fill=panel["color"])
                    d.text((x + 26, y0 - 70), panel["label"], font=font(34, True), fill=INK)
                    frame.paste(scene_image(step, frame_number, scale), (x, y0))
                    d.text((x + sw // 2, y0 + sh + 34), caption(step, panel["phase"], panel.get("lit_target")),
                           font=font(27, True), fill=INK, anchor="ma")
                    tally = sum(int(s["active"] and s["action"] == s["lit"]) for s in panel["steps"][: k + 1])
                    d.text((x + sw // 2, y0 + sh + 84), f"presses on the lit lever so far: {tally}",
                           font=font(24), fill=INK2, anchor="ma")
                out.append_data(np.asarray(frame))
                frame_number += 1
        for _ in range(2 * FPS):
            out.append_data(np.asarray(frame))
    print("wrote", path)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    single(OUT / "1_phase1_conditioning.mp4", [{
        "title": "Phase 1: the agent learns that a scream means pain",
        "subtitle": "Shocked group. Random shocks hit the agent; it screams while it is hurt. The levers only give food.",
        "steps": episode("buf20k", "shocked", 1, 13625), "phase": 1}])
    side_by_side(
        OUT / "2_side_by_side.mp4", "Phase 2: the lit lever now shocks the other rat",
        "Same task, same lamp lit (lever B). Reward is only ever about the agent's own body.",
        {"label": "Shocked in phase 1", "color": SHOCKED, "steps": episode("buf40k", "shocked", 2, 16160), "phase": 2,
         "lit_target": "other"},
        {"label": "Never shocked", "color": NEVER, "steps": episode("buf40k", "never_shocked", 2, 16040), "phase": 2,
         "lit_target": "other"})
    single(OUT / "3_positive_control.mp4", [
        {"title": "Positive control: the lit lever shocks the agent itself",
         "subtitle": "Early in phase 2 (step 16k): it still presses the lit lever sometimes, and gets shocked.",
         "steps": episode("buf20k", "pilot_self_shock", 1, 16088), "phase": 2, "lit_target": "self"},
        {"title": "Positive control: after learning",
         "subtitle": "Later in phase 2 (step 40k): it presses only the unlit lever.",
         "steps": episode("buf20k", "pilot_self_shock", 1, 40182), "phase": 2, "lit_target": "self"}])
