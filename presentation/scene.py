"""
The replay's pixel scene (ui/replay.html, drawScene) in Python, for rendering videos: the same 128 × 72 scene, sprites,
layout and palette. One call draws one recorded step (a numpy record from recording.read); `flick` alternates the
shock flicker. Returns an RGB numpy array, 72 × 128 × 3.
"""

import numpy as np

W, H = 128, 72
C = dict(bg="#101014", wall="#23232c", wallEdge="#3b3b4a", rod="#5b5b66", rodAlt="#45454f", base="#18181d",
         plate="#6e6e7c", plateDark="#4a4a55", bar="#d0d0dc", lampOff="#3d3a26", lampOn="#ffd84a", lampCore="#fff6c2",
         hopper="#6f6150", hopperDark="#4b4136", pellet="#c58a43", fur="#e2ddd2", furShade="#bdb6a8", pink="#e796a4",
         eye="#1b1b1b", dead="#7f7f86", shock="#ffe14d", shockHot="#ffffff", scream="#ff6b8a", mesh="#4a4a58",
         health="#e5484d", healthBg="#3b1d21", energy="#46c37b", energyBg="#18341f", label="#9a9aa8",
         bubble="#f5f0e6", bubbleEdge="#2b2b33", yell="#d42a3c")
SPRITES = {
    "sit": ["..........pp....", "......wwwwppw...", "....wwwwwwwwwk..", "...wwwwwwwwwwwwn", "..wwwwwwwwwwwww.",
            "tt.sssssssssss..", "..t.ff.....ff..."],
    "rear": ["....pp....", "...wwwp...", "..wwwwwk..", "..wwwwwwn.", "..wwwww...", ".wwwwwwff.", ".wwwwww...",
             ".wwwwww...", "wwwwwww...", "wsssssw...", "t.ff.ff...", "tt........"],
    "dead": ["...ff.....ff....", "..ssssssssssssx.", ".wwwwwwwwwwwwwwn", "tt.wwwwwwwwww..."],
}
FONT = {"A": [".#.", "#.#", "###", "#.#", "#.#"], "B": ["##.", "#.#", "##.", "#.#", "##."],
        "H": ["#.#", "#.#", "###", "#.#", "#.#"], "E": ["###", "#..", "##.", "#..", "###"],
        "G": ["###", "#..", "#.#", "#.#", "###"], "!": [".#.", ".#.", ".#.", "...", ".#."]}
AGENT, OTHER = dict(x=3, w=72, home=18, hopper=34), dict(x=81, w=44, home=86, homeFed=96, hopper=112)
TOP, FLOOR, LEVERS = 3, 48, [(8, "A"), (56, "B")]


def _rgb(hex_color: str) -> np.ndarray:
    return np.array([int(hex_color[i:i + 2], 16) for i in (1, 3, 5)], dtype=float)


class _Canvas:
    def __init__(self):
        self.img = np.zeros((H, W, 3))

    def px(self, x, y, w, h, color, alpha=1.0):
        x0, y0, x1, y1 = max(0, int(x)), max(0, int(y)), min(W, int(x + w)), min(H, int(y + h))
        if x1 > x0 and y1 > y0:
            self.img[y0:y1, x0:x1] = self.img[y0:y1, x0:x1] * (1 - alpha) + _rgb(color) * alpha

    def sprite(self, rows, x, y, palette):
        for j, row in enumerate(rows):
            for i, ch in enumerate(row):
                if ch != "." and ch in palette:
                    self.px(x + i, y + j, 1, 1, palette[ch])

    def text(self, s, x, y, color):
        for k, ch in enumerate(s):
            self.sprite([r.replace("#", "x") for r in FONT[ch]], x + 4 * k, y, {"x": color})


def _chamber(cv, c):
    cv.px(c["x"], TOP, c["w"], FLOOR - TOP, C["wall"]); cv.px(c["x"], TOP, c["w"], 1, C["wallEdge"])
    cv.px(c["x"], TOP, 1, 53, C["wallEdge"]); cv.px(c["x"] + c["w"] - 1, TOP, 1, 53, C["wallEdge"])
    cv.px(c["x"], FLOOR + 3, c["w"], 4, C["base"]); cv.px(c["x"], FLOOR + 7, c["w"], 1, C["wallEdge"])


def _floor(cv, c, shocking, flick):
    for x in range(c["x"] + 1, c["x"] + c["w"] - 1):
        on = shocking and (x + flick) % 3 != 0
        cv.px(x, FLOOR, 1, 3, (C["shock"] if flick % 2 else C["shockHot"]) if on else (C["rod"] if x % 2 else C["rodAlt"]))


def _hopper(cv, x, fed):
    cv.px(x, 36, 7, 7, C["hopper"]); cv.px(x + 1, 37, 5, 3, C["hopperDark"])
    cv.px(x - 1, 43, 9, 2, C["hopper"]); cv.px(x, 45, 7, 3, C["hopperDark"])
    if fed:
        cv.px(x + 2, 44, 2, 2, C["pellet"]); cv.px(x + 4, 45, 1, 1, C["pellet"])


def _lever(cv, x, label, lit, active, pressed, glow):
    if lit:
        cv.px(x + 1, 22, 8, 9, C["lampOn"], glow); cv.px(x + 2, 24, 6, 5, C["lampOn"]); cv.px(x + 4, 25, 2, 2, C["lampCore"])
    else:
        cv.px(x + 2, 24, 6, 5, C["lampOff"])
    cv.px(x + 3, 23, 4, 1, C["plateDark"]); cv.px(x, 32, 10, 8, C["plate"]); cv.px(x, 39, 10, 1, C["plateDark"])
    cv.text(label, x + 4, 33, C["plateDark"])
    if active:
        cv.px(x + 1, 41 if pressed else 40, 8, 2, C["bar"])


def _rat(cv, pose, x, alive, shocked, flick):
    if not alive:
        cv.sprite(SPRITES["dead"], x, FLOOR - 4, dict(w=C["dead"], s="#6a6a70", t=C["pink"], f=C["pink"], x=C["eye"], n=C["pink"]))
        return
    rows = SPRITES[pose]; y = FLOOR - len(rows)
    hot = shocked and flick % 2
    cv.sprite(rows, x, y, dict(w=C["shock"] if hot else C["fur"], s=C["shockHot"] if hot else C["furShade"],
                               p=C["pink"], t=C["pink"], f=C["pink"], k=C["eye"], n=C["pink"]))
    if shocked:
        width = len(rows[0])
        for dx, dy in ([(-2, 2), (width + 1, 4), (4, -3)] if flick % 2 else [(-1, 5), (width, 1), (8, -2)]):
            cv.px(x + dx, y + dy, 1, 2, C["shock"]); cv.px(x + dx + 1, y + dy + 2, 1, 1, C["shock"])


def _scream(cv, c, hx, hy, side, flick):
    bw, bh = 23, 9
    bx = hx + 3 if side == "right" else hx - 4 - bw if side == "left" else hx - 12
    bx = max(c["x"] + 1, min(c["x"] + c["w"] - 1 - bw, bx))
    by = hy - bh - 3 - flick % 2
    cv.px(bx + 1, by, bw - 2, bh, C["bubbleEdge"]); cv.px(bx, by + 1, bw, bh - 2, C["bubbleEdge"])
    cv.px(bx + 1, by + 1, bw - 2, bh - 2, C["bubble"])
    tx = max(bx + 2, min(bx + bw - 5, hx - 1))
    cv.px(tx, by + bh - 1, 3, 1, C["bubble"]); cv.px(tx, by + bh, 1, 1, C["bubbleEdge"])
    cv.px(tx + 2, by + bh, 1, 1, C["bubbleEdge"]); cv.px(tx + 1, by + bh, 1, 1, C["bubble"])
    cv.px(tx + 1, by + bh + 1, 1, 1, C["bubbleEdge"])
    cv.text("AAGH!", bx + 2, by + 2, C["yell"])


def _bar(cv, x, y, w, before, after, fill, background, letter, flick):
    cv.text(letter, x, y - 1, C["label"])
    bx, bw = x + 5, w - 5
    cv.px(bx, y, bw, 3, background)
    a, b = round(after * bw), round(before * bw)
    cv.px(bx, y, a, 3, fill)
    if b > a:
        cv.px(bx + a, y, b - a, 3, "#ffffff" if flick % 2 else "#ffb3b8")
    if a > b:
        cv.px(bx + b, y, a - b, 3, "#ffffff")


def draw(step, flick: int = 1) -> np.ndarray:
    """One recorded step as a 72 × 128 RGB image (uint8)."""
    cv = _Canvas()
    cv.px(0, 0, W, H, C["bg"])
    _chamber(cv, AGENT); _chamber(cv, OTHER)
    for y in range(TOP, FLOOR + 8):
        for x in range(75, 81):
            if (x + y) % 2 == 0 or x in (75, 80):
                cv.px(x, y, 1, 1, C["mesh"])
    agent_scream, other_scream = bool(step["agent_scream"]), bool(step["other_scream"])
    _floor(cv, AGENT, agent_scream, flick); _floor(cv, OTHER, other_scream, flick)
    active, action, lit = bool(step["active"]), int(step["action"]), int(step["lit"])
    pressed = active and action != 2
    for k, (x, label) in enumerate(LEVERS):
        _lever(cv, x, label, lit == k, active, pressed and action == k, 0.35 if active else 0.18)
    _hopper(cv, AGENT["hopper"], step["agent_fed"]); _hopper(cv, OTHER["hopper"], step["other_fed"])
    (ah, ae), (oh, oe) = step["agent_after"].astype(float), step["other_after"].astype(float)
    (ah0, ae0), (oh0, oe0) = step["agent_before"].astype(float), step["other_before"].astype(float)
    ax = LEVERS[action][0] if pressed else AGENT["home"]
    ox = OTHER["homeFed"] if step["other_fed"] else OTHER["home"]
    _rat(cv, "rear" if pressed else "sit", ax, ah > 0, agent_scream, flick)
    _rat(cv, "sit", ox, oh > 0, other_scream, flick)
    if agent_scream:
        if pressed:
            _scream(cv, AGENT, ax + 5, FLOOR - 12, "right" if action == 0 else "left", flick)
        else:
            _scream(cv, AGENT, ax + 12, FLOOR - 7, "above", flick)
    if other_scream:
        _scream(cv, OTHER, ox + 12, FLOOR - 7, "above", flick)
        for j in range(flick % 3 + 1):  # the scream carries through the mesh to the agent
            cv.px(73 - 3 * j, 36 - j, 1, 5 + 2 * j, C["scream"])
    _bar(cv, AGENT["x"] + 1, 61, AGENT["w"] - 2, ah0, ah, C["health"], C["healthBg"], "H", flick)
    _bar(cv, AGENT["x"] + 1, 67, AGENT["w"] - 2, ae0, ae, C["energy"], C["energyBg"], "E", flick)
    _bar(cv, OTHER["x"] + 1, 61, OTHER["w"] - 2, oh0, oh, C["health"], C["healthBg"], "H", flick)
    _bar(cv, OTHER["x"] + 1, 67, OTHER["w"] - 2, oe0, oe, C["energy"], C["energyBg"], "E", flick)
    return cv.img.astype(np.uint8)
