"""Deterministic procedural generation of perception probe images.

Each probe family is a function `gen(seed, difficulty) -> (PIL.Image, str)`
where the string is the ground-truth answer. Difficulty is an integer level
that raises the *spatial-frequency / information density* of the scene:
smaller glyphs, more and smaller objects, finer rings, thinner strokes.

Two families are designed to sit on opposite sides of the budget cliff:

- "coarse" tasks (ocr, spatial, color, mlread): the answer can be recovered
  from a handful of vision tokens. We expect them to be budget-robust.
- "fine" tasks (detail, colcnt): the answer requires per-element binding of
  many small features. We expect them to collapse when the budget drops
  below the element count.

Design rule: the question offers a fixed closed set of answer options and the
ground truth is always one of them, so a lenient-but-explicit normalizer in
``tokenbudget.check`` can score the model fairly.

Rendering is intentionally minimal (PIL primitives only): the point is that
the *content* is controlled exactly, and any misreading by the model is a
perception failure, not an annotation artifact.
"""
from __future__ import annotations

import random

from PIL import Image, ImageDraw, ImageFont

W, H = 512, 512

FONT_CANDIDATES = ("arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf", "arialuni.ttf")


def _font(size: int) -> ImageFont.FreeTypeFont | None:
    for name in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return None


# --- named colors used across families --------------------------------
COLOR_NAMES = {
    "red": (200, 40, 40),
    "blue": (40, 80, 200),
    "green": (40, 160, 70),
    "orange": (230, 130, 20),
    "pink": (220, 100, 140),
    "purple": (140, 60, 200),
}


def gen_ocr(seed: int, difficulty: int) -> tuple[Image.Image, str]:
    """A 5-digit number, rendered smaller and under heavier clutter as
    difficulty rises. Expected to stay budget-robust even at d5: the glyphs
    are large enough that a few tokens still capture the digits."""
    rng = random.Random(seed)
    digits = [str(rng.randint(0, 9)) for _ in range(5)]
    answer = "".join(digits)

    font_size = {0: 96, 1: 72, 2: 52, 3: 36, 4: 24, 5: 16}[min(difficulty, 5)]
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    f = _font(font_size)
    text = " ".join(digits)
    try:
        w = d.textlength(text, font=f)
    except Exception:
        w = len(text) * font_size
    if difficulty >= 2:
        for _ in range(difficulty * 40):
            x, y = rng.randrange(W), rng.randrange(H)
            d.point((x, y), fill=(rng.randint(150, 255),) * 3)
    d.text(((W - w) / 2, H / 2 - font_size / 2), text, fill=(0, 0, 0), font=f)
    return img, answer


def gen_count(seed: int, difficulty: int) -> tuple[Image.Image, str]:
    """Small same-color dots; answer is the count. Kept as a coarse reference
    task at low densities (3..8 dots) where a 1.3B model can still subitize.
    """
    n = {0: 3, 1: 4, 2: 5, 3: 6, 4: 7, 5: 8}[min(difficulty, 5)]
    r = {0: 14, 1: 13, 2: 12, 3: 11, 4: 10, 5: 9}[min(difficulty, 5)]
    rng = random.Random(seed)
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    color = (rng.randint(40, 210), rng.randint(40, 210), rng.randint(40, 210))
    placed = []
    tries = 0
    while len(placed) < n and tries < 800:
        x, y = rng.randrange(r, W - r), rng.randrange(r, H - r)
        if all((x - px) ** 2 + (y - py) ** 2 > (3 * r) ** 2 for px, py in placed):
            placed.append((x, y))
        tries += 1
    for (x, y) in placed:
        d.ellipse([x - r, y - r, x + r, y + r], fill=color)
    return img, str(len(placed))


def gen_mlread(seed: int, difficulty: int) -> tuple[Image.Image, str]:
    """K lines of 5-digit strings; answer is the SECOND line. Difficulty adds
    more distractor lines (which a low budget can conflate), so the target
    line must be indexed among K objects."""
    rng = random.Random(seed)
    k_lines = {0: 2, 1: 3, 2: 4, 3: 5, 4: 6, 5: 8}[min(difficulty, 5)]
    fs = {0: 26, 1: 24, 2: 22, 3: 20, 4: 18, 5: 16}[min(difficulty, 5)]
    lines = ["".join(str(rng.randint(0, 9)) for _ in range(5)) for _ in range(k_lines)]
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    f = _font(fs)
    y = H / 2 - k_lines * fs / 2
    for i, line in enumerate(lines):
        try:
            w = d.textlength(line, font=f)
        except Exception:
            w = len(line) * fs
        d.text(((W - w) / 2, y + i * (fs + 6)), line, fill=(0, 0, 0), font=f)
    return img, lines[1]


def gen_colcnt(seed: int, difficulty: int) -> tuple[Image.Image, str]:
    """A dense grid of small colored squares; answer is how many are RED.
    The scene holds N objects, so a budget that cannot afford every square
    must mis-bind colors to positions."""
    n = {0: 8, 1: 12, 2: 16, 3: 20, 4: 24, 5: 30}[min(difficulty, 5)]
    cell = {0: 32, 1: 28, 2: 24, 3: 22, 4: 20, 5: 18}[min(difficulty, 5)]
    rng = random.Random(seed)
    colors = list(COLOR_NAMES.values())
    red = colors[0]
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    side = int(n**0.5) + 1
    gap = 8
    count = 0
    for i in range(n):
        r_, c_ = divmod(i, side)
        x0 = 20 + c_ * (cell + gap)
        y0 = 20 + r_ * (cell + gap)
        if rng.random() < 0.18:
            c = red
            count += 1
        else:
            c = colors[rng.randrange(len(colors))]
        d.rectangle([x0, y0, x0 + cell, y0 + cell], fill=c, outline=(230, 230, 230), width=1)
    return img, str(count)


def gen_spatial(seed: int, difficulty: int) -> tuple[Image.Image, str]:
    """Two colored squares side by side; answer is the color on the LEFT.
    Difficulty shrinks the squares and adds gray distractor dots."""
    rng = random.Random(seed)
    names = list(COLOR_NAMES)
    rng.shuffle(names)
    left_name, right_name = names[0], names[1]
    a = COLOR_NAMES[left_name]
    b = COLOR_NAMES[right_name]
    jitter = min(difficulty, 5) * 14
    y_a = H / 2 + (rng.randrange(-jitter, jitter + 1) if difficulty else 0)
    y_b = H / 2 + (rng.randrange(-jitter, jitter + 1) if difficulty else 0)
    s = {0: 84, 1: 72, 2: 60, 3: 48, 4: 36, 5: 28}[min(difficulty, 5)]
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    if difficulty >= 1:
        for _ in range(difficulty * 14):
            x, y = rng.randrange(W), rng.randrange(H)
            rr = rng.randrange(2, 6)
            d.ellipse([x - rr, y - rr, x + rr, y + rr], fill=(200, 200, 200))
    d.rectangle([W * 0.25 - s / 2, y_a - s / 2, W * 0.25 + s / 2, y_a + s / 2], fill=a)
    d.rectangle([W * 0.75 - s / 2, y_b - s / 2, W * 0.75 + s / 2, y_b + s / 2], fill=b)
    return img, left_name


def gen_color(seed: int, difficulty: int) -> tuple[Image.Image, str]:
    """A single square whose hue is perturbed toward one of two anchors;
    answer is the closer anchor name. Difficulty = smaller hue offset."""
    rng = random.Random(seed)
    anchors = [("red", (200, 40, 40)), ("orange", (230, 130, 20)), ("pink", (220, 100, 140))]
    chosen, ref = rng.choice(anchors)
    step = [40, 28, 20, 14, 10, 7][min(difficulty, 5)]
    next_ref = rng.choice(anchors)
    target = tuple(min(255, max(0, ref[i] + (next_ref[1][i] - ref[i]) * step // 120)) for i in range(3))
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.rectangle([W / 2 - 120, H / 2 - 120, W / 2 + 120, H / 2 + 120], fill=target)
    return img, chosen


def gen_detail(seed: int, difficulty: int) -> tuple[Image.Image, str]:
    """Two rows of ring shapes; one row has a small 1px gap. Answer: top or
    bottom. Difficulty shrinks the rings and thins the stroke, so the gap
    feature falls below the rasterization a small budget provides."""
    rng = random.Random(seed)
    gap = {0: 10, 1: 8, 2: 6, 3: 4, 4: 2, 5: 1}[min(difficulty, 5)]
    r = {0: 66, 1: 60, 2: 54, 3: 48, 4: 42, 5: 36}[min(difficulty, 5)]
    width = {0: 9, 1: 8, 2: 7, 3: 6, 4: 5, 5: 4}[min(difficulty, 5)]
    has_gap_row = rng.randrange(2)
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    for row in range(2):
        y0 = 90 + row * 220
        cx, cy = W / 2, y0 + 70
        if row == has_gap_row:
            d.arc([cx - r, cy - r, cx + r, cy + r], start=20, end=340, fill=(0, 0, 0), width=width)
            d.line([cx - 10, cy - r, cx + 10, cy - r], fill=(255, 255, 255), width=width + 2)
        else:
            d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(0, 0, 0), width=width)
    return img, ("top" if has_gap_row == 0 else "bottom")


def gen_ringgap(seed: int, difficulty: int) -> tuple[Image.Image, str]:
    """Fine variant: gap fixed at 1px, only ring radius and stroke thin.
    This is the budget-cliff family discovered during calibration."""
    rng = random.Random(seed)
    r = {0: 60, 1: 48, 2: 38, 3: 30, 4: 24, 5: 18}[min(difficulty, 5)]
    width = {0: 6, 1: 5, 2: 4, 3: 4, 4: 3, 5: 3}[min(difficulty, 5)]
    has_gap_row = rng.randrange(2)
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    for row in range(2):
        y0 = 90 + row * 220
        cx, cy = W / 2, y0 + 70
        if row == has_gap_row:
            d.arc([cx - r, cy - r, cx + r, cy + r], start=20, end=340, fill=(0, 0, 0), width=width)
            d.line([cx - 8, cy - r, cx + 8, cy - r], fill=(255, 255, 255), width=width + 2)
        else:
            d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(0, 0, 0), width=width)
    return img, ("top" if has_gap_row == 0 else "bottom")


PALETTE = [
    (200, 40, 40),
    (40, 80, 200),
    (40, 160, 70),
    (230, 130, 20),
    (220, 100, 140),
    (140, 60, 200),
]

FAMILY_GENERATORS = {
    "ocr": gen_ocr,
    "count": gen_count,
    "mlread": gen_mlread,
    "colcnt": gen_colcnt,
    "spatial": gen_spatial,
    "color": gen_color,
    "detail": gen_detail,
    "ringgap": gen_ringgap,
}
