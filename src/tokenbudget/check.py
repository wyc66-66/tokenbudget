"""Answer normalization and correctness checks.

The probes ask for constrained outputs (a digit string, a number, a color, a
row). A small MLLM will not always reply with the exact string, so we
normalize leniently per family before comparing to ground truth. Every rule
is a pure function of the model text, kept explicit so failures are
inspectable rather than hidden by a fuzzy matcher.
"""
from __future__ import annotations

import re

_WORD_NUMS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20",
}

_COLOR_WORDS = ["red", "blue", "green", "orange", "pink", "purple"]


def norm_ocr(text: str) -> str:
    return "".join(re.findall(r"\d", text))[:20]


def norm_count(text: str) -> str | None:
    t = text.lower()
    nums = re.findall(r"\d+", t)
    if nums:
        return nums[0]
    # fall back to spelled-out numbers
    for word, val in _WORD_NUMS.items():
        if re.search(rf"\b{word}\b", t):
            return val
    return None


def norm_spatial(text: str) -> str | None:
    t = text.lower()
    for c in _COLOR_WORDS:
        if c in t:
            return c
    return None


def norm_color(text: str) -> str | None:
    t = text.lower()
    for c in ("red", "orange", "pink"):
        if c in t:
            return c
    return None


def norm_detail(text: str) -> str | None:
    t = text.lower()
    if "top" in t:
        return "top"
    if "bottom" in t:
        return "bottom"
    return None


def norm_gridloc(text: str) -> str | None:
    """Normalize 'row X column Y' / 'XY' / 'x y' replies to a 2-digit code."""
    t = text.lower()
    m = re.search(r"(?:row|r)[^\d]{0,3}(\d+)[^\d]{0,6}(?:column|c|col)[^\d]{0,3}(\d+)", t)
    if m:
        return f"{m.group(1)}{m.group(2)}"
    digits = re.findall(r"\d+", t)
    if len(digits) >= 2:
        return "".join(digits[:2])
    if digits:
        d = digits[0]
        if len(d) >= 2:
            return d[:2]
        # a bare digit might be row or col; leave as-is
        return None
    return None


def norm_maxdigit(text: str) -> str | None:
    t = text.lower()
    words = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    for i, w in enumerate(words):
        if re.search(rf"\b{w}\b", t):
            return str(i)
    digits = re.findall(r"\d", t)
    if digits:
        return digits[-1]
    return None


def norm_colcnt(text: str) -> str | None:
    t = text.lower()
    nums = re.findall(r"\d+", t)
    if nums:
        return nums[0]
    words = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve"]
    for i, w in enumerate(words):
        if re.search(rf"\b{w}\b", t):
            return str(i)
    return None


def norm_same(text: str) -> str | None:
    t = text.lower()
    if "same" in t or "identical" in t or "equal" in t:
        return "same"
    if "different" in t or "differ" in t or "not" in t:
        return "different"
    return None


_NORMALIZERS = {
    "ocr": norm_ocr,
    "mlread": norm_ocr,
    "count": norm_count,
    "spatial": norm_spatial,
    "color": norm_color,
    "detail": norm_detail,
    "ringgap": norm_detail,
    "gridloc": norm_gridloc,
    "maxdigit": norm_maxdigit,
    "same": norm_same,
    "colcnt": norm_colcnt,
}


def normalize(family: str, raw: str) -> str | None:
    return _NORMALIZERS[family](raw)


def is_correct(family: str, raw: str, answer: str) -> bool:
    n = normalize(family, raw)
    if n is None:
        return False
    return n == answer
