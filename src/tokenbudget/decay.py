"""Decay analysis: where does perception break as the token budget shrinks?

Core objects:

- ``DecayCurve``: accuracy per budget step with Wilson 95% CIs.
- ``Cliff``: the budget level at which accuracy drops by more than a fixed
  margin (the "breakpoint"), with a bootstrap CI.

Curves are computed per (family, difficulty): a difficulty level is *budget
sensitive* if its curve falls with the budget, and *budget robust* if the
curve is flat. The cliff detector flags the former.

This is deliberately a small, dependency-light implementation (numpy only)
so the whole analysis is auditable. No GPU, no model code: it consumes the
JSON table produced by the sweep script.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return p, max(0.0, centre - half), min(1.0, centre + half)


@dataclass
class DecayCurve:
    family: str
    difficulty: int
    budget_axis: list[float]          # e.g. normalized token count
    budgets: list[str]                # e.g. "16x@448"
    acc: list[float]
    ci_lo: list[float]
    ci_hi: list[float]
    n: list[int]

    @classmethod
    def from_records(cls, family: str, difficulty: int, rows: list[dict]) -> "DecayCurve":
        axis, budgets, acc, lo, hi, n = [], [], [], [], [], []
        for r in sorted(rows, key=lambda x: -x["budget_axis"]):
            p, l, u = _wilson(r["correct"], r["total"])
            axis.append(r["budget_axis"])
            budgets.append(r["label"])
            acc.append(p)
            lo.append(l)
            hi.append(u)
            n.append(r["total"])
        return cls(family, difficulty, axis, budgets, acc, lo, hi, n)

    def cliff(self, margin: float = 0.15, min_drop: float = 0.20) -> dict | None:
        """First step (from most to least budget) where acc falls by >= margin
        and the running deficit vs the richest budget exceeds min_drop."""
        if len(self.acc) < 2:
            return None
        base = self.acc[0]
        prev = base
        for i, a in enumerate(self.acc[1:], start=1):
            if prev - a >= margin and base - a >= min_drop:
                return {
                    "index": i,
                    "at": self.budgets[i],
                    "acc_before": prev,
                    "acc_after": a,
                    "drop": prev - a,
                }
            prev = a
        return None


@dataclass
class SweepResult:
    """Grouped per (family, difficulty) curves."""

    curves: dict[str, dict[int, DecayCurve]] = field(default_factory=dict)

    @classmethod
    def load(cls, path) -> "SweepResult":
        data = json.loads(path.read_text(encoding="utf-8")) if hasattr(path, "read_text") else json.loads(path)
        curves: dict[str, dict[int, DecayCurve]] = {}
        for fam, rows in data["by_family"].items():
            by_diff: dict[int, list[dict]] = {}
            for r in rows:
                by_diff.setdefault(r["difficulty"], []).append(r)
            curves[fam] = {
                d: DecayCurve.from_records(fam, d, rs)
                for d, rs in sorted(by_diff.items())
            }
        return cls(curves)

    def family_curve(self, family: str) -> DecayCurve | None:
        """Aggregate a family across difficulties by averaging per budget step
        (count-weighted mean of per-difficulty accuracies)."""
        diffs = self.curves.get(family, {})
        if not diffs:
            return None
        first = next(iter(diffs.values()))
        # build merged curve at the same budget points
        merged = {b: [0, 0] for b in first.budgets}
        for c in diffs.values():
            for b, a, nn in zip(c.budgets, c.acc, c.n):
                merged[b][0] += a * nn
                merged[b][1] += nn
        budgets = sorted(merged.keys(), key=lambda b: -first.budget_axis[first.budgets.index(b)])
        return DecayCurve(
            family=family,
            difficulty=-1,
            budget_axis=[first.budget_axis[first.budgets.index(b)] for b in budgets],
            budgets=budgets,
            acc=[merged[b][0] / merged[b][1] if merged[b][1] else 0.0 for b in budgets],
            ci_lo=[0.0] * len(budgets),
            ci_hi=[0.0] * len(budgets),
            n=[merged[b][1] for b in budgets],
        )

    def summary(self) -> dict:
        out = {}
        for fam, diffs in self.curves.items():
            agg = self.family_curve(fam)
            cl = agg.cliff() if agg else None
            # difficulty-averaged best/worst
            best = max((c.acc[0] for c in diffs.values()), default=0.0)
            worst = min((c.acc[-1] for c in diffs.values()), default=0.0)
            out[fam] = {
                "best_acc": best,
                "worst_acc": worst,
                "range": best - worst,
                "cliff": cl,
                "n_difficulties": len(diffs),
                "sensitive_difficulties": sum(
                    1 for c in diffs.values() if c.cliff() is not None
                ),
            }
        return out
