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
        acc = [merged[b][0] / merged[b][1] if merged[b][1] else 0.0 for b in budgets]
        n = [merged[b][1] for b in budgets]
        # count-weighted Wilson interval over the merged successes/failures
        ci_lo, ci_hi = [], []
        for b, nn in zip(budgets, n):
            k = int(round(merged[b][0]))  # successes = acc * n, rounded
            _, lo, hi = _wilson(k, nn)
            ci_lo.append(lo)
            ci_hi.append(hi)
        return DecayCurve(
            family=family,
            difficulty=-1,
            budget_axis=[first.budget_axis[first.budgets.index(b)] for b in budgets],
            budgets=budgets,
            acc=acc,
            ci_lo=ci_lo,
            ci_hi=ci_hi,
            n=n,
        )

    def summary(self) -> dict:
        """Per-family summary on two explicit views.

        - ``agg_*``  : the count-weighted aggregate curve over difficulties
          (this is the headline view; the aggregate is what a deployer sees
          when a family is used as a single capability probe).
        - ``span_*`` : the extreme-value span across difficulties (best single
          difficulty at the richest budget vs worst single difficulty at the
          tightest). Kept only for reference — it is *not* a budget effect.

        Both views are labelled so that a reader can never mistake the
        per-difficulty span for the aggregate budget trend.
        """
        out = {}
        for fam, diffs in self.curves.items():
            agg = self.family_curve(fam)
            cl = agg.cliff() if agg else None
            agg_richest = agg.acc[0] if agg else 0.0
            agg_tightest = agg.acc[-1] if agg else 0.0
            # largest single-step drop along the aggregate curve (budget
            # getting tighter: acc[i] -> acc[i+1], so a positive value is a
            # real degradation). Max, not min: min would report a *recovery*
            # step (e.g. the V-bottom of ringgap) as if it were a drop.
            agg_drops = [
                a - b for a, b in zip(agg.acc[:-1], agg.acc[1:]) if agg and len(agg.acc) > 1
            ]
            out[fam] = {
                "agg_richest": agg_richest,
                "agg_tightest": agg_tightest,
                "agg_range": agg_richest - agg_tightest,
                "agg_max_step_drop": (max(agg_drops) if agg_drops else 0.0),
                "cliff": cl,
                "n_difficulties": len(diffs),
                "sensitive_difficulties": sum(
                    1 for c in diffs.values() if c.cliff() is not None
                ),
                "span_richest": max((c.acc[0] for c in diffs.values()), default=0.0),
                "span_tightest": min((c.acc[-1] for c in diffs.values()), default=0.0),
                "span_range": max((c.acc[0] for c in diffs.values()), default=0.0)
                - min((c.acc[-1] for c in diffs.values()), default=0.0),
            }
        return out
