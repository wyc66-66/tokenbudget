"""Tests for TokenBudget normalization and decay/cliff analysis."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tokenbudget.check import is_correct, norm_ocr, normalize
from tokenbudget.decay import DecayCurve, SweepResult


# --------------------------------------------------------------------------- #
# Answer normalization
# --------------------------------------------------------------------------- #
class TestNorm:
    def test_norm_ocr_digits(self):
        assert norm_ocr("The number is 12345.") == "12345"
        assert norm_ocr("42") == "42"
        assert norm_ocr("no digits here") == ""

    def test_norm_count_spelled_out(self):
        assert normalize("count", "There are three circles") == "3"
        assert normalize("count", "answer: 7") == "7"
        assert normalize("count", "none") is None

    def test_norm_color(self):
        assert normalize("color", "It is RED") == "red"
        assert normalize("color", "orange") == "orange"
        assert normalize("color", "unknown") is None

    def test_norm_spatial_family(self):
        assert normalize("spatial", "left of it is a blue object") == "blue"
        assert normalize("spatial", "no color") is None

    def test_is_correct_matches(self):
        assert is_correct("ocr", "123", "123") is True
        assert is_correct("ocr", "12", "123") is False
        assert is_correct("count", "five", "5") is True


# --------------------------------------------------------------------------- #
# Cliff detection + bootstrap
# --------------------------------------------------------------------------- #
class TestDecayCurve:
    def _curve(self, accs, budgets=None, n=20):
        if budgets is None:
            budgets = [f"b{i}" for i in range(len(accs))]
        axis = list(reversed(range(len(accs))))
        return DecayCurve(
            family="t",
            difficulty=0,
            budget_axis=[float(x) for x in axis],
            budgets=budgets,
            acc=accs,
            ci_lo=[0.0] * len(accs),
            ci_hi=[1.0] * len(accs),
            n=[n] * len(accs),
        )

    def test_cliff_detected(self):
        c = self._curve([0.95, 0.90, 0.30, 0.20])
        cl = c.cliff(margin=0.15, min_drop=0.20)
        assert cl is not None
        assert cl["at"] == "b2"
        assert cl["drop"] == pytest.approx(0.60)

    def test_no_cliff_on_flat(self):
        c = self._curve([0.5, 0.5, 0.5])
        assert c.cliff() is None

    def test_no_cliff_on_gradual(self):
        # 0.1 steps never hit the 0.15 one-step margin
        c = self._curve([0.95, 0.85, 0.75, 0.65])
        assert c.cliff(margin=0.15) is None

    def test_cliff_requires_two_points(self):
        c = self._curve([0.9])
        assert c.cliff() is None

    def test_bootstrap_survives_strong_cliff(self):
        # very strong step: bootstrap should usually keep a cliff
        c = self._curve([0.95, 0.05, 0.05, 0.05])
        b = c.cliff_bootstrap(n_boot=200, seed=0)
        assert b["survival_rate"] >= 0.5
        assert b["cliff_index_ci"] is not None

    def test_bootstrap_low_on_flat(self):
        # n=20 resampling makes even a flat 0.5 curve look cliffy ~18% of
        # draws; the honest claim is "near-chance", not "never"
        c = self._curve([0.5, 0.5, 0.5, 0.5])
        b = c.cliff_bootstrap(n_boot=500, seed=0)
        assert b["survival_rate"] < 0.3

    def test_bootstrap_deterministic_seed(self):
        c = self._curve([0.9, 0.3, 0.1])
        a = c.cliff_bootstrap(n_boot=100, seed=1)
        b = c.cliff_bootstrap(n_boot=100, seed=1)
        assert a["survival_rate"] == b["survival_rate"]
        assert a["cliff_index_ci"] == b["cliff_index_ci"]


# --------------------------------------------------------------------------- #
# SweepResult aggregation
# --------------------------------------------------------------------------- #
class TestSweepResult:
    @pytest.fixture()
    def result(self, tmp_path):
        rows = []
        for fam in ("ringgap", "count"):
            for d in (0, 1, 2):
                # ringgap: strong budget drop; count: flat
                base = 0.95 if fam == "ringgap" else 0.5
                for i, label in enumerate(["rich", "mid", "tight"]):
                    acc = max(0.05, base - 0.35 * i) if fam == "ringgap" else base
                    rows.append(
                        {
                            "family": fam,
                            "difficulty": d,
                            "label": label,
                            "mode": "x",
                            "res": 448,
                            "budget_axis": float(3 - i),
                            "correct": int(round(acc * 20)),
                            "total": 20,
                        }
                    )
        path = tmp_path / "sweep.json"
        path.write_text(json.dumps({"by_family": {}}), encoding="utf-8")
        return SweepResult({"ringgap": {0: _mk(rows, "ringgap", 0), 1: _mk(rows, "ringgap", 1), 2: _mk(rows, "ringgap", 2)},
                            "count": {0: _mk(rows, "count", 0), 1: _mk(rows, "count", 1), 2: _mk(rows, "count", 2)}})

    def test_family_curve_aggregation(self, result):
        c = result.family_curve("ringgap")
        assert c is not None
        # richest budget should be highest accuracy
        assert c.acc[0] > c.acc[-1]

    def test_summary_has_cliff_and_bootstrap(self, result):
        s = result.summary()
        assert s["ringgap"]["cliff"] is not None
        assert s["ringgap"]["cliff_bootstrap"] is not None
        # count is flat -> no cliff
        assert s["count"]["cliff"] is None


def _mk(rows, fam, diff):
    """Extract rows for a family/difficulty and build a DecayCurve."""
    from tokenbudget.decay import DecayCurve

    rs = [r for r in rows if r["family"] == fam and r["difficulty"] == diff]
    return DecayCurve.from_records(fam, diff, rs)
