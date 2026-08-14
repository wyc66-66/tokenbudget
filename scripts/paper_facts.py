#!/usr/bin/env python3
"""Extract the factual claims the paper makes, straight from sweep.json."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tokenbudget.decay import SweepResult  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default="data/sweep/sweep.json", type=Path)
    args = ap.parse_args()

    data = json.loads(args.sweep.read_text(encoding="utf-8"))
    result = SweepResult.load(args.sweep)

    print("== budget table ==")
    for label, b in data["budget_table"].items():
        print(f"  {label}: {b['vision_tokens']} vision tokens")

    print("\n== per-family summary ==")
    for fam, s in result.summary().items():
        print(f"  {fam}: best={s['best_acc']:.3f} worst={s['worst_acc']:.3f} range={s['range']:.3f}")
        if s["cliff"]:
            c = s["cliff"]
            print(f"    cliff at {c['at']} (acc {c['acc_before']:.3f} -> {c['acc_after']:.3f}, drop {c['drop']:.3f})")

    # cross-family: how much does the tightest budget hurt vs loosest?
    print("\n== budget-sensitivity ranking (best-worst range) ==")
    rows = []
    for fam, s in result.summary().items():
        rows.append((s["range"], fam))
    for rng, fam in sorted(rows, reverse=True):
        print(f"  {fam}: {rng:.3f}")


if __name__ == "__main__":
    main()
