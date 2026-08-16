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

    print("\n== per-family summary (count-weighted aggregate across difficulties) ==")
    for fam, s in result.summary().items():
        print(
            f"  {fam}: agg {s['agg_richest']:.3f} (richest) -> {s['agg_tightest']:.3f} "
            f"(tightest), range={s['agg_range']:.3f}, max_step_drop={s['agg_max_step_drop']:.3f}"
        )
        if s["cliff"]:
            c = s["cliff"]
            print(f"    cliff at {c['at']} (acc {c['acc_before']:.3f} -> {c['acc_after']:.3f}, drop {c['drop']:.3f})")
        b = s.get("cliff_bootstrap")
        if b:
            surv = b["survival_rate"]
            if b["cliff_index_ci"]:
                print(
                    f"    bootstrap: survival={surv:.3f}, location CI "
                    f"(budget index) {b['cliff_index_ci'][0]}..{b['cliff_index_ci'][1]} "
                    f"median={b.get('cliff_median')}"
                )
            else:
                print(f"    bootstrap: survival={surv:.3f} (no cliff in any draw)")

    # cross-family: how much does the tightest budget hurt vs loosest?
    print("\n== budget-sensitivity ranking (aggregate range) ==")
    rows = []
    for fam, s in result.summary().items():
        rows.append((s["agg_range"], fam))
    for rng, fam in sorted(rows, reverse=True):
        print(f"  {fam}: {rng:.3f}")

    print("\n== per-family per-difficulty extreme span (NOT a budget effect; reference only) ==")
    for fam, s in result.summary().items():
        print(
            f"  {fam}: span_richest={s['span_richest']:.3f} span_tightest={s['span_tightest']:.3f} "
            f"span_range={s['span_range']:.3f}"
        )


if __name__ == "__main__":
    main()
