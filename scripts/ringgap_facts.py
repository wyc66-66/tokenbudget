#!/usr/bin/env python3
"""Extract report facts from the n=200 ringgap re-measurement.

Answers three questions the n=20 sweep could not:

1. Does the V-shaped recovery at 82 tokens survive at n=200? (table + sign test)
2. Is the 347-token dip significant against both neighbours (pairwise z)?
3. Does the cliff *location* stay at 16x@672?
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    lo = (c - z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d
    hi = (c + z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d
    return float(max(0.0, lo)), float(min(1.0, hi))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default="data/sweep_ringgap200/ringgap200.json", type=Path)
    args = ap.parse_args()

    d = json.loads(args.sweep.read_text(encoding="utf-8"))
    greedy = d["greedy"]

    print(f"n seeds = {d['n_seeds']}, sampled decodes = {d['sampled']}")
    print("\n== per-config per-difficulty accuracy (n=200, greedy) ==")
    table: dict[str, dict[int, dict]] = {}
    for r in greedy:
        cfg = r["config"]
        dd = r["difficulty"]
        table.setdefault(cfg, {})[dd] = {"k": r["correct"], "n": r["total"], "rate": r["rate"]}

    configs = sorted(table, key=lambda c: -int(c.split("@")[1]))
    print(f"{'difficulty':>10}" + "".join(f"{c:>14}" for c in configs))
    for dd in sorted(table[configs[0]]):
        row = f"{dd:>10}"
        for c in configs:
            r = table[c][dd]
            lo, hi = wilson(r["k"], r["n"])
            row += f"{r['rate']:>9.2f} [{lo:.2f},{hi:.2f}]"
        print(row)

    # aggregate
    print("\n== aggregate per config ==")
    agg = {}
    for c in configs:
        k = sum(table[c][dd]["k"] for dd in table[c])
        n = sum(table[c][dd]["n"] for dd in table[c])
        lo, hi = wilson(k, n)
        agg[c] = (k, n, k / n)
        print(f"  {c}: {k}/{n} = {k/n:.3f}  Wilson [{lo:.3f}, {hi:.3f}]")

    # pairwise z-tests between neighbouring configs
    print("\n== pairwise two-proportion z-tests (aggregate) ==")
    names = list(agg)
    for i in range(len(names) - 1):
        a, b = names[i], names[i + 1]
        k1, n1, p1 = agg[a]
        k2, n2, p2 = agg[b]
        p = (k1 + k2) / (n1 + n2)
        se = np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
        z = (p1 - p2) / se
        print(f"  {a} vs {b}: diff={p1-p2:+.4f}  z={z:+.2f}")

    # V-shape test: is the 347-token point below BOTH neighbours?
    print("\n== V-shape check ==")
    lo3 = agg[configs[1]][2]  # 16x@672
    lo3_neigh_low = agg[configs[2]][2]  # 16x@448
    lo3_neigh_high = agg[configs[0]][2]  # 4x@1280
    if lo3 < lo3_neigh_low and lo3 < lo3_neigh_high:
        print(f"  V-shape CONFIRMED at n=200: 347-tok ({lo3:.3f}) below 82-tok ({lo3_neigh_low:.3f}) "
              f"and 2598-tok ({lo3_neigh_high:.3f})")
    else:
        print(f"  V-shape NOT confirmed: 347-tok={lo3:.3f}, 82-tok={lo3_neigh_low:.3f}, 2598-tok={lo3_neigh_high:.3f}")

    # Wilson intervals of the two adjacent points
    print("\n== 82-tok vs 347-tok intervals ==")
    for c in (configs[1], configs[2]):
        r = agg[c]
        lo, hi = wilson(r[0], r[1])
        print(f"  {c}: {r[2]:.3f} [{lo:.3f}, {hi:.3f}]")

    # difficulty d5 detail
    print("\n== d5 (hardest) per config ==")
    for c in configs:
        r = table[c][5]
        lo, hi = wilson(r["k"], r["n"])
        print(f"  {c}: {r['rate']:.3f} ({r['k']}/{r['n']}) Wilson [{lo:.3f}, {hi:.3f}]")


if __name__ == "__main__":
    main()
