#!/usr/bin/env python3
"""Build the probe corpus (no GPU needed)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tokenbudget.probes.registry import build_probe_set  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate deterministic perception probe images.")
    ap.add_argument("--out", default="data/probes", type=Path)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--difficulties", type=int, nargs="*", default=list(range(6)))
    ap.add_argument("--families", type=str, nargs="*", default=None)
    ap.add_argument("--size", type=int, default=512)
    args = ap.parse_args()

    info = build_probe_set(
        args.out,
        n_seeds=args.seeds,
        difficulties=tuple(args.difficulties),
        families=tuple(args.families) if args.families else None,
        image_size=args.size,
    )
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
