#!/usr/bin/env python3
"""Re-measure the ringgap cliff cells with larger n and sampled decodes (GPU).

The main sweep used n=20 per (family, difficulty, config) cell with a single
greedy decode; §4.5 of the report flags that the cliff location has real
sampling variance at that n, and §4.6 proposes re-measuring the 347-token
operating point with 200+ seeds and sampled decodes. This script does exactly
that for the one cliff family (ringgap):

- 200 seeds per difficulty (probes built by build_probes.py --seeds 200)
- three budget points: 4x@1280 (2,598 tok baseline), 16x@672 (347 tok, the
  measured cliff), 16x@448 (82 tok, the V-shape recovery side)
- per probe: one greedy decode plus ``--sampled`` temperature decodes

Output JSON (data/sweep_ringgap200/ringgap200.json) mirrors the sweep schema:
per (config, difficulty) correct/total plus a sampled-decodes table with the
per-probe variance, so the V-shape question is answerable either way.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PIL import Image

from tokenbudget.check import is_correct  # noqa: E402
from tokenbudget.engine import MiniCPMEngine  # noqa: E402

CONFIGS = [("4x", 1280), ("16x", 672), ("16x", 448)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probes", default="data/probes_ringgap200", type=Path)
    ap.add_argument("--out", default="data/sweep_ringgap200", type=Path)
    ap.add_argument("--model", default="openbmb/MiniCPM-V-4.6")
    ap.add_argument("--sampled", type=int, default=10,
                    help="number of temperature-sampled decodes per probe (0 = greedy only)")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    manifest = json.loads((args.probes / "manifest.json").read_text(encoding="utf-8"))
    probes = [p for p in manifest if p["family"] == "ringgap"]
    print(f"[focus] {len(probes)} ringgap probes (seeds 0..{max(p['seed'] for p in probes)})", flush=True)

    engine = MiniCPMEngine(args.model)
    args.out.mkdir(parents=True, exist_ok=True)

    import random

    rng = random.Random(args.seed)

    rows = []           # greedy, one per (config, difficulty)
    sampled_rows = []   # per (config, difficulty, probe) sampled decodes
    t0 = time.time()
    for mode, size in CONFIGS:
        label = f"{mode}@{size}"
        per_diff: dict[int, dict] = {}
        for p in probes:
            img = Image.open(args.probes / p["image"]).convert("RGB")
            if size != img.width:
                img = img.resize((size, size))
            raw = engine.ask(img, p["question"], downsample_mode=mode)
            ok = is_correct(p["family"], raw, p["answer"])
            d = per_diff.setdefault(p["difficulty"], {"correct": 0, "total": 0})
            d["total"] += 1
            d["correct"] += int(ok)

            if args.sampled > 0:
                import torch

                sampled_correct = 0
                for si in range(args.sampled):
                    torch.manual_seed(args.seed + si)
                    r_raw = engine.ask(img, p["question"], downsample_mode=mode,
                                       temperature=args.temperature)
                    if is_correct(p["family"], r_raw, p["answer"]):
                        sampled_correct += 1
                sampled_rows.append({
                    "config": label,
                    "difficulty": p["difficulty"],
                    "seed": p["seed"],
                    "greedy": ok,
                    "sampled_n": args.sampled,
                    "sampled_correct": sampled_correct,
                })
        for diff, st in sorted(per_diff.items()):
            rows.append({
                "config": label,
                "difficulty": diff,
                "mode": mode,
                "res": size,
                "correct": st["correct"],
                "total": st["total"],
                "rate": st["correct"] / st["total"],
            })
        print(f"[{label}] {st['total']} probes done in {time.time()-t0:.0f}s", flush=True)

    out = {
        "model": args.model,
        "n_seeds": max(p["seed"] for p in probes) + 1,
        "sampled": args.sampled,
        "temperature": args.temperature,
        "seed": args.seed,
        "greedy": rows,
        "sampled_rows": sampled_rows,
    }
    out_file = args.out / "ringgap200.json"
    out_file.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"[focus] wrote {out_file}")


if __name__ == "__main__":
    main()
