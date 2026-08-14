#!/usr/bin/env python3
"""Sweep the token budget grid against the probe corpus (GPU).

Configurations: downsample mode {4x, 16x} x input resolution {448, 672, 1280}.
For each config we re-run every probe once and record correct/total per
(family, difficulty). Output: a JSON table plus a budget_axis per config
(normalized vision-token count measured by the engine itself).
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

CONFIGS = [
    ("16x", 448, None),
    ("16x", 672, None),
    ("16x", 1280, None),
    ("4x", 448, None),
    ("4x", 672, None),
    ("4x", 1280, None),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probes", default="data/probes", type=Path)
    ap.add_argument("--out", default="data/sweep", type=Path)
    ap.add_argument("--model", default="openbmb/MiniCPM-V-4.6")
    ap.add_argument("--families", type=str, nargs="*", default=None)
    ap.add_argument("--difficulties", type=int, nargs="*", default=None)
    ap.add_argument("--configs", type=str, nargs="*", default=None,
                    help="subset of labels like 16x@448,4x@1280 (default: all)")
    args = ap.parse_args()

    configs = CONFIGS
    if args.configs:
        wanted = set(args.configs)
        configs = [c for c in CONFIGS if f"{c[0]}@{c[1]}" in wanted]

    manifest = json.loads((args.probes / "manifest.json").read_text(encoding="utf-8"))
    if args.families:
        manifest = [p for p in manifest if p["family"] in args.families]
    if args.difficulties:
        manifest = [p for p in manifest if p["difficulty"] in args.difficulties]

    engine = MiniCPMEngine(args.model)
    args.out.mkdir(parents=True, exist_ok=True)

    # measure token budget per config using a reference probe image
    ref = Image.open(args.probes / "images" / f"{manifest[0]['id']}.png").convert("RGB")
    budgets: dict[str, dict] = {}
    for mode, size, slices in configs:
        ref2 = ref.resize((size, size))
        n_tok = engine.token_budget(ref2, downsample_mode=mode)
        budgets[f"{mode}@{size}"] = {"mode": mode, "res": size, "vision_tokens": n_tok}

    rows: dict[str, list[dict]] = {}
    t0 = time.time()
    for mode, size, slices in configs:
        label = f"{mode}@{size}"
        per_bucket: dict[tuple[str, int], dict] = {}
        n_done = 0
        for p in manifest:
            img = Image.open(args.probes / p["image"]).convert("RGB")
            if size != img.width:
                img = img.resize((size, size))
            raw = engine.ask(img, p["question"], downsample_mode=mode)
            ok = is_correct(p["family"], raw, p["answer"])
            key = (p["family"], p["difficulty"])
            d = per_bucket.setdefault(key, {"correct": 0, "total": 0})
            d["total"] += 1
            d["correct"] += int(ok)
            n_done += 1
        fam_stats = []
        for (fam, diff), st in sorted(per_bucket.items()):
            fam_stats.append(
                {
                    "family": fam,
                    "difficulty": diff,
                    "label": label,
                    "mode": mode,
                    "res": size,
                    "budget_axis": budgets[label]["vision_tokens"],
                    "correct": st["correct"],
                    "total": st["total"],
                }
            )
        rows.setdefault("_meta", []).append({"label": label, "budgets": budgets[label]})
        for fs in fam_stats:
            rows.setdefault(fs["family"], []).append(fs)
        print(f"[{label}] {n_done} probes done in {time.time()-t0:.0f}s", flush=True)

    out = {"configs": configs, "budget_table": budgets, "by_family": {}}
    for fam, r in rows.items():
        if fam != "_meta":
            out["by_family"][fam] = r
    out["_meta"] = rows.get("_meta", [])
    (args.out / "sweep.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("wrote", args.out / "sweep.json")


if __name__ == "__main__":
    main()
