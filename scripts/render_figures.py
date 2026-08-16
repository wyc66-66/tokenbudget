#!/usr/bin/env python3
"""Render analysis figures and the arXiv-style paper for tokenbudget."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tokenbudget.decay import SweepResult  # noqa: E402

FAMILY_LABELS = {
    "ocr": "digit reading",
    "count": "counting",
    "mlread": "targeted line reading",
    "colcnt": "per-element color counting",
    "spatial": "spatial layout",
    "color": "fine hue",
    "detail": "gap detection",
    "ringgap": "ring-gap detection",
}

FAMILY_COLORS = {
    "ocr": "#5f8db8",
    "count": "#7fa86a",
    "mlread": "#c8a44c",
    "colcnt": "#c0585a",
    "spatial": "#8a6fb8",
    "color": "#d08ab4",
    "detail": "#5f9e9e",
    "ringgap": "#e08a3c",
}


def _diff_color(d: int, n: int) -> str:
    if n <= 1:
        return "#c05621"
    return plt.cm.RdYlGn(0.9 - 0.9 * (d / max(n - 1, 1)))


def fig_decay(result: SweepResult, out: Path) -> None:
    """One panel per family: accuracy vs budget, one line per difficulty."""
    families = sorted(result.curves)
    n_fam = len(families)
    cols = 3
    rows = (n_fam + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4.6 * rows))
    axes = axes.flatten()
    for i, fam in enumerate(families):
        ax = axes[i]
        diffs = result.curves[fam]
        first = next(iter(diffs.values()))
        budget_axis = first.budget_axis
        for d, c in diffs.items():
            ax.plot(c.budget_axis, c.acc, "-o", ms=4, lw=1.4,
                    color=_diff_color(int(d), len(diffs)),
                    label=f"d{int(d)}")
            ax.fill_between(c.budget_axis, c.ci_lo, c.ci_hi,
                            color=_diff_color(int(d), len(diffs)), alpha=0.10)
        ax.set_xscale("log")
        ax.set_title(FAMILY_LABELS.get(fam, fam), fontsize=12)
        ax.set_xlabel("vision tokens (log)")
        ax.set_ylabel("accuracy")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.3)
        agg = result.family_curve(fam)
        cl = agg.cliff() if agg else None
        if cl:
            ax.axvline(first.budget_axis[first.budgets.index(cl["at"])],
                       color="r", ls="--", alpha=0.7)
        if i == 0:
            ax.legend(fontsize=7, ncol=2)
    for j in range(n_fam, len(axes)):
        axes[j].axis("off")
    fig.suptitle("Per-family accuracy vs visual-token budget (MiniCPM-V 4.6)", fontsize=14)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def fig_budget_table(budget_table: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    labels = list(budget_table.keys())
    toks = [budget_table[k]["vision_tokens"] for k in labels]
    colors = ["#8a4f34" if k.startswith("4x") else "#b98f66" for k in labels]
    bars = ax.barh(range(len(labels)), toks, color=colors)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("vision tokens")
    ax.invert_yaxis()
    for i, t in enumerate(toks):
        ax.text(t + 5, i, str(t), va="center", fontsize=10)
    ax.set_title("Effective vision-token budget per configuration")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def fig_sensitivity_ranking(result: SweepResult, out: Path) -> None:
    """Horizontal bars of best-minus-worst budget accuracy per family."""
    summary = result.summary()
    fams = sorted(summary, key=lambda f: -summary[f]["agg_range"])
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(fams)))
    y = range(len(fams))
    ranges = [summary[f]["agg_range"] for f in fams]
    bars = ax.barh(list(y), ranges, color=[FAMILY_COLORS.get(f, "#888") for f in fams])
    ax.set_yticks(list(y))
    ax.set_yticklabels([FAMILY_LABELS.get(f, f) for f in fams], fontsize=10)
    ax.set_xlabel("best-budget minus worst-budget accuracy (points)")
    ax.set_xlim(0, 1)
    for i, r in enumerate(ranges):
        ax.text(r + 0.01, i, f"{r:.2f}", va="center", fontsize=9)
    ax.axvline(0.15, color="r", ls=":", alpha=0.6)
    ax.text(0.16, len(fams) - 0.4, "cliff threshold (0.15)", fontsize=8, color="r")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default="data/sweep/sweep.json", type=Path)
    ap.add_argument("--figs", default="docs/paper/tokenbudget/figures", type=Path)
    args = ap.parse_args()

    data = json.loads(args.sweep.read_text(encoding="utf-8"))
    result = SweepResult.load(args.sweep)
    args.figs.mkdir(parents=True, exist_ok=True)

    # Names match what scripts/render_tokenbudget_paper.py embeds, so a fresh
    # run regenerates the committed report exactly.
    fig_decay(result, args.figs / "fig1_decay.png")
    fig_budget_table(data["budget_table"], args.figs / "fig2_budget.png")
    fig_sensitivity_ranking(result, args.figs / "fig3_sensitivity.png")

    print(json.dumps(result.summary(), indent=1))


if __name__ == "__main__":
    main()
