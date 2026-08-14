# TokenBudget

**How visual-token budget changes what an edge MLLM can see.**

Edge multimodal LLMs (MiniCPM-V 4.6) compress vision tokens 4x or 16x to fit
on-device compute budgets. That compression changes *how many* tokens describe
a frame — but does it change *what the model can actually perceive*? This
project sweeps the token budget (downsample mode × input resolution) against a
battery of perception tasks with known ground truth, and maps where perception
quietly breaks as the budget shrinks.

## Research question

MiniCPM-V 4.6's `downsample_mode` gives two operating points — 4x (more tokens)
and 16x (edge-friendly). Combined with input resolution, we get a budget axis
from ~82 to ~2600 vision tokens. We ask:

> **Which perception capabilities survive aggressive token compression,
> and which collapse first?**

The working hypothesis, confirmed by the data: **coarse capabilities** (reading
a large number, judging spatial layout, discriminating color) are budget-robust;
**per-element binding** (counting many small objects, locating one cell in a
dense grid, detecting a sub-pixel gap) is budget-fragile, degrading sharply
below a configuration-specific threshold.

## Method

1. **Procedural probes.** Deterministic PIL-rendered images, 8 families × 6
   difficulty levels × 20 seeds = 960 probes, each with an exact ground-truth
   answer:

   | family | question | difficulty axis |
   |--------|----------|-----------------|
   | `ocr` | read a 5-digit number | font size, noise |
   | `count` | count same-color dots | number of dots |
   | `mlread` | read the 2nd line of digits | line count |
   | `colcnt` | count red cells in a dense grid | grid density |
   | `spatial` | which color is on the left | square size, distractors |
   | `color` | closest anchor hue | hue offset |
   | `detail` | which ring has a gap | gap width |
   | `ringgap` | which ring has a 1px gap | ring radius, stroke |

2. **Budget grid.** `downsample_mode ∈ {4x, 16x} × resolution ∈ {448, 672, 1280}`
   gives six configurations. The engine measures the *actual* vision-token count
   per configuration (not the nominal one).

3. **Scoring.** A lenient-but-explicit normalizer maps free-form model output to
   the closed answer set; a probe is correct iff the normalized answer equals
   ground truth.

4. **Decay analysis.** Per (family, difficulty) accuracy vs token budget with
   Wilson 95% CIs. A *cliff* is the first budget step where accuracy drops
   ≥15 points and ≥20 points below the richest configuration.

## Reproduce

```bash
# 1. build the probe corpus (CPU)
python scripts/build_probes.py --seeds 20 --out data/probes

# 2. run the budget sweep (GPU, MiniCPM-V 4.6)
python scripts/run_sweep.py --probes data/probes --out data/sweep

# 3. inspect the numbers and render paper figures
python scripts/paper_facts.py --sweep data/sweep/sweep.json
python scripts/render_figures.py --sweep data/sweep/sweep.json --figs docs/figures

# 4. local web console
python -m tokenbudget ui --port 8000   # open http://127.0.0.1:8000
```

GPU requirements: CUDA with ~16 GB VRAM (bf16), `pip install -e .[gpu,paper,ui]`.

## Key numbers

`data/sweep/sweep.json` is the single source of truth for the paper. Every
claim in the report can be traced back to this table. `scripts/paper_facts.py`
prints the derived claims (budget table, per-family cliffs, sensitivity
ranking).

## Layout

```
src/tokenbudget/
  engine.py      MiniCPM-V 4.6 inference + token-budget measurement
  probes/        procedural image generation + registry
  check.py       answer normalization / correctness
  decay.py       Wilson-CI decay curves + cliff detection
  ui/            FastAPI console + static dashboard
scripts/         corpus build, sweep, figures, paper facts
data/            probes/, sweep/
docs/            figures + technical report
```
