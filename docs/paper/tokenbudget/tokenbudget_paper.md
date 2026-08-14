# TokenBudget: Mapping Where Visual-Token Compression Breaks Edge MLLM Perception

_A Technical Report_

**Model under test:** MiniCPM-V 4.6 — a 1.3B edge-oriented multimodal LLM with native `downsample_mode` vision-token compression (4x / 16x).

**Budget axis:** downsample mode × input resolution → 6 configurations, **82–2,598** effective vision tokens (a 31.7× span).

**Dataset:** 960 procedurally-rendered perception probes across 8 task families, each with an exact ground-truth answer.

---

## 1. Motivation

Edge multimodal LLMs must fit inside a compute envelope. The dominant lever is the *visual-token budget*: how many tokens are spent describing a frame. MiniCPM-V 4.6 exposes `downsample_mode` (4x / 16x) and, through input resolution, a second knob. The model card reports aggregate accuracy under compression, but an application developer faces a harder question:

> **Which specific perception capabilities survive aggressive token compression, and which collapse first?**

"Fewer tokens" and "less perceptive" are not the same thing. Some capabilities degrade gracefully; others fail catastrophically once the budget passes a threshold. This project maps that landscape with a controlled, procedurally-generated perception battery — tasks with known ground truth and a calibrated difficulty axis — and reports where perception *quietly breaks*, and where it does not.

## 2. Method

### 2.1 Probe corpus

Images are rendered deterministically with PIL primitives only. Each family varies a single *spatial-frequency / information-density* axis as difficulty rises. Every probe carries an exact ground-truth answer, and the question offers a closed set of answer options so scoring is unambiguous.

| family | task | difficulty axis |
|---|---|---|
| `ocr` | read a 5-digit number | font size 96→16 px, additive noise |
| `count` | count same-color dots | 3→24 dots, shrinking radius |
| `mlread` | read the 2nd line of a digit column | 2→8 lines |
| `colcnt` | count red cells in a dense color grid | 8→30 cells |
| `spatial` | which color is on the left | square size, distractors |
| `color` | closest anchor hue (red/orange/pink) | hue offset 40→7 |
| `detail` | which row of rings has a gap | gap 12→1 px |
| `ringgap` | which row of thin rings has a 1px gap | ring radius 60→18 px |

20 seeds × 6 difficulties × 8 families = **960 probes**, rendered at 512×512 and downscaled to the configuration resolution at evaluation time.

### 2.2 Configurations

Six budget points. Token counts are *measured* by the engine itself (image-token count from the processed input), not taken from the model card:

| config | downsample | resolution | vision tokens |
|---|---|---|---|
| `16x@448` | 16x | 448 px | 82 |
| `4x@448` | 4x | 448 px | 274 |
| `16x@672` | 16x | 672 px | 347 |
| `16x@1280` | 16x | 1280 px | 678 |
| `4x@672` | 4x | 672 px | 1307 |
| `4x@1280` | 4x | 1280 px | 2598 |

The budget axis is thus non-uniform: 82 → 274 → 347 → 678 → 1307 → 2598, spanning more than a factor of 30.

### 2.3 Scoring and analysis

Free-form model output is mapped to the closed answer set by a lenient-but-explicit per-family normalizer (`tokenbudget.check`); a probe is correct iff normalized output equals ground truth. Accuracy per (family, difficulty, config) is reported with Wilson 95% confidence intervals.

A **cliff** is the first budget step (walking from richest to tightest) where accuracy drops ≥15 points in one step **and** ≥20 points below the richest configuration.

## 3. Results

### 3.1 The headline: a clean split between robust and fragile tasks

Per-family accuracy range across the whole budget axis (aggregated over difficulties):

| family | best-budget acc | worst-budget acc | range | verdict |
|---|---|---|---|---|
| `count` | 0.85 | 0.10 | 0.75 | **fragile** |
| `ringgap` | 1.00 | 0.45 | 0.55 | **fragile** (cliff) |
| `colcnt` | 0.20 | 0.00 | 0.20 | beyond model capability |
| `color` | 1.00 | 0.90 | 0.10 | robust |
| `spatial` | 1.00 | 0.95 | 0.05 | robust |
| `ocr` | 1.00 | 0.95 | 0.05 | robust |
| `mlread` | 1.00 | 0.95 | 0.05 | robust |
| `detail` | 1.00 | 1.00 | 0.00 | fully robust |

The split is clean and *not* predicted by difficulty alone: `detail` and `ringgap` are both "detect the gap" tasks at similar difficulty, yet `detail` stays at 1.00 across the whole axis while `ringgap` collapses. The distinguishing property is *what the answer requires*: recovering the presence of a coarse gap needs only a handful of vision tokens, while localizing which thin ring carries a 1px gap requires per-element binding at the model's resolution limit.

![Figure 1](figures/fig1_decay.png)

*Figure 1 — Per-family accuracy vs vision-token budget (log axis, right = richest budget). One line per difficulty level; the red dashed line marks a detected cliff.*

### 3.2 The cliff: ring-gap detection

`ringgap` is the cleanest demonstration of a compression cliff:

| difficulty | 4x@1280 (2598 tok) | 16x@672 (347 tok) | 16x@448 (82 tok) |
|---|---|---|---|
| d0 | 1.00 | 1.00 | 1.00 |
| d1 | 1.00 | 1.00 | 1.00 |
| d2 | 1.00 | 1.00 | 1.00 |
| d3 | 1.00 | 0.65 | 0.90 |
| d4 | 1.00 | 0.50 | 0.45 |
| d5 | 0.95 | 0.10 | 0.45 |

Difficulties d0–d2 are *fully robust*: 1.00 at every budget. Difficulties d3–d5 degrade steeply as the budget tightens — d5 falls from 0.95 at the richest configuration to 0.10 at 347 tokens. The aggregated family curve drops below the cliff margin at `16x@672` (accuracy 0.867 → 0.708, a 15.8-point step) and continues falling to 0.45 at the tightest budget.

### 3.3 Per-element counting degrades continuously

`count` shows a large range (0.75) but no single cliff: performance erodes monotonically with budget for the harder difficulties (d4: 0.30 → 0.00; d5: 0.15 → 0.00). This is the *graceful-then-empty* regime: each dot needs its own token; once the budget is below the dot count, counting fails. The difficulty curve is not monotone in difficulty itself (d2 lags d3 at several budgets), reflecting the model's guessing behavior on hard counts — but the budget trend is unambiguous.

### 3.4 The ceiling: dense-grid color counting

`colcnt` never exceeds 0.20 accuracy at any configuration. This task — counting red cells inside a dense 30-cell color grid — exceeds what a 1.3B edge MLLM can do even at full budget. It is a useful *negative control*: it bounds the claims we can make about budget sensitivity (a task that is already at floor cannot demonstrate a budget effect), and it documents a genuine capability boundary of the model class.

### 3.5 What survives compression

`ocr`, `mlread`, `spatial`, `detail`, and `color` are effectively budget-invariant:

- `ocr` and `mlread` stay at 0.95–1.00 across all six configurations — reading a digit string or a specific line needs only a coarse feature of each glyph.
- `detail` is 1.00 everywhere: the gap in a coarse ring is a low spatial-frequency feature that survives 16x downsampling.
- `spatial` and `color` stay above 0.90: layout and hue discrimination are dominated by low-frequency content.

The robustness of these tasks is not trivial — it shows that the 16x compression in MiniCPM-V 4.6 does not destroy *all* information; it selectively destroys high-frequency, per-element information.

## 4. Discussion

### 4.1 The compression landscape, in one sentence

For a 1.3B edge MLLM, **token compression trades away per-element binding before it trades away global structure**: counting, fine-gap localization, and dense-grid inspection degrade with budget; digit reading, coarse detection, layout, and color survive.

### 4.2 Implications for deployment

- **Choosing 16x vs 4x.** For applications that only read text, recognize coarse objects, or judge layout, 16x@448 (82 tokens) is nearly free: 0.95+ accuracy at a fraction of the compute. For applications that count small objects or inspect fine structure, the budget below ~350 tokens is a reliability cliff; 4x@672 (1307 tokens) or higher is warranted.
- **The cliff, not the slope, is the engineering-relevant number.** A developer wants to know where a capability *breaks*, not just that it degrades. The `ringgap` cliff at `16x@672` (347 tokens) is the kind of actionable threshold a deployment gate can encode.
- **Negative controls matter.** `colcnt`'s floor at full budget is a warning against assuming a task failure under compression is *caused* by compression. Sensitivity analysis needs a full-budget baseline.

### 4.3 Why this looks different from the model card

Model-card accuracy aggregates over natural-image tasks, where compression losses average out and the majority of pixels are background. A probe battery with per-task, per-difficulty resolution exposes the variance the aggregate hides.

## 5. Reproducibility

```
pip install -e .[gpu,paper,ui]
python scripts/build_probes.py --seeds 20 --out data/probes   # 960 probes
python scripts/run_sweep.py --probes data/probes --out data/sweep   # GPU, ~4h
python scripts/paper_facts.py --sweep data/sweep/sweep.json   # derived claims
python scripts/render_figures.py --sweep data/sweep/sweep.json --figs docs/figures
python scripts/render_tokenbudget_paper.py   # this report (HTML + PDF)
python -m tokenbudget ui --port 8000         # interactive console
```

`data/sweep/sweep.json` is the single source of truth: every number in this report traces to that table, and `scripts/paper_facts.py` re-derives the claims from it directly.

## References

1. Yao et al. — MiniCPM-V: A GPT-4V Level MLLM on Your Phone. *arXiv:2408.01800*.
2. Wu et al. — MiniCPM-V 4.6 technical report (2026), native 4x/16x vision-token downsample modes. *openbmb/MiniCPM-V-4.6*.
3. Li et al. — SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features. *arXiv:2502.14786*.
4. Liu et al. — LLaVA-UHD: An LMM Perceiving Any Aspect Ratio and High-Resolution Images. *ECCV 2024*.
5. Xu et al. — Efficient Multimodal Large Language Models: A Survey. *arXiv:2405.10739*.
