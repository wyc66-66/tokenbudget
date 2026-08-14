# TokenBudget: Where Visual-Token Budget Breaks Edge MLLM Perception

_A Technical Report_

**Model under test:** MiniCPM-V 4.6 (edge-oriented multimodal LLM)
**Budget axis:** downsample mode × input resolution → 6 configurations,
**82–2600** effective vision tokens.

---

## 1. Motivation

On-device multimodal LLMs must fit inside a compute envelope. The dominant
lever is the *visual-token budget*: how many tokens are spent describing a
frame. MiniCPM-V 4.6 exposes `downsample_mode` (4x / 16x) and, through input
resolution, an additional knob. But "fewer tokens" and "less perceptive" are
not the same thing. Some capabilities degrade gracefully; others fail
catastrophically once the budget passes a threshold.

This project maps that landscape for eight perception tasks with known ground
truth, and reports where perception *quietly breaks* — and where it does not.

## 2. Method

### 2.1 Probe corpus

Procedurally rendered images (PIL, deterministic), each with an exact answer:

| family | task | difficulty axis |
|---|---|---|
| `ocr` | read a 5-digit number | font size 96→16 px, noise |
| `count` | count same-color dots | 3→8 dots |
| `mlread` | read the 2nd line of digits | 2→8 lines |
| `colcnt` | count red cells in dense grid | 8→30 cells |
| `spatial` | which color is on the left | square size, distractors |
| `color` | closest anchor hue | hue offset 40→7 |
| `detail` | which ring has a gap | gap 12→1 px |
| `ringgap` | which ring has a 1 px gap | radius 60→18 px, stroke |

20 seeds × 6 difficulties × 8 families = **960 probes**.

### 2.2 Configurations

Six budget points (measured, not nominal, token counts):

| config | vision tokens |
|---|---|
| 16x@448 | 82 |
| 16x@672 | 347 |
| 16x@1280 | 678 |
| 4x@448 | 274 |
| 4x@672 | 1307 |
| 4x@1280 | 2598 |

### 2.3 Scoring and analysis

Free-form model output is mapped to the closed answer set by a lenient
explicit normalizer; a probe is correct iff normalized == ground truth.
Accuracy per (family, difficulty, config) is computed with Wilson 95% CIs.

A **cliff** is the first budget step (walking from richest to tightest) where
accuracy drops ≥15 points in one step **and** ≥20 points below the richest
configuration.

## 3. Results

### 3.1 Budget-sensitivity ranking

_Generated from data/sweep/sweep.json — see `scripts/paper_facts.py`._

### 3.2 Decay curves

_Figures: fig_decay.png, fig_budget.png, fig_sensitivity.png._

### 3.3 What collapses

### 3.4 What survives

## 4. Discussion

## 5. Reproducibility

```
pip install -e .[gpu,paper,ui]
python scripts/build_probes.py --seeds 20 --out data/probes
python scripts/run_sweep.py --probes data/probes --out data/sweep
python scripts/paper_facts.py --sweep data/sweep/sweep.json
python -m tokenbudget ui --port 8000
```

## References

[cite MiniCPM-V, token compression literature, edge deployment]
