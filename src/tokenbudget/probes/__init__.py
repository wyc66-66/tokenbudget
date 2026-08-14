"""Perception probes: procedurally generated vision tasks with known ground truth.

Five families, chosen to stress different parts of a small edge MLLM:
- ocr: reading digits under font size stress
- count: counting small objects at increasing densities
- spatial: relational layout ("is red left of blue")
- color: fine-grained hue discrimination
- detail: tiny feature detection (gap vs no-gap)

Every probe is generated from a fixed random seed and carries its own
ground-truth answer, so the full sweep is deterministic and auditable.
"""

from .registry import build_probe_set, PROBE_FAMILIES

__all__ = ["build_probe_set", "PROBE_FAMILIES"]
