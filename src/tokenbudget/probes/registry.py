"""Probe set registry: build a full deterministic task bank."""
from __future__ import annotations

import json
from pathlib import Path

from .generators import FAMILY_GENERATORS

PROBE_FAMILIES = tuple(sorted(FAMILY_GENERATORS))


def build_probe_set(
    out_dir: Path,
    *,
    n_seeds: int = 20,
    difficulties: tuple[int, ...] = (0, 1, 2, 3, 4, 5),
    families: tuple[str, ...] | None = None,
    image_size: int = 512,
) -> dict:
    """Render every (family, seed, difficulty) combination to disk.

    Returns a manifest: list of dicts {id, family, seed, difficulty,
    image (relpath), answer, question}.
    """
    from PIL import Image

    out_dir = Path(out_dir)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    families = tuple(families) if families else PROBE_FAMILIES
    questions = {
        "ocr": "What number is shown? Answer with the digits only.",
        "count": "How many objects are in the image? Answer with a number only.",
        "mlread": "Several lines of digits are shown. What is the number on the second line? Answer with the digits only.",
        "colcnt": "A grid of small colored squares is shown. How many of them are red? Answer with a number only.",
        "spatial": "Two colored squares are shown side by side. Which color is on the left? Answer with a single color word.",
        "color": "A colored square is shown. Which color is it closest to: red, orange, or pink? Answer with a single word.",
        "detail": "Two rows of ring shapes are shown, one at the top and one at the bottom. Which row has a gap in its outline, top or bottom? Answer with 'top' or 'bottom'.",
        "ringgap": "Two rows of ring shapes are shown, one at the top and one at the bottom. Which row has a gap in its outline, top or bottom? Answer with 'top' or 'bottom'.",
    }

    manifest = []
    for fam in families:
        gen = FAMILY_GENERATORS[fam]
        for seed in range(n_seeds):
            for diff in difficulties:
                img, answer = gen(seed, diff)
                img = img.resize((image_size, image_size), Image.LANCZOS)
                pid = f"{fam}_s{seed:03d}_d{diff}"
                img.save(img_dir / f"{pid}.png")
                manifest.append(
                    {
                        "id": pid,
                        "family": fam,
                        "seed": seed,
                        "difficulty": diff,
                        "image": f"images/{pid}.png",
                        "answer": answer,
                        "question": questions[fam],
                    }
                )

    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    return {"families": list(families), "n_probes": len(manifest)}
