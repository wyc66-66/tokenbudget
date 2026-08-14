"""MiniCPM-V 4.6 inference engine.

Wraps the standard HF Processor + AutoModelForImageTextToText flow. The two
things that matter for this project are both exposed as knobs:

- ``downsample_mode``: "4x" (more vision tokens, slower) or "16x" (default,
  edge-friendly). This is MiniCPM-V 4.6's native vision-token compression.
- ``resize``: optional input resolution; the LLaVA-UHD slicing inside the
  processor turns this into a different token budget.

The engine caches nothing by default; the sweep script decides batching.
"""
from __future__ import annotations

import torch
from PIL import Image


class MiniCPMEngine:
    def __init__(self, model_path: str = "openbmb/MiniCPM-V-4.6"):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path, dtype=torch.bfloat16, attn_implementation="sdpa"
        ).eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.model_path = model_path

    @torch.inference_mode()
    def ask(self, image: Image.Image, question: str, *, downsample_mode: str = "16x", max_slice_nums: int = 9, max_new_tokens: int = 24) -> str:
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": question}]}]
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={"downsample_mode": downsample_mode, "max_slice_nums": max_slice_nums},
        ).to(self.model.device)
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, downsample_mode=downsample_mode)
        return self.processor.decode(out[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True).strip()

    @torch.inference_mode()
    def token_budget(self, image: Image.Image, *, downsample_mode: str = "16x", max_slice_nums: int = 9) -> int:
        """Number of image tokens this input would produce (total - text tokens)."""
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": "x"}]}]
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={"downsample_mode": downsample_mode, "max_slice_nums": max_slice_nums},
        ).to(self.model.device)
        text_ids = self.processor.tokenizer("x", return_tensors="pt").input_ids.shape[-1]
        return inputs["input_ids"].shape[-1] - text_ids
