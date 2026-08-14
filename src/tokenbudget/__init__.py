"""tokenbudget: how visual token budget changes what an edge MLLM can see.

MiniCPM-V 4.6 compresses vision tokens 4x/16x for on-device deployment. We
sweep token budget (downsample mode x input resolution) against a battery of
perception tasks -- from coarse reading to dense per-element binding -- and
locate where perception quietly breaks as the budget shrinks.
"""

__version__ = "0.1.0"
