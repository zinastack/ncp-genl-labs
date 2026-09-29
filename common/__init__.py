"""Shared helpers for the GPU labs (imported by every gpu_lab.py)."""

import os

# Hugging Face libraries read HF_TOKEN. Accept a key exported as HUGGING_FACE too.
# The token is optional: every model used in the labs is public. It lifts anonymous rate limits
# and is required only if you swap in a gated model (e.g. Llama).
if os.environ.get("HUGGING_FACE") and not os.environ.get("HF_TOKEN"):
    os.environ["HF_TOKEN"] = os.environ["HUGGING_FACE"]
