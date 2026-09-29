"""Tiny OpenAI-compatible chat client (standard library only) for the GPU labs.

Backends, in order of preference:
  1. LLM_BASE_URL (+ LLM_MODEL, LLM_API_KEY): any OpenAI-compatible server (NIM, vLLM, ...)
  2. NVIDIA_API_KEY set: NVIDIA's hosted NIM endpoints (https://build.nvidia.com)
  3. otherwise: the local vLLM server started by `make llm-up` on the lab GPU (port 8008)
"""

import json
import os
import urllib.error
import urllib.request

LOCAL_URL = "http://localhost:8008/v1"
LOCAL_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
HOSTED_URL = "https://integrate.api.nvidia.com/v1"

BASE_URL = os.environ.get("LLM_BASE_URL") or (HOSTED_URL if os.environ.get("NVIDIA_API_KEY") else LOCAL_URL)
IS_HOSTED_NIM = BASE_URL.startswith(HOSTED_URL)
MODEL = os.environ.get("LLM_MODEL") or ("meta/llama-3.1-8b-instruct" if IS_HOSTED_NIM else LOCAL_MODEL)
API_KEY = os.environ.get("LLM_API_KEY") or os.environ.get("NVIDIA_API_KEY", "not-needed")


def chat(messages, model=MODEL, **params) -> str:
    """Send a chat completion and return the assistant text.

    `messages` is a string or a list of {"role", "content"}; `params` pass through
    (temperature, top_p, max_tokens, stop, seed, ...). Use guided_choice() for constrained output.
    """
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    params.setdefault("max_tokens", 512)
    body = json.dumps({"model": model, "messages": messages, **params}).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/chat/completions",
        data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"},
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.load(resp)["choices"][0]["message"]["content"]


def guided_choice(choices: list[str]) -> dict:
    """Constrained decoding params: NIM uses the `nvext` extension, vLLM a top-level field."""
    return {"nvext": {"guided_choice": choices}} if IS_HOSTED_NIM else {"guided_choice": choices}


def available() -> bool:
    try:
        req = urllib.request.Request(f"{BASE_URL}/models", headers={"Authorization": f"Bearer {API_KEY}"})
        urllib.request.urlopen(req, timeout=5)
        return True
    except (urllib.error.URLError, OSError):
        return False


def describe() -> str:
    return f"{MODEL} @ {BASE_URL}"
