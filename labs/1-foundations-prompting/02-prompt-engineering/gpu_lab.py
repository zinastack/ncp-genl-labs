# %% [markdown]
# # Lab 02 — GPU: prompting a real model with YOUR decoding code
# Section 1 instance (1× L4/T4). Uses Qwen2.5-0.5B-Instruct locally through transformers
# (override with LAB_MODEL=...). The last cell uses the vLLM server from `make llm-up` if running.
#
# 1. A generation loop built on your apply_temperature / top_k / top_p / repetition penalty
# 2. Constrained decoding (guided_choice) for classification
# 3. Zero-shot vs few-shot accuracy on financial headlines
# 4. Chain-of-thought + self-consistency
# 5. Structured JSON output with parse + retry

# %%
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from common import device as dev  # noqa: E402
from common import llm  # noqa: E402
from common.labimpl import load  # noqa: E402

lab = load(HERE)
DEVICE, HALF = dev.device(), dev.half_dtype()
MODEL = os.environ.get("LAB_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")  # 0.5B is too weak to show few-shot gains
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=HALF).to(DEVICE).eval()
print(dev.summary(), "\nmodel:", MODEL)


def chat_ids(user: str, system: str | None = None) -> torch.Tensor:
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return tok(text, return_tensors="pt").input_ids.to(DEVICE)


# %% 1. Generation loop driven by your sampling functions (KV cache reused every step)
@torch.no_grad()
def generate(prompt_ids, max_new_tokens=60, temperature=0.0, top_k=None, top_p=None,
             repetition_penalty=1.0, seed=0):
    rng = np.random.default_rng(seed)
    out = model(prompt_ids, use_cache=True)
    cache, generated = out.past_key_values, []
    logits = out.logits[0, -1].float().cpu().numpy()
    for _ in range(max_new_tokens):
        if repetition_penalty != 1.0:
            logits = lab.apply_repetition_penalty(logits, generated, repetition_penalty)
        if top_k:
            logits = lab.top_k_filter(logits, top_k)
        if top_p:
            logits = lab.top_p_filter(logits, top_p)
        probs = lab.apply_temperature(logits, temperature)
        next_id = int(np.argmax(probs)) if temperature == 0 else int(rng.choice(len(probs), p=probs))
        if next_id == tok.eos_token_id or next_id in (tok.convert_tokens_to_ids("<|im_end|>"),):
            break
        generated.append(next_id)
        out = model(torch.tensor([[next_id]], device=DEVICE), past_key_values=cache, use_cache=True)
        cache, logits = out.past_key_values, out.logits[0, -1].float().cpu().numpy()
    return tok.decode(generated)


prompt = chat_ids("Write one sentence describing what a GPU does.")
print("greedy        :", generate(prompt))
for s in range(3):
    print(f"T=0.9 top_p=.9 :", generate(prompt, temperature=0.9, top_p=0.9, seed=s))
print("T=1.5 (chaos) :", generate(prompt, temperature=1.5, seed=0))

# %% 2 + 3. Constrained decoding, and zero-shot vs few-shot on a labelled mini set
DATA = [
    ("Chipmaker raises full-year guidance on data-center demand", "positive"),
    ("Retailer files for bankruptcy protection", "negative"),
    ("Central bank leaves interest rates unchanged", "neutral"),
    ("Bank reports record profit but discloses regulator probe", "negative"),
    ("Airline shares jump after fuel costs fall", "positive"),
    ("Automaker recalls 2 million vehicles over brake defect", "negative"),
    ("Company schedules annual shareholder meeting for June", "neutral"),
    ("Software firm beats earnings estimates, stock hits all-time high", "positive"),
    ("Insurer warns of losses after hurricane season", "negative"),
    ("Index closes flat ahead of jobs report", "neutral"),
    ("Pharma group wins approval for new cancer drug", "positive"),
    ("CEO resigns amid accounting investigation", "negative"),
    ("Utility completes previously announced bond offering", "neutral"),
    ("Startup doubles revenue and turns first profit", "positive"),
    ("Mining company halts output after fatal accident", "negative"),
    ("Exchange announces holiday trading hours", "neutral"),
]
FEW_SHOT = [
    ("Profit beats estimates but SEC subpoena disclosed", "negative"),
    ("Record quarterly revenue lifts shares 8%", "positive"),
    ("Board confirms date of quarterly dividend payment", "neutral"),
]
LABELS = ["positive", "negative", "neutral"]
# Chat models often answer "Negative" rather than "negative": allow both spellings of each label.
label_ids = {tok.encode(v, add_special_tokens=False)[0]: label
             for label in LABELS for v in (label, label.capitalize())}
assert len(label_ids) == 6, "each spelling must start with a distinct token for single-step constrained decoding"
INSTRUCTION = ("Classify the sentiment of the financial headline as positive, negative or neutral. "
               "Legal or regulatory risk outweighs earnings news. Answer with one word.")


@torch.no_grad()
def classify(headline, examples):
    ids = chat_ids(lab.build_prompt(INSTRUCTION, examples, headline, "Headline", "Sentiment"))
    logits = model(ids).logits[0, -1].float().cpu().numpy()
    return label_ids[int(np.argmax(lab.constrain_to_choices(logits, list(label_ids))))]


for name, shots in [("zero-shot", []), ("one-shot", FEW_SHOT[:1]), ("few-shot", FEW_SHOT)]:
    preds = [classify(h, shots) for h, _ in DATA]
    acc = np.mean([p == y for p, (_, y) in zip(preds, DATA)])
    print(f"{name:9s} accuracy {acc:.0%}   (output always a valid label thanks to constrained decoding)")

# %% 4. Chain-of-thought + self-consistency (sampling T=0.8, majority vote)
QUESTIONS = [
    ("A node has 8 GPUs. A job needs 44 GPUs. How many nodes must be reserved?", "6"),
    ("A model has 7 billion parameters stored in 2 bytes each. How many GB of weights is that?", "14"),
    ("Batch 16 per GPU, 4 accumulation steps, 8 GPUs. What is the global batch size?", "512"),
]
for q, gold in QUESTIONS:
    ids = chat_ids(q + " Think step by step. End with a final line of the form: The answer is <number>.")
    direct = generate(chat_ids(q + " Reply with only the number."), max_new_tokens=8)
    samples = [generate(ids, max_new_tokens=320, temperature=0.8, top_p=0.95, seed=s) for s in range(5)]
    # Maths-tuned models often write \boxed{6}; normalise that into the format your extractor parses.
    samples = [re.sub(r"\\boxed\{([^}]*)\}", r"The answer is \1.", t) for t in samples]
    answer, agreement = lab.self_consistency(samples)
    print(f"gold={gold:>4} | direct={direct.strip()!r:>8} | self-consistency={answer!r} ({agreement:.0%} agree)")

# %% 5. Structured output: ask for JSON, parse robustly, retry once on failure
request = ('Headline: "Bank reports record profit but discloses regulator probe". Return JSON with keys '
           '"label" (positive|negative|neutral), "risk_flags" (list of strings) and "reasoning" (one sentence).')
for attempt in range(2):
    reply = generate(chat_ids(request, system="You output only valid JSON."), max_new_tokens=120)
    try:
        print(lab.parse_json_output(reply, ("label", "risk_flags")))
        break
    except ValueError as e:
        print(f"attempt {attempt + 1} failed ({e}); retrying with the error in the prompt")
        request += f"\nYour previous reply was invalid ({e}). Return only the JSON object."

# %% Optional: the same guided decoding through an OpenAI-compatible server (vLLM via `make llm-up`, or NIM)
if llm.available():
    print("server:", llm.describe())
    print(llm.chat(f"Sentiment of: {DATA[3][0]}", temperature=0, **llm.guided_choice(LABELS)))
else:
    print("[skipped] no LLM server: run `make llm-up` (vLLM on the GPU) or export NVIDIA_API_KEY")
