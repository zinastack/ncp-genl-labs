# Lab 02 — Solution walkthrough

How each exercise works, why each line exists, worked numbers (computed by running
`solutions.py`), and the exam link. Try the exercise first, then read its section.

The picture: an LLM does not "answer" in one go. At every step it produces **logits**, one score
per vocabulary token, and a **decoding strategy** picks the next token. Prompt engineering controls
*what goes in*; decoding controls *what comes out*.

```
prompt text ──(chat template, ex. 1–2)──► token ids ──► model ──► logits (one per token)
                                                                    │
                    temperature / top-k / top-p / penalties / constraints (ex. 3–5)
                                                                    ▼
                                                             next token → repeat
answer text ──(parse: final answer, JSON, voting; ex. 6, 8)──► your program
```

---

## Exercise 1 — `build_prompt`: zero-, one- and few-shot prompts

**In-context learning:** the model learns the task from examples *inside the prompt*, with no
weight updates. The number of examples gives the name: 0 → zero-shot, 1 → one-shot, k → few-shot.

```
Classify the sentiment of the financial headline ...     ← instruction

Input: Shares soar after record earnings                 ← example 1
Output: positive

Input: Company misses revenue guidance                   ← example 2
Output: negative

Input: Fed holds rates steady                            ← the real query
Output:                                                  ← model continues from here
```

Why this exact layout matters:
- **Consistent labels** (`Input:` / `Output:`) let the model see the pattern and continue it.
- The prompt **ends with `Output:`** so the most natural next tokens are the answer. A trailing
  space or newline changes tokenisation and can hurt quality, which is why the test is strict.
- The code builds a list of blocks and joins them with `"\n\n"`, which is cleaner than adding
  separators by hand, and zero-shot falls out naturally when the example list is empty.

**Exam link:** zero-shot works because of **instruction tuning**. Few-shot helps with a specific
label set, style or edge cases. "No parameter updates allowed" rules out fine-tuning, so the answer
is few-shot, CoT or RAG.

---

## Exercise 2 — `to_chatml`: chat templates

Chat models are trained on conversations wrapped in **special tokens**. ChatML (Qwen and others) looks like:

```
<|im_start|>system
Be brief.<|im_end|>
<|im_start|>user
Hi<|im_end|>
<|im_start|>assistant
                    ← the "generation prompt": the model now writes the assistant turn
```

- Each message becomes `<|im_start|>{role}\n{content}<|im_end|>\n`.
- `add_generation_prompt=True` appends the opening of an assistant turn. **Forget it and the model
  may continue the user's message** instead of answering. That's a classic bug and an exam question.
- In practice you call `tokenizer.apply_chat_template(messages, add_generation_prompt=True)`.
  Every model family has its own template (Llama-3 uses `<|start_header_id|>` etc.), and you must
  use the one the model was trained with.

---

## Exercise 3 — temperature, top-k, top-p: controlling randomness

All three act on the logits before choosing a token. Running example (5 tokens):

```
logits = [2.0, 1.0, 0.5, -1.0, 3.0]
```

### `apply_temperature`: `softmax(logits / T)`

| T | probabilities | effect |
|---|---|---|
| 0.2 | `[0.007, 0.000, 0.000, 0.000, 0.993]` | almost always the top token |
| 0.5 | `[0.117, 0.016, 0.006, 0.000, 0.862]` | sharp |
| 1.0 | `[0.229, 0.084, 0.051, 0.011, 0.624]` | the model's own distribution |
| 2.0 | `[0.253, 0.154, 0.120, 0.056, 0.417]` | flatter |
| 5.0 | `[0.231, 0.189, 0.171, 0.127, 0.282]` | almost uniform, so random |

Dividing by a small T **stretches** the gaps between logits, and softmax then amplifies them.
Dividing by a large T **shrinks** the gaps. `T = 0` would divide by zero, so the code treats it as
**greedy**: one-hot on the argmax (`np.argmax`). That is the limit of `T → 0`.

### `top_k_filter`: keep the k best

`k=2` → `[2.0, -inf, -inf, -inf, 3.0]`. `np.argsort(logits)[-k:]` gives the indices of the k
largest. Everything else becomes `-inf`, so its probability is exactly 0 after softmax. The fixed k
ignores how confident the model is.

### `top_p_filter` (nucleus): keep just enough to reach probability p

Sort the probabilities descending and accumulate:

```
token      4      0      1      2      3
prob     0.624  0.229  0.084  0.051  0.011
cumsum   0.624  0.853  0.937  0.989  1.000
```

- `p = 0.7`: the first cumsum ≥ 0.7 is at position 1, so keep tokens 4 and 0.
- `p = 0.9`: keep 4, 0 and 1.

Code: `np.searchsorted(cumulative, p)` finds that first position, and `+1` makes it inclusive.
Keeping at least the top token is automatic, because even a tiny `p` keeps position 0. Unlike
top-k, **the set adapts**: a confident model keeps 1–2 tokens and an uncertain one keeps many.

> **Exam link:** extraction and classification use T≈0 (deterministic); creative writing uses
> T≈0.7–1.0 with top-p 0.9–0.95. Low temperature gives **consistency, not truth**. It does not fix
> hallucinations; RAG does.

---

## Exercise 4 — `apply_repetition_penalty`: discouraging loops

Small models love to repeat themselves. For every token already generated, the CTRL/Hugging Face
penalty makes it less likely:

```
logits [2.0, 1.0, 0.5, -1.0, 3.0], generated ids [0, 3, 0], penalty 2.0
token 0 (positive 2.0)  → 2.0 / 2 = 1.0
token 3 (negative -1.0) → -1.0 × 2 = -2.0
result [1.0, 1.0, 0.5, -2.0, 3.0]
```

- **Why divide positives but multiply negatives?** Either way the logit moves *down*. Dividing a
  negative number would move it *up* (−1/2 = −0.5), which would reward repetition.
- `set(generated_ids)` penalises each token once, however many times it appeared (token 0 appears twice).
- `.copy()`: never modify the caller's array. The test checks this.
- OpenAI-style *frequency* and *presence* penalties instead **subtract** from the logits (frequency scales with count).

---

## Exercise 5 — `constrain_to_choices`: guided decoding

If the only valid answers are `positive`, `negative` or `neutral`, don't *ask* the model nicely.
**Make every other token impossible:**

```
allowed ids [1, 2] → [-inf, 1.0, 0.5, -inf, -inf] → argmax is token 1
```

The output is valid **by construction**, whatever the model "wanted". This is what `guided_choice`,
`guided_json` and `guided_regex` do in NIM and vLLM at every step (JSON schemas become a grammar
that decides which tokens are allowed next). The GPU lab uses it to guarantee a valid label and
allows both "negative" and "Negative", because chat models often capitalise.

**Exam link:** "the parser crashes because of invalid JSON or extra labels" → constrained or guided decoding plus validation.

---

## Exercise 6 — `extract_final_answer` and `self_consistency`

### Chain-of-thought

Asking the model to *think step by step* lets it write intermediate results ("44 / 8 = 5.5, so 6
nodes") before answering. That helps multi-step problems, but you then need to **find the answer
inside the text**.

### The regex, piece by piece

```python
r"(?:the answer is|answer:)[ \t]*(.*?)[ \t]*(?:\.(?=\s|$)|$)"     flags: IGNORECASE | MULTILINE
```

| part | meaning |
|---|---|
| `(?:the answer is\|answer:)` | either phrase; `(?:…)` groups without capturing |
| `[ \t]*` | optional spaces after it |
| `(.*?)` | **the answer**: lazy, so it stops as early as possible |
| `(?:\.(?=\s\|$)\|$)` | stop at a period *followed by whitespace or end*, or at end of line |

The lazy match plus the "period followed by space" rule is what makes `"The answer is 3.5."`
give `3.5`, not `3`: the dot in `3.5` is followed by `5`, not a space. `MULTILINE` makes `$`
match at every line end. **The last match wins**, because models often self-correct ("…the answer
is 3. Recheck: Answer: 5.").

### Self-consistency = sample several times, then vote

```
"... The answer is 7."   → 7
"Answer: 7"              → 7
"The answer is 8."       → 8
"hmm"                    → (no answer, ignored)
"The answer is 7"        → 7
winner 7, agreement 3/4 = 0.75
```

`Counter.most_common(1)` does the vote (ties go to the first seen). Agreement is a free
**confidence score**: low agreement means the model is unsure.

> **Exam trap:** self-consistency needs **sampling (T > 0)**. With greedy decoding every sample is
> identical and voting adds nothing. In a test run of the GPU lab with Qwen2.5-0.5B, the direct
> answer to "44 GPUs / 8 per node" was 4 (wrong), while CoT with a 5-sample vote gave 6 (right).

---

## Exercise 7 — `select_examples`: dynamic few-shot

Instead of the same 5 examples for every query, pick the **most similar** labelled examples for
*this* query, using embeddings:

```
cosine(a, b) = a·b / (|a| · |b|)          ← compares direction (meaning), not length
```

Why cosine and not a plain dot product? With `q = [1, 0]` and examples `A = [10, 5]`, `B = [1, 0]`:
- dot product: A = 10, B = 1, so A "wins" only because it's a long vector;
- cosine: A = 0.894, B = 1.000, so B wins because it points the same way.

Code: normalise the query and every example to length 1, then one matrix-vector product gives all
cosines at once. `np.argsort(-sims)[:k]` returns the top k, best first.

**Exam link:** dynamic few-shot helps rare classes. Also balance labels and randomise order to
avoid majority-label and recency bias.

---

## Exercise 8 — `parse_json_output`: robust structured output

Models wrap JSON in prose and code fences:

````
Sure! Here you go:
```json
{"label": "negative", "confidence": 0.82, "why": "rate {hike}"}
```
````

A regex like `\{.*\}` breaks on braces inside strings (`"rate {hike}"`). Instead, let Python's
own JSON parser do the work:

```python
decoder = json.JSONDecoder()
for match in re.finditer(r"\{", text):          # try every "{" as a possible start
    obj, _ = decoder.raw_decode(text, match.start())   # parse ONE value from there, ignore the rest
```

`raw_decode` understands strings and nesting, so `{hike}` inside a string is fine. If parsing
fails at one `{`, move on to the next. After success, check the **required keys** and raise
`ValueError` if any is missing, so the caller can **retry** (the GPU lab retries once and puts the
error message into the next prompt).

---

## How this lab maps to exam questions

| If a question mentions… | Think… |
|---|---|
| no labelled examples, the model still does it | zero-shot, thanks to instruction tuning |
| no parameter updates, consistent and explainable | few-shot with curated examples + CoT with structured output |
| multi-step reasoning is wrong | chain-of-thought; self-consistency (needs T > 0) |
| outputs must be deterministic | T = 0 (greedy) or a fixed seed, stop sequences, max_tokens |
| invalid JSON or extra labels | guided/constrained decoding + validation |
| model continues the user's text | missing chat template or generation prompt |
| outdated or confident wrong facts | RAG with citations, not lower temperature |
| p-tuning / prompt tuning | trains soft prompts, so it is PEFT, not prompt engineering |
