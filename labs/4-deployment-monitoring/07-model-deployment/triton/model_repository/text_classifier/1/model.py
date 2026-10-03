"""Triton Python backend: DistilBERT sentiment classifier.

With dynamic batching, Triton may hand `execute` several requests at once. We run them as ONE
batch on the GPU and then split the results back into one response per request.

Environment knobs used by the drills (unset = normal behaviour):
  SLOW_LOAD_SECONDS   sleep in initialize(): a model that takes long to load (Kubernetes startup probes)
  EXTRA_LATENCY_MS    sleep in every execute(): a slower "v2" for the canary drill
"""

import os
import time

import numpy as np
import torch
import triton_python_backend_utils as pb_utils
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_ID = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"


class TritonPythonModel:
    def initialize(self, args):
        # instance_group decides KIND_GPU / KIND_CPU, but a Python model only honours it if the code
        # reads it: Triton passes the kind and device id, and we place the model accordingly.
        kind, device_id = args["model_instance_kind"], args["model_instance_device_id"]
        self.device = f"cuda:{device_id}" if kind == "GPU" and torch.cuda.is_available() else "cpu"
        time.sleep(float(os.environ.get("SLOW_LOAD_SECONDS", "0")))
        self.extra_s = float(os.environ.get("EXTRA_LATENCY_MS", "0")) / 1000

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID).to(self.device).eval()
        if self.device != "cpu":
            self.model.half()
        self.id2label = self.model.config.id2label
        pb_utils.Logger.log_info(f"{args['model_instance_name']} on {self.device}")

    def execute(self, requests):
        texts, sizes = [], []
        for request in requests:
            arr = pb_utils.get_input_tensor_by_name(request, "TEXT").as_numpy()  # (batch, 1) of bytes
            batch = [row[0].decode() if isinstance(row[0], bytes) else str(row[0]) for row in arr]
            texts.extend(batch)
            sizes.append(len(batch))

        with torch.inference_mode():
            enc = self.tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(self.device)
            probs = self.model(**enc).logits.float().softmax(-1)
        scores, idx = probs.max(-1)
        scores, idx = scores.cpu().numpy(), idx.cpu().numpy()
        if self.extra_s:
            time.sleep(self.extra_s)

        responses, start = [], 0
        for n in sizes:
            labels = np.array([[self.id2label[int(i)].encode()] for i in idx[start : start + n]], dtype=np.object_)
            score = scores[start : start + n].reshape(-1, 1).astype(np.float32)
            responses.append(pb_utils.InferenceResponse(output_tensors=[
                pb_utils.Tensor("LABEL", labels),
                pb_utils.Tensor("SCORE", score),
            ]))
            start += n
        return responses
