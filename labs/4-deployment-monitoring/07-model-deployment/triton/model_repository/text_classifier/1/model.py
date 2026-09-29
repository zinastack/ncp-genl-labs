"""Triton Python backend: DistilBERT sentiment classifier.

With dynamic batching, Triton may hand `execute` several requests at once. We run them as ONE
batch on the GPU and then split the results back into one response per request.
"""

import numpy as np
import torch
import triton_python_backend_utils as pb_utils
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_ID = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"


class TritonPythonModel:
    def initialize(self, args):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID).to(self.device).eval()
        if self.device == "cuda":
            self.model.half()
        self.id2label = self.model.config.id2label

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
