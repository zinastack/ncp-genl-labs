"""Ensemble step 1: tokenize to a fixed length (128) so every request in a batch has the same shape."""

import numpy as np
import triton_python_backend_utils as pb_utils
from transformers import AutoTokenizer

MODEL_ID = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
SEQ_LEN = 128


class TritonPythonModel:
    def initialize(self, args):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    def execute(self, requests):
        responses = []
        for request in requests:
            arr = pb_utils.get_input_tensor_by_name(request, "TEXT").as_numpy()  # (batch, 1) bytes
            texts = [row[0].decode() if isinstance(row[0], bytes) else str(row[0]) for row in arr]
            enc = self.tokenizer(texts, padding="max_length", truncation=True, max_length=SEQ_LEN, return_tensors="np")
            responses.append(pb_utils.InferenceResponse(output_tensors=[
                pb_utils.Tensor("input_ids", enc["input_ids"].astype(np.int32)),
                pb_utils.Tensor("attention_mask", enc["attention_mask"].astype(np.int32)),
            ]))
        return responses
